
"""
utilV1,修改了下采样算法sample and groud
"""

import torch
import torch.nn.functional as F
import numpy as np
import io
import warnings

try:
    from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
except (ImportError, OSError):
    pointnet2_utils = None


@torch.jit.script
def _furthest_point_sample_torch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Deterministic FPS fallback matching PointNet++ CUDA tie ordering."""
    batch_size, num_points, _ = xyz.shape
    centroids = torch.empty(
        (batch_size, npoint), dtype=torch.long, device=xyz.device
    )
    min_dist = torch.full(
        (batch_size, num_points),
        float("inf"),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)

    # PointNet++ assigns strided point indices to a power-of-two CUDA block and
    # then reduces per-thread maxima.  Reproducing that ordering matters for
    # symmetric point clouds, where several points can have identical distance.
    block_size = 1
    while block_size * 2 <= num_points and block_size < 512:
        block_size *= 2
    row_count = (num_points + block_size - 1) // block_size
    pad_count = row_count * block_size - num_points

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        delta = xyz - centroid
        dist = (
            delta[:, :, 0] * delta[:, :, 0]
            + delta[:, :, 1] * delta[:, :, 1]
            + delta[:, :, 2] * delta[:, :, 2]
        )
        magnitude = (
            xyz[:, :, 0] * xyz[:, :, 0]
            + xyz[:, :, 1] * xyz[:, :, 1]
            + xyz[:, :, 2] * xyz[:, :, 2]
        )
        valid = magnitude > 1.0e-3
        min_dist = torch.where(valid, torch.minimum(min_dist, dist), min_dist)
        candidates = torch.where(valid, min_dist, torch.full_like(min_dist, -1.0))
        work = F.pad(candidates, (0, pad_count), value=-1.0).view(
            batch_size, row_count, block_size
        )
        thread_best, thread_row = torch.max(work, dim=1)
        _, thread_id = torch.max(thread_best, dim=1)
        farthest = thread_row[batch_indices, thread_id] * block_size + thread_id
    return centroids


_FALLBACK_WARNING_EMITTED = False


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    global _FALLBACK_WARNING_EMITTED
    if pointnet2_utils is not None:
        return pointnet2_utils.furthest_point_sample(xyz, npoint).long()
    if not _FALLBACK_WARNING_EMITTED:
        warnings.warn(
            "pointnet2_ops_lib is unavailable; using the deterministic PyTorch "
            "FPS fallback. Inference remains valid but preprocessing is slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FALLBACK_WARNING_EMITTED = True
    return _furthest_point_sample_torch(xyz, int(npoint))


def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    关键修改：几何距离/FPS/kNN 只使用 xyz[..., :3]。

    xyz:    [B, N, C] (C>=3)
    points: [B, N, D]
    return:
        new_xyz:    [B, S, 3]
        new_points: [B, S, nsample, 2D]（保持你原来的特征拼接方式）
    """
    B, N, C = xyz.shape
    S = npoint
    if C < 3:
        raise ValueError(f"xyz last dim must be >=3, got {C}")

    xyz_pos = xyz[:, :, :3].contiguous()

    fps_idx = furthest_point_sample(xyz_pos, npoint)  # [B,S]
    new_xyz = index_points(xyz_pos, fps_idx)        # [B,S,3]
    center_points = index_points(points, fps_idx)   # [B,S,D]

    idx = knn_point(nsample, xyz_pos, new_xyz)      # [B,S,nsample]
    grouped_points = index_points(points, idx)      # [B,S,nsample,D]
    grouped_points_norm = grouped_points - center_points.view(B, S, 1, -1)

    new_points = torch.cat(
        [
            grouped_points_norm,
            center_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1),
        ],
        dim=-1,
    )
    return new_xyz, new_points

def load_numeric_table_maybe_header(path: str, dtype=np.float32) -> np.ndarray:
    """
    兼容：
    - 表头（含字母）：自动跳过首行
    - // 或 # 开头的注释行：自动跳过
    - 分隔符：自动识别逗号 / 空白(含Tab)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = []
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("//") or s.startswith("#"):
                continue
            raw.append(s)

    if not raw:
        raise ValueError(f"空文件或全是注释: {path}")

    # 识别表头：首个有效行含字母 -> 认为是表头
    first = raw[0]
    if any(ch.isalpha() for ch in first):
        raw = raw[1:]
        if not raw:
            raise ValueError(f"只有表头没有数据: {path}")

    # 识别分隔符：有逗号就按逗号，否则按空白(空格/Tab)
    first_data = raw[0]
    delim = "," if ("," in first_data) else None

    return np.loadtxt(io.StringIO("\n".join(raw)), delimiter=delim, dtype=dtype)
