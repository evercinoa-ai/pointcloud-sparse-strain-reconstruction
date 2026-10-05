import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from util import load_numeric_table_maybe_header


class DisplacementDatasetCached(Dataset):
    """
    只读取离线缓存：
      - cloud: <cache_root>/<split>/cloud_npz/<id>.npz  (xyz[S,3], idx[S])
      - disp : <cache_root>/<split>/disp_npy/<id>.npy   (disp[3*M] or [M,3])

    目录示例：
      data400_new/
        train/
          cloudpoint/
          displacement/
        test/
          cloudpoint/
          displacement/
        cache_np8944/
          train/
            cloud_npz/
            disp_npy/
          test/
            cloud_npz/
            disp_npy/

    设计说明：
      1) 不再使用 split_ratio 随机划分，而是显式读取 split 子目录；
      2) 训练集增强通过“是否纳入 *_augXX 缓存样本”来控制；
      3) 归一化统计量始终基于训练 split 的 clean 样本计算，避免受 aug_repeats 影响；
      4) val/test 默认只读取 clean 样本。
    """

    AUG_PATTERN = re.compile(r"_aug\d+$")

    def __init__(
        self,
        root_dir="data400_new",
        split="train",                  # train / val / test
        is_training_split=None,         # None -> fallback to split == "train"
        cache_root=None,                 # e.g. data400_new/cache_np8944
        disp_dir="displacement",        # 用于读取节点坐标 0.csv
        num_points=8944,
        disp_mean=(0.0, 0.0, 0.0),
        disp_std=(1.0, 1.0, 1.0),
        cloud_mean=None,
        cloud_std=None,
        augment=False,                   # 仅对 train 生效：是否纳入 *_augXX 样本
        seed=42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = str(split)
        self.split_root = os.path.join(root_dir, self.split)

        # cache_root 传“基目录”，内部自动拼 split
        self.cache_root = cache_root or os.path.join(root_dir, f"cache_np{num_points}")
        self.cloud_cache = os.path.join(self.cache_root, self.split, "cloud_npz")
        self.disp_cache = os.path.join(self.cache_root, self.split, "disp_npy")

        # 节点坐标从 <root>/<split>/displacement/0.csv 读取
        self.node_path = os.path.join(self.split_root, disp_dir)
        self.num_nodes = self._infer_num_nodes_from_node_csv()

        self.num_points = int(num_points)
        if is_training_split is None:
            self.is_training = (self.split == "train")
        else:
            self.is_training = bool(is_training_split)
        self.augment = bool(augment) and self.is_training
        self.seed = int(seed)

        # 约束点（与你原代码保持一致）
        self.constraint_points = np.array(
            [[100.0, 100.0, 3000.0],
             [100.0, 100.0, 0.0]],
            dtype=np.float32
        )

        self._validate_cache_files()
        self._build_sample_ids()

        # 统计量：训练集负责计算；验证/测试集用训练集传入
        self.disp_mean = self._as_vec3(disp_mean, "disp_mean")
        self.disp_std = self._as_vec3(disp_std, "disp_std")
        self.cloud_mean = None if cloud_mean is None else np.array(cloud_mean, dtype=np.float32)
        self.cloud_std = None if cloud_std is None else np.array(cloud_std, dtype=np.float32)

        if self.is_training:
            self._compute_global_stats()

        self.node_data = self._load_node_data()  # [4,M] (CPU tensor)

    @staticmethod
    def _as_vec3(x, name: str) -> np.ndarray:
        """允许传入 float / list[3] / np.ndarray(3,)"""
        arr = np.array(x, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3:
            raise ValueError(f"{name} 必须是标量或长度为3的向量，当前 shape={arr.shape}, value={x}")
        return arr.astype(np.float32)

    @classmethod
    def _is_augmented_id(cls, sid: str) -> bool:
        return bool(cls.AUG_PATTERN.search(sid))

    def _infer_num_nodes_from_node_csv(self) -> int:
        disp_files = sorted([f for f in os.listdir(self.node_path) if f.endswith(".csv")])
        if len(disp_files) == 0:
            raise RuntimeError(f"未在 {self.node_path} 中找到任何 displacement csv")
        node_file = os.path.join(self.node_path, disp_files[0])
        node_all = load_numeric_table_maybe_header(node_file, dtype=np.float32)
        if node_all.ndim != 2 or node_all.shape[1] < 4:
            raise RuntimeError(f"node csv 列数异常: {node_all.shape}, file={node_file}")
        return int(node_all.shape[0])

    def _validate_cache_files(self):
        if (not os.path.isdir(self.cloud_cache)) or (not os.path.isdir(self.disp_cache)):
            raise RuntimeError(
                f"缓存目录不存在：\n"
                f"  cloud_cache={self.cloud_cache}\n"
                f"  disp_cache={self.disp_cache}\n"
                f"请先运行 preprocess_fps_cache.py 生成对应 split 的缓存。"
            )

        self.ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.cloud_cache) if f.endswith(".npz")])
        disp_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.disp_cache) if f.endswith(".npy")])

        if self.ids != disp_ids:
            raise RuntimeError("cloud_npz 与 disp_npy 的文件 id 不一致（请检查预处理是否完整）")

        if len(self.ids) == 0:
            raise RuntimeError("缓存为空：未找到任何 .npz/.npy")

        self.clean_ids = [sid for sid in self.ids if not self._is_augmented_id(sid)]
        self.aug_ids = [sid for sid in self.ids if self._is_augmented_id(sid)]

        if len(self.clean_ids) == 0:
            raise RuntimeError("未找到任何 clean 样本（期望至少存在不带 _augXX 后缀的缓存文件）")

    def _build_sample_ids(self):
        # train: augment=True -> clean + aug；augment=False -> 仅 clean
        # val/test: 始终只读 clean
        if self.is_training and self.augment:
            self.sample_ids = list(self.clean_ids) + list(self.aug_ids)
            if len(self.aug_ids) == 0:
                print("[WARN] 当前 train 缓存中未找到 *_augXX 文件，augment=True 但实际只会使用 clean 样本。")
        else:
            self.sample_ids = list(self.clean_ids)

        print(
            f"split={self.split} | clean={len(self.clean_ids)} | aug={len(self.aug_ids)} | "
            f"实际使用样本数={len(self.sample_ids)} | augment={'on' if self.augment else 'off'}"
        )

    def __len__(self):
        return len(self.sample_ids)

    def _compute_global_stats(self):
        """
        统计量只基于 train split 的 clean 样本计算：
        - 避免 aug_repeats 改变 cloud_mean/std
        - disp_mean/std 不会因重复增强样本被重复计权
        """
        sum_xyz = np.zeros(3, dtype=np.float64)
        sumsq_xyz = np.zeros(3, dtype=np.float64)
        count_xyz = 0

        sum_disp = np.zeros(3, dtype=np.float64)
        sumsq_disp = np.zeros(3, dtype=np.float64)
        count_disp = 0

        print("正在基于训练集 clean 缓存计算统计量...")

        for sid in self.clean_ids:
            cloud_file = os.path.join(self.cloud_cache, f"{sid}.npz")
            disp_file = os.path.join(self.disp_cache, f"{sid}.npy")

            xyz = np.load(cloud_file)["xyz"].astype(np.float32)  # [S,3]
            sum_xyz += xyz.sum(axis=0, dtype=np.float64)
            sumsq_xyz += (xyz * xyz).sum(axis=0, dtype=np.float64)
            count_xyz += xyz.shape[0]

            disp = np.load(disp_file).astype(np.float32)

            if disp.ndim == 1:
                if disp.shape[0] != self.num_nodes * 3:
                    raise RuntimeError(
                        f"disp 长度与 num_nodes 不匹配: disp={disp.shape}, num_nodes={self.num_nodes} "
                        f"(期望 {self.num_nodes * 3}) | file={disp_file}"
                    )
                disp3 = disp.reshape(self.num_nodes, 3)
            elif disp.ndim == 2:
                if disp.shape != (self.num_nodes, 3):
                    raise RuntimeError(
                        f"disp shape 异常: disp={disp.shape}, 期望 ({self.num_nodes},3) | file={disp_file}"
                    )
                disp3 = disp
            else:
                raise RuntimeError(f"disp ndim 异常: {disp.shape} | file={disp_file}")

            sum_disp += disp3.sum(axis=0, dtype=np.float64)
            sumsq_disp += (disp3 * disp3).sum(axis=0, dtype=np.float64)
            count_disp += disp3.shape[0]

        mean_xyz = sum_xyz / max(1, count_xyz)
        var_xyz = sumsq_xyz / max(1, count_xyz) - mean_xyz * mean_xyz
        var_xyz = np.maximum(var_xyz, 1e-12)
        std_xyz = np.sqrt(var_xyz)

        self.cloud_mean = mean_xyz.astype(np.float32)
        self.cloud_std = std_xyz.astype(np.float32)

        mean_disp = sum_disp / max(1, count_disp)
        var_disp = sumsq_disp / max(1, count_disp) - mean_disp * mean_disp
        var_disp = np.maximum(var_disp, 1e-12)
        std_disp = np.sqrt(var_disp)

        self.disp_mean = mean_disp.astype(np.float32)
        self.disp_std = std_disp.astype(np.float32)

        print(f"点云统计量 | mean={self.cloud_mean} | std={self.cloud_std}")
        print(f"位移统计量(逐轴) | mean={self.disp_mean} mm | std={self.disp_std} mm")

    def _min_dist_to_constraints(self, pts_norm):
        """
        pts_norm: torch.Tensor [N,3]，已归一化坐标
        返回： [N,1] 最近约束点距离（欧氏距离）
        """
        c = (self.constraint_points - self.cloud_mean) / (self.cloud_std + 1e-8)  # [2,3]
        c0 = torch.tensor(c[0], dtype=pts_norm.dtype)
        c1 = torch.tensor(c[1], dtype=pts_norm.dtype)
        d0 = ((pts_norm - c0) ** 2).sum(dim=1)
        d1 = ((pts_norm - c1) ** 2).sum(dim=1)
        return torch.sqrt(torch.minimum(d0, d1) + 1e-12).unsqueeze(1)

    def _load_node_data(self):
        disp_files = sorted([f for f in os.listdir(self.node_path) if f.endswith(".csv")])
        if len(disp_files) == 0:
            raise RuntimeError(f"未在 {self.node_path} 中找到任何 displacement csv")
        node_file = os.path.join(self.node_path, disp_files[0])
        node_all = load_numeric_table_maybe_header(node_file, dtype=np.float32)
        node_xyz = node_all[:, 1:4]  # NodeIds,x,y,z,...

        if node_xyz.shape[0] != self.num_nodes:
            raise RuntimeError(f"node_xyz 行数与 num_nodes 不一致: {node_xyz.shape[0]} vs {self.num_nodes}")

        node = torch.from_numpy(node_xyz).float()
        node = (node - torch.from_numpy(self.cloud_mean)) / (torch.from_numpy(self.cloud_std) + 1e-8)

        min_dist = self._min_dist_to_constraints(node)
        node = torch.cat([node, min_dist], dim=1)  # [M,4]
        return node.permute(1, 0).contiguous()  # [4,M]

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        cloud_file = os.path.join(self.cloud_cache, f"{sid}.npz")
        disp_file = os.path.join(self.disp_cache, f"{sid}.npy")

        # cloud
        xyz = np.load(cloud_file)["xyz"].astype(np.float32)  # [S,3]
        pts = torch.from_numpy(xyz).float()

        # 注意：随机噪声 + 随机挖空已经在 preprocess_fps_cache.py 中于 FPS 前完成。
        # 这里不再做在线增强，避免与离线缓存增强叠加。
        pts = (pts - torch.from_numpy(self.cloud_mean)) / (torch.from_numpy(self.cloud_std) + 1e-8)
        min_dist = self._min_dist_to_constraints(pts)
        pts = torch.cat([pts, min_dist], dim=1)  # [S,4]
        cloud = pts.permute(1, 0).contiguous()   # [4,S]

        # disp
        disp = np.load(disp_file).astype(np.float32)

        if disp.ndim == 1:
            if disp.shape[0] != self.num_nodes * 3:
                raise RuntimeError(
                    f"disp 长度异常: {disp.shape}, 期望 [{self.num_nodes * 3}] (=M*3), M={self.num_nodes} | file={disp_file}"
                )
            disp_t = torch.from_numpy(disp).float().view(self.num_nodes, 3)  # [M,3]
        elif disp.ndim == 2:
            if disp.shape != (self.num_nodes, 3):
                raise RuntimeError(
                    f"disp shape 异常: {disp.shape}, 期望 ({self.num_nodes},3) | file={disp_file}"
                )
            disp_t = torch.from_numpy(disp).float()
        else:
            raise RuntimeError(f"disp ndim 异常: {disp.shape} | file={disp_file}")

        mu = torch.from_numpy(self.disp_mean).view(1, 3)
        sig = torch.from_numpy(self.disp_std).view(1, 3)
        disp_t = (disp_t - mu) / (sig + 1e-8)

        return cloud, disp_t, self.node_data
