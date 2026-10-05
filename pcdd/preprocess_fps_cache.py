import os
import argparse
import numpy as np
import torch
from util import furthest_point_sample, load_numeric_table_maybe_header


def list_csv(dir_path):
    return sorted([f for f in os.listdir(dir_path) if f.endswith(".csv")])


def ensure_match(cloud_files, disp_files):
    if len(cloud_files) != len(disp_files):
        raise RuntimeError(f"数量不一致: cloud={len(cloud_files)} disp={len(disp_files)}")
    for c, d in zip(cloud_files, disp_files):
        if c != d:
            raise RuntimeError(f"文件名不匹配: {c} vs {d}")


def add_gaussian_noise(xyz, std_mm):
    """对点坐标加入高斯噪声，单位与原始坐标一致（你当前是 mm）。"""
    if std_mm <= 0:
        return xyz.astype(np.float32, copy=True)
    noise = np.random.randn(*xyz.shape).astype(np.float32) * float(std_mm)
    return (xyz.astype(np.float32, copy=False) + noise).astype(np.float32)


def random_holes(
    xyz,
    num_holes_min=1,
    num_holes_max=3,
    radius_min_mm=30.0,
    radius_max_mm=80.0,
    min_keep_points=None,
):
    """
    随机局部挖空：
      - 随机选若干中心点；
      - 将其邻域内点全部删除；
      - 用于模拟局部遮挡。

    注意：本函数应在 FPS 之前调用。
    """
    pts = xyz.astype(np.float32, copy=True)
    if pts.shape[0] == 0:
        return pts

    nh = np.random.randint(int(num_holes_min), int(num_holes_max) + 1)

    for _ in range(nh):
        if pts.shape[0] <= 1:
            break

        center = pts[np.random.randint(0, pts.shape[0])]
        radius = float(np.random.uniform(radius_min_mm, radius_max_mm))

        d2 = np.sum((pts - center[None, :]) ** 2, axis=1)
        keep_mask = d2 > (radius * radius)

        # 防止一次删得过多，导致后续不足以做 FPS
        if min_keep_points is not None and int(keep_mask.sum()) < int(min_keep_points):
            continue

        pts = pts[keep_mask]

    return pts


def save_variant(xyz_use, disp, out_cloud_file, out_disp_file, num_points, device):
    if xyz_use.shape[0] < num_points:
        raise RuntimeError(f"点数不足: N={xyz_use.shape[0]} < num_points={num_points}")

    xyz_t = torch.from_numpy(xyz_use).to(device).contiguous().unsqueeze(0)  # [1,N,3]
    # Prefer the compiled PointNet++ operator when available; otherwise use
    # the deterministic PyTorch implementation provided by util.py.
    idx = furthest_point_sample(xyz_t, num_points)  # [1,S]
    idx = idx.squeeze(0).long().detach().cpu().numpy()  # [S]

    sampled = xyz_use[idx].astype(np.float32)
    np.savez_compressed(out_cloud_file, xyz=sampled, idx=idx)
    np.save(out_disp_file, disp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data400_new")
    ap.add_argument("--split", type=str, default="train", help="train / val / test")
    ap.add_argument("--cloud_dir", type=str, default="cloudpoint")
    ap.add_argument("--disp_dir", type=str, default="displacement")
    ap.add_argument("--num_points", type=int, default=8944)
    ap.add_argument("--out_dir", type=str, default=None, help="默认: <root>/cache_np<num_points>/<split>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Optional comma-separated sample IDs, for example: 160,170,199",
    )

    # 增强总开关：仅建议对 train split 打开
    ap.add_argument("--enable_aug", action="store_true", help="是否额外生成 *_augXX 增强缓存")
    ap.add_argument("--aug_repeats", type=int, default=3, help="每个原始样本额外生成多少份增强样本")
    ap.add_argument("--noise_std_mm", type=float, default=1.0, help="高斯噪声标准差，单位 mm")
    ap.add_argument("--hole_num_min", type=int, default=1)
    ap.add_argument("--hole_num_max", type=int, default=3)
    ap.add_argument("--hole_radius_min_mm", type=float, default=30.0)
    ap.add_argument("--hole_radius_max_mm", type=float, default=80.0)
    ap.add_argument("--min_keep_ratio", type=float, default=1.2, help="挖空后至少保留 num_points * ratio 个点")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 才能运行 FPS 预处理（未检测到 CUDA）")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    split_root = os.path.join(args.root, args.split)
    cloud_path = os.path.join(split_root, args.cloud_dir)
    disp_path = os.path.join(split_root, args.disp_dir)

    cloud_files = list_csv(cloud_path)
    disp_files = list_csv(disp_path)
    if args.sample_ids:
        requested = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        available = {os.path.splitext(name)[0] for name in cloud_files}
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"requested sample IDs were not found: {missing}")
        cloud_files = [name for name in cloud_files if os.path.splitext(name)[0] in requested]
        disp_files = [name for name in disp_files if os.path.splitext(name)[0] in requested]
    ensure_match(cloud_files, disp_files)

    out_root = args.out_dir or os.path.join(args.root, f"cache_np{args.num_points}", args.split)
    out_cloud = os.path.join(out_root, "cloud_npz")
    out_disp = os.path.join(out_root, "disp_npy")
    os.makedirs(out_cloud, exist_ok=True)
    os.makedirs(out_disp, exist_ok=True)

    device = torch.device("cuda")

    ok = 0
    skip = 0
    fail = 0

    if len(disp_files) == 0:
        raise RuntimeError(f"未在 {disp_path} 中找到任何 displacement csv")
    ref_node = load_numeric_table_maybe_header(os.path.join(disp_path, disp_files[0]), dtype=np.float32)
    ref_M = int(ref_node.shape[0])

    for fname in cloud_files:
        in_cloud = os.path.join(cloud_path, fname)
        in_disp = os.path.join(disp_path, fname)
        base = os.path.splitext(fname)[0]

        try:
            # 1) 读取原始点云
            xyz_all = load_numeric_table_maybe_header(in_cloud, dtype=np.float32)
            xyz_raw = xyz_all[:, :3]  # [N,3]
            if xyz_raw.shape[0] < args.num_points:
                raise RuntimeError(f"原始点数不足: N={xyz_raw.shape[0]} < num_points={args.num_points}")

            # 2) 读取对应位移真值
            disp_all = load_numeric_table_maybe_header(in_disp, dtype=np.float32)
            disp = disp_all[:, 4:7]  # [M,3]
            if disp.shape[0] != ref_M:
                raise RuntimeError(f"节点行数不一致: file={fname} M={disp.shape[0]} ref_M={ref_M}")
            disp = disp.reshape(-1).astype(np.float32)  # [3*M]

            # 3) 先构造 clean 版本；若 enable_aug，再构造 aug 版本
            variants = [(base, xyz_raw)]

            # 仅当打开增强时，额外生成 *_augXX
            if args.enable_aug:
                min_keep_points = max(args.num_points, int(round(args.num_points * args.min_keep_ratio)))
                for k in range(args.aug_repeats):
                    xyz_aug = add_gaussian_noise(xyz_raw, args.noise_std_mm)
                    xyz_aug = random_holes(
                        xyz_aug,
                        num_holes_min=args.hole_num_min,
                        num_holes_max=args.hole_num_max,
                        radius_min_mm=args.hole_radius_min_mm,
                        radius_max_mm=args.hole_radius_max_mm,
                        min_keep_points=min_keep_points,
                    )
                    variants.append((f"{base}_aug{k + 1:02d}", xyz_aug))

            # 4) 对每个版本执行 FPS 并保存
            for save_id, xyz_use in variants:
                out_cloud_file = os.path.join(out_cloud, f"{save_id}.npz")
                out_disp_file = os.path.join(out_disp, f"{save_id}.npy")

                if (not args.overwrite) and os.path.exists(out_cloud_file) and os.path.exists(out_disp_file):
                    skip += 1
                    continue

                save_variant(
                    xyz_use=xyz_use,
                    disp=disp,
                    out_cloud_file=out_cloud_file,
                    out_disp_file=out_disp_file,
                    num_points=args.num_points,
                    device=device,
                )
                ok += 1

                if ok % 20 == 0:
                    print(f"[OK] {ok} processed... last={save_id}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {fname}: {e}")

    print("\n==== preprocess done ====")
    print(f"split   : {args.split}")
    print(f"out_root: {out_root}")
    print(f"augment : {'on' if args.enable_aug else 'off'}")
    print(f"ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
