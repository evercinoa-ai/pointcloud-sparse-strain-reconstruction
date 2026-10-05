import os
import re
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_reg_U import PctRegressor
from util import load_numeric_table_maybe_header


PLOT_FONT_FAMILY = '"Times New Roman", SimSun, serif'
PLOT_FONT_SIZE = 14
PLOT_TITLE_FONT_SIZE = PLOT_FONT_SIZE + 2
COLORBAR_TICK_FONT_SIZE = PLOT_FONT_SIZE
COLORBAR_TITLE_FONT_SIZE = PLOT_FONT_SIZE + 2


AUG_PATTERN = re.compile(r"_aug\d+$")


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def list_csv(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到目录: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort(key=natural_key)
    return files


def find_template_disp_csv(data_root, split, disp_dir="displacement"):
    disp_root = os.path.join(data_root, split, disp_dir)
    disp_files = list_csv(disp_root)
    if len(disp_files) == 0:
        raise RuntimeError(f"未在 {disp_root} 中找到任何 displacement csv")
    return os.path.join(disp_root, disp_files[0])


def resolve_cache_base(data_root, num_points, cache_root_from_cfg=None, cache_root_from_cli=None):
    if cache_root_from_cli:
        return cache_root_from_cli
    if cache_root_from_cfg:
        return cache_root_from_cfg
    return os.path.join(data_root, f"cache_np{num_points}")


def resolve_split_cache_root(cache_base, split):
    """
    兼容两种写法：
    1) cache_base = data/cache_np8944           -> 实际 split 缓存为 data/cache_np8944/train
    2) cache_base = data/cache_np8944/train     -> 已经是 split 根目录
    """
    split_cloud = os.path.join(cache_base, split, "cloud_npz")
    split_disp = os.path.join(cache_base, split, "disp_npy")
    if os.path.isdir(split_cloud) and os.path.isdir(split_disp):
        return os.path.join(cache_base, split)

    direct_cloud = os.path.join(cache_base, "cloud_npz")
    direct_disp = os.path.join(cache_base, "disp_npy")
    if os.path.isdir(direct_cloud) and os.path.isdir(direct_disp):
        return cache_base

    raise FileNotFoundError(
        f"未找到有效缓存目录。尝试过：\n"
        f"  1) {os.path.join(cache_base, split)}\n"
        f"  2) {cache_base}"
    )


def list_sample_ids(cache_split_root, include_aug=True):
    cloud_dir = os.path.join(cache_split_root, "cloud_npz")
    disp_dir = os.path.join(cache_split_root, "disp_npy")
    if not os.path.isdir(cloud_dir):
        raise FileNotFoundError(f"找不到 cloud_npz: {cloud_dir}")
    if not os.path.isdir(disp_dir):
        raise FileNotFoundError(f"找不到 disp_npy: {disp_dir}")

    cloud_ids = [os.path.splitext(fn)[0] for fn in os.listdir(cloud_dir) if fn.endswith(".npz")]
    disp_ids = [os.path.splitext(fn)[0] for fn in os.listdir(disp_dir) if fn.endswith(".npy")]
    cloud_ids.sort(key=natural_key)
    disp_ids.sort(key=natural_key)

    if cloud_ids != disp_ids:
        raise RuntimeError("cloud_npz 与 disp_npy 的文件 id 不一致（请检查预处理是否完整）")

    if not include_aug:
        cloud_ids = [sid for sid in cloud_ids if not AUG_PATTERN.search(sid)]

    if not cloud_ids:
        raise RuntimeError(f"缓存为空：{cache_split_root}")
    return cloud_ids


def load_cached_sample(cache_split_root, sample_id, num_nodes):
    cloud_npz = os.path.join(cache_split_root, "cloud_npz", f"{sample_id}.npz")
    disp_npy = os.path.join(cache_split_root, "disp_npy", f"{sample_id}.npy")
    if not os.path.exists(cloud_npz):
        raise FileNotFoundError(f"找不到: {cloud_npz}")
    if not os.path.exists(disp_npy):
        raise FileNotFoundError(f"找不到: {disp_npy}")

    xyz = np.load(cloud_npz)["xyz"].astype(np.float32)  # [S,3]

    disp = np.load(disp_npy).astype(np.float32)
    if disp.ndim == 1:
        if disp.shape[0] != num_nodes * 3:
            raise RuntimeError(f"disp shape 异常: {disp.shape}, 期望 [{num_nodes * 3}] (=M*3), M={num_nodes}")
        disp = disp.reshape(num_nodes, 3)
    elif disp.ndim == 2:
        if disp.shape != (num_nodes, 3):
            raise RuntimeError(f"disp shape 异常: {disp.shape}, 期望 [{num_nodes},3]")
    else:
        raise RuntimeError(f"disp shape 异常: {disp.shape}")

    return xyz, disp


def load_node_xyz_mm(data_root, split, disp_dir="displacement"):
    node_file = find_template_disp_csv(data_root, split, disp_dir=disp_dir)
    node_all = load_numeric_table_maybe_header(node_file, dtype=np.float32)
    if node_all.ndim != 2 or node_all.shape[1] < 4:
        raise RuntimeError(f"节点文件列数异常: {node_all.shape} | file={node_file}")
    node_xyz = node_all[:, 1:4]
    return node_xyz, node_file


def min_dist_feature_norm(xyz_norm, cloud_mean, cloud_std):
    constraint_points_mm = np.array([[100.0, 100.0, 3000.0],
                                     [100.0, 100.0, 0.0]], dtype=np.float32)
    c_norm = (constraint_points_mm - cloud_mean) / (cloud_std + 1e-8)
    d0 = np.sum((xyz_norm - c_norm[0]) ** 2, axis=1)
    d1 = np.sum((xyz_norm - c_norm[1]) ** 2, axis=1)
    return np.sqrt(np.minimum(d0, d1) + 1e-12).astype(np.float32)[:, None]


def make_cloud_input(xyz_mm, cloud_mean, cloud_std):
    xyz_norm = (xyz_mm - cloud_mean) / (cloud_std + 1e-8)
    md = min_dist_feature_norm(xyz_norm, cloud_mean, cloud_std)
    feat = np.concatenate([xyz_norm, md], axis=1)
    return feat.T.astype(np.float32)


def make_node_input(node_xyz_mm, cloud_mean, cloud_std):
    node_norm = (node_xyz_mm - cloud_mean) / (cloud_std + 1e-8)
    md = min_dist_feature_norm(node_norm, cloud_mean, cloud_std)
    feat = np.concatenate([node_norm, md], axis=1)
    return feat.T.astype(np.float32)


def denorm_disp(pred_norm, disp_mean, disp_std):
    return pred_norm * disp_std.reshape(1, 3) + disp_mean.reshape(1, 3)


def compute_metrics_mm(pred_mm, true_mm):
    diff = pred_mm - true_mm
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = float(np.sqrt(np.mean(diff ** 2)))
    mae_axis = np.mean(np.abs(diff), axis=0).tolist()
    rmse_axis = np.sqrt(np.mean(diff ** 2, axis=0)).tolist()
    mag_err = np.linalg.norm(diff, axis=1)
    return {
        "MAE_all(mm)": mae_all,
        "RMSE_all(mm)": rmse_all,
        "MAE_axis(mm)": mae_axis,
        "RMSE_axis(mm)": rmse_axis,
        "MAE_|dU|(mm)": float(np.mean(mag_err)),
        "RMSE_|dU|(mm)": float(np.sqrt(np.mean(mag_err ** 2))),
    }


def build_args_from_config(cfg):
    return SimpleNamespace(**cfg)


def make_compare_figure(node_xyz_mm, u_true, u_pred, title):
    x, y, z = node_xyz_mm[:, 0], node_xyz_mm[:, 1], node_xyz_mm[:, 2]
    u_err = u_pred - u_true

    abs_err = np.abs(u_err)
    mag_true = np.linalg.norm(u_true, axis=1)
    mag_pred = np.linalg.norm(u_pred, axis=1)
    mag_du = np.linalg.norm(u_err, axis=1)

    fields = {
        "Ux": (u_true[:, 0], u_pred[:, 0], abs_err[:, 0]),
        "Uy": (u_true[:, 1], u_pred[:, 1], abs_err[:, 1]),
        "Uz": (u_true[:, 2], u_pred[:, 2], abs_err[:, 2]),
        "|U|": (mag_true, mag_pred, np.abs(mag_pred - mag_true)),
        "|ΔU|": (mag_true, mag_pred, mag_du),
    }

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("True", "Pred", "Abs Error"),
        horizontal_spacing=0.02
    )

    k0 = "Uz"
    c_true, c_pred, c_eabs = fields[k0]

    vmin_tp = float(min(c_true.min(), c_pred.min()))
    vmax_tp = float(max(c_true.max(), c_pred.max()))
    vmax_e = float(max(c_eabs.max(), 1e-12))

    ids = np.arange(node_xyz_mm.shape[0])
    hover = (
        "id=%{customdata}<br>"
        "x=%{x:.3f} mm<br>y=%{y:.3f} mm<br>z=%{z:.3f} mm<br>"
        "val=%{marker.color:.6f} mm<extra></extra>"
    )

    cb_font = dict(family=PLOT_FONT_FAMILY, size=COLORBAR_TITLE_FONT_SIZE)
    cb_tickfont = dict(family=PLOT_FONT_FAMILY, size=COLORBAR_TICK_FONT_SIZE)
    plot_font = dict(family=PLOT_FONT_FAMILY, size=PLOT_FONT_SIZE)
    title_font = dict(family=PLOT_FONT_FAMILY, size=PLOT_TITLE_FONT_SIZE)
    cb_tp = dict(
        x=0.70,
        y=0.5,
        len=0.85,
        thickness=18,
        xanchor="left",
        tickfont=cb_tickfont,
    )
    cb_e = dict(
        x=1.03,
        y=0.5,
        len=0.85,
        thickness=18,
        xanchor="left",
        tickfont=cb_tickfont,
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=3, color=c_true, cmin=vmin_tp, cmax=vmax_tp, colorscale="Viridis", showscale=False),
            customdata=ids,
            hovertemplate=hover,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=3,
                color=c_pred,
                cmin=vmin_tp,
                cmax=vmax_tp,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text=f"{k0} (mm)", font=cb_font), **cb_tp),
            ),
            customdata=ids,
            hovertemplate=hover,
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=3,
                color=c_eabs,
                cmin=0.0,
                cmax=vmax_e,
                colorscale="Jet",
                showscale=True,
                colorbar=dict(title=dict(text="|err| (mm)", font=cb_font), **cb_e),
            ),
            customdata=ids,
            hovertemplate=hover,
        ),
        row=1, col=3
    )

    buttons = []
    for name, (ct, cp, ceabs) in fields.items():
        vmin_tp = float(min(ct.min(), cp.min()))
        vmax_tp = float(max(ct.max(), cp.max()))
        vmax_e = float(max(ceabs.max(), 1e-12))
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {
                        "marker.color": [ct, cp, ceabs],
                        "marker.cmin": [vmin_tp, vmin_tp, 0.0],
                        "marker.cmax": [vmax_tp, vmax_tp, vmax_e],
                    },
                    {
                        "title": title + f" | Field={name}",
                        "annotations": [
                            dict(text="True", x=0.145, y=1.02, xref="paper", yref="paper", showarrow=False, font=plot_font),
                            dict(text="Pred", x=0.505, y=1.02, xref="paper", yref="paper", showarrow=False, font=plot_font),
                            dict(text="Abs Error", x=0.86, y=1.02, xref="paper", yref="paper", showarrow=False, font=plot_font),
                        ],
                    }
                ],
            )
        )

    fig.update_layout(
        title=title + f" | Field={k0}",
        font=plot_font,
        title_font=title_font,
        hoverlabel=dict(font=plot_font),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.08, font=plot_font)],
        showlegend=False,
        autosize=False,
        width=1500,
        height=650,
        margin=dict(l=10, r=260, t=70, b=10),
    )
    fig.update_annotations(font=plot_font)
    fig.update_scenes(
        aspectmode="data",
        xaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
        yaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
        zaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
    )
    return fig


def make_deformed_overlay_figure(node_xyz_mm, u_true, u_pred, title, scale=1.0):
    x0 = node_xyz_mm
    xt = x0 + scale * u_true
    xp = x0 + scale * u_pred

    plot_font = dict(family=PLOT_FONT_FAMILY, size=PLOT_FONT_SIZE)
    title_font = dict(family=PLOT_FONT_FAMILY, size=PLOT_TITLE_FONT_SIZE)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xt[:, 0], y=xt[:, 1], z=xt[:, 2],
        mode="markers",
        marker=dict(size=3),
        name="True deformed"
    ))
    fig.add_trace(go.Scatter3d(
        x=xp[:, 0], y=xp[:, 1], z=xp[:, 2],
        mode="markers",
        marker=dict(size=3),
        name="Pred deformed"
    ))
    fig.update_layout(
        title=f"{title} | Deformed overlay (scale={scale})",
        font=plot_font,
        title_font=title_font,
        hoverlabel=dict(font=plot_font),
        legend=dict(font=plot_font),
        margin=dict(l=10, r=10, t=60, b=10),
        scene=dict(
            aspectmode="data",
            xaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
            yaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
            zaxis=dict(tickfont=plot_font, title=dict(font=plot_font)),
        )
    )
    return fig


def select_sample_id(sample_ids, sample_id=None, sample_index=None):
    if sample_id is not None:
        if sample_id not in sample_ids:
            preview = ", ".join(sample_ids[:20])
            raise ValueError(f"sample_id={sample_id} 不存在。当前可用样本前 20 个：{preview}")
        return sample_id

    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(sample_ids):
            raise IndexError(f"sample_index 越界: {sample_index}，当前样本数={len(sample_ids)}")
        return sample_ids[sample_index]

    clean_ids = [sid for sid in sample_ids if not AUG_PATTERN.search(sid)]
    return clean_ids[0] if clean_ids else sample_ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="包含 best_model.pth/norm_params.json/config.json 的目录")
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"], help="要推理的样本来自哪个 split")
    ap.add_argument("--sample_id", type=str, default=None, help="样本 id，如 160、200、0_aug01")
    ap.add_argument("--sample_index", type=int, default=None, help="按排序后的索引选样本；与 --sample_id 二选一")
    ap.add_argument("--list_ids", action="store_true", help="只列出当前 split 下可用 sample_id 并退出")
    ap.add_argument("--only_clean", action="store_true", help="列样本/默认选样本时只考虑 clean，不包含 *_augXX")
    ap.add_argument("--cache_root", type=str, default=None, help="可选：手动指定缓存根目录（基目录或 split 目录都可）")
    ap.add_argument("--out_dir", type=str, default=None, help="输出目录，默认=ckpt_dir/vis_html/<split>")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--pred_is_norm", action="store_true", help="模型输出是否为归一化位移（默认True）")
    ap.set_defaults(pred_is_norm=True)
    ap.add_argument("--overlay_scale", type=float, default=1.0, help="变形叠加显示的放大倍数（仅显示用）")
    args_cli = ap.parse_args()

    if args_cli.sample_id is not None and args_cli.sample_index is not None:
        raise ValueError("--sample_id 与 --sample_index 不能同时指定")

    ckpt_dir = args_cli.ckpt_dir
    best_pth = os.path.join(ckpt_dir, "best_model.pth")
    norm_json = os.path.join(ckpt_dir, "norm_params.json")
    cfg_json = os.path.join(ckpt_dir, "config.json")

    for p in [best_pth, norm_json, cfg_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到: {p}")

    cfg = load_json(cfg_json)
    norm = load_json(norm_json)

    data_root = cfg.get("data_root", "data400_new")
    num_points = int(cfg.get("num_points", 8944))

    # 默认推理 split：优先 val_split，其次 test，再次 train
    split = args_cli.split or cfg.get("val_split") or cfg.get("test_split") or cfg.get("train_split") or "val"

    cache_base = resolve_cache_base(
        data_root=data_root,
        num_points=num_points,
        cache_root_from_cfg=cfg.get("cache_root", None),
        cache_root_from_cli=args_cli.cache_root,
    )
    cache_split_root = resolve_split_cache_root(cache_base, split)

    sample_ids = list_sample_ids(cache_split_root, include_aug=(not args_cli.only_clean))

    if args_cli.list_ids:
        print(f"split={split} | cache={cache_split_root} | 共 {len(sample_ids)} 个样本")
        for i, sid in enumerate(sample_ids):
            print(f"[{i}] {sid}")
        return

    sample_id = select_sample_id(sample_ids, sample_id=args_cli.sample_id, sample_index=args_cli.sample_index)

    out_dir = args_cli.out_dir or os.path.join(ckpt_dir, "vis_html", split)
    os.makedirs(out_dir, exist_ok=True)

    cloud_mean = np.array(norm["cloud_mean"], dtype=np.float32).reshape(3)
    cloud_std = np.array(norm["cloud_std"], dtype=np.float32).reshape(3)
    disp_mean = np.array(norm["disp_mean"], dtype=np.float32).reshape(3)
    disp_std = np.array(norm["disp_std"], dtype=np.float32).reshape(3)

    node_xyz_mm, node_file = load_node_xyz_mm(data_root, split)
    M = int(node_xyz_mm.shape[0])
    xyz_cloud_mm, disp_true_mm = load_cached_sample(cache_split_root, sample_id, num_nodes=M)

    cloud_in = make_cloud_input(xyz_cloud_mm, cloud_mean, cloud_std)
    node_in = make_node_input(node_xyz_mm, cloud_mean, cloud_std)

    device = torch.device("cpu" if args_cli.cpu or (not torch.cuda.is_available()) else "cuda")

    args_model = build_args_from_config(cfg)
    model = PctRegressor(args_model, num_nodes=M).to(device)
    sd = torch.load(best_pth, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        cloud_t = torch.from_numpy(cloud_in).unsqueeze(0).to(device)
        node_t = torch.from_numpy(node_in).unsqueeze(0).to(device)
        pred = model(cloud_t.float(), node_t.float())

        if pred.dim() == 2 and pred.shape[1] == M * 3:
            pred = pred.view(1, M, 3)
        elif pred.dim() == 3 and pred.shape[1:] == (M, 3):
            pass
        else:
            raise RuntimeError(f"预测输出 shape 异常: {tuple(pred.shape)}, 期望 [1,{M * 3}] 或 [1,{M},3]")

        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

    disp_pred_mm = denorm_disp(pred_np, disp_mean, disp_std) if args_cli.pred_is_norm else pred_np

    metrics = compute_metrics_mm(disp_pred_mm, disp_true_mm)
    metrics["split"] = split
    metrics["sample_id"] = sample_id
    metrics["cache_split_root"] = cache_split_root
    metrics["node_template_file"] = node_file

    print("\n==== Inference Metrics (mm) ====")
    print(f"split:      {split}")
    print(f"sample_id:  {sample_id}")
    print(f"cache_root: {cache_split_root}")
    print(f"node_file:  {node_file}")
    print(f"MAE_all:    {metrics['MAE_all(mm)']:.6f} mm")
    print(f"RMSE_all:   {metrics['RMSE_all(mm)']:.6f} mm")
    print(f"MAE_axis:   {metrics['MAE_axis(mm)']}  (Ux,Uy,Uz)")
    print(f"RMSE_axis:  {metrics['RMSE_axis(mm)']} (Ux,Uy,Uz)")
    print(f"MAE_|dU|:   {metrics['MAE_|dU|(mm)']:.6f} mm")
    print(f"RMSE_|dU|:  {metrics['RMSE_|dU|(mm)']:.6f} mm")
    print("==============================\n")

    stem = f"{split}__{sample_id}"
    with open(os.path.join(out_dir, f"{stem}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    title = f"{os.path.basename(ckpt_dir)} | split={split} | sample={sample_id}"

    fig_cmp = make_compare_figure(node_xyz_mm, disp_true_mm, disp_pred_mm, title)
    out_html_cmp = os.path.join(out_dir, f"{stem}_compare.html")
    fig_cmp.write_html(out_html_cmp, include_plotlyjs="cdn")

    fig_def = make_deformed_overlay_figure(
        node_xyz_mm, disp_true_mm, disp_pred_mm, title, scale=float(args_cli.overlay_scale)
    )
    out_html_def = os.path.join(out_dir, f"{stem}_deformed.html")
    fig_def.write_html(out_html_def, include_plotlyjs="cdn")

    out_csv = os.path.join(out_dir, f"{stem}_pred_vs_true.csv")
    arr = np.concatenate([node_xyz_mm, disp_true_mm, disp_pred_mm, (disp_pred_mm - disp_true_mm)], axis=1)
    header = "x_mm,y_mm,z_mm,true_ux,true_uy,true_uz,pred_ux,pred_uy,pred_uz,err_ux,err_uy,err_uz"
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")

    print(f"[OK] HTML对比: {out_html_cmp}")
    print(f"[OK] HTML变形: {out_html_def}")
    print(f"[OK] 指标JSON: {os.path.join(out_dir, f'{stem}_metrics.json')}")
    print(f"[OK] 对比CSV: {out_csv}")


if __name__ == "__main__":
    main()
