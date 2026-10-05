import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)

from infer_and_viz_html import (
    build_args_from_config,
    compute_metrics_mm,
    denorm_disp,
    list_sample_ids,
    load_cached_sample,
    load_json,
    load_node_xyz_mm,
    make_cloud_input,
    make_node_input,
    resolve_cache_base,
    resolve_split_cache_root,
    select_sample_id,
)
from model_reg_U import PctRegressor


def _resolve_project_path(value, script_dir):
    path = Path(value)
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()


def _parse_indices(raw, num_nodes):
    if not raw:
        return None
    values = []
    for item in raw.split(","):
        idx = int(item.strip())
        if idx < 0 or idx >= num_nodes:
            raise ValueError(f"node index {idx} is outside [0, {num_nodes - 1}]")
        if idx not in values:
            values.append(idx)
    return values


def _nearest_section_node(node_xyz, axis, target):
    span = max(float(np.ptp(node_xyz[:, axis])), 1e-8)
    transverse_axes = [i for i in range(3) if i != axis]
    transverse = node_xyz[:, transverse_axes]
    center = np.median(transverse, axis=0)
    transverse_scale = np.maximum(np.ptp(transverse, axis=0), 1e-8)
    axis_cost = np.abs(node_xyz[:, axis] - target) / span
    transverse_cost = np.linalg.norm((transverse - center) / transverse_scale, axis=1)
    return int(np.argmin(axis_cost + 0.03 * transverse_cost))


def _representative_nodes(node_xyz, disp_true):
    axis = int(np.argmax(np.ptp(node_xyz, axis=0)))
    low = float(np.min(node_xyz[:, axis]))
    high = float(np.max(node_xyz[:, axis]))
    axis_name = "XYZ"[axis]
    fractions = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    candidates = []
    for fraction in fractions:
        target = low + fraction * (high - low)
        candidates.append(
            (
                _nearest_section_node(node_xyz, axis, target),
                f"{axis_name}/L = {fraction:.2f}",
            )
        )
    unique = []
    used = set()
    for idx, label in candidates:
        if idx not in used:
            unique.append((idx, label))
            used.add(idx)
    if len(unique) < 4:
        ordered = np.argsort(np.linalg.norm(disp_true, axis=1))[::-1]
        for idx in ordered:
            idx = int(idx)
            if idx not in used:
                unique.append((idx, f"High-response node {idx}"))
                used.add(idx)
            if len(unique) == 4:
                break
    return unique[:4], axis


def _attention_statistics(weights, token_xyz, node_xyz, top_fraction):
    eps = np.finfo(np.float64).tiny
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), eps)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, eps))))
    token_count = int(weights.size)
    top_k = max(1, int(math.ceil(token_count * top_fraction)))
    top_indices = np.argpartition(weights, -top_k)[-top_k:]
    distances = np.linalg.norm(token_xyz - node_xyz.reshape(1, 3), axis=1)
    return {
        "entropy_normalized": entropy / math.log(token_count),
        "effective_token_count": math.exp(entropy),
        "maximum_weight": float(weights.max()),
        "top_mass": float(weights[top_indices].sum()),
        "weighted_distance_mm": float(np.sum(weights * distances)),
        "top_indices": top_indices,
    }


def _all_node_attention_diagnostics(
    weights,
    token_xyz,
    node_xyz,
    principal_axis,
    top_fraction,
    num_bins=40,
):
    """Summarise whether node-to-point attention follows the target location."""
    weights = np.asarray(weights, dtype=np.float64)
    token_axis = token_xyz[:, principal_axis].astype(np.float64)
    node_axis = node_xyz[:, principal_axis].astype(np.float64)
    low = float(token_axis.min())
    span = max(float(token_axis.max() - low), np.finfo(np.float64).eps)
    token_s = np.clip((token_axis - low) / span, 0.0, 1.0)
    node_s = np.clip((node_axis - low) / span, 0.0, 1.0)

    row_sum = np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
    weights = weights / row_sum
    centroid = weights @ token_s
    variance = np.sum(weights * (token_s.reshape(1, -1) - centroid.reshape(-1, 1)) ** 2, axis=1)
    local_mask = np.abs(token_s.reshape(1, -1) - node_s.reshape(-1, 1)) <= 0.10
    local_mass = np.sum(weights * local_mask, axis=1)
    low_end_mass = weights[:, token_s <= 0.10].sum(axis=1)
    high_end_mass = weights[:, token_s >= 0.90].sum(axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(np.float64).tiny)), axis=1)
    entropy_normalized = entropy / math.log(weights.shape[1])
    top_k = max(1, int(math.ceil(weights.shape[1] * top_fraction)))
    top_mass = np.partition(weights, -top_k, axis=1)[:, -top_k:].sum(axis=1)

    token_bin = np.minimum((token_s * num_bins).astype(np.int64), num_bins - 1)
    node_bin = np.minimum((node_s * num_bins).astype(np.int64), num_bins - 1)
    node_token_mass = np.zeros((weights.shape[0], num_bins), dtype=np.float64)
    for bin_idx in range(num_bins):
        node_token_mass[:, bin_idx] = weights[:, token_bin == bin_idx].sum(axis=1)
    profile = np.full((num_bins, num_bins), np.nan, dtype=np.float64)
    centroid_by_bin = np.full(num_bins, np.nan, dtype=np.float64)
    for bin_idx in range(num_bins):
        rows = node_bin == bin_idx
        if np.any(rows):
            profile[bin_idx] = node_token_mass[rows].mean(axis=0)
            centroid_by_bin[bin_idx] = centroid[rows].mean()

    if np.std(node_s) > 0.0 and np.std(centroid) > 0.0:
        centroid_correlation = float(np.corrcoef(node_s, centroid)[0, 1])
    else:
        centroid_correlation = float("nan")

    def describe(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5.0)),
            "p95": float(np.percentile(values, 95.0)),
        }

    summary = {
        "centroid_target_pearson_r": centroid_correlation,
        "absolute_centroid_offset_normalized": describe(np.abs(centroid - node_s)),
        "attention_centroid_normalized": describe(centroid),
        "attention_spread_normalized": describe(np.sqrt(np.maximum(variance, 0.0))),
        "local_mass_within_0p10_length": describe(local_mass),
        "low_end_mass_first_0p10_length": describe(low_end_mass),
        "high_end_mass_last_0p10_length": describe(high_end_mass),
        "entropy_normalized": describe(entropy_normalized),
        "top_mass": describe(top_mass),
        "top_fraction": float(top_fraction),
        "num_nodes": int(weights.shape[0]),
        "num_tokens": int(weights.shape[1]),
    }
    return summary, profile, centroid_by_bin


def _make_all_node_diagnostic_figure(
    stage_profiles,
    stage_centroids,
    sample_id,
    output_path,
    dpi,
):
    finite_values = np.concatenate(
        [profile[np.isfinite(profile)] for profile in stage_profiles]
    )
    vmax = float(np.percentile(finite_values, 99.0)) if finite_values.size else 1.0
    vmax = max(vmax, np.finfo(np.float64).eps)
    fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.65), constrained_layout=True)
    image_for_colorbar = None
    bin_centers = (np.arange(stage_profiles[0].shape[0]) + 0.5) / stage_profiles[0].shape[0]
    for panel, (ax, profile, centroid, stage_name) in enumerate(
        zip(axes, stage_profiles, stage_centroids, ("Stage 1", "Stage 2")), start=1
    ):
        image_for_colorbar = ax.imshow(
            profile,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="white", linewidth=0.9, label="Local alignment")
        valid = np.isfinite(centroid)
        ax.plot(centroid[valid], bin_centers[valid], color="#66c2ff", linewidth=1.5, label="Attention centroid")
        ax.set_title(f"({chr(96 + panel)}) {stage_name}", fontsize=9.0)
        ax.set_xlabel("Point-token position, $Z/L$ (0: free; 1: fixed)", fontsize=7.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelsize=7.2)
        ax.grid(False)
    axes[0].set_ylabel("Target-node position, $Z/L$", fontsize=8.0)
    axes[1].legend(loc="lower left", fontsize=6.7, framealpha=0.88)
    if image_for_colorbar is not None:
        colorbar = fig.colorbar(image_for_colorbar, ax=axes, shrink=0.84, pad=0.025)
        colorbar.ax.tick_params(labelsize=7.0)
        colorbar.set_label("Mean attention mass per longitudinal bin", fontsize=7.8)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(output_path).with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_figure(
    token_xyz,
    node_xyz,
    selected,
    weights,
    statistics,
    sample_id,
    top_fraction,
    principal_axis,
    output_path,
    dpi,
):
    selected_weights = [weights[idx] for idx, _ in selected]
    colored_weights = []
    for row, stats in zip(selected_weights, statistics):
        colored_weights.extend(row[stats["top_indices"]].tolist())
    vmax = max(colored_weights) if colored_weights else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    transverse_axes = [axis for axis in range(3) if axis != principal_axis]
    transverse_axis = max(
        transverse_axes, key=lambda axis: float(np.ptp(token_xyz[:, axis]))
    )
    axis_names = "XYZ"
    longitudinal = token_xyz[:, principal_axis]
    low = float(longitudinal.min())
    span = max(float(longitudinal.max() - low), 1e-8)
    longitudinal_norm = (longitudinal - low) / span

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.25),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    scatter_for_colorbar = None
    for panel, ((node_idx, label), row, stats) in enumerate(
        zip(selected, selected_weights, statistics), start=1
    ):
        ax = axes.flat[panel - 1]
        ax.scatter(
            longitudinal_norm,
            token_xyz[:, transverse_axis],
            s=7,
            c=row,
            cmap="viridis",
            norm=norm,
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )
        top_idx = stats["top_indices"]
        top_weights = row[top_idx]
        sizes = 16.0 + 75.0 * top_weights / max(float(top_weights.max()), 1e-12)
        scatter_for_colorbar = ax.scatter(
            longitudinal_norm[top_idx],
            token_xyz[top_idx, transverse_axis],
            s=sizes,
            c=top_weights,
            cmap="viridis",
            norm=norm,
            alpha=0.92,
            edgecolors="none",
            rasterized=True,
        )
        target = node_xyz[node_idx]
        target_longitudinal = (float(target[principal_axis]) - low) / span
        ax.scatter(
            [target_longitudinal],
            [target[transverse_axis]],
            marker="*",
            s=180,
            c="#d7191c",
            edgecolors="black",
            linewidths=0.7,
            label="Target node",
        )
        ax.set_title(
            f"({chr(96 + panel)}) Target {label} (node {node_idx})\n"
            f"Top-{100.0 * top_fraction:g}% mass = {stats['top_mass']:.3f}; "
            f"$H_{{n}}$ = {stats['entropy_normalized']:.3f}",
            fontsize=8.4,
        )
        ax.axvline(target_longitudinal, color="#d7191c", linewidth=0.8, alpha=0.35)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel(f"Longitudinal coordinate, {axis_names[principal_axis]}/L", fontsize=8.2)
        ax.set_ylabel(f"Transverse coordinate, {axis_names[transverse_axis]} (mm)", fontsize=8.2)

    if scatter_for_colorbar is not None:
        colorbar = fig.colorbar(
            scatter_for_colorbar,
            ax=fig.axes,
            shrink=0.68,
            pad=0.02,
            aspect=30,
        )
        colorbar.ax.tick_params(labelsize=7.2)
        colorbar.set_label("Mean cross-attention weight (four heads)", fontsize=8.2)
    fig.suptitle(
        f"Stage-2 node-to-point cross-attention | validation sample {sample_id}",
        fontsize=10.5,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(output_path).with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Export real node-to-point cross-attention maps from a trained PCDD checkpoint."
    )
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--node_indices", default=None, help="Comma-separated zero-based node indices")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override the data_root stored in config.json (useful after moving a project)",
    )
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args_cli = parser.parse_args()

    if not 0.0 < args_cli.top_fraction <= 1.0:
        raise ValueError("--top_fraction must be in (0, 1]")
    if args_cli.sample_id is not None and args_cli.sample_index is not None:
        raise ValueError("--sample_id and --sample_index are mutually exclusive")

    script_dir = Path(__file__).resolve().parent
    ckpt_dir = _resolve_project_path(args_cli.ckpt_dir, script_dir)
    cfg = load_json(ckpt_dir / "config.json")
    norm_params = load_json(ckpt_dir / "norm_params.json")
    checkpoint_path = ckpt_dir / "best_model.pth"

    data_root = _resolve_project_path(
        args_cli.data_root or cfg.get("data_root", "data400_new"), script_dir
    )
    num_points = int(cfg.get("num_points", 8944))
    split = (
        args_cli.split
        or cfg.get("val_split")
        or cfg.get("test_split")
        or cfg.get("train_split")
        or "val"
    )

    cache_cfg = cfg.get("cache_root")
    if cache_cfg:
        cache_cfg = str(_resolve_project_path(cache_cfg, script_dir))
    cache_cli = args_cli.cache_root
    if cache_cli:
        cache_cli = str(_resolve_project_path(cache_cli, script_dir))
    cache_base = resolve_cache_base(
        str(data_root),
        num_points,
        cache_root_from_cfg=cache_cfg,
        cache_root_from_cli=cache_cli,
    )
    cache_split_root = resolve_split_cache_root(cache_base, split)
    sample_ids = list_sample_ids(cache_split_root, include_aug=False)
    sample_id = select_sample_id(
        sample_ids,
        sample_id=args_cli.sample_id,
        sample_index=args_cli.sample_index,
    )

    node_xyz_mm, node_file = load_node_xyz_mm(str(data_root), split)
    num_nodes = int(node_xyz_mm.shape[0])
    cloud_xyz_mm, disp_true_mm = load_cached_sample(
        cache_split_root, sample_id, num_nodes=num_nodes
    )

    cloud_mean = np.asarray(norm_params["cloud_mean"], dtype=np.float32).reshape(3)
    cloud_std = np.asarray(norm_params["cloud_std"], dtype=np.float32).reshape(3)
    disp_mean = np.asarray(norm_params["disp_mean"], dtype=np.float32).reshape(3)
    disp_std = np.asarray(norm_params["disp_std"], dtype=np.float32).reshape(3)
    cloud_input = make_cloud_input(cloud_xyz_mm, cloud_mean, cloud_std)
    node_input = make_node_input(node_xyz_mm, cloud_mean, cloud_std)

    device = torch.device(
        "cpu" if args_cli.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = PctRegressor(build_args_from_config(cfg), num_nodes=num_nodes).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    model.eval()

    with torch.inference_mode():
        cloud_tensor = torch.from_numpy(cloud_input).unsqueeze(0).to(device)
        node_tensor = torch.from_numpy(node_input).unsqueeze(0).to(device)
        prediction, attention = model(
            cloud_tensor.float(), node_tensor.float(), return_attention=True
        )

    prediction_norm = prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)
    prediction_mm = denorm_disp(prediction_norm, disp_mean, disp_std)
    metrics = compute_metrics_mm(prediction_mm, disp_true_mm)

    stage_1_heads = attention["node_from_point_stage_1"].squeeze(0).detach().cpu().numpy()
    stage_2_heads = attention["node_from_point_stage_2"].squeeze(0).detach().cpu().numpy()
    token_xyz_norm = (
        attention["point_token_xyz"].squeeze(0).permute(1, 0).detach().cpu().numpy()
    )
    token_xyz_mm = token_xyz_norm * cloud_std.reshape(1, 3) + cloud_mean.reshape(1, 3)
    stage_1_mean = stage_1_heads.mean(axis=0)
    stage_2_mean = stage_2_heads.mean(axis=0)

    selected_indices = _parse_indices(args_cli.node_indices, num_nodes)
    if selected_indices is None:
        selected, principal_axis = _representative_nodes(node_xyz_mm, disp_true_mm)
    else:
        selected = [(idx, f"Selected node {idx}") for idx in selected_indices]
        principal_axis = int(np.argmax(np.ptp(node_xyz_mm, axis=0)))

    stage_1_all_summary, stage_1_profile, stage_1_centroid = _all_node_attention_diagnostics(
        stage_1_mean,
        token_xyz_mm,
        node_xyz_mm,
        principal_axis,
        args_cli.top_fraction,
    )
    stage_2_all_summary, stage_2_profile, stage_2_centroid = _all_node_attention_diagnostics(
        stage_2_mean,
        token_xyz_mm,
        node_xyz_mm,
        principal_axis,
        args_cli.top_fraction,
    )

    stage_1_stats = []
    stage_2_stats = []
    for node_idx, _ in selected:
        stage_1_stats.append(
            _attention_statistics(
                stage_1_mean[node_idx],
                token_xyz_mm,
                node_xyz_mm[node_idx],
                args_cli.top_fraction,
            )
        )
        stage_2_stats.append(
            _attention_statistics(
                stage_2_mean[node_idx],
                token_xyz_mm,
                node_xyz_mm[node_idx],
                args_cli.top_fraction,
            )
        )

    output_dir = (
        _resolve_project_path(args_cli.out_dir, script_dir)
        if args_cli.out_dir
        else ckpt_dir / "attention_maps" / split / str(sample_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{sample_id}_node_to_point_stage2"
    figure_path = output_dir / f"{stem}.png"
    _make_figure(
        token_xyz_mm,
        node_xyz_mm,
        selected,
        stage_2_mean,
        stage_2_stats,
        sample_id,
        args_cli.top_fraction,
        principal_axis,
        figure_path,
        args_cli.dpi,
    )
    diagnostic_figure_path = output_dir / f"{split}_{sample_id}_all_node_attention_diagnostic.png"
    _make_all_node_diagnostic_figure(
        (stage_1_profile, stage_2_profile),
        (stage_1_centroid, stage_2_centroid),
        sample_id,
        diagnostic_figure_path,
        args_cli.dpi,
    )
    diagnostic_data_path = output_dir / f"{split}_{sample_id}_all_node_attention_diagnostic.npz"
    np.savez_compressed(
        diagnostic_data_path,
        stage1_longitudinal_profile=stage_1_profile,
        stage2_longitudinal_profile=stage_2_profile,
        stage1_attention_centroid_by_node_bin=stage_1_centroid,
        stage2_attention_centroid_by_node_bin=stage_2_centroid,
    )

    csv_path = output_dir / f"{stem}_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_index",
                "label",
                "x_mm",
                "y_mm",
                "z_mm",
                "stage1_entropy_normalized",
                "stage2_entropy_normalized",
                "stage1_effective_token_count",
                "stage2_effective_token_count",
                "stage1_top_mass",
                "stage2_top_mass",
                "stage1_weighted_distance_mm",
                "stage2_weighted_distance_mm",
                "stage2_maximum_weight",
            ],
        )
        writer.writeheader()
        for (node_idx, label), stats_1, stats_2 in zip(
            selected, stage_1_stats, stage_2_stats
        ):
            xyz = node_xyz_mm[node_idx]
            writer.writerow(
                {
                    "node_index": node_idx,
                    "label": label,
                    "x_mm": float(xyz[0]),
                    "y_mm": float(xyz[1]),
                    "z_mm": float(xyz[2]),
                    "stage1_entropy_normalized": stats_1["entropy_normalized"],
                    "stage2_entropy_normalized": stats_2["entropy_normalized"],
                    "stage1_effective_token_count": stats_1["effective_token_count"],
                    "stage2_effective_token_count": stats_2["effective_token_count"],
                    "stage1_top_mass": stats_1["top_mass"],
                    "stage2_top_mass": stats_2["top_mass"],
                    "stage1_weighted_distance_mm": stats_1["weighted_distance_mm"],
                    "stage2_weighted_distance_mm": stats_2["weighted_distance_mm"],
                    "stage2_maximum_weight": stats_2["maximum_weight"],
                }
            )

    npz_path = output_dir / f"{stem}_data.npz"
    selected_idx_array = np.asarray([idx for idx, _ in selected], dtype=np.int64)
    np.savez_compressed(
        npz_path,
        sample_id=np.asarray(str(sample_id)),
        selected_node_indices=selected_idx_array,
        selected_node_labels=np.asarray([label for _, label in selected]),
        selected_node_xyz_mm=node_xyz_mm[selected_idx_array],
        point_token_xyz_mm=token_xyz_mm,
        stage1_attention_heads=stage_1_heads[:, selected_idx_array, :],
        stage2_attention_heads=stage_2_heads[:, selected_idx_array, :],
        prediction_mm=prediction_mm[selected_idx_array],
        truth_mm=disp_true_mm[selected_idx_array],
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_strict_load": True,
        "data_root": str(data_root),
        "device": str(device),
        "split": split,
        "sample_id": str(sample_id),
        "num_nodes": num_nodes,
        "num_point_tokens": int(token_xyz_mm.shape[0]),
        "attention_heads": int(stage_2_heads.shape[0]),
        "principal_axis": principal_axis,
        "node_template_file": str(node_file),
        "prediction_metrics_mm": metrics,
        "stage1_row_sum_max_error": float(np.max(np.abs(stage_1_mean.sum(axis=1) - 1.0))),
        "stage2_row_sum_max_error": float(np.max(np.abs(stage_2_mean.sum(axis=1) - 1.0))),
        "all_node_attention_diagnostics": {
            "stage1": stage_1_all_summary,
            "stage2": stage_2_all_summary,
            "interpretation_guardrail": (
                "Attention weights describe learned information routing and should not be "
                "presented as causal physical importance without independent validation."
            ),
        },
        "selected_nodes": [
            {
                "index": idx,
                "label": label,
                "stage1": {k: v for k, v in stats_1.items() if k != "top_indices"},
                "stage2": {k: v for k, v in stats_2.items() if k != "top_indices"},
            }
            for (idx, label), stats_1, stats_2 in zip(
                selected, stage_1_stats, stage_2_stats
            )
        ],
        "outputs": {
            "figure": str(figure_path),
            "figure_svg": str(figure_path.with_suffix(".svg")),
            "all_node_diagnostic_figure": str(diagnostic_figure_path),
            "all_node_diagnostic_figure_svg": str(diagnostic_figure_path.with_suffix(".svg")),
            "all_node_diagnostic_data_npz": str(diagnostic_data_path),
            "statistics_csv": str(csv_path),
            "data_npz": str(npz_path),
        },
    }
    report_path = output_dir / f"{stem}_report.json"
    report["outputs"]["report_json"] = str(report_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
