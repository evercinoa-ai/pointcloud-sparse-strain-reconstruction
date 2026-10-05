#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch TLS-like point cloud simulation for deformed structural OBJ models.

Designed for a dataset layout like:
    data_Lab/
      OBJ/
        0.obj
        1.obj
        ...
      displacement/
        0.csv
        1.csv
        ...

For each case id that exists in both folders (same stem), the script:
1) loads the OBJ surface mesh,
2) samples surface points,
3) simulates one or multiple scan stations,
4) adds configurable three-level errors:
   - range noise + angular noise
   - incidence-angle bias
   - inter-station registration residual
5) merges enabled stations and writes CSV with x,y,z columns.

Compared with the earlier version:
- merged output is written directly as: data_Lab/cloudpoint/0.csv, 1.csv, ...
- no per-case subfolder is created
- per-case summary.json is disabled by default
- batch summary is also disabled by default

No external geometry library is required.

修改tls_sim_config_data_Lab_flat.json中的超参数
python simulate_tls_batch_flat_raytrace.py --config tls_sim_config_data_Lab_raytrace.json --case_ids 5
python simulate_tls_batch_flat_raytrace.py --config tls_sim_config_data_Lab_raytrace.json --use_stations S1 --case_ids 2

stations表示扫描站点位置
simulation表示可见性误差，按法向量

"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def mm_to_model_units(value_mm: float, model_unit: str) -> float:
    if model_unit.lower() == "m":
        return value_mm * 1e-3
    if model_unit.lower() == "mm":
        return value_mm
    raise ValueError(f"Unsupported model_unit={model_unit!r}; expected 'm' or 'mm'.")



def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0



def arcsec_to_rad(arcsec: float) -> float:
    return arcsec * math.pi / (180.0 * 3600.0)



def rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rzm @ rym @ rxm



def write_xyz_csv(path: Path, points: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for row in points:
            writer.writerow([f"{row[0]:.10f}", f"{row[1]:.10f}", f"{row[2]:.10f}"])


# -----------------------------
# OBJ loading and surface sampling
# -----------------------------


def load_obj_mesh(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangulated faces from a Wavefront OBJ file.

    Supports 'v' and 'f'. Polygon faces are fan-triangulated.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    token = p.split("/")[0]
                    if not token:
                        continue
                    vid = int(token)
                    if vid < 0:
                        vid = len(vertices) + vid
                    else:
                        vid = vid - 1
                    idx.append(vid)
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        raise ValueError(f"Failed to read valid vertices/faces from {obj_path}")
    return v, f



def triangle_areas_and_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]  # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    areas = 0.5 * area2
    normals = cross / np.clip(area2[:, None], 1e-15, None)
    return areas, normals



def sample_points_on_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    areas, face_normals = triangle_areas_and_normals(vertices, faces)
    valid = areas > 0
    if not np.any(valid):
        raise ValueError("Mesh has no positive-area triangles.")
    valid_face_ids = np.nonzero(valid)[0]
    areas = areas[valid]
    face_normals = face_normals[valid]
    faces = faces[valid]

    probs = areas / areas.sum()
    chosen = rng.choice(len(faces), size=n_points, replace=True, p=probs)
    tri = vertices[faces[chosen]]
    normals = face_normals[chosen]
    sampled_face_ids = valid_face_ids[chosen]

    u = rng.random(n_points)
    v = rng.random(n_points)
    sqrt_u = np.sqrt(u)
    b0 = 1.0 - sqrt_u
    b1 = sqrt_u * (1.0 - v)
    b2 = sqrt_u * v
    pts = tri[:, 0] * b0[:, None] + tri[:, 1] * b1[:, None] + tri[:, 2] * b2[:, None]
    return pts, normals, sampled_face_ids


# -----------------------------
# Mesh ray tracing (exact visibility)
# -----------------------------


@dataclass
class BVHNode:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: Optional["BVHNode"] = None
    right: Optional["BVHNode"] = None
    tri_indices: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.tri_indices is not None


def build_triangle_data(vertices: np.ndarray, faces: np.ndarray) -> dict:
    triangles = vertices[faces]
    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    return {
        "triangles": triangles,
        "tri_min": tri_min,
        "tri_max": tri_max,
        "centroids": centroids,
        "faces": faces,
    }


def build_bvh(tri_min: np.ndarray, tri_max: np.ndarray, centroids: np.ndarray, tri_indices: np.ndarray, leaf_size: int) -> BVHNode:
    bbox_min = np.min(tri_min[tri_indices], axis=0)
    bbox_max = np.max(tri_max[tri_indices], axis=0)

    if len(tri_indices) <= leaf_size:
        return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, tri_indices=tri_indices.copy())

    c = centroids[tri_indices]
    extent = c.max(axis=0) - c.min(axis=0)
    axis = int(np.argmax(extent))
    order = np.argsort(c[:, axis], kind="mergesort")
    sorted_idx = tri_indices[order]
    mid = len(sorted_idx) // 2
    if mid == 0 or mid == len(sorted_idx):
        return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, tri_indices=tri_indices.copy())

    left = build_bvh(tri_min, tri_max, centroids, sorted_idx[:mid], leaf_size)
    right = build_bvh(tri_min, tri_max, centroids, sorted_idx[mid:], leaf_size)
    return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, left=left, right=right)


def ray_intersects_aabb(origin: np.ndarray, direction: np.ndarray, t_max: float, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    inv_dir = np.empty(3, dtype=float)
    for k in range(3):
        if abs(direction[k]) < 1e-15:
            inv_dir[k] = np.inf
        else:
            inv_dir[k] = 1.0 / direction[k]

    t0 = (bbox_min - origin) * inv_dir
    t1 = (bbox_max - origin) * inv_dir
    tmin = np.maximum.reduce(np.minimum(t0, t1))
    tmax = np.minimum.reduce(np.maximum(t0, t1))
    return (tmax >= max(tmin, 0.0)) and (tmin <= t_max)


def segment_hits_any_triangle(
    origin: np.ndarray,
    target: np.ndarray,
    triangles: np.ndarray,
    bvh_root: BVHNode,
    ignore_face_id: Optional[int] = None,
    eps: float = 1e-9,
) -> bool:
    direction = target - origin
    seg_len = float(np.linalg.norm(direction))
    if seg_len <= 1e-15:
        return False
    direction = direction / seg_len
    t_upper = max(seg_len - eps, 0.0)

    stack = [bvh_root]
    while stack:
        node = stack.pop()
        if not ray_intersects_aabb(origin, direction, t_upper, node.bbox_min, node.bbox_max):
            continue

        if node.is_leaf:
            tri_idx = node.tri_indices
            if ignore_face_id is not None:
                tri_idx = tri_idx[tri_idx != ignore_face_id]
            if tri_idx is None or len(tri_idx) == 0:
                continue

            tris = triangles[tri_idx]
            v0 = tris[:, 0, :]
            e1 = tris[:, 1, :] - v0
            e2 = tris[:, 2, :] - v0
            pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            mask = np.abs(det) > 1e-12
            if not np.any(mask):
                continue

            inv_det = np.zeros_like(det)
            inv_det[mask] = 1.0 / det[mask]
            tvec = origin[None, :] - v0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
            qvec = np.cross(tvec, e1)
            v = np.einsum("ij,ij->i", np.broadcast_to(direction, qvec.shape), qvec) * inv_det
            t = np.einsum("ij,ij->i", e2, qvec) * inv_det

            hit = mask & (u >= -1e-10) & (v >= -1e-10) & ((u + v) <= 1.0 + 1e-10) & (t > eps) & (t < t_upper)
            if np.any(hit):
                return True
        else:
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
    return False


def exact_ray_tracing_visibility(
    points: np.ndarray,
    normals: np.ndarray,
    source_face_ids: np.ndarray,
    station_xyz: np.ndarray,
    frontface_only: bool,
    mesh_rt: dict,
    self_hit_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec = points - station_xyz[None, :]
    r = np.linalg.norm(vec, axis=1)
    ray_dir = vec / np.clip(r[:, None], 1e-15, None)
    cos_view = np.sum(normals * (-ray_dir), axis=1)

    front_mask = np.ones(len(points), dtype=bool)
    if frontface_only:
        front_mask &= cos_view > 0.0

    idx = np.nonzero(front_mask)[0]
    if len(idx) == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        return points[:0], normals[:0], np.empty((0,), dtype=float), empty_i

    visible_idx = []
    triangles = mesh_rt["triangles"]
    bvh_root = mesh_rt["bvh_root"]
    for i in idx:
        if not segment_hits_any_triangle(
            origin=station_xyz,
            target=points[i],
            triangles=triangles,
            bvh_root=bvh_root,
            ignore_face_id=int(source_face_ids[i]),
            eps=self_hit_eps,
        ):
            visible_idx.append(i)

    if not visible_idx:
        empty_i = np.empty((0,), dtype=np.int64)
        return points[:0], normals[:0], np.empty((0,), dtype=float), empty_i

    vis_idx = np.asarray(visible_idx, dtype=np.int64)
    vis_cos = np.clip(cos_view[vis_idx], -1.0, 1.0)
    incidence_deg = np.degrees(np.arccos(vis_cos))
    return points[vis_idx], normals[vis_idx], incidence_deg, vis_idx


# -----------------------------
# Station simulation
# -----------------------------

@dataclass
class Station:
    name: str
    xyz: Tuple[float, float, float]
    enabled: bool = True


@dataclass
class SimResult:
    case_id: str
    n_sampled_surface_points: int
    n_enabled_stations: int
    points_per_station: Dict[str, int]
    n_merged_points: int
    output_csv: str



def cartesian_to_spherical(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return r, azimuth, elevation.

    azimuth = atan2(y, x)
    elevation = atan2(z, sqrt(x^2+y^2))
    """
    r = np.linalg.norm(v, axis=1)
    xy = np.linalg.norm(v[:, :2], axis=1)
    az = np.arctan2(v[:, 1], v[:, 0])
    el = np.arctan2(v[:, 2], np.clip(xy, 1e-15, None))
    return r, az, el



def spherical_to_cartesian(r: np.ndarray, az: np.ndarray, el: np.ndarray) -> np.ndarray:
    cos_el = np.cos(el)
    x = r * cos_el * np.cos(az)
    y = r * cos_el * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1)



def apply_visibility_and_occlusion(
    points: np.ndarray,
    normals: np.ndarray,
    source_face_ids: np.ndarray,
    station_xyz: np.ndarray,
    frontface_only: bool,
    occlusion_enabled: bool,
    az_res_deg: float,
    el_res_deg: float,
    occlusion_mode: str,
    mesh_rt: Optional[dict],
    self_hit_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visible points, normals, incidence angles (deg).

    Supported visibility modes:
    1) angular_zbuffer: front-face filter + angular bin nearest-point approximation
    2) ray_tracing: exact segment-triangle intersection against the OBJ mesh
    """
    if occlusion_enabled and occlusion_mode == "ray_tracing":
        if mesh_rt is None:
            raise ValueError("Ray-tracing visibility requires mesh_rt data.")
        pts, nrm, incidence_deg, _ = exact_ray_tracing_visibility(
            points=points,
            normals=normals,
            source_face_ids=source_face_ids,
            station_xyz=station_xyz,
            frontface_only=frontface_only,
            mesh_rt=mesh_rt,
            self_hit_eps=self_hit_eps,
        )
        return pts, nrm, incidence_deg

    vec = points - station_xyz[None, :]
    r = np.linalg.norm(vec, axis=1)
    ray_dir = vec / np.clip(r[:, None], 1e-15, None)
    cos_view = np.sum(normals * (-ray_dir), axis=1)  # larger => more front-facing

    mask = np.ones(len(points), dtype=bool)
    if frontface_only:
        mask &= cos_view > 0.0

    pts = points[mask]
    nrm = normals[mask]
    vec = vec[mask]
    r = r[mask]
    cos_view = np.clip(cos_view[mask], -1.0, 1.0)
    incidence_deg = np.degrees(np.arccos(np.clip(cos_view, -1.0, 1.0)))

    if not occlusion_enabled or len(pts) == 0:
        return pts, nrm, incidence_deg

    _, az, el = cartesian_to_spherical(vec)
    az_bin = np.floor((az + math.pi) / deg_to_rad(az_res_deg)).astype(np.int64)
    el_bin = np.floor((el + math.pi / 2.0) / deg_to_rad(el_res_deg)).astype(np.int64)

    # keep nearest point in each angular bin
    best: Dict[Tuple[int, int], int] = {}
    for i in range(len(pts)):
        key = (int(az_bin[i]), int(el_bin[i]))
        if key not in best or r[i] < r[best[key]]:
            best[key] = i
    idx = np.array(sorted(best.values()), dtype=np.int64)
    return pts[idx], nrm[idx], incidence_deg[idx]



def apply_measurement_noise(
    points: np.ndarray,
    station_xyz: np.ndarray,
    rng: np.random.Generator,
    range_noise_std: float,
    angle_noise_std_rad: float,
) -> np.ndarray:
    vec = points - station_xyz[None, :]
    r, az, el = cartesian_to_spherical(vec)

    if range_noise_std > 0:
        r = r + rng.normal(0.0, range_noise_std, size=r.shape)
        r = np.clip(r, 1e-9, None)
    if angle_noise_std_rad > 0:
        az = az + rng.normal(0.0, angle_noise_std_rad, size=az.shape)
        el = el + rng.normal(0.0, angle_noise_std_rad, size=el.shape)
        el = np.clip(el, -math.pi / 2 + 1e-6, math.pi / 2 - 1e-6)

    return station_xyz[None, :] + spherical_to_cartesian(r, az, el)



def apply_incidence_bias(
    points: np.ndarray,
    station_xyz: np.ndarray,
    incidence_deg: np.ndarray,
    threshold_deg: float,
    max_bias: float,
    mode: str,
) -> np.ndarray:
    if len(points) == 0 or max_bias <= 0:
        return points

    vec = points - station_xyz[None, :]
    r = np.linalg.norm(vec, axis=1)
    ray_dir = vec / np.clip(r[:, None], 1e-15, None)

    t = np.clip((incidence_deg - threshold_deg) / max(1e-9, 90.0 - threshold_deg), 0.0, 1.0)
    if mode == "linear":
        w = t
    elif mode == "quadratic":
        w = t ** 2
    elif mode == "cubic":
        w = t ** 3
    else:
        raise ValueError(f"Unsupported incidence bias mode: {mode}")

    bias = max_bias * w
    return points + bias[:, None] * ray_dir



def apply_registration_residual(
    points: np.ndarray,
    rng: np.random.Generator,
    translation_std: float,
    rotation_std_deg: float,
) -> np.ndarray:
    if len(points) == 0:
        return points
    t = rng.normal(0.0, translation_std, size=3) if translation_std > 0 else np.zeros(3)
    rot_std = deg_to_rad(rotation_std_deg)
    rxyz = rng.normal(0.0, rot_std, size=3) if rot_std > 0 else np.zeros(3)
    R = rotation_matrix_from_euler_xyz(rxyz[0], rxyz[1], rxyz[2])
    c = points.mean(axis=0)
    return ((points - c[None, :]) @ R.T) + c[None, :] + t[None, :]



def simulate_station_scan(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    source_face_ids: np.ndarray,
    station_xyz: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    mesh_rt: Optional[dict],
) -> Tuple[np.ndarray, dict]:
    sim = cfg["simulation"]
    occ = sim["occlusion"]
    inc = sim["incidence_bias"]

    vis_pts, vis_nrm, incidence_deg = apply_visibility_and_occlusion(
        points=surface_points,
        normals=surface_normals,
        source_face_ids=source_face_ids,
        station_xyz=station_xyz,
        frontface_only=bool(sim.get("frontface_only", True)),
        occlusion_enabled=bool(occ.get("enabled", True)),
        az_res_deg=float(occ.get("azimuth_res_deg", 0.08)),
        el_res_deg=float(occ.get("elevation_res_deg", 0.08)),
        occlusion_mode=str(occ.get("mode", "angular_zbuffer")).lower(),
        mesh_rt=mesh_rt,
        self_hit_eps=float(occ.get("self_hit_eps", 1e-9)),
    )

    pts = apply_measurement_noise(
        vis_pts,
        station_xyz,
        rng,
        range_noise_std=float(sim.get("range_noise_std", 0.0)),
        angle_noise_std_rad=float(sim.get("angle_noise_std_rad", 0.0)),
    )

    if bool(inc.get("enabled", False)):
        pts = apply_incidence_bias(
            pts,
            station_xyz,
            incidence_deg,
            threshold_deg=float(inc.get("angle_threshold_deg", 60.0)),
            max_bias=float(inc.get("max_bias", 0.0)),
            mode=str(inc.get("mode", "quadratic")).lower(),
        )

    meta = {
        "n_visible": int(len(vis_pts)),
        "incidence_deg_mean": float(np.mean(incidence_deg)) if len(incidence_deg) else 0.0,
        "incidence_deg_max": float(np.max(incidence_deg)) if len(incidence_deg) else 0.0,
        "occlusion_mode": str(occ.get("mode", "angular_zbuffer")).lower(),
    }
    return pts, meta


# -----------------------------
# Batch processing
# -----------------------------



def list_common_case_ids(obj_dir: Path, disp_dir: Path) -> List[str]:
    obj_stems = {p.stem for p in obj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".obj"}
    disp_stems = {p.stem for p in disp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"}
    return sorted(obj_stems & disp_stems, key=lambda s: int(s) if s.isdigit() else s)



def validate_and_prepare_config(cfg: dict) -> dict:
    cfg = dict(cfg)  # shallow copy
    model_unit = str(cfg.get("model_unit", "m")).lower()
    if model_unit not in {"m", "mm"}:
        raise ValueError("model_unit must be 'm' or 'mm'.")

    sim = dict(cfg.get("simulation", {}))
    sim["range_noise_std"] = mm_to_model_units(float(sim.get("range_noise_std_mm", 0.0)), model_unit)
    sim["angle_noise_std_rad"] = arcsec_to_rad(float(sim.get("angle_noise_std_arcsec", 0.0)))

    inc = dict(sim.get("incidence_bias", {}))
    inc["max_bias"] = mm_to_model_units(float(inc.get("max_bias_mm", 0.0)), model_unit)
    sim["incidence_bias"] = inc

    reg = dict(sim.get("registration_residual", {}))
    reg["translation_std"] = mm_to_model_units(float(reg.get("translation_std_mm", 0.0)), model_unit)
    sim["registration_residual"] = reg

    occ = dict(sim.get("occlusion", {}))
    occ.setdefault("mode", "angular_zbuffer")
    occ["self_hit_eps"] = mm_to_model_units(float(occ.get("self_hit_eps_mm", 0.02)), model_unit)
    occ.setdefault("bvh_leaf_size", 32)
    sim["occlusion"] = occ
    cfg["simulation"] = sim

    cfg.setdefault("save_per_station_csv", False)
    cfg.setdefault("save_case_summary_json", False)
    cfg.setdefault("save_batch_summary", False)
    return cfg



def get_enabled_stations(cfg: dict, override_names: Optional[Sequence[str]] = None) -> List[Station]:
    stations = []
    requested = set(override_names) if override_names else None
    for s in cfg.get("stations", []):
        name = str(s["name"])
        enabled = bool(s.get("enabled", True))
        if requested is not None:
            enabled = enabled and (name in requested)
        stations.append(Station(name=name, xyz=tuple(map(float, s["xyz"])), enabled=enabled))
    enabled = [s for s in stations if s.enabled]
    if not enabled:
        raise ValueError("No enabled stations after applying config / override.")
    return enabled



def simulate_case(case_id: str, obj_path: Path, out_dir: Path, cfg: dict, stations: List[Station], seed: int) -> SimResult:
    rng = np.random.default_rng(seed)
    vertices, faces = load_obj_mesh(obj_path)
    n_surface_points = int(cfg.get("mesh_sampling", {}).get("n_points", 30000))
    surface_points, surface_normals, sampled_face_ids = sample_points_on_mesh(vertices, faces, n_surface_points, rng)

    occ_cfg = cfg["simulation"]["occlusion"]
    mesh_rt = None
    if bool(occ_cfg.get("enabled", True)) and str(occ_cfg.get("mode", "angular_zbuffer")).lower() == "ray_tracing":
        tri_data = build_triangle_data(vertices, faces)
        tri_indices = np.arange(len(faces), dtype=np.int64)
        bvh_root = build_bvh(
            tri_data["tri_min"],
            tri_data["tri_max"],
            tri_data["centroids"],
            tri_indices,
            leaf_size=int(occ_cfg.get("bvh_leaf_size", 32)),
        )
        mesh_rt = {
            "triangles": tri_data["triangles"],
            "bvh_root": bvh_root,
        }

    station_outputs = []
    points_per_station: Dict[str, int] = {}
    station_meta: Dict[str, dict] = {}
    reg_cfg = cfg["simulation"]["registration_residual"]

    for i, station in enumerate(stations):
        s_rng = np.random.default_rng(seed + 10007 * (i + 1))
        pts, meta = simulate_station_scan(
            surface_points=surface_points,
            surface_normals=surface_normals,
            source_face_ids=sampled_face_ids,
            station_xyz=np.asarray(station.xyz, dtype=float),
            cfg=cfg,
            rng=s_rng,
            mesh_rt=mesh_rt,
        )
        if bool(reg_cfg.get("enabled", False)) and i > 0:
            pts = apply_registration_residual(
                pts,
                s_rng,
                translation_std=float(reg_cfg.get("translation_std", 0.0)),
                rotation_std_deg=float(reg_cfg.get("rotation_std_deg", 0.0)),
            )
        points_per_station[station.name] = int(len(pts))
        station_meta[station.name] = meta
        station_outputs.append(pts)

        if bool(cfg.get("save_per_station_csv", False)):
            write_xyz_csv(out_dir / f"{case_id}_station_{station.name}.csv", pts)

    merged = np.concatenate(station_outputs, axis=0) if station_outputs else np.empty((0, 3), dtype=float)
    out_csv = out_dir / f"{case_id}.csv"
    write_xyz_csv(out_csv, merged)

    if bool(cfg.get("save_case_summary_json", False)):
        summary = {
            "case_id": case_id,
            "obj_path": str(obj_path),
            "n_sampled_surface_points": int(n_surface_points),
            "n_enabled_stations": len(stations),
            "points_per_station": points_per_station,
            "n_merged_points": int(len(merged)),
            "stations": [asdict(s) for s in stations],
            "station_meta": station_meta,
            "config": cfg,
        }
        with (out_dir / f"{case_id}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return SimResult(
        case_id=case_id,
        n_sampled_surface_points=n_surface_points,
        n_enabled_stations=len(stations),
        points_per_station=points_per_station,
        n_merged_points=int(len(merged)),
        output_csv=str(out_csv),
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate TLS-like scan point clouds from deformed OBJ meshes in batch.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional override for data_root in config.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional override for output_dir in config.")
    parser.add_argument("--use_stations", type=str, default=None, help="Comma-separated station names to enable for this run.")
    parser.add_argument("--case_ids", type=str, default=None, help="Comma-separated case ids to process. Default: all common ids.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = validate_and_prepare_config(cfg)

    data_root = Path(args.data_root or cfg.get("data_root", "data_Lab"))
    obj_dir = data_root / cfg.get("obj_dir", "OBJ")
    disp_dir = data_root / cfg.get("displacement_dir", "displacement")
    out_dir = Path(args.out_dir or cfg.get("output_dir", str(data_root / "cloudpoint")))
    ensure_dir(out_dir)

    if not obj_dir.exists():
        raise FileNotFoundError(f"OBJ directory not found: {obj_dir}")
    if not disp_dir.exists():
        raise FileNotFoundError(f"Displacement directory not found: {disp_dir}")

    override_names = [s.strip() for s in args.use_stations.split(",")] if args.use_stations else None
    stations = get_enabled_stations(cfg, override_names=override_names)

    common_ids = list_common_case_ids(obj_dir, disp_dir)
    if args.case_ids:
        requested = {s.strip() for s in args.case_ids.split(",")}
        case_ids = [cid for cid in common_ids if cid in requested]
    else:
        case_ids = common_ids
    if not case_ids:
        raise RuntimeError("No matching case ids found between OBJ and displacement directories.")

    base_seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    results: List[SimResult] = []
    for idx, case_id in enumerate(case_ids):
        obj_path = obj_dir / f"{case_id}.obj"
        if not obj_path.exists():
            candidates = list(obj_dir.glob(f"{case_id}.*"))
            obj_candidates = [p for p in candidates if p.suffix.lower() == ".obj"]
            if not obj_candidates:
                print(f"[WARN] Skip case {case_id}: OBJ not found.")
                continue
            obj_path = obj_candidates[0]

        print(f"[INFO] Processing case {case_id} from {obj_path.name}")
        result = simulate_case(
            case_id=case_id,
            obj_path=obj_path,
            out_dir=out_dir,
            cfg=cfg,
            stations=stations,
            seed=base_seed + idx * 104729,
        )
        results.append(result)

    if not results:
        raise RuntimeError("No cases were processed successfully.")

    if bool(cfg.get("save_batch_summary", False)):
        batch_summary_csv = out_dir / "batch_summary.csv"
        with batch_summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id",
                "n_sampled_surface_points",
                "n_enabled_stations",
                "n_merged_points",
                "output_csv",
            ])
            for r in results:
                writer.writerow([
                    r.case_id,
                    r.n_sampled_surface_points,
                    r.n_enabled_stations,
                    r.n_merged_points,
                    r.output_csv,
                ])

        with (out_dir / "batch_summary.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"[DONE] Batch summary written to: {batch_summary_csv}")

    print(f"[DONE] {len(results)} cases processed.")
    print(f"[DONE] Flat CSV outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
