"""Utility helpers for PINN3 (LBO-hybrid displacement/stress reconstruction).

Conventions
-----------
Units follow Abaqus export:
- length: mm
- stress: MPa (= N/mm^2)
- volume: mm^3

Important: coordinate normalization is ONLY for neural-network inputs.
All physics losses (constitutive, equilibrium) should be computed in
physical coordinates using derivatives precomputed from the physical mesh.

This module is intentionally self-contained (numpy/pandas/torch only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------------
# basic tensor helpers
# -----------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute sum(w*x)/sum(w) with broadcasting.

    Parameters
    ----------
    x : (...,)
    w : (...,) or broadcastable to x
    """
    wsum = torch.sum(w)
    return torch.sum(w * x) / (wsum + eps)


def as_torch(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


# -----------------------------
# stress/strain conversions
# -----------------------------

def sigma33_to_voigt6(sigma33: torch.Tensor) -> torch.Tensor:
    """( ...,3,3 ) -> ( ...,6 ) with ordering [11,22,33,12,13,23]."""
    s11 = sigma33[..., 0, 0]
    s22 = sigma33[..., 1, 1]
    s33 = sigma33[..., 2, 2]
    s12 = sigma33[..., 0, 1]
    s13 = sigma33[..., 0, 2]
    s23 = sigma33[..., 1, 2]
    return torch.stack([s11, s22, s33, s12, s13, s23], dim=-1)


def voigt6_to_sigma33(v6: torch.Tensor) -> torch.Tensor:
    """( ...,6 ) -> ( ...,3,3 ) assuming symmetry."""
    s11, s22, s33, s12, s13, s23 = v6.unbind(dim=-1)
    z = torch.zeros_like(s11)
    row0 = torch.stack([s11, s12, s13], dim=-1)
    row1 = torch.stack([s12, s22, s23], dim=-1)
    row2 = torch.stack([s13, s23, s33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def isotropic_sigma_from_eps_eng(eps6: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """Engineering strain ( ...,6 ) -> stress voigt6 ( ...,6 ).

    eps6 ordering: [E11,E22,E33,E12,E13,E23] where shear components are
    *engineering* gamma_ij (i!=j).

    Output sigma6 uses physical shear stress components [S12,S13,S23].
    """
    E_t = float(E)
    nu_t = float(nu)
    lam = E_t * nu_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
    mu = E_t / (2.0 * (1.0 + nu_t))

    e11, e22, e33, g12, g13, g23 = eps6.unbind(dim=-1)
    tr = e11 + e22 + e33

    s11 = lam * tr + 2.0 * mu * e11
    s22 = lam * tr + 2.0 * mu * e22
    s33 = lam * tr + 2.0 * mu * e33

    # sigma_ij = 2*mu*eps_ij = mu*gamma_ij
    s12 = mu * g12
    s13 = mu * g13
    s23 = mu * g23

    return torch.stack([s11, s22, s33, s12, s13, s23], dim=-1)


def isotropic_eps_from_sigma_voigt(sig6: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """Stress voigt6 ( ...,6 ) -> engineering strain eps6 ( ...,6 )."""
    E_t = float(E)
    nu_t = float(nu)
    s11, s22, s33, s12, s13, s23 = sig6.unbind(dim=-1)

    e11 = (s11 - nu_t * (s22 + s33)) / E_t
    e22 = (s22 - nu_t * (s11 + s33)) / E_t
    e33 = (s33 - nu_t * (s11 + s22)) / E_t

    # gamma_ij = sigma_ij / mu
    mu = E_t / (2.0 * (1.0 + nu_t))
    g12 = s12 / mu
    g13 = s13 / mu
    g23 = s23 / mu

    return torch.stack([e11, e22, e33, g12, g13, g23], dim=-1)


# -----------------------------
# GP coordinate matching (Abaqus CSV -> npz GP order)
# -----------------------------

def _greedy_unique_assignment(dist2: np.ndarray) -> np.ndarray:
    """Greedy one-to-one assignment for a square distance matrix.

    Returns perm p such that src[p[i]] matches tgt[i].
    """
    n = dist2.shape[0]
    # list all pairs (i,j) sorted by distance
    pairs = [(dist2[i, j], i, j) for i in range(n) for j in range(n)]
    pairs.sort(key=lambda x: x[0])

    perm = -np.ones((n,), dtype=np.int64)
    used = np.zeros((n,), dtype=bool)
    for d, i, j in pairs:
        if perm[i] >= 0:
            continue
        if used[j]:
            continue
        perm[i] = j
        used[j] = True
        if used.all():
            break

    if (perm < 0).any():
        raise RuntimeError("GP matching failed: could not build a bijection")
    return perm


def match_gp_by_xyz(
    tgt_xyz: np.ndarray,
    src_xyz: np.ndarray,
    tol_mm: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Match GP points by coordinates.

    Parameters
    ----------
    tgt_xyz : (ngp,3) target coords (npz order)
    src_xyz : (ngp,3) source coords (Abaqus CSV, usually ordered by IP)

    Returns
    -------
    perm : (ngp,) indices into src_xyz so that src_xyz[perm[i]] ~= tgt_xyz[i]
    max_dist : float
    """
    tgt = np.asarray(tgt_xyz, dtype=np.float64)
    src = np.asarray(src_xyz, dtype=np.float64)
    if tgt.shape != src.shape:
        raise ValueError(f"shape mismatch: tgt {tgt.shape} vs src {src.shape}")

    diff = tgt[:, None, :] - src[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)  # (ngp,ngp)
    perm = _greedy_unique_assignment(dist2)
    max_dist = float(np.sqrt(np.max(dist2[np.arange(tgt.shape[0]), perm])))
    if max_dist > float(tol_mm):
        raise RuntimeError(f"GP xyz matching exceeds tol: max_dist={max_dist:.3e} mm > tol={tol_mm:.3e} mm")
    return perm, max_dist


def build_gp_perm_map_from_csv(
    gp_csv_path: str,
    elem_ids_npz: np.ndarray,
    gp_xyz_npz: np.ndarray,
    tol_mm: float = 1e-3,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Build per-element permutation mapping Abaqus (sorted by IP) -> npz gp order.

    Returns perm_map with shape (n_elem, ngp), where
    sigma_npz_order[e, i] = sigma_abaqus_sorted_by_ip[e, perm_map[e,i]].

    Notes
    -----
    - Requires the CSV to contain columns: ElemId, IP, X, Y, Z.
    - Works for ngp=8 (C3D20R) and also ngp=27 if provided.
    """
    df = pd.read_csv(gp_csv_path)
    need = ['ElemId', 'IP', 'X', 'Y', 'Z']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"GP csv missing column {c}: {gp_csv_path}")

    df = df[need].copy()
    df['ElemId'] = df['ElemId'].astype(np.int64)
    df['IP'] = df['IP'].astype(np.int64)

    elem_ids_npz = np.asarray(elem_ids_npz, dtype=np.int64)
    n_elem = int(elem_ids_npz.shape[0])
    gp_xyz_npz = np.asarray(gp_xyz_npz, dtype=np.float64)
    if gp_xyz_npz.ndim != 3:
        raise ValueError(f"gp_xyz_npz must be (n_elem,ngp,3), got {gp_xyz_npz.shape}")
    ngp = int(gp_xyz_npz.shape[1])

    elem_id2idx = {int(eid): i for i, eid in enumerate(elem_ids_npz.tolist())}
    perm_map = -np.ones((n_elem, ngp), dtype=np.int64)

    max_dist_overall = 0.0
    miss = 0

    # enforce a deterministic source order: sorted by IP
    df = df.sort_values(['ElemId', 'IP'])

    for eid, g in df.groupby('ElemId', sort=False):
        eid = int(eid)
        if eid not in elem_id2idx:
            continue
        eidx = elem_id2idx[eid]
        xyz_src = g[['X', 'Y', 'Z']].to_numpy(np.float64)
        if xyz_src.shape[0] != ngp:
            raise RuntimeError(f"ElemId={eid} has {xyz_src.shape[0]} rows, expected ngp={ngp}.")
        perm, md = match_gp_by_xyz(gp_xyz_npz[eidx], xyz_src, tol_mm=tol_mm)
        perm_map[eidx, :] = perm
        if md > max_dist_overall:
            max_dist_overall = md

    miss = int(np.sum(perm_map[:, 0] < 0))
    if miss > 0:
        raise RuntimeError(f"perm_map missing for {miss}/{n_elem} elements. Check GP csv coverage.")

    if verbose:
        print(f"[gp-perm] built by xyz from: {os.path.basename(gp_csv_path)} | max_dist_overall={max_dist_overall:.3e} mm")
    return perm_map, float(max_dist_overall)


def load_gp_sigma_case_with_perm(
    gp_csv_path: str,
    elem_ids_npz: np.ndarray,
    perm_map: np.ndarray,
) -> np.ndarray:
    """Load Abaqus GP stresses and reorder to npz GP order using perm_map.

    Returns
    -------
    sigma_all : (n_elem,ngp,6) float32
    """
    df = pd.read_csv(gp_csv_path)
    need = ['ElemId', 'IP', 'S11', 'S22', 'S33', 'S12', 'S13', 'S23']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"GP csv missing column {c}: {gp_csv_path}")

    df = df[need].copy()
    df['ElemId'] = df['ElemId'].astype(np.int64)
    df['IP'] = df['IP'].astype(np.int64)
    df = df.sort_values(['ElemId', 'IP'])

    elem_ids_npz = np.asarray(elem_ids_npz, dtype=np.int64)
    n_elem = int(elem_ids_npz.shape[0])
    ngp = int(perm_map.shape[1])

    elem_id2idx = {int(eid): i for i, eid in enumerate(elem_ids_npz.tolist())}
    sigma_all = np.zeros((n_elem, ngp, 6), dtype=np.float32)

    for eid, g in df.groupby('ElemId', sort=False):
        eid = int(eid)
        if eid not in elem_id2idx:
            continue
        eidx = elem_id2idx[eid]
        gi = g.to_numpy()
        if gi.shape[0] != ngp:
            raise RuntimeError(f"ElemId={eid} has {gi.shape[0]} rows, expected ngp={ngp}.")
        sig_src = gi[:, 2:8].astype(np.float32)  # (ngp,6)
        sig = sig_src[perm_map[eidx], :]
        sigma_all[eidx, :, :] = sig

    return sigma_all


def load_gp_eps_case_with_perm(
    gp_csv_path: str,
    elem_ids_npz: np.ndarray,
    perm_map: np.ndarray,
) -> np.ndarray:
    """Load Abaqus GP engineering strains and reorder to npz GP order.

    Returns
    -------
    eps_all : (n_elem,ngp,6) float32 with ordering [E11,E22,E33,E12,E13,E23].
    """
    df = pd.read_csv(gp_csv_path)
    need = ['ElemId', 'IP', 'E11', 'E22', 'E33', 'E12', 'E13', 'E23']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"GP csv missing column {c}: {gp_csv_path}")

    df = df[need].copy()
    df['ElemId'] = df['ElemId'].astype(np.int64)
    df['IP'] = df['IP'].astype(np.int64)
    df = df.sort_values(['ElemId', 'IP'])

    elem_ids_npz = np.asarray(elem_ids_npz, dtype=np.int64)
    n_elem = int(elem_ids_npz.shape[0])
    ngp = int(perm_map.shape[1])

    elem_id2idx = {int(eid): i for i, eid in enumerate(elem_ids_npz.tolist())}
    eps_all = np.zeros((n_elem, ngp, 6), dtype=np.float32)

    for eid, g in df.groupby('ElemId', sort=False):
        eid = int(eid)
        if eid not in elem_id2idx:
            continue
        eidx = elem_id2idx[eid]
        gi = g.to_numpy()
        if gi.shape[0] != ngp:
            raise RuntimeError(f"ElemId={eid} has {gi.shape[0]} rows, expected ngp={ngp}.")
        eps_src = gi[:, 2:8].astype(np.float32)  # (ngp,6)
        eps = eps_src[perm_map[eidx], :]
        eps_all[eidx, :, :] = eps

    return eps_all


# -----------------------------
# NPZ / CSV loading & reordering
# -----------------------------

def load_gp_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_phi_node_csv(
    path: str,
    prefix: str,
    n_modes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load node-wise LBO modes from CSV.

    CSV must contain first column node id ("NodeId" or first column),
    and mode columns named like f"{prefix}1"..f"{prefix}{n_modes}".

    Returns node_ids (N,) int64 and phi (N,n_modes) float32.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise RuntimeError(f"bad phi csv: {path}")

    # node id column
    if 'NodeId' in df.columns:
        nid = df['NodeId'].to_numpy(np.int64)
    elif 'NodeIds' in df.columns:
        nid = df['NodeIds'].to_numpy(np.int64)
    else:
        nid = df.iloc[:, 0].to_numpy(np.int64)

    cols = [f"{prefix}{i}" for i in range(1, n_modes + 1)]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"{path}: missing column {c}")
    phi = df[cols].to_numpy(np.float32)
    return nid, phi


def reorder_rows_by_node_ids(
    target_node_ids: np.ndarray,
    src_node_ids: np.ndarray,
    src_vals: np.ndarray,
) -> np.ndarray:
    """Reorder src_vals to match target_node_ids.

    Parameters
    ----------
    target_node_ids : (N,) target ordering
    src_node_ids    : (M,) source ids
    src_vals        : (M,...) values

    Returns
    -------
    out : (N,...) values aligned to target
    """
    tgt = np.asarray(target_node_ids, dtype=np.int64)
    src_ids = np.asarray(src_node_ids, dtype=np.int64)
    idx = {int(n): i for i, n in enumerate(src_ids.tolist())}

    out_shape = (tgt.shape[0],) + tuple(src_vals.shape[1:])
    out = np.zeros(out_shape, dtype=src_vals.dtype)
    for i, nid in enumerate(tgt.tolist()):
        j = idx.get(int(nid), None)
        if j is None:
            raise RuntimeError(f"node id {nid} not found in source")
        out[i] = src_vals[j]
    return out


def load_obs_disp(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    need = ['NodeIds', 'U11', 'U22', 'U33']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path}: missing column {c}")
    ids = df['NodeIds'].to_numpy(np.int64)
    val = df[['U11', 'U22', 'U33']].to_numpy(np.float32)
    return ids, val


def load_obs_strain(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    need = ['NodeIds', 'E11', 'E22', 'E33', 'E12', 'E13', 'E23']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path}: missing column {c}")
    ids = df['NodeIds'].to_numpy(np.int64)
    val = df[['E11', 'E22', 'E33', 'E12', 'E13', 'E23']].to_numpy(np.float32)
    return ids, val


# -----------------------------
# Weak equilibrium assembly
# -----------------------------


def build_node_degree(conn20: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Number of incident elements for each node (based on connectivity)."""
    idx = conn20.reshape(-1)
    ones = torch.ones_like(idx, dtype=torch.int32)
    deg = torch.zeros((n_nodes,), device=conn20.device, dtype=torch.int32)
    deg.index_add_(0, idx, ones)
    return deg


def complete_node_mask_for_elem_batch(
    conn_batch: torch.Tensor,
    deg_total: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Mask of nodes whose all incident elements are included in the current batch."""
    idx = conn_batch.reshape(-1)
    ones = torch.ones_like(idx, dtype=torch.int32)
    deg_b = torch.zeros((n_nodes,), device=conn_batch.device, dtype=torch.int32)
    deg_b.index_add_(0, idx, ones)
    return deg_b == deg_total


def assemble_internal_nodal_forces(
    conn20: torch.Tensor,
    sigma_gp: torch.Tensor,
    w_dNdx: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Assemble internal nodal forces from GP stresses.

    Parameters
    ----------
    conn20  : (B,20) long, 0-based node indices
    sigma_gp: (B,ngp,3,3)
    w_dNdx  : (B,ngp,20,3) equals dN_dx_gp * dV (physical)

    Returns
    -------
    f_int : (n_nodes,3)
    """
    # element nodal force contributions: f_eai = sum_{g,j} sigma_{g,i,j} * dNdx_{g,a,j} * dV_g
    f_e = torch.einsum('bgij,bgaj->bai', sigma_gp, w_dNdx)  # (B,20,3)

    f = torch.zeros((n_nodes, 3), device=sigma_gp.device, dtype=sigma_gp.dtype)
    f.index_add_(0, conn20.reshape(-1), f_e.reshape(-1, 3))
    return f


@dataclass
class WeakEquilibriumLoss:
    """Compute nodal assembled weak equilibrium residual loss."""

    conn20_all: torch.Tensor            # (E,20)
    w_dNdx_all: torch.Tensor            # (E,ngp,20,3)
    deg_total: torch.Tensor             # (n_nodes,)
    boundary_mask: torch.Tensor         # (n_nodes,) bool
    F0: float = 5000.0

    def loss(
        self,
        sigma_gp_all: torch.Tensor,
        elem_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute equilibrium loss.

        Parameters
        ----------
        sigma_gp_all: (E,ngp,3,3)
        elem_batch  : (B,) long indices. If None -> use all.

        Returns
        -------
        loss_eq : scalar tensor
        stats   : dict
        """
        if elem_batch is None:
            conn = self.conn20_all
            sigma = sigma_gp_all
            w_dNdx = self.w_dNdx_all
            complete = torch.ones_like(self.boundary_mask, dtype=torch.bool)
        else:
            conn = self.conn20_all[elem_batch]
            sigma = sigma_gp_all[elem_batch]
            w_dNdx = self.w_dNdx_all[elem_batch]
            complete = complete_node_mask_for_elem_batch(conn, self.deg_total, int(self.deg_total.shape[0]))

        f_int = assemble_internal_nodal_forces(conn, sigma, w_dNdx, n_nodes=int(self.deg_total.shape[0]))

        mask = (~self.boundary_mask) & complete
        r = f_int[mask]
        if r.numel() == 0:
            # no complete nodes in this batch -> return 0 (and warn via stats)
            loss = sigma_gp_all.new_tensor(0.0)
            return loss, {
                'n_nodes_used': 0,
                'mean_res_N': 0.0,
                'p90_res_N': 0.0,
                'max_res_N': 0.0,
            }

        res = torch.linalg.norm(r, dim=1)  # (n_used,)
        loss = (res / float(self.F0)).pow(2).mean()

        # small stats for logging
        with torch.no_grad():
            res_cpu = res.detach().cpu().numpy()
            stats = {
                'n_nodes_used': int(res_cpu.shape[0]),
                'mean_res_N': float(res_cpu.mean()),
                'p90_res_N': float(np.percentile(res_cpu, 90)),
                'max_res_N': float(res_cpu.max()),
            }
        return loss, stats


# -----------------------------
# Mesh utilities
# -----------------------------

def build_unique_edges_from_conn20(conn20: np.ndarray) -> np.ndarray:
    """Build unique undirected edges from a (E,20) connectivity (0-based).

    Edge set includes all pairs (i,j) that appear as an element edge in a hex20.
    This is a heuristic smoothness graph (not a full surface mesh).

    Returns edges (n_edges,2) int64.
    """
    conn20 = np.asarray(conn20, dtype=np.int64)
    if conn20.ndim != 2 or conn20.shape[1] != 20:
        raise ValueError(f"conn20 must be (E,20), got {conn20.shape}")

    # Hex20 has 12 corner edges + edges to mid-side nodes.
    # We construct edges by connecting each midside node to its two end corner nodes.
    # Node ordering assumed Abaqus C3D20/C3D20R standard.
    # corners: 0-7, mids: 8-19
    corner_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    mids_to_corners = [
        (8, 0, 1),
        (9, 1, 2),
        (10, 2, 3),
        (11, 3, 0),
        (12, 4, 5),
        (13, 5, 6),
        (14, 6, 7),
        (15, 7, 4),
        (16, 0, 4),
        (17, 1, 5),
        (18, 2, 6),
        (19, 3, 7),
    ]

    edges_set = set()
    for e in range(conn20.shape[0]):
        c = conn20[e]
        # corner-corner edges
        for a, b in corner_edges:
            i = int(c[a]); j = int(c[b])
            if i == j:
                continue
            if i < j:
                edges_set.add((i, j))
            else:
                edges_set.add((j, i))
        # midside connections
        for m, a, b in mids_to_corners:
            im = int(c[m]); ia = int(c[a]); ib = int(c[b])
            for i, j in [(im, ia), (im, ib)]:
                if i == j:
                    continue
                if i < j:
                    edges_set.add((i, j))
                else:
                    edges_set.add((j, i))

    edges = np.array(sorted(edges_set), dtype=np.int64)
    return edges
