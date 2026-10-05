# PINN3_d20_hybrid_gp_v3_smooth20_pretrain.py
# Pretrain u/s/load nets independently on multi-case truth data.
# Notes:
# - Uses CASES/{OBS_U,OBS_E,FULL,GP} + manifest.csv (cf1~cf3) for truth.
# - Saves normalization (xyz, phi stats, u0/eps0/sigma0/F0) for finetune alignment.

"""
Example (PowerShell):
python .\\PINN3_d20_hybrid_gp_v3_smooth20_pretrain.py `
  --gp_npz Lab-use-solid-20-R-GP_topo_derivs.npz `
  --hybrid_csv Lab-use-solid-20-LBO-R.csv `
  --cases_root CASES `
  --abaqus_ip_perm "0,1,3,2,4,5,7,6" `
  --E 2.05e5 --nu 0.30 `
  --hybrid_modes_u 20 --hybrid_modes_s 50 `
  --cond_force 1 --use_load_net 1 `
  --load_use_disp 1 --load_use_strain 1`
  --epochs_u 4000 --epochs_s 4000 --epochs_f 2000 `
  --val_ratio 0.7 `
  --case_batch 10 --batch_elems 560 `
  --lr_u 1e-3 --lr_s 1e-3 --lr_f 5e-4 `
  --norm_save ckpt/pre_norm.npz `
  --save_u_best ckpt/pre_u_best.pt `
  --save_s_best ckpt/pre_s_best.pt `
  --save_f_best ckpt/pre_f_best.pt `
  --device cuda
"""

import argparse
import glob
import os
import re
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinn3_utils import build_gp_perm_map_from_csv, load_gp_sigma_case_with_perm


# -----------------------------
# 1) utils: Voigt <-> tensor
# -----------------------------

def sigma33_to_voigt6(sig: torch.Tensor) -> torch.Tensor:
    # sig: (...,3,3)
    return torch.stack([
        sig[..., 0, 0], sig[..., 1, 1], sig[..., 2, 2],
        sig[..., 0, 1], sig[..., 0, 2], sig[..., 1, 2]
    ], dim=-1)


# -----------------------------
# 2) MLPs
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=256, depth=5, act="silu"):
        super().__init__()
        if act == "tanh":
            A = nn.Tanh
        elif act == "relu":
            A = nn.ReLU
        else:
            A = nn.SiLU
        layers = [nn.Linear(in_dim, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DispNet(nn.Module):
    # (x,y,z, phi_hybrid[Ku], [cond(3)]) -> u(3)
    def __init__(self, Ku, width=256, depth=5, cond_dim=0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.mlp = MLP(3 + int(Ku) + self.cond_dim, 3, width=width, depth=depth, act="silu")

    def forward(self, xyz, phi, cond=None):
        if self.cond_dim > 0:
            if cond is None:
                cond = xyz.new_zeros((xyz.shape[0], self.cond_dim))
            x = torch.cat([xyz, phi, cond], dim=-1)
        else:
            x = torch.cat([xyz, phi], dim=-1)
        return self.mlp(x)


class StressCoeffNet(nn.Module):
    # (x,y,z, [cond(3)]) -> P (N,3,Ks)
    #
    # Paper-aligned (3D): three coefficient groups p1(x), p2(x), p3(x),
    # each outputs Ks scalars, assembled as sigma_ij = sum_k P[i,k] * d(phi_k)/dx_j.
    #
    # NOTE:
    #   We DO NOT feed phi into s_net. Geometry enters sigma through grad(phi) only.
    #   This matches FinitePINN's strong-form derivation where p = p(x, cond) and
    #   div(sigma) uses partial derivatives w.r.t x with cond held constant.
    def __init__(self, Ks, width=256, depth=6, cond_dim=0):
        super().__init__()
        self.Ks = int(Ks)
        self.cond_dim = int(cond_dim)
        in_dim = 3 + self.cond_dim

        # three groups (like 2D px/py, here 3D p1/p2/p3)
        self.mlp_p1 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")
        self.mlp_p2 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")
        self.mlp_p3 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")

    def forward(self, xyz, cond=None):
        if self.cond_dim > 0:
            if cond is None:
                cond = xyz.new_zeros((xyz.shape[0], self.cond_dim))
            x = torch.cat([xyz, cond], dim=-1)
        else:
            x = xyz
        p1 = self.mlp_p1(x)  # (N,K)
        p2 = self.mlp_p2(x)  # (N,K)
        p3 = self.mlp_p3(x)  # (N,K)
        P = torch.stack([p1, p2, p3], dim=1)  # (N,3,K)
        return P


def _set_encoder_point_mlp(in_dim: int, embed_dim: int, hidden: int, depth: int):
    return MLP(in_dim, embed_dim, width=hidden, depth=depth, act="silu")


class LoadNetSet(nn.Module):
    """Permutation-invariant load net with learned pooling + magnitude stats.
    Returns f_hat (3,) dimensionless; physical force = f_hat * F0.
    """
    def __init__(self, embed_dim=64, hidden=128, depth=3):
        super().__init__()
        self.embed_dim = int(embed_dim)

        self.disp_mlp = _set_encoder_point_mlp(in_dim=6, embed_dim=embed_dim, hidden=hidden, depth=depth)
        self.strain_mlp = _set_encoder_point_mlp(in_dim=9, embed_dim=embed_dim, hidden=hidden, depth=depth)

        self.disp_score = nn.Linear(embed_dim, 1, bias=False)
        self.strain_score = nn.Linear(embed_dim, 1, bias=False)

        # mean(D) + rms(D) for disp, mean(D) + rms(D) for strain => 4D
        self.head = MLP(in_dim=4 * embed_dim, out_dim=3, width=hidden, depth=max(2, depth), act="silu")

    @staticmethod
    def _attn_pool_mean_rms(feat: torch.Tensor, score_layer: nn.Module):
        # feat: (N, D)
        s = score_layer(feat).squeeze(-1)     # (N,)
        w = torch.softmax(s, dim=0).unsqueeze(-1)  # (N,1), sum=1

        mean = (feat * w).sum(dim=0, keepdim=True)  # (1,D)
        rms  = torch.sqrt((feat.pow(2) * w).sum(dim=0, keepdim=True) + 1e-12)  # (1,D)

        return torch.cat([mean, rms], dim=1)  # (1, 2D)

    def forward(self, disp_pts, strain_pts=None):
        device = next(self.parameters()).device
        z2D = torch.zeros((1, 2 * self.embed_dim), device=device)

        if (disp_pts is not None) and (disp_pts.numel() > 0):
            disp_feat = self.disp_mlp(disp_pts)  # (Nu,D)
            du = self._attn_pool_mean_rms(disp_feat, self.disp_score)  # (1,2D)
        else:
            du = z2D

        if (strain_pts is not None) and (strain_pts.numel() > 0):
            strain_feat = self.strain_mlp(strain_pts)  # (Ne,D)
            de = self._attn_pool_mean_rms(strain_feat, self.strain_score)  # (1,2D)
        else:
            de = z2D

        h = torch.cat([du, de], dim=1)  # (1,4D)
        f_hat = self.head(h).squeeze(0) # (3,)
        return f_hat


# -----------------------------
# 3) core ops
# -----------------------------

def assemble_sigma_from_P(P_3K: torch.Tensor, gradphi_K3: torch.Tensor) -> torch.Tensor:
    """
    Paper-aligned assembly:
      P: (N,3,K), gradphi: (N,K,3) -> sigma: (N,3,3)
      sigma_ij = sum_k P[i,k] * d(phi_k)/dx_j
    """
    # einsum: (N,3,K) x (N,K,3) -> (N,3,3)
    sigma = torch.einsum('nik,nkj->nij', P_3K, gradphi_K3)
    return sigma



def weighted_mean(v, w):
    # v: (N,) or (N,*) ; w: (N,1) or (N,)
    if w.dim() == 1:
        w = w.view(-1, 1)
    if v.dim() == 1:
        v = v.view(-1, 1)
    num = (v * w).sum()
    den = w.sum() * v.shape[1]
    return num / (den + 1e-12)


# -----------------------------
# 4) CSV loaders
# -----------------------------

def load_gp_npz(path: str) -> dict:
    pkg = np.load(path, allow_pickle=True)
    return {k: pkg[k] for k in pkg.files}


def load_lbo_phi(csv_path, prefix, n_modes):
    df = pd.read_csv(csv_path)
    if 'NodeIds' not in df.columns:
        raise RuntimeError(f"{csv_path}: missing NodeIds column")
    node_ids = df['NodeIds'].to_numpy(np.int64)

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    mode_cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            mode_cols.append((int(m.group(1)), c))
    if len(mode_cols) == 0:
        raise RuntimeError(f"{csv_path}: no mode columns matched prefix='{prefix}'")

    mode_cols.sort(key=lambda x: x[0])
    cols = [c for _, c in mode_cols[:n_modes]]
    phi = df[cols].to_numpy(np.float64)
    if phi.shape[1] < n_modes:
        raise RuntimeError(f"{csv_path}: requested n_modes={n_modes}, but found {phi.shape[1]}")

    xyz = None
    if any(c.lower() == 'x' for c in df.columns) and any(c.lower() == 'y' for c in df.columns) and any(c.lower() == 'z' for c in df.columns):
        xyz = df[[
            [c for c in df.columns if c.lower() == 'x'][0],
            [c for c in df.columns if c.lower() == 'y'][0],
            [c for c in df.columns if c.lower() == 'z'][0],
        ]].to_numpy(np.float64)
    return node_ids, xyz, phi


def reorder_rows_by_node_ids(src_ids: np.ndarray, src_vals: np.ndarray, dst_ids: np.ndarray, name=''):
    idx = {int(n): i for i, n in enumerate(src_ids)}
    out = np.zeros((dst_ids.shape[0], src_vals.shape[1]), dtype=src_vals.dtype)
    for i, nid in enumerate(dst_ids):
        if int(nid) not in idx:
            raise RuntimeError(f"{name}: missing node id {nid}")
        out[i] = src_vals[idx[int(nid)]]
    return out


def load_obs_disp(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    need = ['NodeIds', 'U11', 'U22', 'U33']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path}: missing column {c}")
    ids = df['NodeIds'].to_numpy(np.int64)
    val = df[['U11', 'U22', 'U33']].to_numpy(np.float64)
    return ids, val


def load_obs_strain(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    need = ['NodeIds', 'E11', 'E22', 'E33', 'E12', 'E13', 'E23']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path}: missing column {c}")
    ids = df['NodeIds'].to_numpy(np.int64)
    val = df[['E11', 'E22', 'E33', 'E12', 'E13', 'E23']].to_numpy(np.float64)
    return ids, val


def read_manifest_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if 'sample_id' not in df.columns:
        raise RuntimeError(f"manifest missing sample_id: {path}")
    for c in ['cf1', 'cf2', 'cf3']:
        if c not in df.columns:
            raise RuntimeError(f"manifest missing {c}: {path}")
    sids = df['sample_id'].astype(str).to_numpy()
    f = df[['cf1', 'cf2', 'cf3']].to_numpy(np.float32)
    return sids, f


def resolve_case_csv(cases_root: str, subdir: str, sample_id: str) -> str:
    p = os.path.join(cases_root, subdir, f"{sample_id}.csv")
    if os.path.exists(p):
        return p
    cand = glob.glob(os.path.join(cases_root, subdir, f"{sample_id}*.csv"))
    if cand:
        return cand[0]
    raise RuntimeError(f"case csv not found: {cases_root}/{subdir}/{sample_id}(.csv)")


def load_gp_sigma_case(csv_path: str, elem_id2idx: Dict[int, int], n_elem: int, abaqus_perm=None) -> np.ndarray:
    df = pd.read_csv(csv_path)
    need = ['ElemId', 'IP', 'S11', 'S22', 'S33', 'S12', 'S13', 'S23']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"GP csv missing column {c}: {csv_path}")
    df = df[need].copy()
    df['ElemId'] = df['ElemId'].astype(np.int64)
    df['IP'] = df['IP'].astype(np.int64)
    df = df.sort_values(['ElemId', 'IP'])

    sigma_all = np.zeros((n_elem, 8, 6), dtype=np.float32)
    for eid, g in df.groupby('ElemId', sort=False):
        eid = int(eid)
        if eid not in elem_id2idx:
            continue
        gi = g.to_numpy()
        if gi.shape[0] != 8:
            raise RuntimeError(f"ElemId={eid} has {gi.shape[0]} rows (need 8).")
        sig = gi[:, 2:8].astype(np.float32)  # (8,6)
        if abaqus_perm is not None:
            sig = sig[abaqus_perm, :]
        sigma_all[elem_id2idx[eid], :, :] = sig
    return sigma_all


# -----------------------------
# 5) normalization I/O
# -----------------------------

def save_norm(path: str, **kwargs):
    d = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.savez(path, **d)


def load_norm(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    return out


# -----------------------------
# 6) main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--gp_npz', type=str, required=True)
    ap.add_argument('--hybrid_csv', type=str, required=True)
    ap.add_argument('--hybrid_prefix', type=str, default='phi_hybrid_')
    ap.add_argument('--hybrid_modes_u', type=int, default=50)
    ap.add_argument('--hybrid_modes_s', type=int, default=50)

    ap.add_argument('--cases_root', type=str, required=True)
    ap.add_argument('--manifest_csv', type=str, default='')
    ap.add_argument('--case_obs_u_subdir', type=str, default='OBS_U')
    ap.add_argument('--case_obs_e_subdir', type=str, default='OBS_E')
    ap.add_argument('--case_full_u_subdir', type=str, default='FULL')
    ap.add_argument('--case_gp_subdir', type=str, default='GP')

    ap.add_argument('--abaqus_ip_perm', type=str, default='', help='e.g. "0,1,3,2,4,5,7,6"')

    ap.add_argument('--E', type=float, required=True)
    ap.add_argument('--nu', type=float, required=True)

    ap.add_argument('--epochs_u', type=int, default=0)
    ap.add_argument('--epochs_s', type=int, default=0)
    ap.add_argument('--epochs_f', type=int, default=0)
    ap.add_argument('--steps_per_epoch', type=int, default=1)
    ap.add_argument('--case_batch', type=int, default=1)
    ap.add_argument('--batch_elems', type=int, default=256)
    ap.add_argument('--u_full_batch_nodes', type=int, default=0, help='0=use all nodes')

    ap.add_argument('--lr_u', type=float, default=1e-3)
    ap.add_argument('--lr_s', type=float, default=1e-3)
    ap.add_argument('--lr_f', type=float, default=1e-3)

    ap.add_argument('--eval_every', type=int, default=200)

    ap.add_argument('--use_load_net', type=int, default=1)
    ap.add_argument('--load_embed', type=int, default=64)
    ap.add_argument('--load_hidden', type=int, default=128)
    ap.add_argument('--load_depth', type=int, default=3)
    ap.add_argument('--load_use_disp', type=int, default=1, help='1: use displacement points as load_net input')
    ap.add_argument('--load_use_strain', type=int, default=1, help='1: use strain points as load_net input')

    ap.add_argument('--cond_force', type=int, default=1)

    ap.add_argument('--norm_xyz', type=int, default=1)
    ap.add_argument('--xyz_scale', type=str, default='rms', choices=['rms', 'maxrange'])
    ap.add_argument('--norm_phi', type=str, default='per_mode', choices=['none', 'global', 'per_mode'])
    ap.add_argument('--u0_from', type=str, default='full', choices=['full', 'obs'])
    ap.add_argument('--norm_save', type=str, default='')
    ap.add_argument('--norm_load', type=str, default='')

    ap.add_argument('--save_u_best', type=str, default='')
    ap.add_argument('--save_s_best', type=str, default='')
    ap.add_argument('--save_f_best', type=str, default='')

    ap.add_argument('--case_seed', type=int, default=0)
    ap.add_argument('--val_ratio', type=float, default=0.2,
                    help='fraction of cases held out for validation (0=disable)')
    ap.add_argument('--val_seed', type=int, default=0,
                    help='random seed for case split (defaults to --case_seed if 0)')
    ap.add_argument('--val_case_report', type=int, default=5,
                    help='print top-K val cases by |Fy error| for load_net (0=disable)')
    ap.add_argument('--ridge_lambda', type=float, default=1e-6,
                    help='ridge baseline (Fy only) lambda')
    # output organization
    ap.add_argument('--ckpt_root', type=str, default='ckpt')
    ap.add_argument('--project', type=str, default='pretrain')
    ap.add_argument('--log_csv', type=str, default='log.csv')

    # Abaqus GP order: prefer coordinate-based matching (requires X,Y,Z columns in GP.csv)
    ap.add_argument('--gp_match_by_xyz', type=int, default=1)
    ap.add_argument('--gp_match_tol_mm', type=float, default=1e-4)

    # Force normalization (used by finetune weak-eq loss too)
    ap.add_argument('--F0_fixed', type=float, default=5000.0,
                    help='if >0, use this constant F0 (N) for normalization; if <=0 use RMS(F) over train cases')

    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = ap.parse_args()

    device = torch.device(args.device)
    # --- output dir ---
    out_dir = os.path.join(args.ckpt_root, args.project)
    os.makedirs(out_dir, exist_ok=True)

    def _to_out(p: str, default_name: str) -> str:
        p = p if p is not None else ''
        if p == '' or p.lower() == 'none':
            return ''
        # if user gave a directory-less filename -> place under out_dir
        if (not os.path.isabs(p)) and (os.path.dirname(p) == ''):
            return os.path.join(out_dir, p)
        return p

    args.save_u_best = _to_out(args.save_u_best, 'u_best.pt')
    args.save_s_best = _to_out(args.save_s_best, 's_best.pt')
    args.save_f_best = _to_out(args.save_f_best, 'f_best.pt')
    args.norm_save = _to_out(args.norm_save, 'norm.npz')

    log_csv_path = _to_out(args.log_csv, 'log.csv')
    if log_csv_path:
        with open(log_csv_path, 'w', encoding='utf-8') as f:
            f.write('ep,split,loss_u,loss_sigma,loss_force\n')

    if int(args.load_use_disp) == 0 and int(args.load_use_strain) == 0:
        raise RuntimeError("--load_use_disp and --load_use_strain cannot both be 0.")
    rng = np.random.RandomState(int(args.case_seed))

    # --- load gp pack (authoritative for topology/geometry/derivs) ---
    pkg = load_gp_npz(args.gp_npz)
    node_ids_npz = np.asarray(pkg['node_ids'], dtype=np.int64)
    node_xyz_npz = np.asarray(pkg['V'], dtype=np.float64)

    # --- load hybrid modes @nodes from csv, then reorder to match npz node_ids ---
    nid_u, _, phi_u = load_lbo_phi(args.hybrid_csv, args.hybrid_prefix, args.hybrid_modes_u)
    nid_s, _, phi_s = load_lbo_phi(args.hybrid_csv, args.hybrid_prefix, args.hybrid_modes_s)
    if not np.array_equal(nid_u, nid_s):
        raise RuntimeError('hybrid_csv NodeIds mismatch between u and sigma loads')

    phi_u = reorder_rows_by_node_ids(nid_u, phi_u, node_ids_npz, name='phi_u')
    phi_s = reorder_rows_by_node_ids(nid_s, phi_s, node_ids_npz, name='phi_s')

    # --- case list ---
    manifest_csv = args.manifest_csv if args.manifest_csv else os.path.join(args.cases_root, 'manifest.csv')
    case_ids, F_case_true = read_manifest_csv(manifest_csv)
    if len(case_ids) == 0:
        raise RuntimeError(f"empty manifest: {manifest_csv}")

    # --- train/val split ---
    n_cases = int(len(case_ids))
    case_ids = np.asarray(case_ids)
    F_case_true = np.asarray(F_case_true)
    val_ratio = float(args.val_ratio)
    if (val_ratio > 0.0) and (n_cases > 1):
        split_seed = int(args.case_seed) if int(args.val_seed) == 0 else int(args.val_seed)
        split_rng = np.random.RandomState(split_seed)
        perm = np.arange(n_cases)
        split_rng.shuffle(perm)
        n_val = int(round(n_cases * val_ratio))
        n_val = max(1, min(n_val, n_cases - 1))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        train_idx = np.arange(n_cases)
        val_idx = np.array([], dtype=np.int64)
    train_case_ids = case_ids[train_idx]
    train_F_true = F_case_true[train_idx]
    val_case_ids = case_ids[val_idx]
    val_F_true = F_case_true[val_idx]
    print(f"[split] train={len(train_case_ids)} val={len(val_case_ids)} (ratio={val_ratio:.2f})")


    # --- Abaqus GP order matching: prefer coordinate-based matching (robust to IP ordering) ---
    perm_map = None
    if int(getattr(args, 'gp_match_by_xyz', 1)) == 1:
        ref_sid = int(train_case_ids[0])
        ref_gp_csv = resolve_case_csv(args.cases_root, args.case_gp_subdir, ref_sid)
        try:
            perm_map, _ = build_gp_perm_map_from_csv(
                ref_gp_csv,
                np.asarray(pkg['elem_ids'], dtype=np.int64),
                pkg['gp_xyz'],
                tol_mm=float(getattr(args, 'gp_match_tol_mm', 1e-4)),
                verbose=True,
            )
        except Exception as e:
            print(f"[warn] gp_match_by_xyz failed -> fallback to abaqus_ip_perm. reason: {e}")
            perm_map = None


    # --- preload OBS layout (assume consistent) ---
    u0_path = resolve_case_csv(args.cases_root, args.case_obs_u_subdir, case_ids[0])
    obs_u_ids, obs_u_val0 = load_obs_disp(u0_path)
    id2idx = {int(nid): i for i, nid in enumerate(node_ids_npz)}
    obs_u_idx = np.array([id2idx[int(n)] for n in obs_u_ids], dtype=np.int64)

    # try OBS_E (optional)OBS
    try:
        e0_path = resolve_case_csv(args.cases_root, args.case_obs_e_subdir, case_ids[0])
        obs_e_ids, obs_e_val0 = load_obs_strain(e0_path)
        obs_e_idx = np.array([id2idx[int(n)] for n in obs_e_ids], dtype=np.int64)
    except Exception:
        obs_e_ids, obs_e_idx = None, None

    # --- tensors ---
    xyz_t = torch.tensor(node_xyz_npz, device=device, dtype=torch.float32)
    phi_u_t = torch.tensor(phi_u, device=device, dtype=torch.float32)
    phi_s_t = torch.tensor(phi_s, device=device, dtype=torch.float32)

    # --- xyz normalization ---
    xyz_center = xyz_t.mean(dim=0, keepdim=True)
    if args.xyz_scale == 'maxrange':
        L0 = (xyz_t.max(dim=0).values - xyz_t.min(dim=0).values).max()
    else:
        L0 = torch.sqrt(((xyz_t - xyz_center) ** 2).sum(dim=1).mean() + 1e-12)
    L0 = torch.clamp(L0, min=1e-6)

    def norm_xyz(x):
        if args.norm_xyz == 0:
            return x
        return (x - xyz_center) / L0

    # --- phi normalization (inputs only) ---
    if args.norm_phi != 'none':
        if args.norm_phi == 'global':
            mu_u = phi_u_t.mean()
            sd_u = phi_u_t.std()
            mu_s = phi_s_t.mean()
            sd_s = phi_s_t.std()
            mu_u = mu_u.view(1, 1)
            sd_u = sd_u.view(1, 1)
            mu_s = mu_s.view(1, 1)
            sd_s = sd_s.view(1, 1)
        else:
            mu_u = phi_u_t.mean(dim=0, keepdim=True)
            sd_u = phi_u_t.std(dim=0, keepdim=True)
            mu_s = phi_s_t.mean(dim=0, keepdim=True)
            sd_s = phi_s_t.std(dim=0, keepdim=True)
        phi_u_in_t = (phi_u_t - mu_u) / (sd_u + 1e-12)
        phi_s_in_t = (phi_s_t - mu_s) / (sd_s + 1e-12)
    else:
        mu_u = torch.zeros((1, phi_u_t.shape[1]), device=device)
        sd_u = torch.ones((1, phi_u_t.shape[1]), device=device)
        mu_s = torch.zeros((1, phi_s_t.shape[1]), device=device)
        sd_s = torch.ones((1, phi_s_t.shape[1]), device=device)
        phi_u_in_t = phi_u_t
        phi_s_in_t = phi_s_t

    # --- gp tensors ---
    gp_xyz_t = torch.tensor(np.asarray(pkg['gp_xyz'], dtype=np.float32), device=device)
    gp_dV_t = torch.tensor(np.asarray(pkg['gp_dV'], dtype=np.float32), device=device)
    Ks = int(args.hybrid_modes_s)
    phi_h_gp_t = torch.tensor(np.asarray(pkg['phi_hybrid_gp'], dtype=np.float32), device=device)[:, :, :Ks]
    grad_h_gp_t = torch.tensor(np.asarray(pkg['grad_phi_hybrid_gp'], dtype=np.float32), device=device)[:, :, :Ks, :]
    if args.norm_phi != 'none':
        phi_h_gp_in_t = (phi_h_gp_t - mu_s) / (sd_s + 1e-12)
    else:
        phi_h_gp_in_t = phi_h_gp_t

    # --- element id map (for GP truth) ---
    if 'elem_ids' not in pkg:
        raise RuntimeError('gp_npz missing elem_ids; needed to align GP truth csv')
    elem_ids_np = np.asarray(pkg['elem_ids'], dtype=np.int64)
    elem_id2idx = {int(eid): i for i, eid in enumerate(elem_ids_np.tolist())}
    n_elem = int(elem_ids_np.shape[0])

    abaqus_perm = None
    if args.abaqus_ip_perm.strip():
        abaqus_perm = [int(x) for x in args.abaqus_ip_perm.strip().split(',')]
        if len(abaqus_perm) != 8:
            raise RuntimeError("--abaqus_ip_perm must have 8 integers (0..7).")

    # --- cache cases ---
    case_cache = {}

    def load_case_cached(sample_id: str, f_np: np.ndarray) -> dict:
        if sample_id in case_cache:
            return case_cache[sample_id]

        # obs_u (align to obs layout)
        obs_u_ids_c, obs_u_val_c = load_obs_disp(resolve_case_csv(args.cases_root, args.case_obs_u_subdir, sample_id))
        row_u = {int(n): i for i, n in enumerate(obs_u_ids_c.tolist())}
        obs_u_val = np.stack([obs_u_val_c[row_u[int(n)]] for n in obs_u_ids.tolist()], axis=0).astype(np.float32)

        # obs_e (optional)
        obs_e_val_t_local = None
        if obs_e_ids is not None:
            obs_e_ids_c, obs_e_val_c = load_obs_strain(resolve_case_csv(args.cases_root, args.case_obs_e_subdir, sample_id))
            row_e = {int(n): i for i, n in enumerate(obs_e_ids_c.tolist())}
            obs_e_val = np.stack([obs_e_val_c[row_e[int(n)]] for n in obs_e_ids.tolist()], axis=0).astype(np.float32)
            obs_e_val_t_local = torch.tensor(obs_e_val, device=device, dtype=torch.float32)

        # full u
        full_u_ids, full_u_val = load_obs_disp(resolve_case_csv(args.cases_root, args.case_full_u_subdir, sample_id))
        full_u_val = reorder_rows_by_node_ids(full_u_ids, full_u_val, node_ids_npz, name='u_full')
        u_full_t_local = torch.tensor(full_u_val.astype(np.float32), device=device)

        # gp sigma (Abaqus order -> npz order)
        gp_csv_path = resolve_case_csv(args.cases_root, args.case_gp_subdir, sample_id)
        if perm_map is not None:
            sigma_gp = load_gp_sigma_case_with_perm(gp_csv_path, elem_ids_np, perm_map)
        else:
            sigma_gp = load_gp_sigma_case(
                gp_csv_path,
                elem_id2idx,
                n_elem,
                abaqus_perm=abaqus_perm
            )
        sigma_gp_t_local = torch.tensor(sigma_gp, device=device, dtype=torch.float32)

        out = dict(
            sample_id=sample_id,
            F_true_t=torch.tensor(f_np.astype(np.float32), device=device),
            obs_u_val_t=torch.tensor(obs_u_val, device=device),
            obs_e_val_t=obs_e_val_t_local,
            u_full_t=u_full_t_local,
            sigma_gp_t=sigma_gp_t_local,
        )
        case_cache[sample_id] = out
        return out

    def load_obs_only(sample_id: str):
        obs_u_ids_c, obs_u_val_c = load_obs_disp(resolve_case_csv(args.cases_root, args.case_obs_u_subdir, sample_id))
        row_u = {int(n): i for i, n in enumerate(obs_u_ids_c.tolist())}
        obs_u_val = np.stack([obs_u_val_c[row_u[int(n)]] for n in obs_u_ids.tolist()], axis=0).astype(np.float32)

        obs_e_val = None
        if obs_e_ids is not None:
            obs_e_ids_c, obs_e_val_c = load_obs_strain(resolve_case_csv(args.cases_root, args.case_obs_e_subdir, sample_id))
            row_e = {int(n): i for i, n in enumerate(obs_e_ids_c.tolist())}
            obs_e_val = np.stack([obs_e_val_c[row_e[int(n)]] for n in obs_e_ids.tolist()], axis=0).astype(np.float32)
        return obs_u_val, obs_e_val

    def build_flat_feat(obs_u_val: Optional[np.ndarray], obs_e_val: Optional[np.ndarray]) -> np.ndarray:
        feats = []
        if int(args.load_use_disp) == 1 and (obs_u_val is not None) and (obs_u_val.size > 0):
            u_in = obs_u_val.astype(np.float64) / float(u0.item() + 1e-12)   # (Nu,3)
            feats.append(u_in.reshape(-1))
        if int(args.load_use_strain) == 1 and (obs_e_val is not None) and (obs_e_val.size > 0):
            e_in = obs_e_val.astype(np.float64) / float(eps0.item() + 1e-12) # (Ne,6)
            feats.append(e_in.reshape(-1))
        if len(feats) == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(feats, axis=0)


    def describe_flat_index(j: int) -> str:
        # map feature index -> (type, nodeid, component)
        off = 0
        if int(args.load_use_disp) == 1:
            Nu = int(obs_u_ids.shape[0])
            if j < Nu * 3:
                pt = j // 3
                comp = j % 3
                comp_name = ["U11", "U22", "U33"][comp]
                nid = int(obs_u_ids[pt])
                return f"disp nid={nid} {comp_name}"
            off += Nu * 3

        if int(args.load_use_strain) == 1 and (obs_e_ids is not None):
            Ne = int(obs_e_ids.shape[0])
            jj = j - off
            if 0 <= jj < Ne * 6:
                pt = jj // 6
                comp = jj % 6
                comp_name = ["E11", "E22", "E33", "E12", "E13", "E23"][comp]
                nid = int(obs_e_ids[pt])
                return f"strain nid={nid} {comp_name}"

        return f"idx={j}"

    # --- normalization scales (u0/eps0/sigma0/F0) ---
    norm_override = None
    if args.norm_load:
        norm_override = load_norm(args.norm_load)

    if norm_override is None:
        # use train-set RMS for scales
        def compute_train_rms_scales():
            # u0 from train cases
            sum_u2 = 0.0
            cnt_u = 0
            for sid in train_case_ids.tolist():
                if args.u0_from == 'full':
                    full_u_ids, full_u_val = load_obs_disp(resolve_case_csv(args.cases_root, args.case_full_u_subdir, sid))
                    full_u_val = reorder_rows_by_node_ids(full_u_ids, full_u_val, node_ids_npz, name='u_full')
                    v = full_u_val.astype(np.float64)
                else:
                    obs_u_ids_c, obs_u_val_c = load_obs_disp(resolve_case_csv(args.cases_root, args.case_obs_u_subdir, sid))
                    row_u = {int(n): i for i, n in enumerate(obs_u_ids_c.tolist())}
                    v = np.stack([obs_u_val_c[row_u[int(n)]] for n in obs_u_ids.tolist()], axis=0).astype(np.float64)
                sum_u2 += float((v * v).sum())
                cnt_u += int(v.size)
            u0_val = np.sqrt(sum_u2 / max(cnt_u, 1))

            # eps0 from train strain sensors if available; fallback to u0/L0
            if obs_e_ids is not None:
                sum_e2 = 0.0
                cnt_e = 0
                for sid in train_case_ids.tolist():
                    obs_e_ids_c, obs_e_val_c = load_obs_strain(resolve_case_csv(args.cases_root, args.case_obs_e_subdir, sid))
                    row_e = {int(n): i for i, n in enumerate(obs_e_ids_c.tolist())}
                    ev = np.stack([obs_e_val_c[row_e[int(n)]] for n in obs_e_ids.tolist()], axis=0).astype(np.float64)
                    sum_e2 += float((ev * ev).sum())
                    cnt_e += int(ev.size)
                eps0_val = np.sqrt(sum_e2 / max(cnt_e, 1))
            else:
                eps0_val = float(u0_val) / float(L0.item())

            # sigma0 from eps0 * E
            sigma0_val = float(args.E) * float(eps0_val)

            # F0 (used as a force normalization constant)
            if getattr(args, 'F0_fixed', 0.0) and float(args.F0_fixed) > 0.0:
                f0_val = float(args.F0_fixed)
            else:
                # fallback: RMS of true force magnitude over train set
                if len(train_F_true) > 0:
                    f_np = np.asarray(train_F_true, dtype=np.float64)
                    f0_val = float(np.sqrt(np.mean((f_np * f_np).sum(axis=1))))
                else:
                    f0_val = 1.0

            return u0_val, eps0_val, sigma0_val, f0_val

        u0_v, eps0_v, sigma0_v, F0_v = compute_train_rms_scales()
        u0 = torch.tensor(u0_v, device=device, dtype=torch.float32)
        eps0 = torch.tensor(eps0_v, device=device, dtype=torch.float32)
        sigma0 = torch.tensor(sigma0_v, device=device, dtype=torch.float32)
        F0 = torch.tensor(F0_v, device=device, dtype=torch.float32)
        print(f"[scale] train RMS u0={u0.item():.6g} eps0={eps0.item():.6g} sigma0={sigma0.item():.6g} F0={F0.item():.6g}")
    else:
        xyz_center = torch.tensor(norm_override['xyz_center'], device=device, dtype=torch.float32)
        L0 = torch.tensor(norm_override['L0'], device=device, dtype=torch.float32)
        mu_u = torch.tensor(norm_override['phi_u_mu'], device=device, dtype=torch.float32)
        sd_u = torch.tensor(norm_override['phi_u_sd'], device=device, dtype=torch.float32)
        mu_s = torch.tensor(norm_override['phi_s_mu'], device=device, dtype=torch.float32)
        sd_s = torch.tensor(norm_override['phi_s_sd'], device=device, dtype=torch.float32)
        u0 = torch.tensor(norm_override['u0'], device=device, dtype=torch.float32)
        eps0 = torch.tensor(norm_override['eps0'], device=device, dtype=torch.float32)
        sigma0 = torch.tensor(norm_override['sigma0'], device=device, dtype=torch.float32)
        F0 = torch.tensor(norm_override['F0'], device=device, dtype=torch.float32)

        def norm_xyz(x):
            if args.norm_xyz == 0:
                return x
            return (x - xyz_center) / L0

        # recompute normalized inputs using loaded stats
        if args.norm_phi != 'none':
            phi_u_in_t = (phi_u_t - mu_u) / (sd_u + 1e-12)
            phi_s_in_t = (phi_s_t - mu_s) / (sd_s + 1e-12)
            phi_h_gp_in_t = (phi_h_gp_t - mu_s) / (sd_s + 1e-12)
        else:
            phi_u_in_t = phi_u_t
            phi_s_in_t = phi_s_t
            phi_h_gp_in_t = phi_h_gp_t

    # save normalization if requested
    if args.norm_save:
        save_norm(
            args.norm_save,
            xyz_center=xyz_center,
            L0=L0,
            phi_u_mu=mu_u,
            phi_u_sd=sd_u,
            phi_s_mu=mu_s,
            phi_s_sd=sd_s,
            u0=u0,
            eps0=eps0,
            sigma0=sigma0,
            F0=F0,
            norm_phi=args.norm_phi,
            E=float(args.E),
            nu=float(args.nu),
        )
        print(f"[norm] saved -> {args.norm_save}")

    # --- precompute normalized xyz ---
    xyz_in_t = norm_xyz(xyz_t)

    # --- ridge baseline on Fy (train->val) ---
    def _build_ridge_feat_from_obs(obs_u_val, obs_e_val):
        u_feat = np.zeros((3,), dtype=np.float64)
        if int(args.load_use_disp) == 1 and (obs_u_val is not None) and (obs_u_val.size > 0):
            u_in = obs_u_val / float(u0.item() + 1e-12)
            u_feat = np.sqrt((u_in * u_in).mean(axis=0) + 1e-12)
        e_feat = np.zeros((6,), dtype=np.float64)
        if int(args.load_use_strain) == 1 and (obs_e_val is not None) and (obs_e_val.size > 0):
            e_in = obs_e_val / float(eps0.item() + 1e-12)
            e_feat = np.sqrt((e_in * e_in).mean(axis=0) + 1e-12)
        return np.concatenate([u_feat, e_feat], axis=0)

    def run_ridge_baseline_fy():
        if len(train_case_ids) == 0:
            return
        if int(args.load_use_strain) == 1 and obs_e_ids is None:
            print("[ridge] load_use_strain=1 but OBS_E missing; strain features set to 0.")
        X_tr = []
        y_tr = []
        for sid, f_np in zip(train_case_ids.tolist(), train_F_true.tolist()):
            obs_u_val, obs_e_val = load_obs_only(sid)
            X_tr.append(_build_ridge_feat_from_obs(obs_u_val, obs_e_val))
            y_tr.append(float(f_np[1]))
        X_tr = np.asarray(X_tr, dtype=np.float64)
        y_tr = np.asarray(y_tr, dtype=np.float64)

        Xb = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1), dtype=np.float64)], axis=1)
        lam = float(args.ridge_lambda)
        A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float64)
        b = Xb.T @ y_tr
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        mae_tr = float(np.mean(np.abs(Xb @ w - y_tr)))
        if len(val_case_ids) > 0:
            X_va = []
            y_va = []
            for sid, f_np in zip(val_case_ids.tolist(), val_F_true.tolist()):
                obs_u_val, obs_e_val = load_obs_only(sid)
                X_va.append(_build_ridge_feat_from_obs(obs_u_val, obs_e_val))
                y_va.append(float(f_np[1]))
            X_va = np.asarray(X_va, dtype=np.float64)
            y_va = np.asarray(y_va, dtype=np.float64)
            Xb_va = np.concatenate([X_va, np.ones((X_va.shape[0], 1), dtype=np.float64)], axis=1)
            mae_va = float(np.mean(np.abs(Xb_va @ w - y_va)))
            print(f"[ridge] Fy baseline: lambda={lam:g} train_mae={mae_tr:.3e} val_mae={mae_va:.3e}")
        else:
            print(f"[ridge] Fy baseline: lambda={lam:g} train_mae={mae_tr:.3e} val_mae=NA")


    def baseline_best_single_feature(topk=10):
        # build X (N,D) from flat features
        Xtr, ytr = [], []
        Xva, yva = [], []

        for sid, f_np in zip(train_case_ids.tolist(), train_F_true.tolist()):
            obs_u_val, obs_e_val = load_obs_only(sid)
            Xtr.append(build_flat_feat(obs_u_val, obs_e_val))
            ytr.append(float(f_np[1]))
        for sid, f_np in zip(val_case_ids.tolist(), val_F_true.tolist()):
            obs_u_val, obs_e_val = load_obs_only(sid)
            Xva.append(build_flat_feat(obs_u_val, obs_e_val))
            yva.append(float(f_np[1]))

        Xtr = np.asarray(Xtr, dtype=np.float64)
        ytr = np.asarray(ytr, dtype=np.float64)
        Xva = np.asarray(Xva, dtype=np.float64)
        yva = np.asarray(yva, dtype=np.float64)

        # vectorized linear fit per feature: y = a*x + b
        mu_x = Xtr.mean(axis=0)
        mu_y = ytr.mean()
        Xc = Xtr - mu_x
        yc = ytr - mu_y
        var = (Xc * Xc).mean(axis=0) + 1e-12
        cov = (Xc * yc[:, None]).mean(axis=0)
        a = cov / var
        b = mu_y - a * mu_x

        pred_va = Xva * a[None, :] + b[None, :]
        mae_feat = np.mean(np.abs(pred_va - yva[:, None]), axis=0)

        idx = np.argsort(mae_feat)[:min(int(topk), mae_feat.shape[0])]
        print(f"[baseline-single] best {len(idx)} features (val MAE in N):")
        for k, j in enumerate(idx.tolist(), 1):
            print(f"  #{k} mae={mae_feat[j]:.3e} | {describe_flat_index(j)}")

        best_j = int(idx[0])
        pred_tr = Xtr[:, best_j] * a[best_j] + b[best_j]
        pred_va1 = Xva[:, best_j] * a[best_j] + b[best_j]
        mae_tr = float(np.mean(np.abs(pred_tr - ytr)))
        mae_va = float(np.mean(np.abs(pred_va1 - yva)))
        print(f"[baseline-single] choose {describe_flat_index(best_j)} | train_mae={mae_tr:.3e} val_mae={mae_va:.3e}")

    def baseline_dual_ridge_flat(lam: float):
        # Xtr: (N,D), N~160, D~9000+ ; use dual ridge to avoid (D,D) solve.
        Xtr, ytr, sid_tr = [], [], []
        Xva, yva, sid_va = [], [], []

        for sid, f_np in zip(train_case_ids.tolist(), train_F_true.tolist()):
            obs_u_val, obs_e_val = load_obs_only(sid)
            Xtr.append(build_flat_feat(obs_u_val, obs_e_val))
            ytr.append(float(f_np[1]))
            sid_tr.append(sid)
        for sid, f_np in zip(val_case_ids.tolist(), val_F_true.tolist()):
            obs_u_val, obs_e_val = load_obs_only(sid)
            Xva.append(build_flat_feat(obs_u_val, obs_e_val))
            yva.append(float(f_np[1]))
            sid_va.append(sid)

        Xtr = np.asarray(Xtr, dtype=np.float64)
        ytr = np.asarray(ytr, dtype=np.float64)
        Xva = np.asarray(Xva, dtype=np.float64)
        yva = np.asarray(yva, dtype=np.float64)

        # standardize per feature
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-12
        Xtr_s = (Xtr - mu) / sd
        Xva_s = (Xva - mu) / sd

        y_mean = ytr.mean()
        ytr_c = ytr - y_mean

        # dual solve: alpha = (K + lam I)^-1 y
        K = Xtr_s @ Xtr_s.T  # (N,N)
        A = K + float(lam) * np.eye(K.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(A, ytr_c)

        ytr_pred = (K @ alpha) + y_mean
        Kva = Xva_s @ Xtr_s.T
        yva_pred = (Kva @ alpha) + y_mean

        mae_tr = float(np.mean(np.abs(ytr_pred - ytr)))
        mae_va = float(np.mean(np.abs(yva_pred - yva)))
        print(f"[baseline-ridge-flat] lam={lam:g} train_mae={mae_tr:.3e} val_mae={mae_va:.3e}")

        # worst-k on val
        e = np.abs(yva_pred - yva)
        order = np.argsort(-e)
        k = min(int(args.val_case_report), len(order)) if int(args.val_case_report) > 0 else 0
        if k > 0:
            p50 = float(np.percentile(e, 50))
            p90 = float(np.percentile(e, 90))
            mx = float(e.max())
            print(f"[baseline-ridge-flat][Fy-err] p50={p50:.3e} p90={p90:.3e} max={mx:.3e} (N)")
            print(f"[baseline-ridge-flat] worst {k}:")
            for i in order[:k]:
                print(f"  sid={sid_va[i]} true={yva[i]:.3e} pred={yva_pred[i]:.3e} abs={e[i]:.3e} rel={e[i]/(abs(yva[i])+1e-12):.3e}")

    run_ridge_baseline_fy()
    baseline_best_single_feature(topk=10) # 强 baseline A
    baseline_dual_ridge_flat(float(args.ridge_lambda))  # 强 baseline B
    # --- nets ---
    cond_dim = 3 if (int(args.cond_force) == 1) else 0
    u_net = DispNet(args.hybrid_modes_u, cond_dim=cond_dim).to(device)
    s_net = StressCoeffNet(args.hybrid_modes_s, cond_dim=cond_dim).to(device)
    load_net = None
    if int(args.use_load_net) == 1:
        load_net = LoadNetSet(embed_dim=int(args.load_embed), hidden=int(args.load_hidden), depth=int(args.load_depth)).to(device)

    opt_u = torch.optim.AdamW(u_net.parameters(), lr=float(args.lr_u))
    opt_s = torch.optim.AdamW(s_net.parameters(), lr=float(args.lr_s))
    opt_f = torch.optim.AdamW(load_net.parameters(), lr=float(args.lr_f)) if load_net is not None else None

    # --- case sampler ---
    train_db = [(train_case_ids[i], train_F_true[i]) for i in range(len(train_case_ids))]
    val_db = [(val_case_ids[i], val_F_true[i]) for i in range(len(val_case_ids))]

    def sample_case_batch(batch_size: int):
        idx = rng.choice(len(train_db), size=batch_size, replace=True)
        return [train_db[i] for i in idx]

    # --- helpers ---
    def make_cond(F_true: torch.Tensor, n: int):
        if cond_dim <= 0:
            return None
        c = (F_true / (F0 + 1e-12)).view(1, 3)
        return c.expand(n, 3)

    def maybe_sample_nodes(N: int):
        if int(args.u_full_batch_nodes) and int(args.u_full_batch_nodes) > 0:
            n = int(args.u_full_batch_nodes)
            ridx = torch.randint(0, N, (n,), device=device)
            return ridx
        return None

    all_elem_ids = torch.arange(n_elem, device=device, dtype=torch.long)

    def eval_u_val():
        if len(val_db) == 0:
            return None
        u_net.eval()
        sum_abs = torch.zeros((3,), device=device)
        count = 0
        with torch.no_grad():
            for sid, f_np in val_db:
                c = load_case_cached(sid, f_np)
                cond_b = make_cond(c['F_true_t'], xyz_in_t.shape[0])
                u_pred = u_net(xyz_in_t, phi_u_in_t, cond_b) if cond_dim > 0 else u_net(xyz_in_t, phi_u_in_t)
                diff = (u_pred - c['u_full_t']).abs()
                sum_abs += diff.sum(dim=0)
                count += int(diff.shape[0])
        if count == 0:
            return None
        return (sum_abs / float(count)).detach()

    def eval_s_val():
        if len(val_db) == 0:
            return None
        s_net.eval()
        sum_abs = torch.zeros((6,), device=device)
        count = 0
        chunk = max(1, int(args.batch_elems))
        with torch.no_grad():
            for sid, f_np in val_db:
                c = load_case_cached(sid, f_np)
                sigma_true = c['sigma_gp_t']
                for s in range(0, n_elem, chunk):
                    eids = all_elem_ids[s:s + chunk]
                    xgp = gp_xyz_t[eids].reshape(-1, 3)
                    xgp_in = norm_xyz(xgp)
                    phi_gp = phi_h_gp_in_t[eids].reshape(-1, Ks)
                    gradphi_gp = grad_h_gp_t[eids].reshape(-1, Ks, 3)
                    cond_gp = make_cond(c['F_true_t'], xgp_in.shape[0]) if cond_dim > 0 else None

                    P = s_net(xgp_in, cond_gp) if cond_dim > 0 else s_net(xgp_in)
                    sigma = assemble_sigma_from_P(P, gradphi_gp)
                    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
                    sigma6 = sigma33_to_voigt6(sigma)

                    sigma_true_flat = sigma_true[eids].reshape(-1, 6)
                    diff = (sigma6 - sigma_true_flat).abs()
                    sum_abs += diff.sum(dim=0)
                    count += int(diff.shape[0])
        if count == 0:
            return None
        return (sum_abs / float(count)).detach()

    def make_load_pts(c: dict):
        """Build disp_pts / strain_pts for load_net, consistent for train/eval."""
        disp_pts = None
        if int(args.load_use_disp) == 1:
            disp_pts = torch.cat([xyz_in_t[obs_u_idx], c['obs_u_val_t'] / (u0 + 1e-12)], dim=1)  # (Nu,6)

        strain_pts = None
        if int(args.load_use_strain) == 1 and (obs_e_idx is not None) and (c['obs_e_val_t'] is not None):
            strain_pts = torch.cat([xyz_in_t[obs_e_idx], c['obs_e_val_t'] / (eps0 + 1e-12)], dim=1)  # (Ne,9)

        return disp_pts, strain_pts


    def fhat_to_Fpred(f_hat: torch.Tensor):
        """Map load_net output (1 or 3 dims) to physical force vector (3,) in N."""
        f_hat = f_hat.view(-1)
        if f_hat.numel() == 1:
            fy = f_hat[0]
            f_hat3 = torch.stack([fy * 0.0, fy, fy * 0.0])  # [0, Fy_hat, 0]
        else:
            f_hat3 = f_hat
        return f_hat3 * F0    


    def eval_f_val():
        if (load_net is None) or (len(val_db) == 0):
            return None

        load_net.eval()
        sum_abs = torch.zeros((3,), device=device)
        count = 0

        # per-case for diagnostics
        fy_err = []
        cases = []  # (sid, Fy_true, Fy_pred, abs_err, rel_err)

        with torch.no_grad():
            for sid, f_np in val_db:
                c = load_case_cached(sid, f_np)

                disp_pts, strain_pts = make_load_pts(c)
                f_hat = load_net(disp_pts, strain_pts)  # (1,) or (3,)
                F_pred = fhat_to_Fpred(f_hat)           # (3,) in N
                diff = (F_pred - c['F_true_t']).abs()   # (3,) in N

                sum_abs += diff
                count += 1

                Fy_t = float(c['F_true_t'][1].item())
                Fy_p = float(F_pred[1].item())
                ae = abs(Fy_p - Fy_t)
                re = ae / (abs(Fy_t) + 1e-12)
                fy_err.append(ae)
                cases.append((sid, Fy_t, Fy_p, ae, re))

        if count == 0:
            return None

        # print diagnostics
        if int(args.val_case_report) > 0 and len(cases) > 0:
            e = np.asarray(fy_err, dtype=np.float64)
            p50 = float(np.percentile(e, 50))
            p90 = float(np.percentile(e, 90))
            mx = float(e.max())
            print(f"[f][Fy-err] p50={p50:.3e} p90={p90:.3e} max={mx:.3e} (N)")

            cases.sort(key=lambda x: -x[3])  # by abs_err desc
            k = min(int(args.val_case_report), len(cases))
            top = cases[:k]
            msg = "[f][case] worst_Fy: " + ", ".join(
                [f"{sid}: true={Fy_t:.3e} pred={Fy_p:.3e} abs={ae:.3e} rel={re:.3e}"
                for sid, Fy_t, Fy_p, ae, re in top]
            )
            print(msg)

        return (sum_abs / float(count)).detach()

    # --- training: u_net ---
    best_u = float('inf')
    for ep in range(1, int(args.epochs_u) + 1):
        u_net.train()
        total_loss = 0.0
        for _ in range(int(args.steps_per_epoch)):
            batch = sample_case_batch(int(args.case_batch))
            opt_u.zero_grad()
            loss_acc = 0.0
            for sid, f_np in batch:
                c = load_case_cached(sid, f_np)
                u_true = c['u_full_t']
                idx = maybe_sample_nodes(u_true.shape[0])
                if idx is None:
                    u_true_b = u_true
                    xyz_b = xyz_in_t
                    phi_b = phi_u_in_t
                    cond_b = make_cond(c['F_true_t'], xyz_in_t.shape[0])
                else:
                    u_true_b = u_true[idx]
                    xyz_b = xyz_in_t[idx]
                    phi_b = phi_u_in_t[idx]
                    cond_b = make_cond(c['F_true_t'], idx.shape[0])

                u_pred = u_net(xyz_b, phi_b, cond_b) if cond_dim > 0 else u_net(xyz_b, phi_b)
                loss = ((u_pred - u_true_b) / (u0 + 1e-12)).pow(2).mean()
                loss_acc = loss_acc + loss
            loss_acc = loss_acc / float(len(batch))
            loss_acc.backward()
            opt_u.step()
            total_loss += float(loss_acc.item())

        if (ep % int(args.eval_every) == 0) or (ep == 1):
            avg = total_loss / max(1, int(args.steps_per_epoch))
            msg = f"[u] ep={ep} loss={avg:.4e}"
            val_mae = eval_u_val()
            if val_mae is not None:
                msg += f" | val_mae(u1,u2,u3)=({val_mae[0].item():.3e},{val_mae[1].item():.3e},{val_mae[2].item():.3e})"
            print(msg)
            if log_csv_path:
                with open(log_csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{ep},u,{avg},0,0\n")
            if avg < best_u and args.save_u_best:
                best_u = avg
                os.makedirs(os.path.dirname(args.save_u_best), exist_ok=True) if os.path.dirname(args.save_u_best) else None
                torch.save({'epoch': ep, 'loss': best_u, 'u_net': u_net.state_dict()}, args.save_u_best)

    # --- training: s_net ---
    best_s = float('inf')
    for ep in range(1, int(args.epochs_s) + 1):
        s_net.train()
        total_loss = 0.0
        for _ in range(int(args.steps_per_epoch)):
            batch = sample_case_batch(int(args.case_batch))
            opt_s.zero_grad()
            loss_acc = 0.0
            for sid, f_np in batch:
                c = load_case_cached(sid, f_np)
                sigma_true = c['sigma_gp_t']
                # sample element batch
                elem_ids = torch.randint(0, n_elem, (int(args.batch_elems),), device=device)
                xgp = gp_xyz_t[elem_ids].reshape(-1, 3)
                xgp_in = norm_xyz(xgp)
                phi_gp = phi_h_gp_in_t[elem_ids].reshape(-1, Ks)
                gradphi_gp = grad_h_gp_t[elem_ids].reshape(-1, Ks, 3)

                cond_gp = None
                if cond_dim > 0:
                    cond_gp = make_cond(c['F_true_t'], xgp_in.shape[0])

                P = s_net(xgp_in, cond_gp) if cond_dim > 0 else s_net(xgp_in)
                sigma = assemble_sigma_from_P(P, gradphi_gp)
                sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
                sigma6 = sigma33_to_voigt6(sigma)

                sigma_true_flat = sigma_true[elem_ids].reshape(-1, 6)
                w = gp_dV_t[elem_ids].reshape(-1, 1)
                diff = (sigma6 - sigma_true_flat) / (sigma0 + 1e-12)
                diff_n = diff[:, :3]
                diff_s = diff[:, 3:]
                loss_n = weighted_mean(diff_n.pow(2), w)
                loss_s = weighted_mean(diff_s.pow(2), w)
                loss = 0.5 * (loss_n + loss_s)
                loss_acc = loss_acc + loss
            loss_acc = loss_acc / float(len(batch))
            loss_acc.backward()
            opt_s.step()
            total_loss += float(loss_acc.item())

        if (ep % int(args.eval_every) == 0) or (ep == 1):
            avg = total_loss / max(1, int(args.steps_per_epoch))
            msg = f"[s] ep={ep} loss={avg:.4e}"
            val_mae = eval_s_val()
            if val_mae is not None:
                msg += (" | val_mae(S11,S22,S33,S12,S13,S23)="
                        f"({val_mae[0].item():.3e},{val_mae[1].item():.3e},{val_mae[2].item():.3e},"
                        f"{val_mae[3].item():.3e},{val_mae[4].item():.3e},{val_mae[5].item():.3e})")
            print(msg)
            if log_csv_path:
                with open(log_csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{ep},s,0,{avg},0\n")
            if avg < best_s and args.save_s_best:
                best_s = avg
                os.makedirs(os.path.dirname(args.save_s_best), exist_ok=True) if os.path.dirname(args.save_s_best) else None
                torch.save({'epoch': ep, 'loss': best_s, 's_net': s_net.state_dict()}, args.save_s_best)

    # --- training: load_net ---
    if load_net is not None:
        best_f = float('inf')
        for ep in range(1, int(args.epochs_f) + 1):
            load_net.train()
            total_loss = 0.0
            for _ in range(int(args.steps_per_epoch)):
                batch = sample_case_batch(int(args.case_batch))
                opt_f.zero_grad()
                loss_acc = 0.0
                for sid, f_np in batch:
                    c = load_case_cached(sid, f_np)
                    disp_pts = None
                    if int(args.load_use_disp) == 1:
                        disp_pts = torch.cat([xyz_in_t[obs_u_idx], c['obs_u_val_t'] / (u0 + 1e-12)], dim=1)
                    if int(args.load_use_strain) == 1 and (obs_e_idx is not None) and (c['obs_e_val_t'] is not None):
                        strain_pts = torch.cat([xyz_in_t[obs_e_idx], c['obs_e_val_t'] / (eps0 + 1e-12)], dim=1)
                    else:
                        strain_pts = None
                    f_hat = load_net(disp_pts, strain_pts)
                    f_true_hat = c['F_true_t'] / (F0 + 1e-12)
                    loss = (f_hat - f_true_hat).pow(2).mean()
                    loss_acc = loss_acc + loss
                loss_acc = loss_acc / float(len(batch))
                loss_acc.backward()
                opt_f.step()
                total_loss += float(loss_acc.item())

            if (ep % int(args.eval_every) == 0) or (ep == 1):
                avg = total_loss / max(1, int(args.steps_per_epoch))
                msg = f"[f] ep={ep} loss={avg:.4e}"
                val_mae = eval_f_val()
                if val_mae is not None:
                    msg += f" | val_mae(Fx,Fy,Fz)=({val_mae[0].item():.3e},{val_mae[1].item():.3e},{val_mae[2].item():.3e})"
                print(msg)
            if log_csv_path:
                with open(log_csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{ep},f,0,0,{avg}\n")
                if avg < best_f and args.save_f_best:
                    best_f = avg
                    os.makedirs(os.path.dirname(args.save_f_best), exist_ok=True) if os.path.dirname(args.save_f_best) else None
                    torch.save({'epoch': ep, 'loss': best_f, 'load_net': load_net.state_dict()}, args.save_f_best)


if __name__ == '__main__':
    main()
