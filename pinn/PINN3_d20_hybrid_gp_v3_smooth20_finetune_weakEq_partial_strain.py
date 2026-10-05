"""PINN3 finetune (physics) with assembled weak-equilibrium residual.

What this script enforces
-------------------------
1) Data loss on observed displacement nodes (Dirichlet-like measurements)
2) Constitutive consistency at Gauss points:
      sigma_snet(gp)  ~=  C : eps(unet)(gp)
3) Assembled weak equilibrium residual on nodes (excluding Dirichlet/Neumann sets):
      r_a = || \sum_e \int_{\Omega_e} (\nabla N_a)^T sigma \, dV ||

Changes vs older versions
-------------------------
- Removed strong-form PDE (div sigma) loss.
- Removed global virtual-work loss.

Coordinate normalization policy
------------------------------
- xyz is normalized ONLY for neural-network inputs.
- All derivatives/integration weights used in losses come from the gp_npz
  export in physical coordinates (mm) and must NOT be re-normalized.

Requirements for gp_npz
-----------------------
Export with: hybrid_lbo_from_inp_solid_C3D20R_export_fast_v5(3)_with_dNdx.py
and ensure gp_npz contains keys:
- dN_dx_gp        (E,ngp,20,3)
- gp_dV           (E,ngp)
- set_BDIR_idx    (nBdir,)
- set_BNEU_idx    (nBneu,)
- grad_phi_hybrid_gp (E,ngp,K,3)

Logs & checkpoints
------------------
Saved to: ckpt_root/project/
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinn3_utils import (
    as_torch,
    load_gp_npz,
    load_phi_node_csv,
    reorder_rows_by_node_ids,
    load_obs_disp,
    load_obs_strain,
    build_gp_perm_map_from_csv,
    load_gp_sigma_case_with_perm,
    load_gp_eps_case_with_perm,
    sigma33_to_voigt6,
    voigt6_to_sigma33,
    isotropic_sigma_from_eps_eng,
    isotropic_eps_from_sigma_voigt,
    build_unique_edges_from_conn20,
    build_node_degree,
    WeakEquilibriumLoss,
    assemble_internal_nodal_forces,
    complete_node_mask_for_elem_batch,
)


# -----------------------------
# Networks (kept close to existing style)
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 5, act: str = "silu"):
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
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DispNet(nn.Module):
    # (xyz_norm(3), phi_hybrid_norm(Ku), [cond(3)]) -> u(3)
    def __init__(self, Ku: int, width: int = 256, depth: int = 5, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.mlp = MLP(3 + int(Ku) + self.cond_dim, 3, width=width, depth=depth, act="silu")

    def forward(self, xyz: torch.Tensor, phi: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cond_dim > 0:
            if cond is None:
                cond = xyz.new_zeros((xyz.shape[0], self.cond_dim))
            x = torch.cat([xyz, phi, cond], dim=-1)
        else:
            x = torch.cat([xyz, phi], dim=-1)
        return self.mlp(x)


class StressCoeffNet(nn.Module):
    # (xyz_norm(3), [cond(3)]) -> P (N,3,Ks)
    # 3 groups in 3D (paper-aligned)
    def __init__(self, Ks: int, width: int = 256, depth: int = 6, cond_dim: int = 0):
        super().__init__()
        self.Ks = int(Ks)
        self.cond_dim = int(cond_dim)
        in_dim = 3 + self.cond_dim
        self.mlp_p1 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")
        self.mlp_p2 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")
        self.mlp_p3 = MLP(in_dim, self.Ks, width=width, depth=depth, act="silu")

    def forward(self, xyz: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cond_dim > 0:
            if cond is None:
                cond = xyz.new_zeros((xyz.shape[0], self.cond_dim))
            x = torch.cat([xyz, cond], dim=-1)
        else:
            x = xyz
        p1 = self.mlp_p1(x)
        p2 = self.mlp_p2(x)
        p3 = self.mlp_p3(x)
        return torch.stack([p1, p2, p3], dim=1)  # (N,3,K)


def _set_encoder_point_mlp(in_dim: int, embed_dim: int, hidden: int, depth: int) -> nn.Module:
    # small point-wise MLP for set encoding
    return MLP(in_dim, embed_dim, width=hidden, depth=depth, act='silu')


class LoadNetSet(nn.Module):
    """Set encoder to infer global force from sparse observations.

    This is kept for future use. In this finetune script it has **no loss** by default.
    You can load a pretrained load_net and use it to provide the conditioning force.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 128, depth: int = 3,
                 use_disp: bool = True, use_strain: bool = True):
        super().__init__()
        self.use_disp = bool(use_disp)
        self.use_strain = bool(use_strain)

        self.disp_mlp = _set_encoder_point_mlp(in_dim=6, embed_dim=embed_dim, hidden=hidden, depth=depth)
        self.strain_mlp = _set_encoder_point_mlp(in_dim=9, embed_dim=embed_dim, hidden=hidden, depth=depth)

        in_dim = 0
        if self.use_disp:
            in_dim += embed_dim
        if self.use_strain:
            in_dim += embed_dim
        self.head = MLP(in_dim, 3, width=hidden, depth=depth, act='silu')

    def forward(self,
                disp_set: Optional[torch.Tensor],
                strain_set: Optional[torch.Tensor]) -> torch.Tensor:
        # disp_set: (n_obs_u, 6) -> [x,y,z, u1,u2,u3]
        # strain_set: (n_obs_e, 9) -> [x,y,z, e11,e22,e33,e12,e13,e23]
        embeds = []
        if self.use_disp:
            if disp_set is None or disp_set.numel() == 0:
                raise RuntimeError('LoadNetSet: use_disp=True but disp_set is empty')
            e = self.disp_mlp(disp_set)  # (n,embed)
            embeds.append(e.mean(dim=0, keepdim=True))
        if self.use_strain:
            if strain_set is None or strain_set.numel() == 0:
                raise RuntimeError('LoadNetSet: use_strain=True but strain_set is empty')
            e = self.strain_mlp(strain_set)
            embeds.append(e.mean(dim=0, keepdim=True))
        x = torch.cat(embeds, dim=-1)
        return self.head(x)  # (1,3) normalized force (F/F0)


def assemble_sigma_from_P(P_3K: torch.Tensor, gradphi_K3: torch.Tensor) -> torch.Tensor:
    # sigma_ij = sum_k P[i,k] * d(phi_k)/dx_j
    return torch.einsum('nik,nkj->nij', P_3K, gradphi_K3)


# -----------------------------
# Helpers
# -----------------------------

def parse_force_vec(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"force must be 'Fx,Fy,Fz', got: {s}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def norm_xyz_np(xyz: np.ndarray, center: np.ndarray, L0: float) -> np.ndarray:
    return (xyz - center[None, :]) / float(L0)


def compute_rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))


def compute_wmape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    # Use |truth| as weights so the metric stays well-defined for signed fields.
    pred64 = pred.detach().to(torch.float64)
    true64 = true.detach().to(torch.float64)
    denom = true64.abs().sum()
    if float(denom.item()) <= eps:
        return float('nan')
    num = (pred64 - true64).abs().sum()
    return float((num / denom * 100.0).item())


def maybe_makedirs(p: str):
    if p and (not os.path.isdir(p)):
        os.makedirs(p, exist_ok=True)


def save_ckpt(
    path: str,
    u_net: nn.Module,
    s_net: nn.Module,
    load_net: Optional[nn.Module],
    extra: Dict,
    opt_u: Optional[torch.optim.Optimizer] = None,
    opt_s: Optional[torch.optim.Optimizer] = None,
    opt_j: Optional[torch.optim.Optimizer] = None,
):
    maybe_makedirs(os.path.dirname(path))
    ckpt = {
        'u_net': u_net.state_dict(),
        's_net': s_net.state_dict(),
    }
    if load_net is not None:
        ckpt['load_net'] = load_net.state_dict()
    if opt_u is not None:
        ckpt['opt_u'] = opt_u.state_dict()
    if opt_s is not None:
        ckpt['opt_s'] = opt_s.state_dict()
    if opt_j is not None:
        ckpt['opt_j'] = opt_j.state_dict()
    ckpt.update(extra)
    torch.save(ckpt, path)


def ensure_csv_header(path: str, header: str, append_if_exists: bool = False):
    if not path:
        return
    maybe_makedirs(os.path.dirname(path))
    if append_if_exists and os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)


def load_obs_strain_partial_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load partial strain observations with per-component masking.

    Supported CSV layout:
      - required node-id column: NodeId / NodeID / node_id / nodeids
      - optional strain columns (engineering strain convention):
          E11, E22, E33, E12, E13, E23
        aliases also supported, e.g. exx, eyy, ezz, exy, exz, eyz,
        g12/gamma12, g13/gamma13, g23/gamma23.

    Missing columns, empty strings, or NaN cells are treated as "unobserved".

    Returns
    -------
    node_ids : (n,) int64
    eps_val  : (n,6) float32, unobserved entries filled with 0
    eps_mask : (n,6) bool, True only for observed entries
    """
    def _norm(s: str) -> str:
        return ''.join(ch for ch in str(s).strip().lower() if ch.isalnum())

    comp_aliases = {
        0: ('e11', 'eps11', 'exx', 'strain11', 's11'),
        1: ('e22', 'eps22', 'eyy', 'strain22', 's22'),
        2: ('e33', 'eps33', 'ezz', 'strain33', 's33'),
        3: ('e12', 'eps12', 'exy', 'g12', 'gamma12', 'strain12', 's12'),
        4: ('e13', 'eps13', 'exz', 'g13', 'gamma13', 'strain13', 's13'),
        5: ('e23', 'eps23', 'eyz', 'g23', 'gamma23', 'strain23', 's23'),
    }
    node_aliases = ('nodeid', 'nodeids', 'node', 'node_id', 'nid', 'id')

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f'strain_csv has no header: {csv_path}')

        field_map = {_norm(k): k for k in reader.fieldnames if k is not None}

        node_key = None
        for a in node_aliases:
            if _norm(a) in field_map:
                node_key = field_map[_norm(a)]
                break
        if node_key is None:
            raise RuntimeError(
                f"strain_csv must contain a node id column, got headers={reader.fieldnames}"
            )

        comp_keys = {}
        for j, aliases in comp_aliases.items():
            for a in aliases:
                if _norm(a) in field_map:
                    comp_keys[j] = field_map[_norm(a)]
                    break

        node_ids = []
        eps_val = []
        eps_mask = []

        for row in reader:
            raw_nid = row.get(node_key, '')
            if raw_nid is None or str(raw_nid).strip() == '':
                continue
            node_ids.append(int(float(str(raw_nid).strip())))

            v = np.zeros((6,), dtype=np.float32)
            m = np.zeros((6,), dtype=bool)
            for j in range(6):
                key = comp_keys.get(j, None)
                if key is None:
                    continue
                raw = row.get(key, '')
                if raw is None:
                    continue
                s = str(raw).strip()
                if s == '' or s.lower() == 'nan':
                    continue
                v[j] = float(s)
                m[j] = True
            eps_val.append(v)
            eps_mask.append(m)

    if len(node_ids) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 6), dtype=np.float32), np.zeros((0, 6), dtype=bool)

    node_ids = np.asarray(node_ids, dtype=np.int64)
    eps_val = np.asarray(eps_val, dtype=np.float32)
    eps_mask = np.asarray(eps_mask, dtype=bool)

    if not np.any(eps_mask):
        raise RuntimeError(f'strain_csv has no valid observed strain entries: {csv_path}')

    return node_ids, eps_val, eps_mask


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument('--gp_npz', type=str, required=True)
    ap.add_argument('--hybrid_node_csv', type=str, required=True)
    ap.add_argument('--hybrid_prefix', type=str, default='phi_hybrid_')
    ap.add_argument('--Ku', type=int, default=20)
    ap.add_argument('--Ks', type=int, default=20)

    ap.add_argument('--disp_csv', type=str, required=True)
    ap.add_argument('--strain_csv', type=str, default='')
    # optional validation against Abaqus GP stress export (requires X,Y,Z columns)
    ap.add_argument('--abaqus_gp_csv', type=str, default='')
    ap.add_argument('--gp_match_tol_mm', type=float, default=1e-4)
    ap.add_argument('--eval_every', type=int, default=0,
                    help='if >0, evaluate all enabled metrics every N steps (and at step=1)')
    ap.add_argument('--true_disp_csv', type=str, default='',
                    help='optional full-field nodal displacement truth csv (NodeIds,U11,U22,U33)')
    ap.add_argument('--true_csv', type=str, default='eval_true.csv',
                    help='CSV name/path for true-eval logs; relative path is placed under ckpt_root/project')
    ap.add_argument('--save_best_true', type=str, default='ckpt_best_true.pt',
                    help='checkpoint name/path for best true metric; relative path is placed under ckpt_root/project')
    ap.add_argument('--best_true_metric', type=str, default='eps_s',
                    choices=['eps_s', 'eps_u', 'u', 'sigma_wmape', 'u_wmape', 'eps_s_wmape'],
                    help='metric used for best-true checkpoint selection')


    ap.add_argument('--norm_load', type=str, default='',
                    help='npz saved by pretrain; overrides xyz/phi stats/u0/eps0/sigma0/F0')

    ap.add_argument('--ckpt_root', type=str, default='ckpt')
    ap.add_argument('--project', type=str, default='ft_weakEq')
    ap.add_argument('--save_best', type=str, default='ckpt_best.pt')
    ap.add_argument('--save_last', type=str, default='ckpt_last.pt')
    ap.add_argument('--log_csv', type=str, default='log.csv')

    ap.add_argument('--init_u_ckpt', type=str, default='')
    ap.add_argument('--init_s_ckpt', type=str, default='')
    ap.add_argument('--init_f_ckpt', type=str, default='')
    ap.add_argument('--resume_ckpt', type=str, default='',
                    help='optional finetune checkpoint to resume from; restores model/optimizer states and continues training')

    # material
    ap.add_argument('--E', type=float, default=2.06e5)
    ap.add_argument('--nu', type=float, default=0.3)

    # conditioning
    ap.add_argument('--cond_force', type=int, default=1)
    ap.add_argument('--use_load_net', type=int, default=0)
    ap.add_argument('--load_use_disp', type=int, default=1)
    ap.add_argument('--load_use_strain', type=int, default=0)

    ap.add_argument('--force_known', type=str, default='0,15000,0')
    ap.add_argument('--F0', type=float, default=5000.0)

    # training
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--epochs', type=int, default=20000)
    ap.add_argument('--epochs_u', type=int, default=0)
    ap.add_argument('--epochs_s', type=int, default=0)
    ap.add_argument('--epochs_joint', type=int, default=0)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_u', type=float, default=0.0)
    ap.add_argument('--lr_s', type=float, default=0.0)
    ap.add_argument('--lr_joint', type=float, default=0.0)

    ap.add_argument('--w_u', type=float, default=1.0)
    ap.add_argument('--w_usmooth', type=float, default=0.0)
    ap.add_argument('--w_const', type=float, default=1.0)
    ap.add_argument('--w_strain_data', type=float, default=0.0)
    ap.add_argument('--w_eq', type=float, default=1.0)

    ap.add_argument('--batch_elems', type=int, default=0,
                    help='elements per iter for const/strain_data; 0 -> all')
    ap.add_argument('--batch_elems_eq', type=int, default=0,
                    help='elements per iter for weak-eq loss; 0 -> all')

    ap.add_argument('--log_every', type=int, default=50)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # --- ckpt/log paths ---
    out_dir = os.path.join(args.ckpt_root, args.project)
    maybe_makedirs(out_dir)
    save_best_path = os.path.join(out_dir, args.save_best)
    save_last_path = os.path.join(out_dir, args.save_last)
    log_csv_path = os.path.join(out_dir, args.log_csv) if (args.log_csv and args.log_csv.lower() != 'none') else ''
    true_csv_path = os.path.join(out_dir, args.true_csv) if (args.true_csv and args.true_csv.lower() != 'none') else ''
    save_best_true_path = os.path.join(out_dir, args.save_best_true) if (args.save_best_true and args.save_best_true.lower() != 'none') else ''

    resume_ckpt = None
    if args.resume_ckpt:
        if not os.path.isfile(args.resume_ckpt):
            raise RuntimeError(f"resume_ckpt not found: {args.resume_ckpt}")
        resume_ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        print(f"[resume] loaded checkpoint: {args.resume_ckpt}")

    # --- load gp pack ---
    pkg = load_gp_npz(args.gp_npz)

    req_keys = ['V', 'T', 'node_ids', 'elem_ids', 'gp_xyz', 'gp_dV', 'grad_phi_hybrid_gp', 'dN_dx_gp',
                'set_BDIR_idx', 'set_BNEU_idx']
    for k in req_keys:
        if k not in pkg:
            raise RuntimeError(f"gp_npz missing key '{k}'. Re-export with the _with_dNdx script.")

    V = np.asarray(pkg['V'], dtype=np.float64)          # (nNode,3) mm
    node_ids = np.asarray(pkg['node_ids'], dtype=np.int64)
    conn20 = np.asarray(pkg['T'], dtype=np.int64)       # (nElem,20) 0-based
    elem_ids = np.asarray(pkg['elem_ids'], dtype=np.int64)

    n_nodes = int(V.shape[0])
    n_elem = int(conn20.shape[0])
    gp_xyz = np.asarray(pkg['gp_xyz'], dtype=np.float32)  # (E,ngp,3)
    ngp = int(gp_xyz.shape[1])

    # optional Abaqus GP truth tensors (single csv, one-time reordered by gp xyz)
    sigma_true_gp_np = None
    eps_true_gp_np = None
    if args.abaqus_gp_csv and os.path.isfile(args.abaqus_gp_csv):
        try:
            perm_map_gp, md = build_gp_perm_map_from_csv(
                args.abaqus_gp_csv,
                elem_ids,
                gp_xyz,
                tol_mm=float(args.gp_match_tol_mm),
                verbose=True,
            )
            sigma_true_gp_np = load_gp_sigma_case_with_perm(args.abaqus_gp_csv, elem_ids, perm_map_gp)  # (E,ngp,6)
            print(f"[eval] loaded Abaqus GP sigma truth: {args.abaqus_gp_csv}")
            try:
                eps_true_gp_np = load_gp_eps_case_with_perm(args.abaqus_gp_csv, elem_ids, perm_map_gp)  # (E,ngp,6)
                print(f"[eval] loaded Abaqus GP strain truth: {args.abaqus_gp_csv}")
            except Exception as e_eps:
                print(f"[warn] failed to load GP strain truth (E11..E23) from {args.abaqus_gp_csv}: {e_eps}")
                eps_true_gp_np = None
        except Exception as e:
            print(f"[warn] failed to build/reuse gp xyz permutation from abaqus_gp_csv -> skip gp truth eval. reason: {e}")
            sigma_true_gp_np = None
            eps_true_gp_np = None

    # boundary sets (compact indices)
    bdir_idx = np.asarray(pkg['set_BDIR_idx'], dtype=np.int64)
    bneu_idx = np.asarray(pkg['set_BNEU_idx'], dtype=np.int64)
    boundary_mask = np.zeros((n_nodes,), dtype=bool)
    boundary_mask[bdir_idx] = True
    boundary_mask[bneu_idx] = True

    # --- load hybrid modes at nodes, reorder to match gp_npz node_ids ---
    nid_phi, phi_node = load_phi_node_csv(args.hybrid_node_csv, args.hybrid_prefix, args.Ku)
    phi_node = reorder_rows_by_node_ids(node_ids, nid_phi, phi_node).astype(np.float32)

    # --- normalize xyz and phi for NN input ONLY ---
    if args.norm_load and os.path.isfile(args.norm_load):
        zn = np.load(args.norm_load, allow_pickle=False)
        xyz_center = np.asarray(zn['xyz_center'], dtype=np.float64)
        L0 = float(zn['L0'])
        if 'phi_mean_u' in zn.files:
            phi_mean = np.asarray(zn['phi_mean_u'], dtype=np.float32)
            phi_std = np.asarray(zn['phi_std_u'], dtype=np.float32)
        elif 'phi_u_mu' in zn.files:
            # backward-compatible with older pretrain scripts
            phi_mean = np.asarray(zn['phi_u_mu'], dtype=np.float32)
            phi_std = np.asarray(zn['phi_u_sd'], dtype=np.float32)
        else:
            raise RuntimeError(f"norm_load missing phi stats for u: {args.norm_load}")
        u0 = float(zn['u0'])
        eps0 = float(zn['eps0'])
        sigma0 = float(zn['sigma0'])
        F0 = float(zn['F0']) if 'F0' in zn.files else float(args.F0)
    else:
        xyz_center = V.mean(axis=0)
        bbox = V.max(axis=0) - V.min(axis=0)
        L0 = float(np.max(bbox))
        phi_mean = phi_node.mean(axis=0)
        phi_std = phi_node.std(axis=0) + 1e-12
        # u0 from observations
        obs_ids0, obs_u0 = load_obs_disp(args.disp_csv)
        u0 = compute_rms(obs_u0)
        eps0 = u0 / (L0 + 1e-12)
        sigma0 = args.E * eps0
        F0 = float(args.F0)

    xyz_in = norm_xyz_np(V, xyz_center, L0).astype(np.float32)
    phi_in = ((phi_node - phi_mean[None, :]) / phi_std[None, :]).astype(np.float32)

    # precompute normalized gp xyz for s_net input
    gp_xyz_norm = norm_xyz_np(gp_xyz.reshape(-1, 3), xyz_center, L0).astype(np.float32)

    # --- obs data -> indices in compact mesh ---
    obs_u_ids, obs_u_val = load_obs_disp(args.disp_csv)
    id2idx = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    obs_u_idx = np.array([id2idx[int(n)] for n in obs_u_ids.tolist()], dtype=np.int64)

    true_u_idx = None
    true_u_val = None
    if args.true_disp_csv:
        if os.path.isfile(args.true_disp_csv):
            true_u_ids, true_u_val = load_obs_disp(args.true_disp_csv)
            true_u_idx = np.array([id2idx[int(n)] for n in true_u_ids.tolist()], dtype=np.int64)
        else:
            print(f"[warn] true_disp_csv not found -> skip nodal displacement MAE eval: {args.true_disp_csv}")

    obs_eps_idx = None
    obs_eps_val = None
    obs_eps_mask = None
    if args.strain_csv and os.path.isfile(args.strain_csv):
        obs_eps_ids, obs_eps_val, obs_eps_mask = load_obs_strain_partial_csv(args.strain_csv)
        obs_eps_idx = np.array([id2idx[int(n)] for n in obs_eps_ids.tolist()], dtype=np.int64)

    # --- tensors ---
    dtype = torch.float32

    sigma_true_gp_t = as_torch(sigma_true_gp_np, device, dtype) if sigma_true_gp_np is not None else None
    eps_true_gp_t = as_torch(eps_true_gp_np, device, dtype) if eps_true_gp_np is not None else None
    gp_xyz_norm_t = as_torch(gp_xyz_norm, device, dtype)

    xyz_in_t = as_torch(xyz_in, device, dtype)
    phi_in_t = as_torch(phi_in, device, dtype)
    # --- robustify: some pipelines may add a singleton batch dim (1,N,*) ---
    if xyz_in_t.dim() == 3:
        if xyz_in_t.shape[0] == 1:
            xyz_in_t = xyz_in_t.squeeze(0)
        else:
            raise RuntimeError(f"xyz_in_t has unexpected shape {tuple(xyz_in_t.shape)}; expected (N,3)")

    if phi_in_t.dim() == 3:
        if phi_in_t.shape[0] == 1:
            phi_in_t = phi_in_t.squeeze(0)
        else:
            raise RuntimeError(f"phi_in_t has unexpected shape {tuple(phi_in_t.shape)}; expected (N,Ku)")
    obs_u_idx_t = torch.as_tensor(obs_u_idx, device=device, dtype=torch.long)
    obs_u_val_t = as_torch(obs_u_val, device, dtype)
    obs_eps_idx_t = torch.as_tensor(obs_eps_idx, device=device, dtype=torch.long) if obs_eps_idx is not None else None
    obs_eps_val_t = as_torch(obs_eps_val, device, dtype) if obs_eps_val is not None else None
    obs_eps_mask_t = torch.as_tensor(obs_eps_mask, device=device, dtype=torch.bool) if obs_eps_mask is not None else None
    true_u_idx_t = torch.as_tensor(true_u_idx, device=device, dtype=torch.long) if true_u_idx is not None else None
    true_u_val_t = as_torch(true_u_val, device, dtype) if true_u_val is not None else None

    conn20_t = torch.as_tensor(conn20, device=device, dtype=torch.long)

    # GP tensors (physics)
    gradphi_gp = np.asarray(pkg['grad_phi_hybrid_gp'], dtype=np.float32)
    if gradphi_gp.shape[2] < args.Ks:
        raise RuntimeError(f"grad_phi_hybrid_gp has K={gradphi_gp.shape[2]} < Ks={args.Ks}")
    gradphi_gp_t = as_torch(gradphi_gp[:, :, :args.Ks, :], device, dtype)  # (E,ngp,Ks,3)

    dN_dx_gp_t = as_torch(np.asarray(pkg['dN_dx_gp'], dtype=np.float32), device, dtype)            # (E,ngp,20,3)
    gp_dV_t = as_torch(np.asarray(pkg['gp_dV'], dtype=np.float32), device, dtype)                  # (E,ngp)

    # extrap for strain_data (requires ngp==8)
    if args.w_strain_data > 0.0:
        if 'gp_to_corner8_extrap' not in pkg:
            raise RuntimeError("gp_npz missing required key 'gp_to_corner8_extrap' for strain_data loss")
        if ngp != 8:
            raise RuntimeError("strain_data loss currently assumes ngp=8 (C3D20R)")

        E_extrap_t = as_torch(np.asarray(pkg['gp_to_corner8_extrap'], dtype=np.float32), device, dtype)  # (8,8)

        # corner node indices (prefer new key; fallback to old NodeId-based key)
        if 'corner_nodeidx' in pkg:
            corner_nodeidx_np = np.asarray(pkg['corner_nodeidx'], dtype=np.int64)
        elif 'elem_corner_node_ids' in pkg:
            # Old export key stores Abaqus NodeIds; map to compact 0-based node indices.
            corner_node_ids = np.asarray(pkg['elem_corner_node_ids'], dtype=np.int64)
            id2idx = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
            corner_nodeidx_np = np.empty_like(corner_node_ids, dtype=np.int64)
            missing_node_ids = []
            for i in range(corner_node_ids.shape[0]):
                for j in range(corner_node_ids.shape[1]):
                    nid = int(corner_node_ids[i, j])
                    k = id2idx.get(nid, None)
                    if k is None:
                        missing_node_ids.append(nid)
                        corner_nodeidx_np[i, j] = -1
                    else:
                        corner_nodeidx_np[i, j] = k
            if missing_node_ids:
                miss_u = sorted(set(missing_node_ids))
                raise RuntimeError(
                    f"old key 'elem_corner_node_ids' contains NodeIds not found in gp_npz['node_ids']; "
                    f"missing={len(miss_u)} (examples: {miss_u[:8]})"
                )
        else:
            raise RuntimeError(
                "gp_npz missing corner connectivity for strain_data loss: "
                "need 'corner_nodeidx' (new) or 'elem_corner_node_ids' (old)"
            )

        # corner weights (prefer new key; fallback to old key)
        if 'corner_w' in pkg:
            corner_w_np = np.asarray(pkg['corner_w'], dtype=np.float32)
        elif 'elem_corner_volw' in pkg:
            corner_w_np = np.asarray(pkg['elem_corner_volw'], dtype=np.float32)
        else:
            raise RuntimeError(
                "gp_npz missing corner weights for strain_data loss: "
                "need 'corner_w' (new) or 'elem_corner_volw' (old)"
            )

        # shape checks
        if corner_nodeidx_np.shape != (n_elem, 8):
            raise RuntimeError(
                f"corner_nodeidx shape mismatch: got {corner_nodeidx_np.shape}, expected ({n_elem}, 8)"
            )
        if corner_w_np.shape != (n_elem, 8):
            raise RuntimeError(
                f"corner_w shape mismatch: got {corner_w_np.shape}, expected ({n_elem}, 8)"
            )

        # index range checks
        cmin = int(np.min(corner_nodeidx_np))
        cmax = int(np.max(corner_nodeidx_np))
        if cmin < 0 or cmax >= n_nodes:
            raise RuntimeError(
                f"corner_nodeidx out of range: min={cmin}, max={cmax}, valid=[0,{n_nodes - 1}]"
            )

        corner_nodeidx_t = torch.as_tensor(corner_nodeidx_np, device=device, dtype=torch.long)  # (E,8)
        corner_w_t = as_torch(corner_w_np, device, dtype)  # (E,8)

    # weak-eq precompute
    w_dNdx_t = dN_dx_gp_t * gp_dV_t.unsqueeze(-1).unsqueeze(-1)  # (E,ngp,20,3)
    deg_total_t = build_node_degree(conn20_t, n_nodes)
    boundary_mask_t = torch.as_tensor(boundary_mask, device=device, dtype=torch.bool)
    weak_eq = WeakEquilibriumLoss(
        conn20_all=conn20_t,
        w_dNdx_all=w_dNdx_t,
        deg_total=deg_total_t,
        boundary_mask=boundary_mask_t,
        F0=float(F0),
    )

    # smoothness graph
    edges_t = None
    if args.w_usmooth > 0.0:
        edges = build_unique_edges_from_conn20(conn20)
        edges_t = torch.as_tensor(edges, device=device, dtype=torch.long)

    # --- conditioning force (F/F0) ---
    cond_dim = 3 if int(args.cond_force) == 1 else 0

    # known force
    F_known = parse_force_vec(args.force_known)
    F_known_hat = F_known / float(F0)
    F_known_t = as_torch(F_known.reshape(1, 3), device, dtype)
    F_known_hat_t = as_torch(F_known_hat.reshape(1, 3), device, dtype)

    # optional load_net
    load_net: Optional[LoadNetSet] = None
    F_hat_t: Optional[torch.Tensor] = None

    if cond_dim > 0 and int(args.use_load_net) == 1:
        load_net = LoadNetSet(use_disp=bool(args.load_use_disp), use_strain=bool(args.load_use_strain)).to(device)
        if resume_ckpt is not None and 'load_net' in resume_ckpt:
            load_net.load_state_dict(resume_ckpt['load_net'], strict=False)
            print(f"[resume] restored load_net from {args.resume_ckpt}")
        elif args.init_f_ckpt and os.path.isfile(args.init_f_ckpt):
            ck = torch.load(args.init_f_ckpt, map_location='cpu')
            if 'load_net' in ck:
                load_net.load_state_dict(ck['load_net'], strict=False)
            else:
                load_net.load_state_dict(ck, strict=False)
            print(f"[init] loaded load_net from {args.init_f_ckpt}")
        load_net.eval()

        # build input sets (normalized xyz + normalized u/eps)
        xyz_obs_u = xyz_in[obs_u_idx]
        u_obs_hat = obs_u_val / float(u0)
        disp_set = np.concatenate([xyz_obs_u, u_obs_hat], axis=1).astype(np.float32)  # (n_obs_u,6)
        disp_set_t = as_torch(disp_set, device, dtype)

        strain_set_t = None
        if bool(args.load_use_strain):
            if obs_eps_idx is None:
                raise RuntimeError('load_use_strain=1 but strain_csv not provided')
            if obs_eps_mask is None or (not np.all(obs_eps_mask)):
                raise RuntimeError(
                    'load_use_strain=1 currently requires full 6 strain components at each observed strain node. '
                    'For partial-direction strain data, set --load_use_strain 0, or extend LoadNetSet to consume an observation mask.'
                )
            xyz_obs_e = xyz_in[obs_eps_idx]
            eps_obs_hat = obs_eps_val / float(eps0)
            strain_set = np.concatenate([xyz_obs_e, eps_obs_hat], axis=1).astype(np.float32)  # (n_obs_e,9)
            strain_set_t = as_torch(strain_set, device, dtype)

        with torch.no_grad():
            F_hat_t = load_net(disp_set_t if bool(args.load_use_disp) else None,
                               strain_set_t if bool(args.load_use_strain) else None)  # (1,3)
        print(f"[cond] load_net predicted F/F0 = {F_hat_t.detach().cpu().numpy().reshape(-1)}")

    if cond_dim > 0 and F_hat_t is None:
        F_hat_t = F_known_hat_t
        print(f"[cond] using known F/F0 = {F_hat_t.detach().cpu().numpy().reshape(-1)}")

    cond_node_t = None
    if cond_dim > 0:
        cond_node_t = F_hat_t.repeat(n_nodes, 1)

    # --- build nets ---
    u_net = DispNet(Ku=args.Ku, width=256, depth=5, cond_dim=cond_dim).to(device)
    s_net = StressCoeffNet(Ks=args.Ks, width=256, depth=6, cond_dim=cond_dim).to(device)

    # init ckpts (pretrain)
    if resume_ckpt is not None:
        if ('u_net' not in resume_ckpt) or ('s_net' not in resume_ckpt):
            raise RuntimeError(
                f"resume_ckpt must contain both 'u_net' and 's_net' states: {args.resume_ckpt}"
            )
        u_net.load_state_dict(resume_ckpt['u_net'], strict=False)
        s_net.load_state_dict(resume_ckpt['s_net'], strict=False)
        print(f"[resume] restored u_net and s_net from {args.resume_ckpt}")
    else:
        if args.init_u_ckpt and os.path.isfile(args.init_u_ckpt):
            ck = torch.load(args.init_u_ckpt, map_location='cpu')
            if 'u_net' in ck:
                u_net.load_state_dict(ck['u_net'], strict=False)
            else:
                u_net.load_state_dict(ck, strict=False)
            print(f"[init] loaded u_net from {args.init_u_ckpt}")
        if args.init_s_ckpt and os.path.isfile(args.init_s_ckpt):
            ck = torch.load(args.init_s_ckpt, map_location='cpu')
            if 's_net' in ck:
                s_net.load_state_dict(ck['s_net'], strict=False)
            else:
                s_net.load_state_dict(ck, strict=False)
            print(f"[init] loaded s_net from {args.init_s_ckpt}")

    # --- optimizers ---
    lr_u = args.lr_u if args.lr_u > 0 else args.lr
    lr_s = args.lr_s if args.lr_s > 0 else args.lr
    lr_j = args.lr_joint if args.lr_joint > 0 else args.lr

    opt_u = torch.optim.Adam(u_net.parameters(), lr=lr_u)
    opt_s = torch.optim.Adam(s_net.parameters(), lr=lr_s)
    opt_j = torch.optim.Adam(list(u_net.parameters()) + list(s_net.parameters()), lr=lr_j)

    if resume_ckpt is not None:
        for opt_name, opt in (('opt_u', opt_u), ('opt_s', opt_s), ('opt_j', opt_j)):
            if opt_name in resume_ckpt:
                try:
                    opt.load_state_dict(resume_ckpt[opt_name])
                    print(f"[resume] restored {opt_name} from {args.resume_ckpt}")
                except Exception as e:
                    print(f"[warn] failed to restore {opt_name} from {args.resume_ckpt}: {e}")
            else:
                print(f"[resume] {opt_name} not found in {args.resume_ckpt}; optimizer state will start fresh")
    for pg in opt_u.param_groups:
        pg['lr'] = lr_u
    for pg in opt_s.param_groups:
        pg['lr'] = lr_s
    for pg in opt_j.param_groups:
        pg['lr'] = lr_j

    # stage epochs
    if args.epochs_u > 0 or args.epochs_s > 0 or args.epochs_joint > 0:
        epochs_u = int(args.epochs_u)
        epochs_s = int(args.epochs_s)
        epochs_joint = int(args.epochs_joint)
        if epochs_joint <= 0:
            epochs_joint = max(0, int(args.epochs) - epochs_u - epochs_s)
    else:
        epochs_u = 0
        epochs_s = 0
        epochs_joint = int(args.epochs)

    best = float(resume_ckpt.get('best', float('inf'))) if isinstance(resume_ckpt, dict) else float('inf')
    best_true = float(resume_ckpt.get('best_true', float('inf'))) if isinstance(resume_ckpt, dict) else float('inf')
    step = int(resume_ckpt.get('step', 0)) if isinstance(resume_ckpt, dict) else 0
    resume_stage = str(resume_ckpt.get('stage', '')) if isinstance(resume_ckpt, dict) else ''
    resume_epoch_in_stage = int(resume_ckpt.get('epoch_in_stage', 0)) if isinstance(resume_ckpt, dict) else 0

    if resume_ckpt is not None:
        print(
            f"[resume] continue from step={step} stage={resume_stage or 'unknown'} "
            f"epoch={resume_epoch_in_stage} best={best:.3e} best_true={best_true:.3e}"
        )

    # logging
    ensure_csv_header(
        log_csv_path,
        'step,stage,loss_total,loss_u,loss_usmooth,loss_const,loss_strain_data,loss_eq,eq_nodes_used,eq_meanN\n',
        append_if_exists=(resume_ckpt is not None),
    )
    ensure_csv_header(
        true_csv_path,
        'step,stage,tag,Fx_N,Fy_N,Fz_N,'
        'sigma_mse_norm_gp,sigma_rmse_gp,sigma_wmape_pct_gp,'
        'u_mae_node,u_wmape_pct_node,'
        'eps_mae_u_gp,eps_mae_s_gp,eps_s_wmape_pct_gp\n',
        append_if_exists=(resume_ckpt is not None),
    )

    # element sampling weights
    elem_vol = gp_dV_t.sum(dim=1).detach().cpu().numpy()  # (E,)
    vol_prob = elem_vol / (elem_vol.sum() + 1e-12)

    def sample_elem_idx(batch_elems: int) -> Optional[torch.Tensor]:
        if batch_elems <= 0 or batch_elems >= n_elem:
            return None
        idx = np.random.choice(n_elem, size=batch_elems, replace=False, p=vol_prob)
        return torch.as_tensor(idx, device=device, dtype=torch.long)

    t0 = time.time()
    train_state = {'stage': 'init', 'epoch_in_stage': 0}

    def save_training_ckpt(path: str, extra: Optional[Dict] = None):
        payload = {
            'step': int(step),
            'best': float(best),
            'best_true': float(best_true),
            'stage': str(train_state['stage']),
            'epoch_in_stage': int(train_state['epoch_in_stage']),
        }
        if extra:
            payload.update(extra)
        save_ckpt(
            path,
            u_net,
            s_net,
            load_net,
            extra=payload,
            opt_u=opt_u,
            opt_s=opt_s,
            opt_j=opt_j,
        )

    # helper: u -> eps_gp
    def compute_u_and_epsgp(
        require_grad: bool,
        cond_node_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_nodes = cond_node_t if cond_node_override is None else cond_node_override
        if require_grad:
            u_all = u_net(xyz_in_t, phi_in_t, cond=cond_nodes)  # (nNode,3)
        else:
            with torch.no_grad():
                u_all = u_net(xyz_in_t, phi_in_t, cond=cond_nodes)

        u_elem = u_all[conn20_t]  # (E,20,3)
        grad_u = torch.einsum('eai,egaj->egij', u_elem, dN_dx_gp_t)  # (E,ngp,3,3)

        e11 = grad_u[..., 0, 0]
        e22 = grad_u[..., 1, 1]
        e33 = grad_u[..., 2, 2]
        g12 = grad_u[..., 0, 1] + grad_u[..., 1, 0]
        g13 = grad_u[..., 0, 2] + grad_u[..., 2, 0]
        g23 = grad_u[..., 1, 2] + grad_u[..., 2, 1]
        eps6 = torch.stack([e11, e22, e33, g12, g13, g23], dim=-1)
        return u_all, eps6

    # helper: s_net -> sigma (raw & sym) for a set of elements
    def compute_sigma_gp(
        elem_idx: Optional[torch.Tensor],
        cond_hat_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_hat = F_hat_t if cond_hat_override is None else cond_hat_override
        if elem_idx is None:
            gradphi = gradphi_gp_t
            xgp_t = gp_xyz_norm_t.reshape(-1, 3)
            cond_gp = cond_hat.repeat(n_elem * ngp, 1) if cond_dim > 0 else None
            E_used = n_elem
        else:
            gradphi = gradphi_gp_t[elem_idx]
            xgp_t = gp_xyz_norm_t.view(n_elem, ngp, 3)[elem_idx].reshape(-1, 3)
            cond_gp = cond_hat.repeat(int(elem_idx.shape[0]) * ngp, 1) if cond_dim > 0 else None
            E_used = int(elem_idx.shape[0])

        P = s_net(xgp_t, cond=cond_gp)  # (E_used*ngp,3,K)
        P = P.view(E_used, ngp, 3, args.Ks)

        sigma_raw = assemble_sigma_from_P(
            P.reshape(-1, 3, args.Ks),
            gradphi.reshape(-1, args.Ks, 3)
        ).view(E_used, ngp, 3, 3)

        sigma_sym = 0.5 * (sigma_raw + sigma_raw.transpose(-1, -2))
        return sigma_raw, sigma_sym

    def should_eval(step_idx: int) -> bool:
        return (args.eval_every > 0) and ((step_idx == 1) or (step_idx % args.eval_every == 0))

    has_eval_targets = (sigma_true_gp_t is not None) or (eps_true_gp_t is not None) or (true_u_idx_t is not None)
    if args.eval_every > 0 and not has_eval_targets:
        print("[warn] eval_every>0 but no eval targets found (need --abaqus_gp_csv and/or --true_disp_csv).")

    def eval_all_metrics(tag: str, stage: str, step_idx: int, force_run: bool = False):
        nonlocal best_true
        if (not force_run) and (not should_eval(step_idx)):
            return None
        if not has_eval_targets:
            return None

        u_was_training = u_net.training
        s_was_training = s_net.training
        u_net.eval()
        s_net.eval()

        # Fixed eval force source: known force.
        cond_eval_node = None
        cond_eval_hat = None
        Fx = Fy = Fz = 0.0
        if cond_dim > 0:
            cond_eval_hat = F_known_hat_t
            cond_eval_node = cond_eval_hat.repeat(n_nodes, 1)
            Fx, Fy, Fz = [float(v) for v in F_known_t.reshape(-1).detach().cpu().numpy().tolist()]

        sigma_mse_norm = None
        sigma_rmse = None
        sigma_wmape = None
        u_mae = None
        u_wmape = None
        eps_mae_u = None
        eps_mae_s = None
        eps_s_wmape = None

        with torch.no_grad():
            if sigma_true_gp_t is not None:
                _, sigma_sym_full = compute_sigma_gp(elem_idx=None, cond_hat_override=cond_eval_hat)
                sig_pred6 = sigma33_to_voigt6(sigma_sym_full)  # (E,ngp,6)
                diff_s = sig_pred6 - sigma_true_gp_t
                sigma_mse_norm = float((diff_s / float(sigma0)).pow(2).mean().item())
                sigma_rmse = float(torch.sqrt((diff_s * diff_s).mean()).item())
                sigma_wmape = compute_wmape(sig_pred6, sigma_true_gp_t)

            if true_u_idx_t is not None:
                u_all_eval = u_net(xyz_in_t, phi_in_t, cond=cond_eval_node)
                u_pred_true = u_all_eval[true_u_idx_t]
                u_mae = float((u_pred_true - true_u_val_t).abs().mean().item())
                u_wmape = compute_wmape(u_pred_true, true_u_val_t)

            if eps_true_gp_t is not None:
                _, eps_u_gp_full = compute_u_and_epsgp(require_grad=False, cond_node_override=cond_eval_node)
                eps_mae_u = float((eps_u_gp_full - eps_true_gp_t).abs().mean().item())

                _, sigma_sym_full = compute_sigma_gp(elem_idx=None, cond_hat_override=cond_eval_hat)
                eps_s_gp_full = isotropic_eps_from_sigma_voigt(sigma33_to_voigt6(sigma_sym_full), args.E, args.nu)
                eps_mae_s = float((eps_s_gp_full - eps_true_gp_t).abs().mean().item())
                eps_s_wmape = compute_wmape(eps_s_gp_full, eps_true_gp_t)

        if u_was_training:
            u_net.train()
        if s_was_training:
            s_net.train()

        def _fmt(v: Optional[float]) -> str:
            return "NA" if v is None else f"{v:.3e}"

        print(
            f"[eval] {tag} stage={stage} F=({Fx:.3e},{Fy:.3e},{Fz:.3e}) "
            f"sigma_rmse={_fmt(sigma_rmse)} MPa sigma_mse_n={_fmt(sigma_mse_norm)} sigma_wmape={_fmt(sigma_wmape)}% "
            f"u_mae={_fmt(u_mae)} u_wmape={_fmt(u_wmape)}% "
            f"eps_u_mae={_fmt(eps_mae_u)} eps_s_mae={_fmt(eps_mae_s)} eps_s_wmape={_fmt(eps_s_wmape)}%"
        )

        if true_csv_path:
            with open(true_csv_path, 'a', encoding='utf-8') as f:
                def _csv(v: Optional[float]) -> str:
                    return "nan" if v is None else f"{float(v):.8e}"
                f.write(
                    f"{step_idx},{stage},{tag},{Fx:.8e},{Fy:.8e},{Fz:.8e},"
                    f"{_csv(sigma_mse_norm)},{_csv(sigma_rmse)},{_csv(sigma_wmape)},"
                    f"{_csv(u_mae)},{_csv(u_wmape)},"
                    f"{_csv(eps_mae_u)},{_csv(eps_mae_s)},{_csv(eps_s_wmape)}\n"
                )

        metric_name = str(args.best_true_metric).lower()
        metric_val = None
        if metric_name == 'eps_s':
            metric_val = eps_mae_s
        elif metric_name == 'eps_u':
            metric_val = eps_mae_u
        elif metric_name == 'u':
            metric_val = u_mae
        elif metric_name == 'sigma_wmape':
            metric_val = sigma_wmape
        elif metric_name == 'u_wmape':
            metric_val = u_wmape
        elif metric_name == 'eps_s_wmape':
            metric_val = eps_s_wmape

        if (metric_val is not None) and np.isfinite(metric_val):
            if metric_val < best_true:
                best_true = float(metric_val)
                if save_best_true_path:
                    save_training_ckpt(
                        save_best_true_path,
                        extra={
                            'step': int(step_idx),
                            'best_true': float(best_true),
                            'best_true_metric': metric_name,
                        },
                    )
                    print(f"[best-true] step={step_idx} {metric_name}={metric_val:.6g} -> {save_best_true_path}")

        return {
            'sigma_mse_norm': sigma_mse_norm,
            'sigma_rmse': sigma_rmse,
            'sigma_wmape': sigma_wmape,
            'u_mae': u_mae,
            'u_wmape': u_wmape,
            'eps_mae_u': eps_mae_u,
            'eps_mae_s': eps_mae_s,
            'eps_s_wmape': eps_s_wmape,
        }

    # -----------------------------
    # Stage 1: u only
    # -----------------------------
    for ep in range(1, epochs_u + 1):
        step += 1
        train_state['stage'] = 'u'
        train_state['epoch_in_stage'] = ep
        u_net.train(); s_net.eval()

        opt_u.zero_grad(set_to_none=True)

        u_all = u_net(xyz_in_t, phi_in_t, cond=cond_node_t)
        u_pred_obs = u_all[obs_u_idx_t]
        loss_u = ((u_pred_obs - obs_u_val_t) / float(u0)).pow(2).mean()

        loss_us = u_all.new_tensor(0.0)
        if edges_t is not None:
            du = u_all[edges_t[:, 0]] - u_all[edges_t[:, 1]]
            loss_us = (du / float(u0)).pow(2).mean()

        loss = args.w_u * loss_u + args.w_usmooth * loss_us
        loss.backward()
        opt_u.step()

        if should_eval(step):
            eval_all_metrics(f"step={step}", stage='u', step_idx=step)

        if (step % args.log_every) == 0 or ep == 1 or ep == epochs_u:
            print(f"[u] step={step} loss={loss.item():.3e} u={loss_u.item():.3e} us={loss_us.item():.3e}")

        if log_csv_path:
            with open(log_csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{step},u,{loss.item()},{loss_u.item()},{loss_us.item()},0,0,0,0,0\n")

    # -----------------------------
    # Stage 2: s only (u frozen)
    # -----------------------------
    for ep in range(1, epochs_s + 1):
        step += 1
        train_state['stage'] = 's'
        train_state['epoch_in_stage'] = ep
        u_net.eval(); s_net.train()

        opt_s.zero_grad(set_to_none=True)

        elem_idx = sample_elem_idx(args.batch_elems)
        elem_idx_eq = sample_elem_idx(args.batch_elems_eq)

        # u->eps (no grad)
        _, eps_u_gp = compute_u_and_epsgp(require_grad=False)
        eps_u_gp_used = eps_u_gp if elem_idx is None else eps_u_gp[elem_idx]

        # s_net -> sigma
        sigma_raw, sigma_sym = compute_sigma_gp(elem_idx)

        # constitutive target from u
        sig_from_u6 = isotropic_sigma_from_eps_eng(eps_u_gp_used, args.E, args.nu)
        sig_from_u33 = voigt6_to_sigma33(sig_from_u6)
        gp_dV_used = gp_dV_t if elem_idx is None else gp_dV_t[elem_idx]
        diff2 = ((sigma_raw - sig_from_u33) / float(sigma0)).pow(2).view(sigma_raw.shape[0], ngp, -1).mean(dim=-1)
        loss_const = (diff2 * gp_dV_used).sum() / (gp_dV_used.sum() + 1e-12)

        # strain data loss (optional)
        loss_sd = sigma_raw.new_tensor(0.0)
        if args.w_strain_data > 0.0 and obs_eps_idx is not None:
            sig6 = sigma33_to_voigt6(sigma_sym)
            eps6 = isotropic_eps_from_sigma_voigt(sig6, args.E, args.nu)  # (E_used,ngp,6)

            eps_corner = torch.einsum('cg,egk->eck', E_extrap_t, eps6)  # (E_used,8,6)

            acc = torch.zeros((n_nodes, 6), device=device, dtype=dtype)
            wacc = torch.zeros((n_nodes, 1), device=device, dtype=dtype)

            if elem_idx is None:
                corner_idx = corner_nodeidx_t
                cw = corner_w_t
            else:
                corner_idx = corner_nodeidx_t[elem_idx]
                cw = corner_w_t[elem_idx]

            idx_flat = corner_idx.reshape(-1)
            val_flat = (eps_corner * cw.unsqueeze(-1)).reshape(-1, 6)
            acc.index_add_(0, idx_flat, val_flat)
            wacc.index_add_(0, idx_flat, cw.reshape(-1, 1))
            eps_node = acc / (wacc + 1e-12)

            eps_pred_obs = eps_node[obs_eps_idx_t]
            diff_sd = ((eps_pred_obs - obs_eps_val_t) / float(eps0)).pow(2)
            n_valid_sd = int(obs_eps_mask_t.sum().item())
            loss_sd = diff_sd[obs_eps_mask_t].mean() if n_valid_sd > 0 else sigma_raw.new_tensor(0.0)

        # weak equilibrium loss
        if args.w_eq <= 0.0:
            loss_eq = sigma_raw.new_tensor(0.0)
            eq_stats = {'n_nodes_used': 0, 'mean_res_N': 0.0}
        else:
            if elem_idx_eq is None:
                # need sigma for all elements
                _, sigma_sym_full = compute_sigma_gp(elem_idx=None)
                loss_eq, eq_stats = weak_eq.loss(sigma_sym_full, elem_batch=None)
            else:
                # compute sigma only for eq batch and assemble on complete nodes of that batch
                _, sigma_sym_eq = compute_sigma_gp(elem_idx=elem_idx_eq)
                conn_b = conn20_t[elem_idx_eq]
                w_b = w_dNdx_t[elem_idx_eq]
                f_int = assemble_internal_nodal_forces(conn_b, sigma_sym_eq, w_b, n_nodes)
                complete = complete_node_mask_for_elem_batch(conn_b, deg_total_t, n_nodes)
                mask = (~boundary_mask_t) & complete
                r = torch.linalg.norm(f_int[mask], dim=1)
                loss_eq = (r / float(F0)).pow(2).mean() if r.numel() > 0 else sigma_sym_eq.new_tensor(0.0)
                with torch.no_grad():
                    eq_stats = {
                        'n_nodes_used': int(r.numel()),
                        'mean_res_N': float(r.detach().cpu().numpy().mean()) if r.numel() > 0 else 0.0,
                    }

        loss = args.w_const * loss_const + args.w_strain_data * loss_sd + args.w_eq * loss_eq
        loss.backward()
        opt_s.step()

        if should_eval(step):
            eval_all_metrics(f"step={step}", stage='s', step_idx=step)

        if (step % args.log_every) == 0 or ep == 1 or ep == epochs_s:
            print(
                f"[s] step={step} loss={loss.item():.3e} const={loss_const.item():.3e} "
                f"sd={loss_sd.item():.3e} eq={loss_eq.item():.3e} "
                f"eq_nodes={eq_stats.get('n_nodes_used',0)} eq_meanN={eq_stats.get('mean_res_N',0.0):.2e}"
            )

        if log_csv_path:
            with open(log_csv_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"{step},s,{loss.item()},0,0,{loss_const.item()},{loss_sd.item()},{loss_eq.item()},"
                    f"{eq_stats.get('n_nodes_used',0)},{eq_stats.get('mean_res_N',0.0)}\n"
                )

    # -----------------------------
    # Stage 3: joint
    # -----------------------------
    for ep in range(1, epochs_joint + 1):
        step += 1
        train_state['stage'] = 'j'
        train_state['epoch_in_stage'] = ep
        u_net.train(); s_net.train()

        opt_j.zero_grad(set_to_none=True)

        # u
        u_all, eps_u_gp = compute_u_and_epsgp(require_grad=True)
        u_pred_obs = u_all[obs_u_idx_t]
        loss_u = ((u_pred_obs - obs_u_val_t) / float(u0)).pow(2).mean()

        loss_us = u_all.new_tensor(0.0)
        if edges_t is not None:
            du = u_all[edges_t[:, 0]] - u_all[edges_t[:, 1]]
            loss_us = (du / float(u0)).pow(2).mean()

        # element subsampling for const/sd
        elem_idx = sample_elem_idx(args.batch_elems)
        elem_idx_eq = sample_elem_idx(args.batch_elems_eq)

        eps_u_gp_used = eps_u_gp if elem_idx is None else eps_u_gp[elem_idx]

        sigma_raw, sigma_sym = compute_sigma_gp(elem_idx)

        sig_from_u6 = isotropic_sigma_from_eps_eng(eps_u_gp_used, args.E, args.nu)
        sig_from_u33 = voigt6_to_sigma33(sig_from_u6)
        gp_dV_used = gp_dV_t if elem_idx is None else gp_dV_t[elem_idx]
        diff2 = ((sigma_raw - sig_from_u33) / float(sigma0)).pow(2).view(sigma_raw.shape[0], ngp, -1).mean(dim=-1)
        loss_const = (diff2 * gp_dV_used).sum() / (gp_dV_used.sum() + 1e-12)

        loss_sd = sigma_raw.new_tensor(0.0)
        if args.w_strain_data > 0.0 and obs_eps_idx is not None:
            sig6 = sigma33_to_voigt6(sigma_sym)
            eps6 = isotropic_eps_from_sigma_voigt(sig6, args.E, args.nu)
            eps_corner = torch.einsum('cg,egk->eck', E_extrap_t, eps6)

            acc = torch.zeros((n_nodes, 6), device=device, dtype=dtype)
            wacc = torch.zeros((n_nodes, 1), device=device, dtype=dtype)

            if elem_idx is None:
                corner_idx = corner_nodeidx_t
                cw = corner_w_t
            else:
                corner_idx = corner_nodeidx_t[elem_idx]
                cw = corner_w_t[elem_idx]

            idx_flat = corner_idx.reshape(-1)
            val_flat = (eps_corner * cw.unsqueeze(-1)).reshape(-1, 6)
            acc.index_add_(0, idx_flat, val_flat)
            wacc.index_add_(0, idx_flat, cw.reshape(-1, 1))
            eps_node = acc / (wacc + 1e-12)

            eps_pred_obs = eps_node[obs_eps_idx_t]
            diff_sd = ((eps_pred_obs - obs_eps_val_t) / float(eps0)).pow(2)
            n_valid_sd = int(obs_eps_mask_t.sum().item())
            loss_sd = diff_sd[obs_eps_mask_t].mean() if n_valid_sd > 0 else sigma_raw.new_tensor(0.0)

        # weak eq
        if args.w_eq <= 0.0:
            loss_eq = sigma_raw.new_tensor(0.0)
            eq_stats = {'n_nodes_used': 0, 'mean_res_N': 0.0}
        else:
            if elem_idx_eq is None:
                _, sigma_sym_full = compute_sigma_gp(elem_idx=None)
                loss_eq, eq_stats = weak_eq.loss(sigma_sym_full, elem_batch=None)
            else:
                _, sigma_sym_eq = compute_sigma_gp(elem_idx=elem_idx_eq)
                conn_b = conn20_t[elem_idx_eq]
                w_b = w_dNdx_t[elem_idx_eq]
                f_int = assemble_internal_nodal_forces(conn_b, sigma_sym_eq, w_b, n_nodes)
                complete = complete_node_mask_for_elem_batch(conn_b, deg_total_t, n_nodes)
                mask = (~boundary_mask_t) & complete
                r = torch.linalg.norm(f_int[mask], dim=1)
                loss_eq = (r / float(F0)).pow(2).mean() if r.numel() > 0 else sigma_sym_eq.new_tensor(0.0)
                with torch.no_grad():
                    eq_stats = {
                        'n_nodes_used': int(r.numel()),
                        'mean_res_N': float(r.detach().cpu().numpy().mean()) if r.numel() > 0 else 0.0,
                    }

        loss = (
            args.w_u * loss_u +
            args.w_usmooth * loss_us +
            args.w_const * loss_const +
            args.w_strain_data * loss_sd +
            args.w_eq * loss_eq
        )
        loss.backward()
        opt_j.step()

        if should_eval(step):
            eval_all_metrics(f"step={step}", stage='j', step_idx=step)

        # save best by total loss
        if loss.item() < best:
            best = float(loss.item())
            save_training_ckpt(save_best_path)

        if (step % args.log_every) == 0 or ep == 1 or ep == epochs_joint:
            print(
                f"[j] step={step} loss={loss.item():.3e} u={loss_u.item():.3e} us={loss_us.item():.3e} "
                f"const={loss_const.item():.3e} sd={loss_sd.item():.3e} eq={loss_eq.item():.3e} "
                f"eq_nodes={eq_stats.get('n_nodes_used',0)} eq_meanN={eq_stats.get('mean_res_N',0.0):.2e}"
            )

        if log_csv_path:
            with open(log_csv_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"{step},j,{loss.item()},{loss_u.item()},{loss_us.item()},{loss_const.item()},{loss_sd.item()},{loss_eq.item()},"
                    f"{eq_stats.get('n_nodes_used',0)},{eq_stats.get('mean_res_N',0.0)}\n"
                )

    # save last
    train_state['stage'] = 'done'
    train_state['epoch_in_stage'] = 0
    save_training_ckpt(save_last_path)

    dt = time.time() - t0
    print(f"[done] steps={step} best={best:.3e} time={dt/60:.1f} min")
    print(f"[saved] best={save_best_path}")
    print(f"[saved] last={save_last_path}")
    if log_csv_path:
        print(f"[log] {log_csv_path}")
    if true_csv_path:
        print(f"[eval-log] {true_csv_path}")
    if save_best_true_path:
        print(f"[saved] best_true={save_best_true_path}")


if __name__ == '__main__':
    main()
