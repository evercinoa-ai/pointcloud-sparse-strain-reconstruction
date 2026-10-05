#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Abaqus .inp (六面体实体单元：C3D8R/C3D8/C3D20R/C3D20) 计算“体域 Hybrid LBO”特征函数

- 体域标量 Laplacian 特征问题（FEM）：
    K φ = λ M φ

- general：整块外边界自然 Neumann ⇒ 存在常数零模态
- specific：对 (BDIR ∪ BNEU) 施零 Dirichlet，其余自然 Neumann
- hybrid：φ_h = φ_g + φ_s（逐列相加），λ_h 取 λ_g（与旧脚本一致）

保持兼容的输出字段/CSV列名（不要改）：
- 输出 dict keys: 'V','T','phi_general','lambda_general','phi_specific','lambda_specific','phi_hybrid','lambda_hybrid','sets_used'
- export_csv 列名：x,y,z, phi_{hybrid|general|specific}_i, phi_*_i_dx,dy,dz, dxx,dxy,dxz, dyx,dyy,dyz, dzx,dzy,dzz（前缀由 prefix 决定）

边界节点集约定（来自 Abaqus *NSET）：
- BDIR：Dirichlet（位移约束边界节点）
- BNEU：Neumann（受外力/已知表面力边界节点）——在 specific LBO 中也视作“非自由边界”，施零 Dirichlet
- Set-1：全部外表面节点（用于推导自由边界节点集 BFREE = Set-1 - BDIR - BNEU）

运行示例
python ".\hybrid_lbo_from_inp_solid_C3D20R_export_fast_v5(3).py" data\Lab-use-solid-D20-R.inp `
  --modes 50 `
  --surfset Set-1 `
  --bdir Dirichlet `
  --bneu Neumann `
  --gauss_order 2 `
  --export_csv Lab-use-solid-20-LBO-R.csv `
  --export_gp_npz Lab-use-solid-20-R-GP_topo_derivs.npz `
  --trace_anchor_hybrid none `
  --verbose 1

python ".\hybrid_lbo_from_inp_solid_C3D20R_export_fast_v5(3).py" data\Lab-use-solid-D20-R.inp `
  --modes 50 `
  --surfset Set-1 `
  --bdir Dirichlet `
  --bneu __NONE__ `
  --gauss_order 2 `
  --export_csv Lab-use-solid-20-LBO-R-v.csv `
  --export_gp_npz Lab-use-solid-20-R-GP_topo_derivs-v.npz `
  --mode_scale grad_rms `
  --scale_target 1.0 `
  --scale_clip 1e-6,1e6 `
  --trace_anchor_hybrid none `
  --verbose 1
"""

import argparse
import re
import numpy as np
from collections import defaultdict

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lobpcg, spilu, LinearOperator

# 最近一次从 .inp 解析到的单元类型（用于自动选择积分阶次等）
_LAST_ELEM_TYPE = None

# 记录 .inp 中检测到的六面体单元类型（用于自动选择积分阶次等）
_LAST_ELEM_TYPE = ""
_LAST_ELEM_IDS = None  # 最近一次从 .inp 解析到的六面体元素 ID 顺序（与 T 行对应）

# ------------------------- 0) 质量评估工具 -------------------------
def eval_eig_quality(K, M, vals, vecs, eps=1e-30):
    # vecs: (n,k), vals: (k,)
    Kv = K @ vecs
    Mv = M @ vecs
    R = Kv - Mv * vals.reshape(1, -1)
    rnorm = np.sqrt(np.sum(R * R, axis=0))
    knorm = np.sqrt(np.sum(Kv * Kv, axis=0))
    rel = rnorm / np.maximum(knorm, eps)

    G = (vecs.T @ (M @ vecs))
    mnorm = np.diag(G)
    off = G - np.diag(np.diag(G))
    orth_max_offdiag = float(np.max(np.abs(off))) if off.size else 0.0
    return rnorm, rel, mnorm, orth_max_offdiag


def write_quality_csv(path, tag, vals, rnorm, rel, mnorm, orth_max_offdiag):
    import pandas as pd
    df = pd.DataFrame({
        "tag": tag,
        "mode": np.arange(1, len(vals) + 1),
        "lambda": vals,
        "res_norm": rnorm,
        "rel_res": rel,
        "mnorm": mnorm,
        "orth_max_offdiag(all_modes)": orth_max_offdiag,
    })
    df.to_csv(path, index=False)
    print(f"[eig_report] Wrote {path}")



# ------------------------- 0) .inp 读取 -------------------------

def read_abaqus_inp_solid(inp_path, hex_types=("C3D8R", "C3D8", "C3D20R", "C3D20")):
    """
    读取 Abaqus .inp：*NODE、*ELEMENT、*NSET

    返回：
      V: (n,3) 节点坐标（按 node_ids_order 的顺序）
      T: (m,nloc) 六面体连通性（0-based 索引），nloc=8 或 20
      node_sets_0: {NAME: np.ndarray(0-based node indices)}
      node_ids_order: 原 Abaqus 节点 ID 顺序（便于回溯）
    """
    nodes = {}                     # nid -> (x,y,z)
    node_ids_order = []            # 节点出现顺序
    hex_elems = []                 # 每个元素存 nloc 个 node id（Abaqus id）
    hex_eids = []                  # 每个六面体元素的 Abaqus 元素 ID（与 hex_elems 同序）
    node_sets = defaultdict(set)   # NSET: name -> set(node_id)

    # 记录“首次遇到的六面体单元类型”，并要求全局一致（否则 T 无法是一个矩阵）
    global _LAST_ELEM_TYPE
    elem_type_found = None
    nloc_found = None

    # Abaqus 常见六面体单元的节点数
    _ETYPE_NLOC = {
        "C3D8": 8, "C3D8R": 8,
        "C3D20": 20, "C3D20R": 20,
    }

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    hex_types_u = tuple(t.upper() for t in hex_types)

    while i < len(lines):
        raw = lines[i].strip()
        u = raw.upper()

        # *NODE
        if u.startswith("*NODE"):
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                parts = [p.strip() for p in lines[i].split(",")]
                parts = [p for p in parts if p != ""]
                if len(parts) >= 4:
                    nid = int(parts[0])
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    nodes[nid] = (x, y, z)
                    node_ids_order.append(nid)
                i += 1
            continue

        # *ELEMENT
        if u.startswith("*ELEMENT"):
            m = re.search(r"TYPE\s*=\s*([A-Za-z0-9]+)", raw, flags=re.IGNORECASE)
            etype = m.group(1).upper() if m else ""
            is_hex = etype in hex_types_u
            nloc = _ETYPE_NLOC.get(etype, None) if is_hex else None

            if is_hex:
                if elem_type_found is None:
                    elem_type_found = etype
                    nloc_found = int(nloc)
                    _LAST_ELEM_TYPE = etype
                else:
                    if etype != elem_type_found:
                        raise RuntimeError(
                            f"Mixed hex element types detected: first={elem_type_found}, now={etype}. "
                            "Please export/mesh with a single hex type (e.g., all C3D20R)."
                        )

            i += 1
            pending_eid = None
            pending_nodes = []
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                line = lines[i].rstrip("\n")
                # 保留空字段（用于识别续行）；但这里用状态机更可靠
                parts = [p.strip() for p in line.split(",")]
                nums = [p for p in parts if p != ""]
                if not nums:
                    i += 1
                    continue

                # 六面体元素可以换行：首行包含 eid，续行不包含 eid
                if pending_eid is None:
                    pending_eid = int(nums[0])         # 元素 ID（不使用，仅用于解析）
                    pending_nodes = [int(x) for x in nums[1:]]
                else:
                    pending_nodes.extend([int(x) for x in nums])

                if is_hex and len(pending_nodes) >= nloc_found:
                    hex_elems.append(pending_nodes[:nloc_found])
                    hex_eids.append(int(pending_eid))
                    pending_eid = None
                    pending_nodes = []

                i += 1

            if pending_eid is not None and is_hex:
                print(f"[warn] ELEMENT block ended with incomplete element EID={pending_eid}, collected {len(pending_nodes)} nodes.")
            continue

        # *NSET
        if u.startswith("*NSET"):
            m = re.search(r"NSET\s*=\s*([^, \n\r]+)", raw, flags=re.IGNORECASE)
            setname = m.group(1) if m else None
            gen_flag = "GENERATE" in u

            i += 1
            cur = []
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                row = lines[i].strip()
                if row == "":
                    i += 1
                    continue
                if gen_flag:
                    nums = [p.strip() for p in row.split(",") if p.strip() != ""]
                    if len(nums) >= 3:
                        a, b, step = map(int, nums[:3])
                        cur.extend(range(a, b + 1, step))
                else:
                    nums = [p.strip() for p in row.split(",") if p.strip() != ""]
                    for p in nums:
                        if re.fullmatch(r"[-+]?\d+", p):
                            cur.append(int(p))
                i += 1
            if setname and cur:
                node_sets[setname.upper()].update(cur)
            continue

        i += 1

    if not node_ids_order:
        node_ids_order = sorted(nodes.keys())

    # id -> 0-based 索引
    id2idx = {nid: k for k, nid in enumerate(node_ids_order)}
    V = np.array([nodes[nid] for nid in node_ids_order], dtype=float)

    if not hex_elems:
        raise RuntimeError(
            f"No hex elements found. Expected types in {hex_types_u}. "
            "Check your .inp ELEMENT blocks."
        )

    # 六面体单元连通性：Abaqus node id -> 0-based idx
    T = np.array([[id2idx[n] for n in conn] for conn in hex_elems], dtype=int)
    # --- 与 T 行一一对应的 Abaqus 元素 ID（用于和 ODB/EL PRINT 对齐） ---
    global _LAST_ELEM_IDS
    _LAST_ELEM_IDS = np.asarray(hex_eids, dtype=np.int64) if len(hex_eids) == T.shape[0] else None
    if T.ndim != 2 or T.shape[1] not in (8, 20):
        raise RuntimeError(f"Parsed hex connectivity has shape {T.shape}; expected (m,8) or (m,20).")

    node_sets_0 = {
        name: np.array([id2idx[n] for n in sorted(s) if n in id2idx], dtype=int)
        for name, s in node_sets.items()
    }
    return V, T, node_sets_0, node_ids_order


def clean_and_compact_solid(V, T, nsets, verbose=True):
    """
    删除未被任何单元使用的节点，并重映射单元连通性和 NSET（0-based）
    """
    used = np.zeros(V.shape[0], dtype=bool)
    used[np.unique(T.flatten())] = True
    n_dropped = int((~used).sum())

    old2new = -np.ones(V.shape[0], dtype=int)
    old2new[used] = np.arange(int(used.sum()))

    V2 = V[used]
    T2 = old2new[T]

    nsets2 = {}
    for name, idxs in nsets.items():
        idxs = np.asarray(idxs, dtype=int)
        idxs_used = idxs[used[idxs]]
        nsets2[name] = old2new[idxs_used]

    if verbose:
        print(f"[clean] removed {n_dropped} unused nodes; final: {V2.shape[0]} nodes, {T2.shape[0]} hex elements.")
    return V2, T2, nsets2, used


# ------------------------- 1) 六面体局部形函数/积分 -------------------------

# C3D8(Abaqus) 常用节点顺序：
# 1(-1,-1,-1),2( 1,-1,-1),3( 1, 1,-1),4(-1, 1,-1),
# 5(-1,-1, 1),6( 1,-1, 1),7( 1, 1, 1),8(-1, 1, 1)
_HEX_SIGNS = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=float)


def _hex8_shape_and_derivs(xi, eta, zeta):
    """C3D8/C3D8R：参考单元 [-1,1]^3 上 trilinear 形函数与一阶导。"""
    s = _HEX_SIGNS
    N = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dN_dxi   = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dN_deta  = 0.125 * (1 + s[:, 0] * xi) * s[:, 1] * (1 + s[:, 2] * zeta)
    dN_dzeta = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) * s[:, 2]
    dN = np.stack([dN_dxi, dN_deta, dN_dzeta], axis=1)
    return N, dN


def _gauss_1d(order: int):
    """1D Gauss-Legendre 点与权重（order=1/2/3/4）。"""
    order = int(order)
    if order == 1:
        return np.array([0.0]), np.array([2.0])
    if order == 2:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    if order == 3:
        a = np.sqrt(3.0 / 5.0)
        return np.array([-a, 0.0, a]), np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    if order == 4:
        # 标准 4 点
        a1 = np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0)
        a2 = np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-a2, -a1, a1, a2]), np.array([w2, w1, w1, w2])
    raise ValueError(f"Unsupported Gauss order={order}. Use 1/2/3/4.")


def _auto_gauss_order(elem_type: str, nloc: int) -> int:
    """根据单元类型/阶次选择每个方向的 Gauss 阶次。

    - 8 节点（线性）默认 2x2x2（与旧脚本一致）。
    - 20 节点（二次）
        * 若 type 以 'R' 结尾（C3D20R）：默认 2x2x2（reduced integration 名义一致）
        * 否则（C3D20）：默认 3x3x3

    注：这里装配的是“纯 Laplacian”的 K/M，没有 Abaqus 的 hourglass 稳定项。
        如果 C3D20R 出现大量伪模态/花斑，建议把 gauss_order 强制到 3。
    """
    nloc = int(nloc)
    et = (elem_type or "").upper()
    if nloc == 8:
        return 2
    if nloc == 20:
        return 2 if et.endswith("R") else 3
    raise ValueError(f"Unsupported nloc={nloc}. Only 8 or 20 are supported.")




# ------------------------- Gauss 点（按 Abaqus 常用编号顺序） -------------------------

def hex_gauss_points_abaqus(order: int):
    """返回 Hex 参考单元 [-1,1]^3 上的 Gauss 点与 3D 权重。

    目标：对齐 Abaqus 的常见编号习惯：先编号靠近 1-2-3-4 面（zeta<0）的一层，再编号另一层；
    同一层内按 (xi,eta) 走一圈：(-,-)->(+,-)->(+,+)->(-,+)。

    - order=2: 2x2x2 (8 points) —— C3D20R reduced integration 常用。
    - order=3/4: 给出“分层 + 行优先”的稳定顺序（不保证与 Abaqus 手册图完全一致，但可用坐标匹配重排）。
    """
    gps1, ws1 = _gauss_1d(int(order))
    gps1 = np.asarray(gps1, float)
    ws1  = np.asarray(ws1, float)

    # 3D weight = wx*wy*wz
    pts = []
    w3 = []

    if int(order) == 2:
        # _gauss_1d(2) 通常返回 [-a, +a]，这里强制取正的 a，避免点顺序被符号翻转
        a = float(abs(gps1[0]))

        # layer zeta=-a then +a; within layer: (-,-)->(+,-)->(+,+)->(-,+)
        layer = [(-a, -a), ( a, -a), ( a,  a), (-a,  a)]

        for zeta, wz in [(-a, ws1[0]), (a, ws1[1])]:
            for (xi, eta) in layer:
                wx = ws1[0] if xi < 0 else ws1[1]
                wy = ws1[0] if eta < 0 else ws1[1]
                pts.append((xi, eta, zeta))
                w3.append(float(wx * wy * wz))

        return np.asarray(pts, float), np.asarray(w3, float)    

    # order>=3：分层（zeta 从负到正），每层内 eta 行优先、xi 列优先
    for zeta, wz in zip(gps1, ws1):
        for eta, wy in zip(gps1, ws1):
            for xi, wx in zip(gps1, ws1):
                pts.append((float(xi), float(eta), float(zeta)))
                w3.append(float(wx * wy * wz))
    return np.asarray(pts, float), np.asarray(w3, float)


# C3D20/C3D20R 的 20 个节点在自然坐标中的位置（与 hex20_shape_and_derivs 的节点顺序一致）
_NODE20_NAT = np.array([
    # 1..8 corners
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    # 9..12 edges on z=-1
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],
    # 13..16 edges on z=+1
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],
    # 17..20 vertical edges (z=0)
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0],
], dtype=float)


def build_gp_topology_and_phi_pack(
    V, T, node_ids,
    phi_hybrid,
    phi_specific=None,
    *,
    lambda_hybrid=None,
    lambda_specific=None,
    elem_type=None,
    gauss_order: int = 0,
    verbose=False,
    cond_max: float = 1e10,
    rcond_lstsq: float = 1e-12,
    trace_anchor_hybrid: str = "lambda",
    trace_anchor_specific: str = "lambda",
):
    """为 Hex20 网格导出 Gauss 点数据包（C3D20R 常用 2x2x2=8 点）：

    输出（写入 npz）：
      - mesh/topology: V,T,node_ids,elem_ids,elem_node_ids,elem_corner_node_ids
      - gp geometry: gp_xyz,gp_detJ,gp_dV,elem_vol
      - hybrid modes on GP: phi_hybrid_gp, grad_phi_hybrid_gp, hessdiag_phi_hybrid_gp
      - specific modes on GP: phi_specific_gp, grad_phi_specific_gp, hessdiag_phi_specific_gp
      - extrap/weights: gp_to_corner8_extrap, elem_corner_volw

    数值稳定性策略（仅需二阶对角项）：
      1) grad：等参映射精确计算  dphi/dx = sum_i phi_i * dNi/dx
      2) dxx,dyy,dzz：单元内用 8 个 GP (x,y,z,phi) 拟合“可分离二次多项式"
         phi ≈ c0 + c1 x + c2 y + c3 z + c4 x^2 + c5 y^2 + c6 z^2
         得 dxx=2c4, dyy=2c5, dzz=2c6。
         若提供 lambda，则用特征关系对 trace 做锚定修正：
             dxx+dyy+dzz  →  -lambda * mean(phi_gp)
         条件数过大时回退：仅用 trace/3 的 isotropic 分配。

    约定：
      - 仅实现 go=2（8 GP）导出。
      - 积分点顺序：先 zeta<0 层(1~4)，再 zeta>0 层(5~8)；层内 (-,-)->(+,-)->(+,+)->(-,+)

    返回 dict 适合 np.savez_compressed。
    """
    et = (elem_type or _LAST_ELEM_TYPE or '').upper()
    nloc = int(T.shape[1])
    go = _auto_gauss_order(et, nloc) if int(gauss_order) == 0 else int(gauss_order)
    if nloc != 20:
        raise RuntimeError(f"GP export only implemented for Hex20 (nloc=20). Got nloc={nloc}.")
    if go != 2:
        raise RuntimeError(f"GP derivative export currently targets C3D20R (gauss_order=2, 8 pts). Got go={go}.")

    gp_nat, gp_w3 = hex_gauss_points_abaqus(go)   # (8,3), (8,)
    ngp = int(gp_nat.shape[0])

    # 预计算：Hex20 形函数在 gp 处的值与一阶导（参考坐标）
    N20_gp = np.zeros((ngp, 20), dtype=float)
    dN20_gp = np.zeros((ngp, 20, 3), dtype=float)
    for g in range(ngp):
        xi, eta, zeta = gp_nat[g]
        N, dN = hex20_shape_and_derivs(xi, eta, zeta)
        N20_gp[g] = N
        dN20_gp[g] = dN

    # 预计算：Hex8 形函数在积分点处（用于“积分点 -> 8角点”外推）
    N8_gp = np.zeros((8, 8), dtype=float)
    for g in range(8):
        xi, eta, zeta = gp_nat[g]
        N8, _ = _hex8_shape_and_derivs(xi, eta, zeta)
        N8_gp[g] = N8
    gp_to_corner8 = np.linalg.inv(N8_gp).astype(np.float32)  # (8,8)

    # -------- 分配输出数组 --------
    nE = int(T.shape[0])
    Kh = int(phi_hybrid.shape[1])
    Ks = int(phi_specific.shape[1]) if phi_specific is not None else 0

    gp_xyz  = np.zeros((nE, ngp, 3), dtype=np.float32)
    gp_detJ = np.zeros((nE, ngp), dtype=np.float32)
    gp_dV   = np.zeros((nE, ngp), dtype=np.float32)

    # shape function gradients at GP in physical coordinates (mm): dN/dx
    dN_dx_gp = np.zeros((nE, ngp, 20, 3), dtype=np.float32)
    phi_h_gp = np.zeros((nE, ngp, Kh), dtype=np.float32)
    phi_s_gp = np.zeros((nE, ngp, Ks), dtype=np.float32) if phi_specific is not None else np.zeros((0,0,0), dtype=np.float32)

    # hybrid/specific 一阶导：[...,k,:] = [dx,dy,dz]
    grad_h_gp = np.zeros((nE, ngp, Kh, 3), dtype=np.float32)
    grad_s_gp = np.zeros((nE, ngp, Ks, 3), dtype=np.float32) if phi_specific is not None else np.zeros((0,0,0,0), dtype=np.float32)

    # hybrid/specific 二阶导对角项：[...,k,:] = [dxx,dyy,dzz]
    hessdiag_h_gp = np.zeros((nE, ngp, Kh, 3), dtype=np.float32)
    hessdiag_s_gp = np.zeros((nE, ngp, Ks, 3), dtype=np.float32) if phi_specific is not None else np.zeros((0,0,0,0), dtype=np.float32)

    fit_cond = np.zeros((nE,), dtype=np.float32)

    # 角点体积权重（用于跨单元角点值体积加权平均）
    elem_corner_volw = np.zeros((nE, 8), dtype=np.float32)
    elem_vol = np.zeros((nE,), dtype=np.float32)

    def _norm_anchor_mode(name: str, mode: str) -> str:
        m = str(mode).strip().lower()
        if m not in ("lambda", "none"):
            raise ValueError(f"trace_anchor_{name} must be 'lambda' or 'none', got '{mode}'")
        return m

    anchor_h = _norm_anchor_mode("hybrid", trace_anchor_hybrid)
    anchor_s = _norm_anchor_mode("specific", trace_anchor_specific)

    # lambda：用于 trace 锚定（可关闭；对 hybrid 建议关闭做诊断）
    if (anchor_h == "lambda") and (lambda_hybrid is not None):
        lam_h = np.asarray(lambda_hybrid, dtype=float).reshape(-1)
        if lam_h.shape[0] < Kh:
            raise RuntimeError(f"lambda_hybrid length {lam_h.shape[0]} < Kh {Kh}")
        lam_h = lam_h[:Kh]
    else:
        lam_h = None

    if (anchor_s == "lambda") and (phi_specific is not None) and (lambda_specific is not None):
        lam_s = np.asarray(lambda_specific, dtype=float).reshape(-1)
        if lam_s.shape[0] < Ks:
            raise RuntimeError(f"lambda_specific length {lam_s.shape[0]} < Ks {Ks}")
        lam_s = lam_s[:Ks]
    else:
        lam_s = None

    # --- paper-style 2nd derivatives on GP (via shape-function gradients applied to nodal first-derivs) ---
    # NOTE: This avoids separable polynomial fitting and does NOT require switching to 27 integration points.
    if verbose:
        print(f"[gp-export] computing nodal gradients for GP Hessian (etype={et}, go={go}) ...")
    grads_h_node = _mode_gradients(V, T, phi_hybrid, elem_type=et, gauss_order=go)  # (nNode,Kh,3)
    grads_s_node = _mode_gradients(V, T, phi_specific, elem_type=et, gauss_order=go) if phi_specific is not None else None

    for e in range(nE):
        conn = T[e]
        Xe = V[conn]  # (20,3)

        Phi_h_e = phi_hybrid[conn]  # (20,Kh)
        Phi_h_gp_e = (N20_gp @ Phi_h_e).astype(np.float32)  # (8,Kh)
        phi_h_gp[e] = Phi_h_gp_e

        if phi_specific is not None:
            Phi_s_e = phi_specific[conn]  # (20,Ks)
            Phi_s_gp_e = (N20_gp @ Phi_s_e).astype(np.float32)  # (8,Ks)
            phi_s_gp[e] = Phi_s_gp_e
        else:
            Phi_s_e = None
            Phi_s_gp_e = None
                # element nodal gradients (for 2nd derivatives)
        G_h_e = grads_h_node[conn]  # (20,Kh,3)
        G_s_e = grads_s_node[conn] if grads_s_node is not None else None

        # --- gp loop: geometry + grad ---
        for g in range(ngp):
            dN_ref = dN20_gp[g]  # (20,3) wrt (xi,eta,zeta)
            J = Xe.T @ dN_ref    # (3,3)  # dx/d(xi,eta,zeta)
            detJ = float(np.linalg.det(J))
            if detJ <= 0:
                raise RuntimeError(f"Invalid detJ={detJ:.3e} at elem {e} gp {g+1}. Check connectivity/order.")

            # dN/dx = solve(J^T, dN_ref^T)^T  (= dN_ref @ inv(J))
            dN_dx = np.linalg.solve(J.T, dN_ref.T).T

            dN_dx_gp[e, g] = dN_dx.astype(np.float32)
            xgp = N20_gp[g] @ Xe
            dV = detJ * float(gp_w3[g])

            gp_xyz[e, g] = xgp.astype(np.float32)
            gp_detJ[e, g] = np.float32(detJ)
            gp_dV[e, g] = np.float32(dV)

            # grad hybrid: (Kh,3)
            grad_h = (dN_dx.T @ Phi_h_e).T
            grad_h_gp[e, g] = grad_h.astype(np.float32)

            if Phi_s_e is not None:
                grad_s = (dN_dx.T @ Phi_s_e).T
                grad_s_gp[e, g] = grad_s.astype(np.float32)

            # --- diag Hessian on this GP (paper-style):
            # dxx = d/dx(dphi/dx) = Σ_i (dphi/dx)_i * dNi/dx
            # dyy = d/dy(dphi/dy) = Σ_i (dphi/dy)_i * dNi/dy
            # dzz = d/dz(dphi/dz) = Σ_i (dphi/dz)_i * dNi/dz
            dxx_h = (dN_dx[:, 0] @ G_h_e[:, :, 0])
            dyy_h = (dN_dx[:, 1] @ G_h_e[:, :, 1])
            dzz_h = (dN_dx[:, 2] @ G_h_e[:, :, 2])
            hessdiag_h_gp[e, g, :, 0] = dxx_h.astype(np.float32)
            hessdiag_h_gp[e, g, :, 1] = dyy_h.astype(np.float32)
            hessdiag_h_gp[e, g, :, 2] = dzz_h.astype(np.float32)

            if G_s_e is not None:
                dxx_s = (dN_dx[:, 0] @ G_s_e[:, :, 0])
                dyy_s = (dN_dx[:, 1] @ G_s_e[:, :, 1])
                dzz_s = (dN_dx[:, 2] @ G_s_e[:, :, 2])
                hessdiag_s_gp[e, g, :, 0] = dxx_s.astype(np.float32)
                hessdiag_s_gp[e, g, :, 1] = dyy_s.astype(np.float32)
                hessdiag_s_gp[e, g, :, 2] = dzz_s.astype(np.float32)

            elem_corner_volw[e] += (N8_gp[g] * dV).astype(np.float32)
            elem_vol[e] += np.float32(dV)

        # --- optional: trace anchor for Laplacian (kept for backward compatibility) ---
        # 注意：LBO 特征关系 Δφ = -λφ 在离散 FE 上是“弱形式”成立；
        # 这里沿用旧逻辑：对每个单元/每个模态，仅在“平均意义”下锚定 trace。
        fit_cond[e] = np.float32(0.0)

        if lam_h is not None:
            trace_fit = (
                hessdiag_h_gp[e, :, :, 0].astype(float) +
                hessdiag_h_gp[e, :, :, 1].astype(float) +
                hessdiag_h_gp[e, :, :, 2].astype(float)
            ).mean(axis=0)
            phi_mean = Phi_h_gp_e.astype(float).mean(axis=0)
            trace_target = -lam_h * phi_mean
            delta = ((trace_target - trace_fit) / 3.0).astype(np.float32)
            hessdiag_h_gp[e, :, :, 0] += delta[None, :]
            hessdiag_h_gp[e, :, :, 1] += delta[None, :]
            hessdiag_h_gp[e, :, :, 2] += delta[None, :]

        if (Phi_s_gp_e is not None) and (lam_s is not None):
            trace_fit_s = (
                hessdiag_s_gp[e, :, :, 0].astype(float) +
                hessdiag_s_gp[e, :, :, 1].astype(float) +
                hessdiag_s_gp[e, :, :, 2].astype(float)
            ).mean(axis=0)
            phi_mean_s = Phi_s_gp_e.astype(float).mean(axis=0)
            trace_target_s = -lam_s * phi_mean_s
            delta_s = ((trace_target_s - trace_fit_s) / 3.0).astype(np.float32)
            hessdiag_s_gp[e, :, :, 0] += delta_s[None, :]
            hessdiag_s_gp[e, :, :, 1] += delta_s[None, :]
            hessdiag_s_gp[e, :, :, 2] += delta_s[None, :]

    elem_node_ids = node_ids[T].astype(np.int64)   # (nE,20)
    elem_corner_node_ids = elem_node_ids[:, :8].copy()  # (nE,8)

    elem_ids = (_LAST_ELEM_IDS if _LAST_ELEM_IDS is not None else (np.arange(nE, dtype=np.int64) + 1))

    out = {
        # mesh/topology
        'V': V.astype(np.float32),
        'T': T.astype(np.int32),
        'node_ids': np.asarray(node_ids, dtype=np.int64),
        'elem_ids': np.asarray(elem_ids, dtype=np.int64),
        'elem_node_ids': elem_node_ids,
        'elem_corner_node_ids': elem_corner_node_ids,
        'corner_local_idx_in_hex20': np.arange(8, dtype=np.int16),

        # gp ordering
        'go_used': np.int32(go),
        'gp_ip': (np.arange(1, ngp+1, dtype=np.int16)),
        'gp_nat': gp_nat.astype(np.float32),
        'gp_w3': gp_w3.astype(np.float32),

        # per-element gp geometry
        'gp_xyz': gp_xyz,
        'gp_detJ': gp_detJ,
        'gp_dV': gp_dV,
        # dN/dx at GP (elem,gp,node,xyz)
        'dN_dx_gp': dN_dx_gp,
        'dN_dx_order': np.array(['dN/dx','dN/dy','dN/dz']),
        'elem_vol': elem_vol,

        # element weights for corner averaging
        'elem_corner_volw': elem_corner_volw,

        # interpolation/extrapolation operators
        'node_to_gp_N20': N20_gp.astype(np.float32),
        'gp_to_corner8_extrap': gp_to_corner8,

        # gp modal values
        'phi_hybrid_gp': phi_h_gp,
        'phi_specific_gp': phi_s_gp,

        # gp derivatives (hybrid + specific)
        'grad_phi_hybrid_gp': grad_h_gp,
        'hessdiag_phi_hybrid_gp': hessdiag_h_gp,
        'grad_phi_specific_gp': grad_s_gp,
        'hessdiag_phi_specific_gp': hessdiag_s_gp,
        'hessdiag_order': np.array(['dxx','dyy','dzz']),
        'fit_cond_separable_quad': fit_cond,
        'trace_anchor_hybrid': np.array([anchor_h], dtype='U16'),
        'trace_anchor_specific': np.array([anchor_s], dtype='U16'),
    }

    if verbose:
        print(f"[gp-export] nElem={nE}, ngp={ngp}, Kh={Kh}, Ks={Ks}, go={go} | grad_h={grad_h_gp.shape} | grad_s={(grad_s_gp.shape if phi_specific is not None else None)}")
        print(f"[gp-export] hessdiag_h={hessdiag_h_gp.shape} | hessdiag_s={(hessdiag_s_gp.shape if phi_specific is not None else None)}")
        print("[gp-export] mapping: element index e -> Abaqus elem_ids[e]; gp index g -> gp_ip[g].")

    return out

def hex20_shape_and_derivs(xi, eta, zeta):
    """C3D20/C3D20R：20 节点 serendipity Hex 形函数与一阶导（自然坐标）。

    Abaqus C3D20 节点顺序（自然坐标）：
      1..8 角点：(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)
      9..16 边中点（z=-1 与 z=+1 两层）
      17..20 竖向边中点（z=0，xi/eta 取 ±1）
    """
    sx = _HEX_SIGNS[:, 0]
    sy = _HEX_SIGNS[:, 1]
    sz = _HEX_SIGNS[:, 2]

    # --- 8 corners ---
    A = (1.0 + sx * xi)
    B = (1.0 + sy * eta)
    C = (1.0 + sz * zeta)
    D = (sx * xi + sy * eta + sz * zeta - 2.0)
    Nc = 0.125 * A * B * C * D

    dNc_dxi   = (sx / 8.0) * B * C * (D + A)
    dNc_deta  = (sy / 8.0) * A * C * (D + B)
    dNc_dzeta = (sz / 8.0) * A * B * (D + C)
    dNc = np.stack([dNc_dxi, dNc_deta, dNc_dzeta], axis=1)  # (8,3)

    # --- 12 mid-edge nodes ---
    N = np.zeros(20, dtype=float)
    dN = np.zeros((20, 3), dtype=float)
    N[:8] = Nc
    dN[:8, :] = dNc

    # 9: (0,-1,-1)  xi-edge, (sy,sz)=(-1,-1)
    # 10:(1,0,-1)   eta-edge, (sx,sz)=(+1,-1)
    # 11:(0,1,-1)   xi-edge, (sy,sz)=(+1,-1)
    # 12:(-1,0,-1)  eta-edge, (sx,sz)=(-1,-1)
    # 13:(0,-1,1)   xi-edge, (sy,sz)=(-1,+1)
    # 14:(1,0,1)    eta-edge, (sx,sz)=(+1,+1)
    # 15:(0,1,1)    xi-edge, (sy,sz)=(+1,+1)
    # 16:(-1,0,1)   eta-edge, (sx,sz)=(-1,+1)
    # 17:(-1,-1,0)  zeta-edge,(sx,sy)=(-1,-1)
    # 18:(1,-1,0)   zeta-edge,(sx,sy)=(+1,-1)
    # 19:(1,1,0)    zeta-edge,(sx,sy)=(+1,+1)
    # 20:(-1,1,0)   zeta-edge,(sx,sy)=(-1,+1)

    # xi-edges (nodes 9,11,13,15)
    for idx, (sy_e, sz_e) in zip([8, 10, 12, 14], [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]):
        fac_y = (1.0 + sy_e * eta)
        fac_z = (1.0 + sz_e * zeta)
        N[idx] = 0.25 * (1.0 - xi * xi) * fac_y * fac_z
        dN[idx, 0] = -0.5 * xi * fac_y * fac_z
        dN[idx, 1] = 0.25 * (1.0 - xi * xi) * sy_e * fac_z
        dN[idx, 2] = 0.25 * (1.0 - xi * xi) * fac_y * sz_e

    # eta-edges (nodes 10,12,14,16)
    for idx, (sx_e, sz_e) in zip([9, 11, 13, 15], [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]):
        fac_x = (1.0 + sx_e * xi)
        fac_z = (1.0 + sz_e * zeta)
        N[idx] = 0.25 * (1.0 - eta * eta) * fac_x * fac_z
        dN[idx, 0] = 0.25 * (1.0 - eta * eta) * sx_e * fac_z
        dN[idx, 1] = -0.5 * eta * fac_x * fac_z
        dN[idx, 2] = 0.25 * (1.0 - eta * eta) * fac_x * sz_e

    # zeta-edges (nodes 17..20)
    for idx, (sx_e, sy_e) in zip([16, 17, 18, 19], [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]):
        fac_x = (1.0 + sx_e * xi)
        fac_y = (1.0 + sy_e * eta)
        N[idx] = 0.25 * (1.0 - zeta * zeta) * fac_x * fac_y
        dN[idx, 0] = 0.25 * (1.0 - zeta * zeta) * sx_e * fac_y
        dN[idx, 1] = 0.25 * (1.0 - zeta * zeta) * fac_x * sy_e
        dN[idx, 2] = -0.5 * zeta * fac_x * fac_y

    return N, dN


def hex_shape_and_derivs(xi, eta, zeta, nnode: int = 8):
    """按 nnode(8/20) 选择 Hex 形函数。"""
    nnode = int(nnode)
    if nnode == 8:
        return _hex8_shape_and_derivs(xi, eta, zeta)
    if nnode == 20:
        return hex20_shape_and_derivs(xi, eta, zeta)
    raise ValueError(f"Unsupported nnode={nnode}. Only 8 or 20 are supported.")


def hex_local_KM(Xe, det_tol=1e-14, elem_type=None, gauss_order: int = 0):
    """计算单个六面体单元的局部刚度/质量矩阵（标量 Laplacian）。

    Ke_ij = ∫ (∇N_i · ∇N_j) dΩ
    Me_ij = ∫ (N_i N_j) dΩ

    Xe: (nloc,3)
    - nloc=8:  C3D8/C3D8R
    - nloc=20: C3D20/C3D20R
    """
    nloc = int(Xe.shape[0])
    go = _auto_gauss_order(elem_type, nloc) if int(gauss_order) == 0 else int(gauss_order)
    gps, ws = _gauss_1d(go)

    Ke = np.zeros((nloc, nloc), dtype=float)
    Me = np.zeros((nloc, nloc), dtype=float)

    for xi, wx in zip(gps, ws):
        for eta, wy in zip(gps, ws):
            for zeta, wz in zip(gps, ws):
                N, dN_ref = hex_shape_and_derivs(xi, eta, zeta, nnode=nloc)

                # Jacobian: J = dx/d(xi,eta,zeta) = Xe^T @ dN_ref
                J = Xe.T @ dN_ref  # (3,3)
                detJ = float(np.linalg.det(J))
                if abs(detJ) < det_tol:
                    raise RuntimeError(f"Degenerate hex element with |detJ|={abs(detJ):.3e}")
                invJ = np.linalg.inv(J)

                # 物理坐标梯度：∇_x N = dN_ref @ invJ
                gradN = dN_ref @ invJ  # (nloc,3)

                dV = abs(detJ) * float(wx * wy * wz)
                Ke += (gradN @ gradN.T) * dV
                Me += np.outer(N, N) * dV

    return Ke, Me


def assemble_KM_hex(V, T, elem_type=None, gauss_order: int = 0):
    """装配全局 K, M（稀疏矩阵 CSR）。

    支持 Hex8/Hex20：T.shape[1] 自动决定 nloc。
    gauss_order=0 表示按 elem_type 自动选择（8 节点 2；20 节点 C3D20R->2, C3D20->3）。
    """
    n_nodes = V.shape[0]
    n_elem = T.shape[0]
    nloc = int(T.shape[1])
    nn = nloc * nloc

    nnz_est = n_elem * nn
    I = np.empty(nnz_est, dtype=np.int32)
    J = np.empty(nnz_est, dtype=np.int32)
    VK = np.empty(nnz_est, dtype=np.float64)
    VM = np.empty(nnz_est, dtype=np.float64)

    p = 0
    for e in range(n_elem):
        conn = T[e]
        Xe = V[conn]  # (nloc,3)

        Ke, Me = hex_local_KM(Xe, elem_type=elem_type, gauss_order=gauss_order)
        ii = np.repeat(conn, nloc)
        jj = np.tile(conn, nloc)

        I[p:p+nn] = ii
        J[p:p+nn] = jj
        VK[p:p+nn] = Ke.reshape(-1)
        VM[p:p+nn] = Me.reshape(-1)
        p += nn

    K = coo_matrix((VK, (I, J)), shape=(n_nodes, n_nodes)).tocsr()
    M = coo_matrix((VM, (I, J)), shape=(n_nodes, n_nodes)).tocsr()
    return K, M


def apply_dirichlet(K, M, dirichlet_nodes):
    """
    删去 Dirichlet 节点对应自由度（与旧脚本一致）
    """
    mask = np.ones(K.shape[0], dtype=bool)
    if len(dirichlet_nodes):
        mask[np.array(sorted(list(dirichlet_nodes)), dtype=int)] = False
    Kb = K[mask][:, mask].tocsr()
    Mb = M[mask][:, mask].tocsr()
    return Kb, Mb, mask


# ------------------------- 2) 特征值求解 -------------------------

def constant_mode(M):
    ones = np.ones(M.shape[0])
    nrm = np.sqrt(ones @ (M @ ones))
    return ones / nrm


def lobpcg_smallest(K, Mmass, k, Y=None, maxiter=500, tol=1e-9, seed=0):
    """
    用 LOBPCG 求最小 k 个特征对:  K x = λ M x
    Y 用于 deflate（例如常数零模态）
    """
    n = K.shape[0]
    k = max(1, min(int(k), n - 1))

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, k))

    dK = K.diagonal().astype(float)

    # 为了避免纯 Neumann 情况 K 奇异导致 ILU 失败，给 K 加一个很小的对角 shift
    pos = dK[dK > 0]
    shift = 1e-8 * float(np.median(pos)) if pos.size else 1e-8
    Ailu = (K + diags([shift], [0], shape=K.shape)).tocsc()

    try:
        ilu = spilu(Ailu, drop_tol=1e-4, fill_factor=10)

        def _prec_matvec(x):
            return ilu.solve(x)

        def _prec_matmat(X):
                        return np.column_stack([ilu.solve(X[:, i]) for i in range(X.shape[1])])

        Prec = LinearOperator(K.shape, matvec=_prec_matvec, matmat=_prec_matmat, dtype=float)

    except Exception:
        # ILU 失败就退回 Jacobi，保证流程不崩
        d = np.maximum(dK, 1e-12)
        Prec = diags(1.0 / d)

    vals, vecs = lobpcg(A=K, X=X0, B=Mmass, M=Prec, Y=Y,
                        largest=False, tol=tol, maxiter=maxiter, verbosityLevel=0)

    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]

    # M-正交归一
    for j in range(vecs.shape[1]):
        nj = np.sqrt(vecs[:, j].T @ (Mmass @ vecs[:, j]))
        if nj > 0:
            vecs[:, j] /= nj
    return vals, vecs

def m_orthonormalize(Q, Mmass, eps=1e-14):
    """
    对列向量做 M-内积正交归一化（Modified Gram-Schmidt）
    使得 Q^T M Q ≈ I
    """
    if Q is None or Q.size == 0:
        return Q
    Q = np.array(Q, dtype=float, copy=True)
    k = Q.shape[1]
    for j in range(k):
        v = Q[:, j].copy()
        for i in range(j):
            qi = Q[:, i]
            rij = float(qi.T @ (Mmass @ v))
            v -= qi * rij
        nrm = float(np.sqrt(max(v.T @ (Mmass @ v), 0.0)))
        if nrm < eps:
            Q[:, j] = 0.0
        else:
            Q[:, j] = v / nrm
    # 删除全零列（极端情况下避免 lobpcg 报错）
    keep = np.linalg.norm(Q, axis=0) > 0
    return Q[:, keep] if np.any(keep) else None


def lobpcg_smallest_batched(K, Mmass, k_total, Y0=None, batch_size=50,
                            maxiter=500, tol=1e-9, seed=0, verbose=False):
    """
    分批求解最小 k_total 个特征对：Kx=λMx
    - 每批求 batch_size 个
    - 将已求得模态加入约束子空间 Y（deflation），避免重复、提升收敛
    - 最终返回按特征值升序排列的 (vals, vecs)

    说明：
    - Y0 可以是常数零模态（general LBO）或其它先验约束
    """
    n = K.shape[0]
    k_total = int(k_total)
    if k_total <= 0:
        return np.array([], dtype=float), np.zeros((n, 0), dtype=float)

    # 约束子空间
    Y = None
    if Y0 is not None and Y0.size > 0:
        Y = m_orthonormalize(Y0, Mmass)

    vals_all = []
    vecs_all = []

    remaining = k_total
    cur_seed = seed

    while remaining > 0:
        # lobpcg 要求：k_batch < n - dim(Y)
        ydim = 0 if (Y is None) else Y.shape[1]
        max_k = max(1, n - ydim - 1)
        k_batch = min(int(batch_size), remaining, max_k)

        if k_batch <= 0:
            break

        if verbose:
            print(f"[lobpcg-batch] solving batch k={k_batch}, remaining={remaining}, Ydim={ydim}, n={n}")

        vals_b, vecs_b = lobpcg_smallest(
            K, Mmass, k=k_batch, Y=Y, maxiter=maxiter, tol=tol, seed=cur_seed
        )
        cur_seed += 1

        vals_all.append(vals_b)
        vecs_all.append(vecs_b)

        # 更新约束子空间：Y <- [Y, vecs_b]
        if Y is None:
            Y = vecs_b
        else:
            Y = np.hstack([Y, vecs_b])
        Y = m_orthonormalize(Y, Mmass)

        remaining -= k_batch

    if not vals_all:
        return np.array([], dtype=float), np.zeros((n, 0), dtype=float)

    vals = np.concatenate(vals_all)
    vecs = np.hstack(vecs_all)

    # 全局排序（保险）
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # 再做一次 M-归一（兼容旧逻辑）
    for j in range(vecs.shape[1]):
        nj = np.sqrt(vecs[:, j].T @ (Mmass @ vecs[:, j]))
        if nj > 0:
            vecs[:, j] /= nj

    return vals, vecs
# ------------------------- 3) Hybrid LBO 主流程 -------------------------

def compute_hybrid_lbo_from_inp(inp_path,
                                n_modes=50,
                                nset_dirichlet="BDIR",
                                nset_neumann="BNEU",
                                nset_free="BFREE",
                                nset_surface="SET-1",
                                include_const_general=True,
                                gauss_order: int = 0,
                                batch_size=50,
                                verbose=True):
    """
    保持同名函数，便于你替换调用。

    nset_surface: “全部外表面节点集”(Set-1)，用于推导：
        BFREE = Set-1 - BDIR - BNEU
    若找不到 Set-1，则退化为 ALL - D - N（仅用于日志核对，不推荐）。
    """
    V, T, nsets, node_ids = read_abaqus_inp_solid(inp_path)
    V, T, nsets, used_mask = clean_and_compact_solid(V, T, nsets, verbose=verbose)
    # --- 保留与 compact 后 V 对齐的 Abaqus NodeIds ---
    node_ids = np.asarray(node_ids, dtype=np.int64)[used_mask]

    et = (_LAST_ELEM_TYPE or "").upper()
    nloc = int(T.shape[1])
    go_used = _auto_gauss_order(et, nloc) if int(gauss_order) == 0 else int(gauss_order)
    if verbose:
        print(f"[assemble] assembling K,M for 3D solid Laplacian ... (etype={et}, nloc={nloc}, gauss_order={go_used})")
    K, M = assemble_KM_hex(V, T, elem_type=et, gauss_order=go_used)

    D = set(nsets.get(nset_dirichlet.upper(), np.array([], int)).tolist())
    N = set(nsets.get(nset_neumann.upper(),  np.array([], int)).tolist())
    F = set(nsets.get(nset_free.upper(),     np.array([], int)).tolist())

    if len(F) == 0:
        surf = set(nsets.get(nset_surface.upper(), np.array([], int)).tolist())
        if len(surf) == 0:
            if verbose:
                print(f"[warn] surface nset '{nset_surface}' not found; BFREE := ALL - D - N (only for checking).")
            surf = set(range(V.shape[0]))
        F = surf - D - N

    # 1) General：全 Neumann（保留常数模态）
    if include_const_general:
        phi0 = constant_mode(M)[:, None]
        k_rest = max(int(n_modes) - 1, 0)
        if k_rest > 0:
            evals_g_rest, evecs_g_rest = lobpcg_smallest_batched(
                K, M, k_total=k_rest, Y0=phi0, batch_size=batch_size,
                maxiter=2000, tol=1e-9, seed=0, verbose=False    #定义特征值求解函数的收敛精度与最大迭代次数
            )
            evecs_g = np.hstack([phi0, evecs_g_rest])
            evals_g = np.concatenate(([0.0], evals_g_rest))
        else:
            evecs_g = phi0
            evals_g = np.array([0.0])
    else:
        evals_g, evecs_g = lobpcg_smallest(K, M, k=n_modes, Y=None)

    # --- eig quality report: general ---
    rnorm_g, rel_g, mnorm_g, orth_g = eval_eig_quality(K, M, evals_g, evecs_g)
    write_quality_csv("eig_general.csv", "general", evals_g, rnorm_g, rel_g, mnorm_g, orth_g)

    # 2) Specific：Dirichlet on (D ∪ N)，其余自然 Neumann
    D_union_N = np.array(sorted(list(D.union(N))), dtype=int)
    if len(D_union_N) > 0:
        Kb, Mb, mask = apply_dirichlet(K, M, D_union_N)
        if Kb.shape[0] <= 1:
            raise RuntimeError("After Dirichlet reduction, dofs <= 1; check BDIR/BNEU.")
        k_spec = min(int(n_modes), Kb.shape[0] - 1)
        evals_s_bc, evecs_s_bc = lobpcg_smallest_batched(
            Kb, Mb, k_total=k_spec, Y0=None,
            batch_size=batch_size, maxiter=2000, tol=1e-9, seed=0, verbose=False
        )

        evecs_s = np.zeros((K.shape[0], int(n_modes)))
        evecs_s[:, :k_spec][mask, :] = evecs_s_bc
        evals_s = np.zeros(int(n_modes)); evals_s[:k_spec] = evals_s_bc

        for j in range(int(n_modes)):
            nj = np.sqrt(evecs_s[:, j].T @ (M @ evecs_s[:, j]))
            if nj > 0:
                evecs_s[:, j] /= nj
    else:
        evals_s = np.zeros_like(evals_g)
        evecs_s = np.zeros_like(evecs_g)

    # --- eig quality report: specific (reduced system) ---
    if len(D_union_N) > 0:
        rnorm_s, rel_s, mnorm_s, orth_s = eval_eig_quality(Kb, Mb, evals_s_bc, evecs_s_bc)
        write_quality_csv("eig_specific.csv", "specific", evals_s_bc, rnorm_s, rel_s, mnorm_s, orth_s)
    else:
        if verbose:
            print('[info] D∪N empty -> skip specific eig quality report')

    # 3) Hybrid：逐列相加，λ取 general（与旧脚本一致）
    min_cols = min(evecs_g.shape[1], evecs_s.shape[1]) if evecs_s.size else evecs_g.shape[1]
    phi_h = evecs_g[:, :min_cols] + (evecs_s[:, :min_cols] if evecs_s.size else 0.0)
    lam_h = evals_g[:min_cols]

    info_sets = {"BDIR": sorted(list(D)), "BNEU": sorted(list(N)), "BFREE": sorted(list(F))}
    if verbose:
        print(f"[info] mesh: {V.shape[0]} nodes, {T.shape[0]} hex; hybrid modes: {phi_h.shape[1]}")
        print(f"[info] sets_used sizes: BDIR={len(D)}, BNEU={len(N)}, BFREE={len(F)}")

    return {
        "V": V, "T": T,"node_ids": node_ids,
        "phi_general": evecs_g, "lambda_general": evals_g,
        "phi_specific": evecs_s, "lambda_specific": evals_s,
        "phi_hybrid": phi_h, "lambda_hybrid": lam_h,
        "sets_used": info_sets
    }



# ------------------------- 4) 导数：grad / Hessian（保持旧脚本列名） -------------------------

def _element_gradients_hex(V, T, values, elem_type=None, gauss_order: int = 0):
    """
    计算每个单元的“体积平均梯度”：
      grad_e = (1/Vol_e) ∫_Ωe ∇v dΩ

    values: (nNode,k) 标量场（k 个模态）
    返回：
      grad_e: (nElem,k,3)
      vol_e : (nElem,)
    """
    n_elem = T.shape[0]
    nloc = int(T.shape[1])
    k = values.shape[1]

    et = (elem_type or _LAST_ELEM_TYPE or "").upper()
    go = _auto_gauss_order(et, nloc) if int(gauss_order) == 0 else int(gauss_order)
    gps, ws = _gauss_1d(go)

    grad_e = np.zeros((n_elem, k, 3), dtype=float)
    vol_e = np.zeros(n_elem, dtype=float)

    for e in range(n_elem):
        conn = T[e]
        Xe = V[conn]
        ve = values[conn, :]  # (nloc,k)

        sum_grad = np.zeros((k, 3), dtype=float)
        sum_vol = 0.0

        for xi, wx in zip(gps, ws):
            for eta, wy in zip(gps, ws):
                for zeta, wz in zip(gps, ws):
                    _, dN_ref = hex_shape_and_derivs(xi, eta, zeta, nnode=nloc)
                    J = Xe.T @ dN_ref
                    detJ = float(np.linalg.det(J))
                    invJ = np.linalg.inv(J)
                    gradN = dN_ref @ invJ
                    dV = abs(detJ) * float(wx * wy * wz)

                    # ∇v = Σ_i v_i ∇N_i = ve^T @ gradN
                    g = ve.T @ gradN  # (k,3)
                    sum_grad += g * dV
                    sum_vol += dV

        grad_e[e] = sum_grad / max(sum_vol, 1e-18)
        vol_e[e] = sum_vol

    return grad_e, vol_e


def _mode_gradients(V, T, modes, elem_type=None, gauss_order: int = 0):
    """
    节点梯度：单元体积平均梯度 -> 体积加权平均到节点
    返回 grads: (nNode,k,3)
    """
    n_nodes, k = modes.shape
    grad_e, vol_e = _element_gradients_hex(V, T, modes, elem_type=elem_type, gauss_order=gauss_order)

    grads = np.zeros((n_nodes, k, 3), dtype=float)
    wsum = np.zeros(n_nodes, dtype=float)

    for e, conn in enumerate(T):
        w = vol_e[e]
        grads[conn] += grad_e[e][None, :, :] * w
        wsum[conn] += w

    grads /= np.maximum(wsum, 1e-18)[:, None, None]
    return grads


def _mode_hessians(V, T, modes, elem_type=None, gauss_order: int = 0):
    """
    近似 Hessian：沿用旧脚本思路 —— “对一阶梯度再取梯度”
    输出 (nNode,k,3,3)，其中 hess[:,:,i,j] = ∂/∂x_j ( ∂phi/∂x_i )
    """
    grads = _mode_gradients(V, T, modes, elem_type=elem_type, gauss_order=gauss_order)  # (n,k,3)
    n_nodes, k, _ = grads.shape
    hess = np.zeros((n_nodes, k, 3, 3), dtype=float)
    for axis in range(3):
        comp_grad = _mode_gradients(V, T, grads[:, :, axis], elem_type=elem_type, gauss_order=gauss_order)
        hess[:, :, axis, :] = comp_grad
    return hess


def export_csv(path, V, T, modes, node_ids=None, prefix="phi_hybrid_", compute_hessian=True,
               elem_type=None, gauss_order: int = 0, grads=None, hess=None):
    """
    CSV 输出列名/顺序保持与你现有脚本一致（不要改）：
    x,y,z,
    phi_hybrid_1, phi_hybrid_1_dx, phi_hybrid_1_dy, phi_hybrid_1_dz,
    phi_hybrid_1_dxx, ..., phi_hybrid_1_dzz,
    phi_hybrid_2, ...

    注：体网格的 grad/hess 是“单元积分平均 + 体积加权到节点”的近似。
    """
    if grads is None:
        grads = _mode_gradients(V, T, modes, elem_type=elem_type, gauss_order=gauss_order)
    if compute_hessian:
        if hess is None:
            hess = _mode_hessians(V, T, modes, elem_type=elem_type, gauss_order=gauss_order)
    else:
        # 仍输出列，但全置 0（与旧 CLI 语义一致）
        hess = np.zeros((V.shape[0], modes.shape[1], 3, 3), dtype=float)

    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1, 1)
        header = ["NodeIds", "x", "y", "z"]
        data_cols = [node_ids, V]
    else:
        header = ["x", "y", "z"]
        data_cols = [V]

    for i in range(modes.shape[1]):
        base = f"{prefix}{i+1}"
        header.extend([base, f"{base}_dx", f"{base}_dy", f"{base}_dz",
                       f"{base}_dxx", f"{base}_dxy", f"{base}_dxz",
                       f"{base}_dyx", f"{base}_dyy", f"{base}_dyz",
                       f"{base}_dzx", f"{base}_dzy", f"{base}_dzz"])
        mode_col = modes[:, i][:, None]
        grad_col = grads[:, i, :]
        hess_col = hess[:, i, :, :].reshape(V.shape[0], 9)
        data_cols.append(np.hstack([mode_col, grad_col, hess_col]))

    arr = np.column_stack(data_cols)
    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="")
    print(f"[export_csv] Wrote {path}")

def export_mode_values_csv(path, V, modes, node_ids=None, prefix="phi_specific_"):
    """
    仅导出节点模态“数值”，不导出 grad/hess。
    列：NodeIds,x,y,z, phi_specific_1,phi_specific_2,...
    """
    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1, 1)
        header = ["NodeIds", "x", "y", "z"] + [f"{prefix}{i+1}" for i in range(modes.shape[1])]
        arr = np.column_stack([node_ids, V, modes])
        fmts = ["%d"] + ["%.16e"] * (3 + modes.shape[1])
    else:
        header = ["x", "y", "z"] + [f"{prefix}{i+1}" for i in range(modes.shape[1])]
        arr = np.column_stack([V, modes])
        fmts = ["%.16e"] * (3 + modes.shape[1])

    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="", fmt=fmts)
    print(f"[export_mode_values_csv] Wrote {path}")


def _parse_clip(s: str, default=(1e-6, 1e6)):
    try:
        a, b = [float(x.strip()) for x in str(s).split(",")]
        lo, hi = (a, b) if a <= b else (b, a)
        return lo, hi
    except Exception:
        return float(default[0]), float(default[1])

def compute_mode_scale_from_gp(pkg, kind: str, target: float, clip=(1e-6, 1e6), eps=1e-30):
    """
    Compute per-mode alpha_k from GP arrays in pkg:
      kind='grad_rms': use RMS of grad_phi_hybrid_gp[...,k,:]
      kind='phi_rms' : use RMS of phi_hybrid_gp[...,k]
    Fallback: if metric too small, alpha_k=1.
    """
    kind = str(kind).lower()
    lo, hi = float(clip[0]), float(clip[1])

    phi = pkg["phi_hybrid_gp"].astype(np.float64)          # (E,8,K)
    grad = pkg["grad_phi_hybrid_gp"].astype(np.float64)    # (E,8,K,3)
    K = phi.shape[2]

    phi_rms = np.sqrt(np.mean(phi * phi, axis=(0,1)) + eps)              # (K,)
    grad_rms = np.sqrt(np.mean(grad * grad, axis=(0,1,3)) + eps)         # (K,)

    if kind == "grad_rms":
        base = grad_rms
    elif kind == "phi_rms":
        base = phi_rms
    else:
        alpha = np.ones((K,), dtype=np.float64)
        return alpha, 1.0/alpha, phi_rms, grad_rms

    alpha = target / np.maximum(base, eps)
    alpha = np.clip(alpha, lo, hi)

    # if base is essentially zero (e.g., constant mode grad), keep alpha=1
    alpha = np.where(base > 1e-20, alpha, 1.0)

    inv_alpha = 1.0 / np.maximum(alpha, eps)
    return alpha, inv_alpha, phi_rms, grad_rms

def apply_mode_scale_to_res_and_pkg(res, pkg, alpha):
    """
    Apply alpha_k to:
      - res['phi_general'], res['phi_specific'], res['phi_hybrid'] (first Kh columns)
      - pkg GP arrays for hybrid and specific (phi/grad/hessdiag)
    Keep CSV columns unchanged (only values change).
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(1, -1)
    Kh = alpha.shape[1]

    # --- node modes ---
    res["phi_general"][:, :Kh] *= alpha
    res["phi_specific"][:, :Kh] *= alpha
    # recompute hybrid to keep exact relation (scaled)
    res["phi_hybrid"] = res["phi_general"][:, :Kh] + res["phi_specific"][:, :Kh]

    # --- gp arrays (hybrid) ---
    a_phi = alpha.reshape(1, 1, Kh).astype(np.float32)
    a_g3  = alpha.reshape(1, 1, Kh, 1).astype(np.float32)

    pkg["phi_hybrid_gp"] *= a_phi
    pkg["grad_phi_hybrid_gp"] *= a_g3
    pkg["hessdiag_phi_hybrid_gp"] *= a_g3

    # --- gp arrays (specific) if exist ---
    if "phi_specific_gp" in pkg and pkg["phi_specific_gp"].size:
        pkg["phi_specific_gp"] *= a_phi
    if "grad_phi_specific_gp" in pkg and pkg["grad_phi_specific_gp"].size:
        pkg["grad_phi_specific_gp"] *= a_g3
    if "hessdiag_phi_specific_gp" in pkg and pkg["hessdiag_phi_specific_gp"].size:
        pkg["hessdiag_phi_specific_gp"] *= a_g3

def save_mode_scale_npz(path, alpha, inv_alpha, phi_rms, grad_rms, kind, target, clip):
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        path,
        mode_scale_kind=np.array([str(kind)], dtype=object),
        mode_scale_target=np.float32(target),
        mode_scale_clip=np.array([float(clip[0]), float(clip[1])], dtype=np.float32),
        mode_scale_alpha=np.asarray(alpha, dtype=np.float32),
        mode_scale_inv_alpha=np.asarray(inv_alpha, dtype=np.float32),
        mode_scale_phi_rms=np.asarray(phi_rms, dtype=np.float32),
        mode_scale_grad_rms=np.asarray(grad_rms, dtype=np.float32),
    )
    print(f"[mode-scale] wrote {path}")

# ------------------------- 5) CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Hybrid LBO (3D solid) from Abaqus .inp with Hex8/Hex20")
    ap.add_argument("inp", help="Abaqus .inp path (solid mesh with C3D8R/C3D8/C3D20R/C3D20)")

    ap.add_argument("--modes", type=int, default=50, help="Number of hybrid modes")
    ap.add_argument("--bdir", default="BDIR", help="NSET name for Dirichlet boundary nodes")
    ap.add_argument("--bneu", default="BNEU", help="NSET name for Neumann(load) boundary nodes (treated as non-free in specific LBO)")
    ap.add_argument("--bfree", default="BFREE", help="Optional NSET name for free boundary nodes (if absent, computed from surfset - bdir - bneu)")
    ap.add_argument("--surfset", default="SET-1", help="NSET name containing ALL exterior surface nodes (default: Set-1)")

    ap.add_argument("--no_hessian", type=int, default=0, help="1=skip Hessian compute (still output columns filled with 0), 0=compute")
    ap.add_argument("--gauss_order", type=int, default=0, help="Gauss order per direction. 0=auto (Hex8->2; Hex20: C3D20R->2, C3D20->3). If C3D20R has spurious patterns, try 3.")
    ap.add_argument("--export_csv", default=None, help="Export CSV. If endswith .csv, writes HYBRID to that path and also writes *_general.csv and *_specific.csv. If no .csv, treated as prefix and writes *_hybrid.csv, *_general.csv, *_specific.csv.")
    ap.add_argument("--export_specific_node_csv", default=None,
                    help="Export node-level specific LBO values (NO deriv). "
                         "If omitted but --export_csv is set, auto writes <base>_specific_node.csv.")

    ap.add_argument("--export_gp_npz", default=None,
                    help="Export Gauss-point hybrid LBO and topology to a .npz (recommended for stress branch & extrapolation).")
    ap.add_argument("--trace_anchor_hybrid", type=str, default="lambda", choices=["lambda", "none"],
                    help="How to anchor Laplacian trace for HYBRID modes when fitting hessdiag on GP: "
                         "lambda | none. For inverse problems, try 'none' to avoid bias.")
    ap.add_argument("--trace_anchor_specific", type=str, default="lambda", choices=["lambda", "none"],
                    help="How to anchor Laplacian trace for SPECIFIC modes when fitting hessdiag on GP: "
                         "lambda | none.")

    # ---- per-mode scaling (export-side) ----
    ap.add_argument("--mode_scale", type=str, default="none",
                    choices=["none", "grad_rms", "phi_rms"],
                    help="Per-mode scaling applied at export: "
                         "none | grad_rms (target RMS of grad_phi) | phi_rms (target RMS of phi). "
                         "Scaling multiplies phi/grad/hessdiag by same alpha_k.")
    ap.add_argument("--scale_target", type=float, default=1.0,
                    help="Target RMS after scaling (per mode).")
    ap.add_argument("--scale_clip", type=str, default="1e-6,1e6",
                    help='Clip range for alpha_k, format "lo,hi".')
    ap.add_argument("--export_scale", default=None,
                    help="Optional: save mode scaling factors to a standalone npz (alpha, inv_alpha, rms stats).")

    ap.add_argument("--verbose", type=int, default=1, help="1=print info, 0=silent")
    ap.add_argument("--batch", type=int, default=50, help="LOBPCG batch size for eigen solve (e.g., 25/50/80)")

    args = ap.parse_args()

    res = compute_hybrid_lbo_from_inp(
        inp_path=args.inp,
        n_modes=args.modes,
        nset_dirichlet=args.bdir,
        batch_size=args.batch,
        nset_neumann=args.bneu,
        nset_free=args.bfree,
        nset_surface=args.surfset,
        include_const_general=True,
        gauss_order=int(args.gauss_order),
        verbose=bool(args.verbose),
    )

    V, T = res["V"], res["T"]

    # --- optional: export Gauss-point hybrid/specific LBO + topology (for stress branch & extrapolation) ---
    # parse clip
    clip_lo, clip_hi = _parse_clip(args.scale_clip, default=(1e-6, 1e6))

    pkg = None
    if args.export_gp_npz:
        et = (_LAST_ELEM_TYPE or "").upper()
        go_used = _auto_gauss_order(et, int(T.shape[1])) if int(args.gauss_order) == 0 else int(args.gauss_order)
        pkg = build_gp_topology_and_phi_pack(
            V, T, res["node_ids"], res["phi_hybrid"], phi_specific=res["phi_specific"],
            lambda_hybrid=res["lambda_hybrid"],
            lambda_specific=res["lambda_specific"],
            elem_type=et, gauss_order=go_used, verbose=bool(args.verbose),
            trace_anchor_hybrid=args.trace_anchor_hybrid,
            trace_anchor_specific=args.trace_anchor_specific,
        )

        # bookkeeping
        pkg["lambda_hybrid"] = res["lambda_hybrid"].astype(np.float32)
        pkg["lambda_specific"] = res["lambda_specific"].astype(np.float32)


        # boundary node sets (0-based node indices) for training-time masks
        try:
            sets = res.get("sets_used", {})
            pkg["set_BDIR_idx"] = np.asarray(sets.get("BDIR", []), dtype=np.int32)
            pkg["set_BNEU_idx"] = np.asarray(sets.get("BNEU", []), dtype=np.int32)
        except Exception as _e:
            if bool(args.verbose):
                print(f"[gp-export] warning: failed to attach boundary sets: {_e}")
        # --- per-mode scaling (export-side) ---
        if args.mode_scale != "none":
            alpha, inv_alpha, phi_rms, grad_rms = compute_mode_scale_from_gp(
                pkg, kind=args.mode_scale, target=float(args.scale_target),
                clip=(clip_lo, clip_hi)
            )
            apply_mode_scale_to_res_and_pkg(res, pkg, alpha)

            # store scale info into the SAME gp npz
            # 用字符串 dtype，避免 object（后续 np.load(allow_pickle=False) 更安全）
            pkg["mode_scale_kind"] = np.array([args.mode_scale], dtype='U16')
            pkg["mode_scale_target"] = np.float32(args.scale_target)
            pkg["mode_scale_clip"] = np.array([clip_lo, clip_hi], dtype=np.float32)
            pkg["mode_scale_alpha"] = np.asarray(alpha, dtype=np.float32)
            pkg["mode_scale_inv_alpha"] = np.asarray(inv_alpha, dtype=np.float32)
            pkg["mode_scale_phi_rms"] = np.asarray(phi_rms, dtype=np.float32)
            pkg["mode_scale_grad_rms"] = np.asarray(grad_rms, dtype=np.float32)

            # 仅当用户显式要求时才额外导出 scale 文件
            if args.export_scale:
                save_mode_scale_npz(args.export_scale, alpha, inv_alpha, phi_rms, grad_rms,
                                    args.mode_scale, args.scale_target, (clip_lo, clip_hi))

        np.savez_compressed(args.export_gp_npz, **pkg)
        if bool(args.verbose):
            print(f"[gp-export] saved npz -> {args.export_gp_npz}")

    # --- node CSV export: keep ONLY hybrid (format identical to v1) ---
    if args.export_csv:
        out = args.export_csv
        compute_hess = (args.no_hessian == 0)
        if out.lower().endswith(".csv"):
            out_hybrid = out
        else:
            out_hybrid = out + "_hybrid.csv"

        et = (_LAST_ELEM_TYPE or "").upper()
        go_used = _auto_gauss_order(et, int(T.shape[1])) if int(args.gauss_order) == 0 else int(args.gauss_order)
        k_h = res["phi_hybrid"].shape[1]

        if bool(args.verbose):
            print(f"[deriv] computing grads/hess (hybrid via general+specific) (etype={et}, nloc={int(T.shape[1])}, gauss_order={go_used}) ...")

        grads_g = _mode_gradients(V, T, res["phi_general"][:, :k_h], elem_type=et, gauss_order=go_used)
        grads_s = _mode_gradients(V, T, res["phi_specific"][:, :k_h], elem_type=et, gauss_order=go_used)
        grads_h = grads_g + grads_s

        if compute_hess:
            hess_g = _mode_hessians(V, T, res["phi_general"][:, :k_h], elem_type=et, gauss_order=go_used)
            hess_s = _mode_hessians(V, T, res["phi_specific"][:, :k_h], elem_type=et, gauss_order=go_used)
            hess_h = hess_g + hess_s
        else:
            hess_h = None

        export_csv(out_hybrid, V, T, res["phi_hybrid"], node_ids=res["node_ids"], prefix="phi_hybrid_",
                   compute_hessian=compute_hess, elem_type=et, gauss_order=go_used, grads=grads_h, hess=hess_h)
        # --- NEW: export node-level specific values (no deriv), for FE-consistent GP gradients later ---
        base = out[:-4] if out.lower().endswith(".csv") else out
        out_spec_node = args.export_specific_node_csv
        if out_spec_node is None:
            out_spec_node = base + "_specific_node.csv"

        k_h = res["phi_hybrid"].shape[1]
        export_mode_values_csv(out_spec_node, V, res["phi_specific"][:, :k_h],
                               node_ids=res["node_ids"], prefix="phi_specific_")

if __name__ == "__main__":
    main()
