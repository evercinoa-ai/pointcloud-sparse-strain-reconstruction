"""Controlled, fixed-budget observation ablation; original sources are read-only.

This is a NEW protocol, not a replay of a truth-selected historic checkpoint.
Only eight simulated strain observations and frozen PCDD predictions enter training.
Evaluation targets are read by a separate post-processing script after training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "pinn"
ORIGINAL = SOURCE / "PINN3_d20_hybrid_gp_v3_smooth20_finetune_weakEq_partial_strain.py"
sys.path.insert(0, str(SOURCE))
import pinn3_utils as U

spec = importlib.util.spec_from_file_location("original_finetune", ORIGINAL)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", required=True)
    ap.add_argument("--phi", required=True)
    ap.add_argument("--cases", default="190,181,173,160,180,170,199,183")
    ap.add_argument("--variants", default="strain_only,displacement_only,fusion")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--label", default="controlled_graph5000_seed42")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--strain-kind", default="axial", choices=["axial", "full"])
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="Use TF32 matrix multiplication; physics tensors remain float32")
    ap.add_argument("--cuda-graph", action="store_true", help="Uniform graph-captured Adam protocol, not bitwise equivalent to eager Adam")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "GPU required for the declared timing protocol"
    gp = dict(np.load(args.gp, allow_pickle=False))
    V = gp["V"].astype(np.float64)
    ids = gp["node_ids"].astype(np.int64)
    conn = gp["T"].astype(np.int64)
    n = len(ids)
    ne, ng = gp["gp_dV"].shape
    idx = {int(nid): k for k, nid in enumerate(ids)}
    bdir = gp["set_BDIR_idx"].astype(np.int64)
    bneu = gp["set_BNEU_idx"].astype(np.int64)
    assert n == 3623 and ne == 480 and ng == 8
    assert np.allclose(V[bdir, 2], 2000.)
    assert np.allclose(V[bneu, 2], 0.)
    assert np.all(gp["gp_dV"] > 0)
    pid, ph = U.load_phi_node_csv(args.phi, "phi_hybrid_", 6)
    ph = U.reorder_rows_by_node_ids(ids, pid, ph).astype(np.float64)
    center = V.mean(0)
    L0 = np.ptp(V, axis=0).max()
    # Fixed engineering scales, identical across all cases/variants; no truth RMS.
    u0, eps0, sigma0, F0 = 1., 1e-4, 20.6, 5000.
    E, nu = 206000., 0.3
    phi_sd = np.maximum(ph.std(0), 1e-12)
    gradphi = gp["grad_phi_hybrid_gp"][:, :, :20, :].astype(np.float64)
    # Pure invertible parameter rescaling: geometry-only RMS, no observed labels.
    grad_rms = np.sqrt(np.mean(gradphi**2, axis=(0, 1, 3)))
    assert np.all(grad_rms > 1e-14), grad_rms

    def ten(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    xyz = ten((V - center) / L0)
    phi = ten((ph - ph.mean(0)) / phi_sd)
    xgp = ten((gp["gp_xyz"].reshape(-1, 3) - center) / L0)
    gphi = ten(gradphi / grad_rms[None, None, :, None])
    con = ten(conn, torch.long)
    dn = ten(gp["dN_dx_gp"])
    dv = ten(gp["gp_dV"])
    wd = dn * dv[:, :, None, None]
    extrap = ten(gp["gp_to_corner8_extrap"])
    if "corner_nodeidx" in gp:
        cidx = gp["corner_nodeidx"].astype(np.int64)
    else:
        cidx = np.vectorize(idx.__getitem__)(gp["elem_corner_node_ids"])
    cw = gp.get("corner_w", gp.get("elem_corner_volw"))
    if cw is None:
        raise ValueError("Missing original corner volume weights")
    cidx_t = ten(cidx.reshape(-1), torch.long)
    cw_t = ten(cw)
    weight_sum = torch.zeros((n, 1), device=device)
    weight_sum.index_add_(0, cidx_t, cw_t.reshape(-1, 1))
    interior = np.ones(n, bool)
    interior[bdir] = False
    interior[bneu] = False
    int_t = ten(np.flatnonzero(interior), torch.long)
    # Hard root kinematics, retaining the original calibrated rotational support.
    distance_root = ten((2000. - V[:, 2])[:, None] / 2000.)
    manifest_path = ROOT / "fe_inputs" / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_meta = {int(c["case_id"]): c for c in manifest["cases"]}
    effective_force = {c: float(m["force_N"]) for c, m in cases_meta.items()}
    support_theta = {c: float(m["theta_x_rad"]) for c, m in cases_meta.items()}
    assert set(map(int, args.cases.split(","))) <= set(cases_meta)
    assert args.strain_kind == "axial"
    rp = np.array([61., 61.5, 2000.])  # original JOB INP assembly reference point 1

    out = ROOT / args.label
    out.mkdir(exist_ok=True)
    protocol = {
        "status": "new controlled numerical validation protocol; not independent test or historic replication",
        "manifest_sha256": sha(manifest_path),
        "experimental_runner_sha256": sha(ROOT.parent / "run_controlled_ablation.py"),
        "source": str(ORIGINAL), "source_sha256": sha(ORIGINAL),
        "utilities_sha256": sha(SOURCE / "pinn3_utils.py"),
        "runner_sha256": sha(__file__), "gp_sha256": sha(args.gp), "phi_sha256": sha(args.phi),
        "arguments": vars(args), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "mesh_nodes": n, "elements": ne, "gauss_per_element": ng,
        "architecture": {"Ku": 6, "Ks": 20, "width": args.width, "depth_u": 5, "depth_s": 6},
        "scales": {"u0_mm": u0, "eps0": eps0, "sigma0_MPa": sigma0, "F0_N": F0},
        "weights": {"displacement": 1., "strain": .5, "constitutive": .1, "equilibrium": 5.},
        "E_MPa": E, "nu": nu, "gradient_rms_rescaling": grad_rms.tolist(),
        "checkpoint_selection": "last step only, never evaluation targets",
        "initialization": "same random seed, no supervised or historic pretrained checkpoint",
        "integration": "all elements, physical integration-volume-normalized constitutive loss",
        "support": {"type": "fixed calibrated rotational support", "reference_point_mm": rp.tolist(),
                    "force_N": {str(k): v for k, v in effective_force.items()}, "theta_x_rad": {str(k): v for k, v in support_theta.items()},
                    "note": "rotation is obtained from existing calibrated connector anchors; not independent validation"},
        "limitations": ["eight simulated axial strain inputs and six withheld gauges", "single structure and load direction", "fixed calibrated support", "validation cases rather than independent test set", "one initialization and fixed optimization budget", "frozen PCDD predictions with recorded cache provenance"],
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        # A resume may not alter scientific parameters or overwrite prior completed runs.
        if not args.resume:
            raise FileExistsError(f"Output exists; use a new label or explicit resume: {out}")
        for key in ["manifest_sha256", "experimental_runner_sha256", "runner_sha256", "source_sha256", "utilities_sha256", "gp_sha256", "phi_sha256", "architecture", "scales", "weights", "support", "E_MPa", "nu"]:
            assert existing[key] == protocol[key], key
        for key in ["steps", "lr", "width", "strain_kind", "tf32", "cuda_graph"]:
            assert existing["arguments"][key] == protocol["arguments"][key], key
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    for case in map(int, args.cases.split(",")):
        prior_path = ROOT / "pcdd" / f"case{case}_pcdd_prior.csv"
        strain_path = Path(cases_meta[case]["input_axial_csv"])
        obs = pd.read_csv(prior_path)
        ui = np.array([idx[int(x)] for x in obs.NodeIds])
        assert np.max(np.abs(V[ui] - obs[["x", "y", "z"]].to_numpy())) < 1e-7
        uidx = ten(ui, torch.long)
        uobs = ten(obs[["U11", "U22", "U33"]].to_numpy())
        es = pd.read_csv(strain_path)
        ei = np.array([idx[int(x)] for x in es.NodeIds])
        assert np.all(weight_sum[ten(ei, torch.long)].cpu().numpy() > 0), "strain node lacks corner reconstruction"
        eidx = ten(ei, torch.long)
        values = es[["E11", "E22", "E33", "E12", "E13", "E23"]].to_numpy()
        emask = ten(np.flatnonzero(np.isfinite(values).reshape(-1)), torch.long)
        eobs = ten(np.nan_to_num(values))
        rotation = np.zeros_like(V)
        rotation[:, 1] = -support_theta[case] * (V[:, 2] - rp[2])
        rotation[:, 2] = support_theta[case] * (V[:, 1] - rp[1])
        urigid = ten(rotation)
        for seed in map(int, args.seeds.split(",")):
            for variant in args.variants.split(","):
                assert variant in ["strain_only", "displacement_only", "fusion"]
                rd = out / f"case{case}_{variant}_seed{seed}"
                if (rd / "completed.json").exists() and args.resume:
                    previous = json.loads((rd / "completed.json").read_text(encoding="utf-8"))
                    assert previous["steps"] == args.steps
                    assert previous["prior_sha256"] == sha(prior_path)
                    assert previous["strain_input_sha256"] == sha(strain_path)
                    print("SKIP completed", rd.name, flush=True)
                    continue
                rd.mkdir(exist_ok=True)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                unet = M.DispNet(6, width=args.width, depth=5).to(device)
                snet = M.StressCoeffNet(20, width=args.width, depth=6).to(device)
                opt = torch.optim.Adam(list(unet.parameters()) + list(snet.parameters()), lr=args.lr, capturable=args.cuda_graph)
                init_hash = hashlib.sha256()
                for parameter in list(unet.parameters()) + list(snet.parameters()):
                    init_hash.update(parameter.detach().cpu().numpy().tobytes())
                weight_u = 0. if variant == "strain_only" else 1.
                weight_s = 0. if variant == "displacement_only" else .5
                def forward():
                    displacement = urigid + distance_root * unet(xyz, phi)
                    gradu = torch.einsum("eai,egaj->egij", displacement[con], dn)
                    epsu = torch.stack([gradu[..., 0, 0], gradu[..., 1, 1], gradu[..., 2, 2],
                                        gradu[..., 0, 1] + gradu[..., 1, 0],
                                        gradu[..., 0, 2] + gradu[..., 2, 0],
                                        gradu[..., 1, 2] + gradu[..., 2, 1]], dim=-1)
                    coef = snet(xgp).view(ne, ng, 3, 20)
                    raw = sigma0 * torch.einsum("egik,egkj->egij", coef, gphi)
                    sym = .5 * (raw + raw.transpose(-1, -2))
                    epss = U.isotropic_eps_from_sigma_voigt(U.sigma33_to_voigt6(sym), E, nu)
                    epcorner = torch.einsum("cg,egk->eck", extrap, epss)
                    acc = torch.zeros((n, 6), device=device)
                    acc.index_add_(0, cidx_t, (epcorner * cw_t[:, :, None]).reshape(-1, 6))
                    epnode = acc / weight_sum.clamp_min(1e-12)
                    return displacement, epsu, raw, sym, epss, epnode
                torch.cuda.synchronize()
                started = time.perf_counter()
                startstep = 0
                prior_elapsed = 0.
                if args.resume and (rd / "resume.pt").exists():
                    old = torch.load(rd / "resume.pt", map_location=device, weights_only=False)
                    unet.load_state_dict(old["u_net"])
                    snet.load_state_dict(old["s_net"])
                    opt.load_state_dict(old["optimizer"])
                    startstep = old["step"]
                    prior_elapsed = old["elapsed_s"]
                logmode = "a" if startstep else "w"
                with (rd / "training_log.csv").open(logmode, newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not startstep:
                        writer.writerow(["step", "elapsed_s", "total", "displacement", "strain", "constitutive", "equilibrium", "interior_residual_mean_N"])
                    def optimization_step():
                        opt.zero_grad(set_to_none=True)
                        up, eu, raw, sym, epss, epnode = forward()
                        lu = ((up[uidx] - uobs) / u0).square().mean()
                        ls = (((epnode[eidx] - eobs) / eps0).square()).reshape(-1)[emask].mean()
                        sigma_u = U.voigt6_to_sigma33(U.isotropic_sigma_from_eps_eng(eu, E, nu))
                        lc = ((((raw - sigma_u) / sigma0).square().mean((-1, -2))) * dv).sum() / dv.sum()
                        fint = U.assemble_internal_nodal_forces(con, sym, wd, n)
                        le = ((fint[int_t] / F0).square().sum(-1)).mean()
                        loss = weight_u * lu + weight_s * ls + .1 * lc + 5. * le
                        loss.backward()
                        opt.step()
                        return loss, lu, ls, lc, le, fint
                    warmup_stream = torch.cuda.Stream()
                    warmup_stream.wait_stream(torch.cuda.current_stream())
                    graph, captured_values = None, None
                    for step in range(startstep + 1, args.steps + 1):
                        local_step = step - startstep
                        if not args.cuda_graph:
                            values_out = optimization_step()
                        elif local_step <= 3:
                            with torch.cuda.stream(warmup_stream):
                                values_out = optimization_step()
                            torch.cuda.current_stream().wait_stream(warmup_stream)
                        elif local_step == 4:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=warmup_stream):
                                captured_values = optimization_step()
                            graph.replay()
                            values_out = captured_values
                        else:
                            graph.replay()
                            values_out = captured_values
                        loss, lu, ls, lc, le, fint = values_out
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"nonfinite loss {rd.name} step={step}")
                        if step == 1 or step % args.log_every == 0 or step == args.steps:
                            torch.cuda.synchronize()
                            elapsed = prior_elapsed + time.perf_counter() - started
                            row = [step, elapsed, loss.item(), lu.item(), ls.item(), lc.item(), le.item(), fint[int_t].norm(dim=1).mean().item()]
                            writer.writerow(row)
                            f.flush()
                            print(rd.name, f"step={step} sec={elapsed:.1f} loss={row[2]:.5g} u={row[3]:.4g} s={row[4]:.4g} c={row[5]:.4g} eq={row[6]:.4g}", flush=True)
                        if step % 1000 == 0:
                            torch.save({"u_net": unet.state_dict(), "s_net": snet.state_dict(), "optimizer": opt.state_dict(), "step": step, "elapsed_s": prior_elapsed + time.perf_counter() - started}, rd / "resume.pt")
                torch.cuda.synchronize()
                elapsed = prior_elapsed + time.perf_counter() - started
                actual_steps = sorted({int(state["step"].item()) for state in opt.state.values()})
                assert actual_steps == [args.steps], actual_steps
                with torch.no_grad():
                    up, eu, raw, sym, epss, epnode = forward()
                    arrays = {"node_ids": ids, "xyz": V, "u_mm": up.cpu().numpy(),
                              "eps_from_stress_node": epnode.cpu().numpy(),
                              "eps_from_stress_gp": epss.cpu().numpy(), "eps_from_u_gp": eu.cpu().numpy(),
                              "sigma_gp_MPa": U.sigma33_to_voigt6(sym).cpu().numpy(),
                              "corner_weight_sum": weight_sum.cpu().numpy()}
                    np.savez_compressed(rd / "predictions_last.npz", **arrays)
                torch.save({"u_net": unet.state_dict(), "s_net": snet.state_dict(), "step": args.steps}, rd / "checkpoint_last.pt")
                done = {"case": case, "variant": variant, "seed": seed, "steps": args.steps,
                        "training_wall_s": elapsed, "timing_includes_log_and_periodic_checkpoint": True,
                        "actual_adam_updates": actual_steps, "initial_parameter_sha256": init_hash.hexdigest(),
                        "cuda_graph": args.cuda_graph,
                        "prior_sha256": sha(prior_path), "strain_input_sha256": sha(strain_path),
                        "support_force_N": effective_force[case], "support_theta_rad": support_theta[case],
                        "evaluation_targets_seen_by_training": False,
                        "selected_checkpoint": "predeclared final step"}
                (rd / "completed.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
                print("COMPLETED", rd.name, f"{elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()


