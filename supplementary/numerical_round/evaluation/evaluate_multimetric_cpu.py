"""Independent CPU-only numerical and experimental multi-metric evaluation.

Reads only final predictions and provenance. It does not train, select a
checkpoint, modify inputs, or modify the manuscript. Scientific source files
remain read-only. CSV products are reproducible computational intermediates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
NUM = HERE.parent
ROOT = NUM.parent
SOURCE = ROOT.parent / "data" / "reference"  # External full-field references; not bundled.
CASES = [190, 181, 173, 160, 180, 170, 199, 183]
E = ["E11", "E22", "E33", "E12", "E13", "E23"]
S = ["S11", "S22", "S33", "S12", "S13", "S23"]
U = ["U11", "U22", "U33"]
SOURCES: dict[str, str] = {}
RECORDS: list[dict] = []
POINTS: list[dict] = []
RUN_AUDITS: list[dict] = []
REFERENCE_CACHE: dict = {}
MANIFEST_CACHE: dict = {}


def sha(path):
    path = Path(path)
    if str(path) not in SOURCES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        SOURCES[str(path)] = digest.hexdigest()
    return SOURCES[str(path)]


def csv_read(path):
    sha(path)
    return pd.read_csv(path)


def json_read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_context(dataset, case):
    if dataset not in MANIFEST_CACHE:
        if dataset == "numerical":
            manifest = json_read(NUM / "fe_inputs" / "case_manifest.json")
            assert manifest["case_ids"] == CASES
            MANIFEST_CACHE[dataset] = {int(row["case_id"]): row for row in manifest["cases"]}
        else:
            states = csv_read(ROOT / "experimental_audit" / "state_manifest.csv")
            MANIFEST_CACHE[dataset] = {int(row["state_index"]) - 1: row for row in states.to_dict("records")}
    data = MANIFEST_CACHE[dataset][case]
    force = float(data["force_N"]) if dataset == "numerical" else float(data["effective_load_kN"]) * 1000
    return {"force_N": force, "load_kN": force / 1000,
            "load_kind": "archived_FE_applied_force" if dataset == "numerical" else "effective_load_inferred_using_strain"}


def metrics(prediction, truth, weights=None):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape or not p.size:
        raise ValueError(f"Empty or mismatched evaluation arrays: {p.shape}, {t.shape}")
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        raise ValueError("Nonfinite prediction/reference cannot be silently omitted.")
    w = np.ones_like(t) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), t.shape)
    if not (np.isfinite(w).all() and np.all(w > 0)):
        raise ValueError("Metric weights must be finite and strictly positive.")
    delta = p - t
    sw = float(w.sum())
    sae = float((w * np.abs(delta)).sum())
    sse = float((w * delta**2).sum())
    sat = float((w * np.abs(t)).sum())
    return {
        "MAE": sae / sw, "RMSE": math.sqrt(sse / sw),
        "WMAPE_percent": None if sat == 0 else 100 * sae / sat,
        "max_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_truth": sat / sw,
        "signed_bias": float((w * delta).sum()) / sw,
        "scalar_count": int(t.size), "sum_weight": sw,
        "sum_weighted_absolute_error": sae, "sum_weighted_squared_error": sse,
        "sum_weighted_absolute_truth": sat,
        "zero_WMAPE_denominator": sat == 0,
    }


def self_test():
    a = metrics([[1, 3], [2, 0]], [[0, 1], [2, 2]], [[1], [3]])
    assert a["MAE"] == 9 / 8 and math.isclose(a["RMSE"], math.sqrt(17 / 8))
    assert math.isclose(a["WMAPE_percent"], 900 / 13)
    assert a["mean_absolute_truth"] == 13 / 8
    b = metrics([1], [0])
    assert b["WMAPE_percent"] is None and b["MAE"] == 1
    c = metrics([2e-20], [1e-20])
    assert math.isclose(c["WMAPE_percent"], 100)  # no epsilon added
    assert metrics([1, 2], [1, 2])["max_absolute_error"] == 0
    try:
        metrics([np.nan], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite prediction accepted.")
    return True


def record(context, name, p, t, unit, weights=None, components=None, reference_kind="FE_numerical_reference"):
    REC = {**context, "metric_group": name, "component": "all", "unit": unit,
           "reference_kind": reference_kind, "weighting": "integration_volume" if weights is not None else "equal_scalar",
           **metrics(p, t, weights)}
    RECORDS.append(REC)
    if components:
        for col, label in enumerate(components):
            wc = None if weights is None else np.asarray(weights)[..., 0]
            RECORDS.append({**context, "metric_group": name, "component": label, "unit": unit,
                            "reference_kind": reference_kind, "weighting": REC["weighting"],
                            **metrics(np.asarray(p)[..., col], np.asarray(t)[..., col], wc)})
    return REC


def reorder_by_id(frame, ids, fields):
    assert frame.NodeIds.is_unique and len(set(map(int, ids))) == len(ids)
    indexed = frame.set_index("NodeIds")
    missing = set(map(int, ids)) - set(map(int, indexed.index))
    assert not missing, sorted(missing)
    return indexed.loc[list(map(int, ids)), fields].to_numpy(dtype=float)


def gp_reference(case, gp, pack_path):
    key = case, str(pack_path)
    if key in REFERENCE_CACHE:
        return REFERENCE_CACHE[key]
    frame = csv_read(SOURCE / "CASES3" / "GP" / f"{case}.csv")
    assert not frame.duplicated(["ElemId", "IP"]).any()
    elem_ids = gp["elem_ids"].astype(int)
    xyz = np.asarray(gp["gp_xyz"], dtype=float)
    ne, ng, _ = xyz.shape
    assert (ne, ng) == (480, 8) and len(frame) == ne * ng
    assert set(map(int, frame.ElemId)) == set(map(int, elem_ids))
    source_index = pd.MultiIndex.from_product([elem_ids, range(1, ng + 1)], names=["ElemId", "IP"])
    ordered = frame.set_index(["ElemId", "IP"]).loc[source_index]
    fields = ["X", "Y", "Z", "IVOL", *S, *E]
    original = ordered[fields].to_numpy(dtype=float).reshape(ne, ng, len(fields))
    # Independent coordinate pairing within each element avoids an unverified
    # assumption that natural Gauss order equals Abaqus IP 1..8 order.
    dist2 = ((xyz[:, :, None, :] - original[:, None, :, :3])**2).sum(-1)
    permutation = dist2.argmin(axis=-1)
    assert np.all(np.sort(permutation, axis=1) == np.arange(ng)[None, :])
    paired = original[np.arange(ne)[:, None], permutation]
    coordinate_error = float(np.max(np.linalg.norm(xyz - paired[..., :3], axis=-1)))
    assert coordinate_error < 2e-4, coordinate_error
    dv = np.asarray(gp["gp_dV"], dtype=float)
    volume_relative_error = float(np.max(np.abs(dv - paired[..., 3]) / paired[..., 3]))
    assert np.all(dv > 0) and volume_relative_error < 1e-5, volume_relative_error
    result = {"stress": paired[..., 4:10], "strain": paired[..., 10:16], "volume": dv,
              "audit": {"case": case, "max_coordinate_error_mm": coordinate_error,
                        "max_IVOL_relative_difference": volume_relative_error,
                        "unique_source_IP_orders_in_pack_order": np.unique(permutation + 1, axis=0).tolist(),
                        "pairing": "ElemId then unique within-element coordinate matching; IP recorded after pairing"}}
    REFERENCE_CACHE[key] = result
    return result


def run_one(completed, dataset, protocol):
    meta = json_read(completed)
    case = int(meta["case"])
    context = {"dataset": dataset, "case": case, "variant": meta["variant"], "seed": int(meta["seed"]),
               "steps": int(meta["steps"]), "training_wall_s": float(meta["training_wall_s"]),
               **case_context(dataset, case)}
    run = completed.parent
    prediction_path = run / "predictions_last.npz"
    sha(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as loaded:
        z = {key: loaded[key] for key in loaded.files}
    ids = z["node_ids"].astype(int)
    assert len(ids) == len(set(ids)) == 3623
    index = {int(node): i for i, node in enumerate(ids)}
    assert context["steps"] == int(protocol["arguments"]["steps"])
    if "actual_adam_updates" in meta:
        assert meta["actual_adam_updates"] == [context["steps"]]
    assert meta["selected_checkpoint"] == "predeclared final step"
    assert meta.get("evaluation_targets_seen_by_training", False) is False
    assert math.isclose(meta["support_force_N"], context["force_N"], rel_tol=1e-14, abs_tol=1e-10)
    if dataset == "numerical":
        assert case in CASES
        holdout_root = NUM / "fe_inputs"
        prior_path = NUM / "pcdd" / f"case{case}_pcdd_prior.csv"
        full_path = SOURCE / "CASES3" / "FULL" / f"{case}.csv"
    else:
        assert case in range(5)
        holdout_root = ROOT / "experimental_audit"
        prior_path = ROOT / "pcdd_audit" / f"lab_case{case}_pcdd_prior.csv"
        full_path = SOURCE / "Lab" / "FULL" / f"{case}.csv"
    holdout_path = holdout_root / f"case{case}_holdout_axial.csv"
    input_path = holdout_root / f"case{case}_input_axial.csv"
    df = csv_read(holdout_path)
    inputs = csv_read(input_path)
    assert len(df) == 6 and len(inputs) == 8
    assert set(map(int, df.NodeIds)).isdisjoint(set(map(int, inputs.NodeIds)))
    if "strain_input_sha256" in meta:
        assert meta["strain_input_sha256"] == sha(input_path)
    if "prior_sha256" in meta:
        assert meta["prior_sha256"] == sha(prior_path)
    if dataset == "numerical":
        source_meta = MANIFEST_CACHE[dataset][case]
        assert source_meta["input_axial_sha256"] == sha(input_path)
        assert source_meta["holdout_axial_sha256"] == sha(holdout_path)
        assert source_meta["source_hashes"]["FULL"] == sha(full_path)
        assert source_meta["source_hashes"]["GP"] == sha(SOURCE / "CASES3" / "GP" / f"{case}.csv")
        assert math.isclose(meta["support_theta_rad"], source_meta["theta_x_rad"], rel_tol=1e-14)
    assert np.isfinite(inputs[E].to_numpy()).sum() == 8
    assert np.isfinite(inputs[E].to_numpy())[:, 2].all()
    input_rows = np.array([index[int(node)] for node in inputs.NodeIds])
    assert np.max(abs(z["xyz"][input_rows] - inputs[["x", "y", "z"]].to_numpy())) < 1e-7
    rows = np.array([index[int(node)] for node in df.NodeIds])
    assert np.all(z["corner_weight_sum"][rows] > 0)
    assert np.max(np.abs(z["xyz"][rows] - df[["x", "y", "z"]].to_numpy())) < 1e-7
    value = df[E].to_numpy(dtype=float)
    mask = np.isfinite(value)
    assert np.array_equal(np.flatnonzero(mask.any(0)), [2]) and mask.sum() == 6
    pred = np.asarray(z["eps_from_stress_node"][rows], dtype=float)
    ref_kind = "FE_numerical_reference" if dataset == "numerical" else "measured_axial_strain_conditional_on_existing_calibration"
    primary = record(context, "heldout_axial", pred[mask] * 1e6, value[mask] * 1e6, "microstrain", reference_kind=ref_kind)
    for i in range(6):
        POINTS.append({**context, "NodeIds": int(df.NodeIds.iloc[i]), "component": "E33",
                       "reference_microstrain": value[i, 2] * 1e6,
                       "prediction_microstrain": pred[i, 2] * 1e6,
                       "error_microstrain": (pred[i, 2] - value[i, 2]) * 1e6,
                       "reference_kind": ref_kind})
    if dataset == "experimental":
        full_obs = csv_read(holdout_root / f"case{case}_holdout_full.csv")
        ir = np.array([index[int(node)] for node in full_obs.NodeIds])
        ev = full_obs[E].to_numpy(dtype=float)
        em = np.isfinite(ev)
        record(context, "heldout_full_shear_sensitivity", z["eps_from_stress_node"][ir][em].astype(float) * 1e6,
               ev[em] * 1e6, "microstrain", reference_kind="measured_strain_with_unverified_reconstructed_shear_convention")

    full = csv_read(full_path)
    assert len(full) == 3623
    true_xyz = reorder_by_id(full, ids, ["x", "y", "z"])
    assert np.max(abs(true_xyz - z["xyz"])) < 1e-7
    true_u = reorder_by_id(full, ids, U)
    displacement_reference = "FE_numerical_reference" if dataset == "numerical" else "calibrated_FE_reference_not_independent_experimental_displacement_truth"
    record(context, "displacement_all_3623_nodes", z["u_mm"], true_u, "mm", components=U, reference_kind=displacement_reference)
    prior = csv_read(prior_path)
    assert len(prior) == 3194 and prior.NodeIds.is_unique
    qi = np.array([index[int(node)] for node in prior.NodeIds])
    assert np.max(abs(z["xyz"][qi] - prior[["x", "y", "z"]].to_numpy())) < 1e-7
    record(context, "displacement_3194_query_nodes", z["u_mm"][qi], true_u[qi], "mm", components=U, reference_kind=displacement_reference)
    record(context, "PCDD_prior_3194_query_nodes", prior[U].to_numpy(), true_u[qi], "mm", components=U, reference_kind=displacement_reference)
    if dataset == "numerical":
        pack_path = Path(protocol["arguments"]["gp"])
        sha(pack_path)
        with np.load(pack_path, allow_pickle=False) as loaded:
            gp = {key: loaded[key] for key in loaded.files}
        assert np.array_equal(gp["node_ids"], ids) and np.max(abs(gp["V"] - z["xyz"])) < 1e-7
        reference = gp_reference(case, gp, pack_path)
        # Reconstruct reference corner strain independently from raw FE GP data;
        # this verifies the six-channel CSV values, not just their filenames.
        corner_idx = np.asarray(gp.get("corner_nodeidx", [[index[int(n)] for n in row] for row in gp["elem_corner_node_ids"]]), dtype=int)
        corner_w = np.asarray(gp.get("corner_w", gp["elem_corner_volw"]), dtype=float)
        corner_eps = np.einsum("cg,egk->eck", gp["gp_to_corner8_extrap"], reference["strain"])
        numerator, denominator = np.zeros((len(ids), 6)), np.zeros(len(ids))
        np.add.at(numerator, corner_idx.ravel(), (corner_eps * corner_w[..., None]).reshape(-1, 6))
        np.add.at(denominator, corner_idx.ravel(), corner_w.ravel())
        corner_truth = numerator[rows] / denominator[rows, None]
        input_truth = numerator[input_rows] / denominator[input_rows, None]
        assert np.allclose(corner_truth[:, 2], value[:, 2], rtol=1e-10, atol=1e-15)
        assert np.allclose(input_truth[:, 2], inputs.E33.to_numpy(), rtol=1e-10, atol=1e-15)
        weight = reference["volume"][..., None]
        for field, name in [("eps_from_stress_gp", "GP_strain_from_stress"), ("eps_from_u_gp", "GP_strain_from_displacement")]:
            record(context, name, np.asarray(z[field], dtype=float) * 1e6, reference["strain"] * 1e6,
                   "microstrain", weights=weight, components=E)
        record(context, "GP_stress", z["sigma_gp_MPa"], reference["stress"], "MPa", weights=weight, components=S)
    RUN_AUDITS.append({**context, "prediction_sha256": sha(prediction_path), "input_sha256": sha(input_path),
                       "heldout_sha256": sha(holdout_path), "initial_parameter_sha256": meta.get("initial_parameter_sha256"),
                       "primary_value_count": 6, "direct_input_holdout_node_overlap": False,
                       "primary_WMAPE_percent": primary["WMAPE_percent"]})


def summarize(records):
    frame = pd.DataFrame(records)
    result = []
    keys = ["dataset", "variant", "metric_group", "component", "unit", "reference_kind", "weighting"]
    for key, group in frame.groupby(keys, sort=False):
        sw = group.sum_weight.sum()
        sat = group.sum_weighted_absolute_truth.sum()
        result.append({**dict(zip(keys, key)), "n_runs": len(group), "n_cases": group.case.nunique(),
                       "total_scalar_comparisons": int(group.scalar_count.sum()),
                       "mean_case_MAE": group.MAE.mean(), "mean_case_RMSE": group.RMSE.mean(),
                       "mean_case_WMAPE_percent": group.WMAPE_percent.mean(),
                       "n_defined_case_WMAPE": int(group.WMAPE_percent.notna().sum()),
                       "pooled_MAE": group.sum_weighted_absolute_error.sum() / sw,
                       "pooled_RMSE": math.sqrt(group.sum_weighted_squared_error.sum() / sw),
                       "pooled_WMAPE_percent": None if sat == 0 else group.sum_weighted_absolute_error.sum() / sat * 100,
                       "maximum_absolute_error": group.max_absolute_error.max(),
                       "pooled_mean_absolute_truth": sat / sw})
    return result


def relative_comparisons(records):
    primary = pd.DataFrame([r for r in records if r["metric_group"] == "heldout_axial" and r["component"] == "all"])
    result = []
    for (dataset, case, seed), group in primary.groupby(["dataset", "case", "seed"], sort=False):
        by_variant = {row["variant"]: row for row in group.to_dict("records")}
        if "fusion" not in by_variant:
            continue
        for baseline in ["strain_only", "displacement_only"]:
            if baseline not in by_variant:
                continue
            for metric in ["MAE", "RMSE", "WMAPE_percent", "max_absolute_error"]:
                b, f = by_variant[baseline][metric], by_variant["fusion"][metric]
                result.append({"dataset": dataset, "case": case, "seed": seed, "baseline": baseline,
                               "metric": metric, "baseline_value": b, "fusion_value": f,
                               "absolute_change_fusion_minus_baseline": f - b,
                               "relative_reduction_percent": None if b == 0 else (b - f) / b * 100})
    return result


def aggregate_relative_comparisons(summaries):
    frame = pd.DataFrame(summaries)
    primary = frame[(frame.metric_group == "heldout_axial") & (frame.component == "all")]
    result = []
    for dataset, group in primary.groupby("dataset", sort=False):
        by_variant = {row["variant"]: row for row in group.to_dict("records")}
        if "fusion" not in by_variant:
            continue
        for baseline in ["strain_only", "displacement_only"]:
            if baseline not in by_variant:
                continue
            for aggregate in ["mean_case", "pooled"]:
                for metric in ["MAE", "RMSE", "WMAPE_percent"]:
                    field = aggregate + "_" + metric
                    b, f = by_variant[baseline][field], by_variant["fusion"][field]
                    result.append({"dataset": dataset, "baseline": baseline, "aggregation": aggregate,
                                   "metric": metric, "n_cases": by_variant["fusion"]["n_cases"],
                                   "baseline_value": b, "fusion_value": f,
                                   "absolute_change_fusion_minus_baseline": f - b,
                                   "relative_reduction_percent": None if b == 0 else (b - f) / b * 100})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical-run", type=Path, default=NUM / "controlled_graph5000_seed42")
    parser.add_argument("--experimental-run", type=Path, default=ROOT / "controlled_graph5000_seed42")
    parser.add_argument("--experimental-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Weighted metrics, zero-denominator, nonzero-small-denominator and nonfinite checks passed.")
        return
    jobs = [("experimental", args.experimental_run, set(range(5)), 15)]
    if not args.experimental_only:
        jobs.append(("numerical", args.numerical_run, set(CASES), 24))
    coverage = []
    for dataset, run, expected_cases, expected_runs in jobs:
        if not (run / "protocol.json").exists():
            if args.allow_partial:
                coverage.append({"dataset": dataset, "completed_runs": 0, "expected_runs": expected_runs, "complete": False})
                continue
            raise FileNotFoundError(run / "protocol.json")
        protocol = json_read(run / "protocol.json")
        completed_files = sorted(run.glob("case*/completed.json"))
        if not args.allow_partial:
            assert len(completed_files) == expected_runs, (dataset, len(completed_files), expected_runs)
        seen = set()
        for completed in completed_files:
            meta = json_read(completed)
            assert int(meta["case"]) in expected_cases
            key = int(meta["case"]), meta["variant"], int(meta["seed"])
            assert key not in seen and meta["variant"] in ["strain_only", "displacement_only", "fusion"]
            assert int(meta["seed"]) == 42 and int(meta["steps"]) == 5000
            seen.add(key)
            run_one(completed, dataset, protocol)
        coverage.append({"dataset": dataset, "completed_runs": len(seen), "expected_runs": expected_runs,
                         "complete": len(seen) == expected_runs})
    if not RECORDS:
        raise RuntimeError("No completed results to evaluate.")
    summaries = summarize(RECORDS)
    comparisons = relative_comparisons(RECORDS)
    aggregate_comparisons = aggregate_relative_comparisons(summaries)
    frame = pd.DataFrame(RECORDS)
    primary = frame[(frame.metric_group == "heldout_axial") & (frame.component == "all")]
    for dataset in ["experimental", "numerical"]:
        primary[primary.dataset == dataset].to_csv(HERE / f"{dataset}_case_multimetrics.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(HERE / "metric_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(POINTS).to_csv(HERE / "heldout_pointwise.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(HERE / "metric_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(HERE / "fusion_relative_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aggregate_comparisons).to_csv(HERE / "fusion_aggregate_comparisons.csv", index=False, encoding="utf-8-sig")
    initial_hashes = sorted(set(row["initial_parameter_sha256"] for row in RUN_AUDITS))
    assert len(initial_hashes) == 1 and initial_hashes[0]
    audit = {
        "evaluation": "independent_cpu_multimetric", "script_sha256": sha(__file__),
        "GPU_used": False, "training_or_checkpoint_selection_performed": False,
        "coverage": coverage, "numerical_evaluation_requested": not args.experimental_only,
        "common_initial_parameter_sha256": initial_hashes[0],
        "primary": "Six axial E33 heldout values per case, all five experimental and all eight numerical cases retained.",
        "numerical_case_roles": "Original PCDD validation cases 160-199; not upgraded to independent test cases.",
        "metric_definitions": {
            "MAE": "sum(w*abs(pred-truth))/sum(w)", "RMSE": "sqrt(sum(w*(pred-truth)^2)/sum(w))",
            "WMAPE_percent": "100*sum(w*abs(pred-truth))/sum(w*abs(truth)); exactly zero denominator gives NA; no epsilon added",
            "max_absolute_error": "max(abs(pred-truth)), not multiplied by integration volume",
            "mean_absolute_truth": "sum(w*abs(truth))/sum(w)",
            "GP_weights": "gp_dV broadcast to each of six engineering-strain or Voigt-stress components; all-component averages divide by 6*sum(gp_dV). This is component-wise, not a Frobenius tensor norm.",
            "node_displacement_weights": "Equal scalar weights, three displacement components, 3623 nodes and 3194 query nodes kept separate.",
            "pooled": "Sum sufficient statistics within a dataset and metric only; numerical and experimental data are never pooled together.",
        },
        "limitations": [
            "Experimental FE-reference displacement diagnostics are not independently measured full-field accuracy.",
            "Existing calibrated support and reconstructed sensor mapping limit full-pipeline independence.",
            "Experimental seven-channel values remain secondary shear-convention sensitivity; six axial channels are primary.",
            "Every adverse case is retained. No four-case subset is promoted in the new manuscript plan.",
            "A lower mean error does not establish uniformly superior fusion accuracy or statistical significance.",
        ],
        "GP_reference_alignment": [v["audit"] for v in REFERENCE_CACHE.values()],
        "run_audits": RUN_AUDITS, "sources_sha256": SOURCES,
    }
    (HERE / "multimetric_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    overview = pd.DataFrame(summaries)
    print(overview[(overview.metric_group == "heldout_axial") & (overview.component == "all")].to_string(index=False))
    print(json.dumps({"coverage": coverage, "evaluation_directory": str(HERE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
