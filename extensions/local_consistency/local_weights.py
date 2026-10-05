"""Input-only longitudinal local-consistency weights.

This module reads no files and accepts no reference labels. The score is a
local leave-center-out polynomial residual, not a calibrated probability or a
general displacement-error estimator. Smooth polynomial bias can be invisible.

Coordinate convention is fixed: z is the longitudinal axis; a line contains
points with exactly equal nominal (x, y). Arbitrary rotation of the coordinate
system is NOT supported. Root membership must be fixed from geometry by the
caller, not obtained from evaluation targets.
"""
from __future__ import annotations

import numpy as np

WINDOW_POINTS = 9
POLYNOMIAL_DEGREE = 3
LSTSQ_RCOND = 1e-12
CONDITION_LIMIT = 1e8
Q_FLOOR = .05

FLAG_DEFINITIONS = {
    "insufficient_line_points": 1,
    "duplicate_longitudinal_coordinate": 2,
    "ill_conditioned_polynomial_design": 4,
    "nonfinite_response_in_window": 8,
    "root_weight_fixed": 16,
    "nonfinite_score": 32,
}


def _validate(xyz, u, root_mask, threshold):
    xyz_array = np.asarray(xyz, dtype=np.float64)
    response_array = np.asarray(u, dtype=np.float64)
    mask = np.asarray(root_mask)
    if xyz_array.ndim != 2 or xyz_array.shape[1:] != (3,) or len(xyz_array) == 0:
        raise ValueError("xyz must have nonempty shape (N, 3)")
    if response_array.shape != xyz_array.shape:
        raise ValueError("u must have the same (N, 3) shape as xyz")
    if mask.shape != (len(xyz_array),) or mask.dtype.kind != "b":
        raise ValueError("root_mask must be a boolean geometry-only array of shape (N,)")
    if not np.isfinite(xyz_array).all():
        raise ValueError("Nominal geometry must be finite")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive, finite, and externally frozen")
    return xyz_array, response_array, mask, threshold


def compute_local_weights(xyz, u, root_mask, threshold):
    """Return one fixed scalar weight per node without mutating inputs.

    A sorted longitudinal line must have at least nine distinct z values. At
    node j, take a contiguous nine-point window, centered when possible and
    shifted one-sided at the ends. Fit each displacement component by a cubic
    on the other eight nodes, then use the Euclidean prediction residual at j.
    The design uses (z - z_j) / max(abs(z - z_j)). No coordinate or response
    threshold is learned here.

    ``valid`` describes successful score calculation, including root nodes.
    ``eligible`` is valid AND non-root. Only eligible q values are divided by
    their own mean. Invalid/fallback and root weights stay exactly one. Thus
    all non-root weights have mean one, without scaling fallback weights away
    from one. ``q`` is the unnormalized score transform, and is one at invalid
    points. Missing scores are NaN, never zero-error claims.

    Zero flags mean a valid non-root score. Flags are a uint16 bitmask whose
    meanings are returned separately. A valid root has the root-fixed bit set.
    """
    geometry, response, root, threshold = _validate(xyz, u, root_mask, threshold)
    n = len(geometry)
    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    flags = np.zeros(n, dtype=np.uint16)
    flags[root] |= FLAG_DEFINITIONS["root_weight_fixed"]
    # Exact nominal coordinate pairs avoid an unstated geometric tolerance.
    _, inverse = np.unique(geometry[:, :2], axis=0, return_inverse=True)
    line_count = int(inverse.max()) + 1
    for group in range(line_count):
        indices = np.flatnonzero(inverse == group)
        indices = indices[np.argsort(geometry[indices, 2], kind="stable")]
        if len(indices) < WINDOW_POINTS:
            flags[indices] |= FLAG_DEFINITIONS["insufficient_line_points"]
            continue
        longitudinal = geometry[indices, 2]
        if np.any(np.diff(longitudinal) <= 0):
            flags[indices] |= FLAG_DEFINITIONS["duplicate_longitudinal_coordinate"]
            continue
        for position, index in enumerate(indices):
            start = min(max(position - WINDOW_POINTS // 2, 0), len(indices) - WINDOW_POINTS)
            window = indices[start:start + WINDOW_POINTS]
            if not np.isfinite(response[window]).all():
                flags[index] |= FLAG_DEFINITIONS["nonfinite_response_in_window"]
                continue
            neighbors = window[window != index]
            dz = geometry[neighbors, 2] - geometry[index, 2]
            scale = float(np.max(np.abs(dz)))
            if not np.isfinite(scale) or scale <= 0:
                flags[index] |= FLAG_DEFINITIONS["ill_conditioned_polynomial_design"]
                continue
            normalized = dz / scale
            design = np.column_stack([normalized ** power for power in range(POLYNOMIAL_DEGREE + 1)])
            try:
                singular = np.linalg.svd(design, compute_uv=False)
                if singular[-1] <= 0 or singular[0] / singular[-1] > CONDITION_LIMIT:
                    flags[index] |= FLAG_DEFINITIONS["ill_conditioned_polynomial_design"]
                    continue
                coefficients, _, rank, _ = np.linalg.lstsq(design, response[neighbors], rcond=LSTSQ_RCOND)
            except np.linalg.LinAlgError:
                flags[index] |= FLAG_DEFINITIONS["ill_conditioned_polynomial_design"]
                continue
            if rank != POLYNOMIAL_DEGREE + 1:
                flags[index] |= FLAG_DEFINITIONS["ill_conditioned_polynomial_design"]
                continue
            residual = response[index] - coefficients[0]
            score = float(np.linalg.norm(residual))
            if not np.isfinite(score):
                flags[index] |= FLAG_DEFINITIONS["nonfinite_score"]
                continue
            scores[index], valid[index] = score, True
    q = np.ones(n, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        q[valid] = np.clip(1. / (1. + (scores[valid] / threshold) ** 2), Q_FLOOR, 1.)
    eligible = valid & ~root
    weights = np.ones(n, dtype=np.float64)
    if np.any(eligible):
        mean_q = float(q[eligible].mean())
        weights[eligible] = q[eligible] / mean_q
    else:
        mean_q = None
    assert np.isfinite(weights).all() and np.all(weights > 0)
    assert np.array_equal(weights[root | ~valid], np.ones(np.count_nonzero(root | ~valid)))
    nonroot_count = int(np.count_nonzero(~root))
    if nonroot_count:
        assert abs(float(weights[~root].mean()) - 1.) < 1e-12
    diagnostics = {
        "nodes": n, "longitudinal_lines": line_count, "root_nodes": int(root.sum()),
        "valid_scores": int(valid.sum()), "eligible_weighted_nodes": int(eligible.sum()),
        "fallback_nonroot_nodes": int(np.count_nonzero(~root & ~valid)),
        "threshold_mm": threshold, "window_points_including_center": WINDOW_POINTS,
        "fit_points_excluding_center": WINDOW_POINTS - 1, "polynomial_degree": POLYNOMIAL_DEGREE,
        "condition_limit": CONDITION_LIMIT, "q_floor": Q_FLOOR, "eligible_mean_q": mean_q,
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "nonroot_weight_mean": float(weights[~root].mean()) if nonroot_count else None,
        "effective_weight_count": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "sum_weights": float(weights.sum()),
        "normalization": "Eligible valid non-root nodes have mean weight 1; fallback and root remain exactly 1.",
        "scope": "Local consistency only; not calibrated confidence or identifiable smooth-bias magnitude.",
    }
    return {"scores": scores, "valid": valid, "weights": weights, "flags": flags,
            "q": q, "eligible": eligible, "flag_definitions": dict(FLAG_DEFINITIONS),
            "diagnostics": diagnostics}
