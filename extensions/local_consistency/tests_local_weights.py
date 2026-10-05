"""Synthetic unit tests only; no data files, labels, FE fields, or GPU read."""
from __future__ import annotations

import inspect
import json
import numpy as np
from local_weights import compute_local_weights, FLAG_DEFINITIONS


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def require_close(a, b, message, tolerance=2e-11):
    difference = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    check(difference <= tolerance, (message, difference))
    return difference


def geometry():
    xyz = np.array([[x, y, z] for x, y in [(0., 0.), (61., 61.5), (122., 123.)]
                    for z in np.arange(0., 2001., 25.)])
    return xyz, xyz[:, 2] == 2000.


def main():
    xyz, root = geometry()
    z = xyz[:, 2] / 2000.
    clean = np.column_stack([.2 + .3 * z - .1 * z ** 2 + .01 * z ** 3,
                             5. * (1. - z) ** 2 + .5 * (1. - z) ** 3,
                             .003 * (xyz[:, 1] - 61.5) + .1 * z ** 2])
    cases = []
    for name, u in [
        ("zero_field", np.zeros_like(xyz)),
        ("affine_field", xyz @ np.array([[.0001, .0002, -.0003], [.002, .001, 0.], [0., -.0004, .0002]])
         + np.array([.1, -.3, .5])),
        ("rigid_field", np.cross(np.array([.001, -.002, .0005]), xyz - np.array([61., 61.5, 2000.]))
         + np.array([.3, .2, -.1])),
        ("cubic_field", clean),
    ]:
        result = compute_local_weights(xyz, u, root, .1)
        check(result["valid"].all(), name + " validity")
        error = require_close(result["scores"], np.zeros(len(xyz)), name + " scores")
        require_close(result["weights"], np.ones(len(xyz)), name + " weights")
        cases.append({"test": name, "max_score_mm": error})
    perturbed = clean.copy()
    center = int(np.flatnonzero((xyz[:, 0] == 61.) & (xyz[:, 2] == 1000.))[0])
    perturbed[center] += np.array([.3, -1., .2])
    xyz_copy, response_copy, root_copy = xyz.copy(), perturbed.copy(), root.copy()
    result = compute_local_weights(xyz, perturbed, root, .1)
    check(result["scores"][center] > 1., "local spike should be observed")
    check(result["weights"][center] < 1., "spike should receive lower-than-uniform weight")
    check(np.array_equal(result["weights"][root], np.ones(root.sum())), "root fixed")
    require_close(result["weights"][~root].mean(), 1., "nonroot mass")
    require_close(result["weights"].sum(), len(xyz), "total mass")
    check(np.array_equal(xyz, xyz_copy) and np.array_equal(perturbed, response_copy)
          and np.array_equal(root, root_copy), "inputs mutated")
    cases.append({"test": "local_perturbation_root_normalization_input_immutability",
                  "spike_score_mm": float(result["scores"][center]),
                  "spike_weight": float(result["weights"][center])})
    order = np.random.default_rng(90210).permutation(len(xyz))
    shuffled = compute_local_weights(xyz[order], perturbed[order], root[order], .1)
    require_close(shuffled["scores"], result["scores"][order], "node-order scores")
    require_close(shuffled["weights"], result["weights"][order], "node-order weights")
    check(np.array_equal(shuffled["flags"], result["flags"][order]), "node-order flags")
    cases.append({"test": "node_permutation"})
    shifted = compute_local_weights(xyz + np.array([1000., -2000., 10000.]), perturbed, root, .1)
    require_close(shifted["scores"], result["scores"], "coordinate origin scores")
    require_close(shifted["weights"], result["weights"], "coordinate origin weights")
    cases.append({"test": "coordinate_translation_not_general_rotation"})
    rigid_added = perturbed + np.cross(np.array([.001, -.002, .0005]), xyz) + np.array([.3, .2, -.1])
    rigid_result = compute_local_weights(xyz, rigid_added, root, .1)
    require_close(rigid_result["scores"], result["scores"], "added rigid displacement scores")
    require_close(rigid_result["weights"], result["weights"], "added rigid displacement weights")
    cases.append({"test": "additive_rigid_displacement_invariance"})
    # A response-vector rotation alone preserves its Euclidean residual. This
    # does not rotate geometry and must not be called geometry-rotation invariance.
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    vector_rotated = compute_local_weights(xyz, perturbed @ rotation.T, root, .1)
    require_close(vector_rotated["scores"], result["scores"], "response-vector rotation")
    cases.append({"test": "orthogonal_response_vector_rotation_only"})
    small_xyz = np.array([[0., 0., z] for z in range(8)], dtype=float)
    small_root = np.arange(8) == 7
    no_valid = compute_local_weights(small_xyz, np.zeros_like(small_xyz), small_root, .1)
    check(not no_valid["valid"].any() and np.array_equal(no_valid["weights"], np.ones(8)), "all fallback")
    check(np.isnan(no_valid["scores"]).all(), "missing scores must not be zero")
    cases.append({"test": "no_valid_neighborhood_all_uniform"})
    combined_xyz = np.vstack([xyz, small_xyz + np.array([200., 300., 0.])])
    combined_u = np.vstack([perturbed, np.zeros_like(small_xyz)])
    combined_root = np.r_[root, small_root]
    mixed = compute_local_weights(combined_xyz, combined_u, combined_root, .1)
    check(np.array_equal(mixed["weights"][-8:], np.ones(8)), "fallback changed by normalization")
    require_close(mixed["weights"][~combined_root].mean(), 1., "mixed nonroot mean")
    require_close(mixed["weights"][:len(xyz)], result["weights"], "adding invalid line should not rescale valid lines")
    cases.append({"test": "fallback_fixed_one_valid_subset_normalization"})
    duplicate_xyz = xyz[:10].copy()
    duplicate_xyz[5, 2] = duplicate_xyz[4, 2]
    duplicate = compute_local_weights(duplicate_xyz, clean[:10], np.zeros(10, bool), .1)
    check(not duplicate["valid"].any(), "duplicate z should be rejected")
    check(np.all(duplicate["flags"] & FLAG_DEFINITIONS["duplicate_longitudinal_coordinate"]), "duplicate flag")
    cases.append({"test": "duplicate_z_fallback"})
    bad_geometry = np.column_stack([np.zeros(9), np.zeros(9), np.r_[np.arange(8) * 1e-17, 1.]])
    ill = compute_local_weights(bad_geometry, np.zeros_like(bad_geometry), np.zeros(9, bool), .1)
    check(not ill["valid"].any(), "ill-conditioned cubic should be rejected")
    check(np.all(ill["flags"] & FLAG_DEFINITIONS["ill_conditioned_polynomial_design"]), "condition flag")
    cases.append({"test": "ill_conditioned_neighborhood_fallback"})
    nonfinite_u = clean.copy()
    nonfinite_u[center, 0] = np.nan
    nonfinite = compute_local_weights(xyz, nonfinite_u, root, .1)
    check(not nonfinite["valid"][center] and nonfinite["weights"][center] == 1., "nonfinite fallback")
    check(nonfinite["flags"][center] & FLAG_DEFINITIONS["nonfinite_response_in_window"], "nonfinite flag")
    cases.append({"test": "nonfinite_response_flagged_not_silently_zero"})
    all_root = compute_local_weights(xyz, perturbed, np.ones(len(xyz), bool), .1)
    check(np.array_equal(all_root["weights"], np.ones(len(xyz))), "all roots fixed")
    cases.append({"test": "all_root_fixed"})
    for threshold in (0., -.1, np.nan, np.inf):
        try:
            compute_local_weights(xyz, clean, root, threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid threshold accepted")
    cases.append({"test": "invalid_threshold_rejected"})
    signature = list(inspect.signature(compute_local_weights).parameters)
    check(signature == ["xyz", "u", "root_mask", "threshold"], "label-like input added")
    cases.append({"test": "label_free_interface", "parameters": signature})
    print(json.dumps({"passed": True, "test_count": len(cases), "tests": cases,
                      "arbitrary_geometry_rotation_supported": False,
                      "field_accuracy_or_uncertainty_calibration_claimed": False}, indent=2))


if __name__ == "__main__":
    main()
