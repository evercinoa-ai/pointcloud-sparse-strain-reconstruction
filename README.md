# Point-cloud and sparse-strain structural state reconstruction

Research source code accompanying the authors' Automation in Construction submission. The workflow combines point-cloud displacement decoding (PCDD) with a hybrid Laplace--Beltrami operator (LBO) geometry representation and displacement/stress reconstruction constrained by sparse strains, constitutive consistency and equilibrium.

This repository contains an independently assembled publication copy. Research equations, network definitions and optimization objectives are unchanged; three experiment/evaluation entry points use repository-relative paths instead of workstation-specific paths. See `SOURCE_MANIFEST.json` for file hashes and adaptation records.

## Scope

| Directory | Contents |
| --- | --- |
| `pcdd/` | Network, cached dataset loader, FPS preprocessing, training, inference, attention export and virtual TLS utility |
| `pinn/` | Solid hybrid-LBO construction, displacement/stress networks, pretraining, sparse-strain fine-tuning and mechanics utilities |
| `supplementary/` | Fixed-budget three-variant ablation runners for five experimental states and eight numerical validation cases; independent multi-metric evaluation |
| `data/geometry/` | Verified specimen geometry, integration-point/LBO pack and a value-preserving six-mode nodal-column extract |
| `results/` | Archived computed metrics and withheld-gauge comparisons for all five experimental and eight numerical cases, without case removal |
| `extensions/local_consistency/` | Input-only local-consistency weighting and synthetic unit tests; an isolated development extension, not an end-to-end validation claim |

Only experimental case 0 and numerical case 170 observation arrays are included as real input examples. These are not synthetic replacements for experimental evidence. The full raw point clouds, 15,000-point caches, pretrained checkpoints, all remaining observation arrays and full-field FE reference files are not bundled. Thus, the small examples and software checks can run locally, but this package alone does not reproduce every manuscript figure or retrain the original PCDD checkpoint.

## Environment

The local publication checks used Python 3.12.14, PyTorch 2.12.0+cu130 and the versions in `requirements.txt`. CUDA is not required for the smoke tests. The fixed-budget ablation runners require CUDA because the recorded timing protocol was GPU-based.

```shell
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -B tests/smoke_test.py
python -B supplementary/numerical_round/evaluation/evaluate_multimetric_cpu.py --self-test
python -B extensions/local_consistency/tests_local_weights.py
```

For CPU-only software checks, install a compatible CPU build of PyTorch instead. Other dependency versions/platforms have not been validated here. The optional PointNet++ CUDA extension is not redistributed; when absent, `pcdd/util.py` uses its existing deterministic PyTorch FPS fallback. FPS ordering and architecture settings must match a checkpoint before scientific predictions are compared.

## Real-input example reconstruction

Run from the repository root. These commands are examples; full 5,000-step training was not repeated during publication packaging. The supplied priors are archived PCDD predictions, not FE displacement targets.

```shell
python supplementary/run_controlled_ablation.py --gp data/geometry/Lab_GP20.npz --phi data/geometry/Lab_LBO6_nodes.csv --cases 0 --variants strain_only,displacement_only,fusion --steps 5000 --seeds 42 --cuda-graph --label example_case0
python supplementary/numerical_round/run_numerical_ablation.py --gp data/geometry/Lab_GP20.npz --phi data/geometry/Lab_LBO6_nodes.csv --cases 170 --variants strain_only,displacement_only,fusion --steps 5000 --seeds 42 --cuda-graph --label example_case170
```

The complete numerical case order is `190,181,173,160,180,170,199,183`; the experimental states are `0,1,2,3,4`. Running all cases requires the other inputs in the same directory structure. Exact historical timing is hardware-dependent. The calibrated support parameters are retained; the experimental FE references are not independent full-field measurements. A 5,000-step terminal checkpoint is used, without selecting a checkpoint from withheld errors. Numerical cases belong to the original validation partition and are not reclassified as independent test cases.

The constitutive loss uses physical integration-volume normalization: the sum of squared, stress-scale-normalized tensor discrepancies multiplied by `gp_dV`, divided by the sum of `gp_dV`. This is implemented in both the original fine-tuning code and the two ablation runners.

`Lab_LBO6_nodes.csv` preserves the original strings in `NodeIds,x,y,z,phi_hybrid_1,...,phi_hybrid_6`. The other nodal derivative/Hessian columns were omitted because the example runners obtain 20-mode integration-point gradients from `Lab_GP20.npz`. The geometry INP is a preprocessing geometry, not a replacement for the original material/support FE model.

## PCDD data preparation, training and attention

The intended partition is sample IDs 0--159 for training and 160--199 for validation. Files must already be placed into their explicit `train` and `val` subdirectories. This repository does not perform a random resplit. Each split requires paired `cloudpoint/<id>.csv` and `displacement/<id>.csv` files. Confirm units, query-node order, geometry, loading direction and normalization against the selected checkpoint.

```shell
python pcdd/preprocess_fps_cache.py --root external/data_Lab_split --split train --num_points 15000 --seed 42
python pcdd/preprocess_fps_cache.py --root external/data_Lab_split --split val --num_points 15000 --seed 42
python pcdd/train_reg.py --data_root external/data_Lab_split --train_split train --val_split val --num_points 15000 --num_point_tokens 1024 --exp_name reproducibility_run
python pcdd/export_attention_maps.py --ckpt_dir external/checkpoints/DataScan4 --data_root external/data_Lab_split --cache_root external/data_Lab_split/cache_np15000 --split val --sample_id 170 --out_dir outputs/attention170
```

These explicit arguments override historical generic defaults (`8944` points, `test` as the validation directory name, and `512` point tokens). The training command demonstrates the interface, not a claim to reproduce the original trained weights from the incomplete example data. Attention export requires `best_model.pth`, `config.json` and `norm_params.json` from the matching checkpoint. Do not substitute a different mesh/load dataset merely because its file dimensions agree. Attention is a model interpretation output, not a calibrated error probability.

## Evaluation and data boundaries

The independent evaluator implements MAE, RMSE, WMAPE, maximum absolute error and signed bias. WMAPE is undefined when the absolute-reference denominator is exactly zero; small nonzero denominators are not silently replaced by an epsilon. Experimental and numerical cases are aggregated separately. `results/` contains archived computations, not fresh runs made during repository preparation.

Full evaluation requires original run predictions and metadata, all input/withheld files, experimental state metadata and aligned FE reference tables. The default external reference location is `data/reference/{Lab,CASES3}`. No assertion of complete end-to-end reproducibility is made until those separate assets are released and verified.

The local-consistency extension assumes a fixed longitudinal z axis and exact nominal (x,y) lines. Its weights are heuristic local-consistency scores; they cannot identify every smooth bias and are not calibrated uncertainty estimates. Its synthetic unit tests do not establish full-pipeline or field-scale performance.

## Attribution and rights

See `NOTICE.md` and `third_party/PCT_Pytorch_MIT_LICENSE.txt`. No general open-source license is assigned here to the authors' original additions or research data. Public availability does not itself grant a reuse license for those materials. The identified upstream MIT terms continue to apply to their covered code. Consult the corresponding author regarding reuse and the availability of omitted research assets.
