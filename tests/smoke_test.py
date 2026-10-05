"""CPU software checks; random tensors are not research validation data."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pcdd"))
sys.path.insert(0, str(ROOT / "pinn"))
from model_reg_U import PctRegressor
import pinn3_utils as mechanics

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = PctRegressor(SimpleNamespace(dropout=0.0, num_point_tokens=16), num_nodes=8).eval()
    with torch.no_grad():
        out, attention = model(torch.rand(1, 4, 64) + .1, torch.rand(1, 4, 8) + .1, return_attention=True)
    assert out.shape == (1, 8, 3) and torch.isfinite(out).all()
    for key in ["node_from_point_stage_1", "node_from_point_stage_2"]:
        matrix = attention[key]
        assert matrix.shape == (1, 4, 8, 16)
        assert torch.allclose(matrix.sum(-1), torch.ones(1, 4, 8), atol=1e-6)
    pinn = load("pinn_finetune", ROOT / "pinn/PINN3_d20_hybrid_gp_v3_smooth20_finetune_weakEq_partial_strain.py")
    unet = pinn.DispNet(6, width=16, depth=2)
    snet = pinn.StressCoeffNet(20, width=16, depth=2)
    xyz, phi = torch.rand(4, 3), torch.rand(4, 6)
    displacement = unet(xyz, phi)
    stress = pinn.assemble_sigma_from_P(snet(xyz), torch.rand(4, 20, 3))
    assert displacement.shape == (4, 3) and stress.shape == (4, 3, 3)
    eps = torch.randn(4, 6) * 1e-4
    recovered = mechanics.isotropic_eps_from_sigma_voigt(mechanics.isotropic_sigma_from_eps_eng(eps, 206000., .3), 206000., .3)
    assert torch.allclose(eps, recovered, atol=1e-10, rtol=1e-5)
    dv = torch.tensor([1., 2., 3., 4.])
    errors = stress.square().mean((-1, -2))
    loss = (errors * dv).sum() / dv.sum()
    assert torch.allclose(loss, (errors * (dv * 7)).sum() / (dv * 7).sum())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in snet.parameters())
    gp = np.load(ROOT / "data/geometry/Lab_GP20.npz", allow_pickle=False)
    assert gp["V"].shape == (3623, 3) and np.all(gp["gp_dV"] > 0)
    ids, modes = mechanics.load_phi_node_csv(str(ROOT / "data/geometry/Lab_LBO6_nodes.csv"), "phi_hybrid_", 6)
    modes = mechanics.reorder_rows_by_node_ids(gp["node_ids"], ids, modes)
    assert modes.shape == (3623, 6) and np.isfinite(modes).all()
    print(json.dumps({"passed": True, "device": "cpu", "PCDD_random_tensor_forward": True,
                     "two_attention_row_sums": True, "PINN_networks_and_backward": True,
                     "constitutive_roundtrip": True, "integration_volume_normalization_invariance": True,
                     "bundled_geometry_and_modes": True, "scientific_training_repeated": False}, indent=2))

if __name__ == "__main__":
    main()
