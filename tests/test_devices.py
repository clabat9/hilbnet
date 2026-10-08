"""Results on an accelerator (MPS or CUDA) match the CPU.

Skipped when no accelerator is available (for example in CI). The MPS checks
matter because MPS silently ignores ``index_add_`` on non-contiguous views;
``hilbnet.utils`` and ``hilbnet.layers`` work around it, and these tests guard
the workaround.
"""
from __future__ import annotations

import pytest
import torch

from hilbnet.bundle_validation import BundleTransportModule
from hilbnet.forecasters import HilbNetForecaster, STGNNConvForecaster, masked_mae
from hilbnet.utils import apply_sheaf_laplacian_from_transport, build_householder_transport_maps

ACCELERATORS = [name for name, available in (("mps", torch.backends.mps.is_available()),
                                             ("cuda", torch.cuda.is_available())) if available]


def _make_graph():
    torch.manual_seed(42)
    B, N, T, F = 3, 4, 5, 2
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    edge_weights = torch.tensor([1.0, 0.8, 1.2, 0.9])
    transport_maps = build_householder_transport_maps(torch.randn(edge_index.shape[1], T, T))
    return torch.randn(B, N, T, F), transport_maps, edge_index, edge_weights


def _make_forecaster(kind: str, device: str):
    """Build on CPU, then move, so CPU and device models share weights."""
    torch.manual_seed(0)
    common = dict(n_nodes=4, time_steps_in=12, time_steps_out=12,
                  edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]]),
                  edge_weights=torch.tensor([1.0, 0.8, 1.2, 0.9]),
                  in_features=[2, 16, 32], activation=torch.nn.ReLU(), kappa=[2, 2])
    if kind == "stgnn_conv":
        model = STGNNConvForecaster(**common, kernel_size=3)
    else:
        model = HilbNetForecaster(**common, transport_param_type=kind,
                                  transport_init="identity_plus_noise",
                                  num_householder_reflections=8, reg_param=0.01)
    x = torch.randn(5, 4, 12, 2)
    y = 50.0 + 10.0 * torch.randn(5, 4, 12)
    return model.to(device), x.to(device), y.to(device)


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("assume_orthogonal", [True, False])
def test_laplacian_application_matches_cpu(device, assume_orthogonal):
    x, transport_maps, edge_index, edge_weights = _make_graph()
    cpu_out = apply_sheaf_laplacian_from_transport(
        x, transport_maps, edge_index, edge_weights, assume_orthogonal=assume_orthogonal)
    dev_out = apply_sheaf_laplacian_from_transport(
        x.to(device), transport_maps.to(device), edge_index.to(device),
        edge_weights.to(device), assume_orthogonal=assume_orthogonal)
    assert torch.allclose(cpu_out, dev_out.cpu(), atol=1e-5)


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("kind", ["direct", "circulant", "stgnn_conv"])
def test_forecaster_forward_matches_cpu(device, kind):
    model_cpu, x_cpu, _ = _make_forecaster(kind, "cpu")
    model_dev, x_dev, _ = _make_forecaster(kind, device)
    with torch.no_grad():
        assert torch.allclose(model_cpu(x_cpu), model_dev(x_dev).cpu(), atol=1e-4)


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("kind", ["direct", "circulant", "stgnn_conv"])
def test_forecaster_loss_and_backward_on_device(device, kind):
    model, x, y = _make_forecaster(kind, device)
    loss = masked_mae(model(x) * 10.0 + 50.0, y, torch.ones_like(y))
    loss = loss + 0.01 * model.kernel_penalty(x)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("variant", ["direct", "circulant", "frozen_id"])
def test_bundle_transport_module_matches_cpu(device, variant):
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    torch.manual_seed(0)
    module_cpu = BundleTransportModule(variant, edge_index=edge_index, stalk_dim=10,
                                       num_reflections=16)
    torch.manual_seed(0)
    module_dev = BundleTransportModule(variant, edge_index=edge_index, stalk_dim=10,
                                       num_reflections=16).to(device)
    assert torch.allclose(module_cpu(), module_dev().cpu(), atol=1e-5)
