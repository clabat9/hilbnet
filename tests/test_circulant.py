import pytest
import torch

from hilbnet.circulant_transport import (
    CirculantTransportParam,
    _band_assignment,
    build_circulant_transport_maps,
    procrustes_to_circulant_phases,
)


# ── Basic construction: orthogonality, structure, determinant ──────────────


@pytest.mark.parametrize("T", [10, 11, 188, 189])
def test_circulant_is_orthogonal(T):
    torch.manual_seed(0)
    E = 4
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free)
    P = build_circulant_transport_maps(phases, T)
    assert P.shape == (E, T, T)
    gram = P.transpose(-1, -2) @ P
    eye = torch.eye(T).unsqueeze(0).expand(E, T, T)
    assert torch.allclose(gram, eye, atol=1e-4)


@pytest.mark.parametrize("T", [10, 11])
def test_circulant_structure(T):
    torch.manual_seed(0)
    E = 3
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free)
    P = build_circulant_transport_maps(phases, T)
    # P[e, i, j] = P[e, (i+1) % T, (j+1) % T] for circulants.
    shifted = P.roll(shifts=(1, 1), dims=(-2, -1))
    assert torch.allclose(P, shifted, atol=1e-5)


@pytest.mark.parametrize("T", [10, 11])
def test_shift_commutation(T):
    """P @ S ≈ S @ P where S is the cyclic shift matrix."""
    torch.manual_seed(0)
    E = 3
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free)
    P = build_circulant_transport_maps(phases, T)
    # S: cyclic shift, S[i, j] = 1 iff j = (i-1) mod T.
    S = torch.zeros(T, T)
    for i in range(T):
        S[i, (i - 1) % T] = 1.0
    lhs = P @ S
    rhs = S @ P
    assert torch.allclose(lhs, rhs, atol=1e-5)


@pytest.mark.parametrize("T", [10, 11])
def test_det_is_plus_one(T):
    """Circulant matrices from our construction lie in SO(T)'s identity component."""
    torch.manual_seed(0)
    E = 3
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free)
    P = build_circulant_transport_maps(phases, T)
    dets = torch.linalg.det(P)
    assert torch.allclose(dets, torch.ones(E), atol=1e-4)


@pytest.mark.parametrize("T", [10, 11])
def test_identity_init_yields_identity(T):
    """phases = 0  ⇒  P = I (because all eigenvalues = 1)."""
    E = 3
    num_free = (T - 1) // 2
    phases = torch.zeros(E, num_free)
    P = build_circulant_transport_maps(phases, T)
    eye = torch.eye(T).unsqueeze(0).expand(E, T, T)
    assert torch.allclose(P, eye, atol=1e-5)


def test_gradient_flow():
    """Backprop reaches the phase parameters."""
    torch.manual_seed(0)
    T, E = 10, 4
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free, requires_grad=True)
    P = build_circulant_transport_maps(phases, T)
    loss = P.pow(2).sum()
    loss.backward()
    assert phases.grad is not None
    assert phases.grad.abs().sum().item() > 0


# ── Banded variant ─────────────────────────────────────────────────────────


def test_band_assignment_is_linear_and_surjective():
    num_free, K = 20, 4
    ba = _band_assignment(num_free, K)
    assert ba.shape == (num_free,)
    # each band appears at least once, and indices are non-decreasing.
    assert set(ba.tolist()) == set(range(K))
    assert (ba[1:] >= ba[:-1]).all().item()


def test_banded_expands_to_piecewise_constant_phases():
    """Banded phases should expand to per-frequency phases constant within each band."""
    torch.manual_seed(0)
    T, E, K = 20, 3, 4
    num_free = (T - 1) // 2  # = 9
    ba = _band_assignment(num_free, K)
    banded_phases = torch.randn(E, K)

    P_banded = build_circulant_transport_maps(banded_phases, T, band_assignment=ba)
    # Expand manually and check equality.
    phases_full = banded_phases[:, ba]  # [E, num_free]
    P_full = build_circulant_transport_maps(phases_full, T)
    assert torch.allclose(P_banded, P_full, atol=1e-5)


# ── Procrustes → circulant projection ─────────────────────────────────────


def test_procrustes_roundtrip_on_known_circulant():
    """Known circulant target → projection extracts its exact phases."""
    torch.manual_seed(0)
    T, E = 10, 3
    num_free = (T - 1) // 2
    true_phases = torch.randn(E, num_free)
    targets = build_circulant_transport_maps(true_phases, T)

    recovered_phases, ba, residual = procrustes_to_circulant_phases(targets, T)
    assert ba is None

    # Phases should match up to sign/wrap.  Easiest check: the rebuilt matrix.
    rebuilt = build_circulant_transport_maps(recovered_phases, T)
    assert torch.allclose(rebuilt, targets, atol=1e-4)
    assert residual < 1e-4


def test_procrustes_on_non_circulant_target_has_nontrivial_residual():
    """For a generic orthogonal target (non-circulant), projection produces a
    circulant approximation with nonzero Frobenius residual."""
    torch.manual_seed(0)
    T, E = 10, 3
    # Generic random orthogonal via QR.
    A = torch.randn(E, T, T)
    Q, _ = torch.linalg.qr(A)
    # Ensure det = +1.
    dets = torch.linalg.det(Q)
    Q[dets < 0, :, 0] *= -1
    phases, ba, residual = procrustes_to_circulant_phases(Q, T)
    assert phases.shape == (E, (T - 1) // 2)
    assert residual > 1e-3  # generic orthogonal is not circulant


def test_procrustes_banded_roundtrip():
    """Known banded-circulant target → projection recovers it (up to banding)."""
    torch.manual_seed(0)
    T, E, K = 20, 3, 4
    num_free = (T - 1) // 2
    ba = _band_assignment(num_free, K)
    true_banded = torch.randn(E, K)
    # Build target using banded phases.
    targets = build_circulant_transport_maps(true_banded, T, band_assignment=ba)

    recovered, ba_out, residual = procrustes_to_circulant_phases(targets, T, num_bands=K)
    assert recovered.shape == (E, K)
    assert ba_out is not None

    rebuilt = build_circulant_transport_maps(recovered, T, band_assignment=ba_out)
    assert torch.allclose(rebuilt, targets, atol=1e-4)


# ── CirculantTransportParam module ─────────────────────────────────────────


def test_module_identity_init():
    T, E = 20, 5
    mod = CirculantTransportParam(num_edges=E, stalk_dim=T, init="identity")
    P = mod()
    eye = torch.eye(T).unsqueeze(0).expand(E, T, T)
    assert torch.allclose(P, eye, atol=1e-5)


def test_module_shapes_full_and_banded():
    T, E = 20, 5
    K = 4
    mod_full = CirculantTransportParam(num_edges=E, stalk_dim=T)
    assert mod_full.phases.shape == (E, (T - 1) // 2)

    mod_banded = CirculantTransportParam(num_edges=E, stalk_dim=T, num_bands=K)
    assert mod_banded.phases.shape == (E, K)
    assert mod_banded.band_assignment.shape == ((T - 1) // 2,)


def test_module_forward_is_orthogonal_after_noise_init():
    torch.manual_seed(0)
    T, E = 30, 4
    mod = CirculantTransportParam(num_edges=E, stalk_dim=T, init="identity_plus_noise", init_scale=0.5)
    P = mod()
    gram = P.transpose(-1, -2) @ P
    eye = torch.eye(T).unsqueeze(0).expand(E, T, T)
    assert torch.allclose(gram, eye, atol=1e-4)


def test_module_gradient_flows_to_phases():
    torch.manual_seed(0)
    T, E = 20, 4
    mod = CirculantTransportParam(num_edges=E, stalk_dim=T, init="identity_plus_noise")
    P = mod()
    loss = P.pow(2).sum()
    loss.backward()
    assert mod.phases.grad is not None
    assert mod.phases.grad.abs().sum().item() > 0


# ── Compatibility with existing sheaf-Laplacian machinery ──────────────────


def test_circulant_transports_work_with_apply_sheaf_laplacian():
    from hilbnet.utils import apply_sheaf_laplacian_from_transport

    torch.manual_seed(0)
    B, N, T, F = 2, 4, 10, 3
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    E = edge_index.shape[1]
    num_free = (T - 1) // 2
    phases = torch.randn(E, num_free)
    P = build_circulant_transport_maps(phases, T)

    x = torch.randn(B, N, T, F)
    out_general = apply_sheaf_laplacian_from_transport(
        x, transport_maps=P, edge_index=edge_index, assume_orthogonal=False,
    )
    out_ortho = apply_sheaf_laplacian_from_transport(
        x, transport_maps=P, edge_index=edge_index, assume_orthogonal=True,
    )
    assert out_general.shape == (B, N, T, F)
    assert torch.allclose(out_general, out_ortho, atol=1e-5)
