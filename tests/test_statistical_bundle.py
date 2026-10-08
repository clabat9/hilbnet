"""Tests for ``hilbnet.statistical_bundle``.

Coverage spans the four layers of the module:

1. **Linear-algebra utilities**: Lyapunov solver, matrix sqrt / inverse sqrt,
   ``sym_to_vec``/``vec_to_sym`` round-trip and Frobenius isometry.
2. **Wasserstein distance on Sym++(n)**: symmetry, identity of
   indiscernibles, triangle inequality (probabilistic check), and the closed
   form for diagonal matrices.
3. **Levi-Civita parallel transport**: trivial-geodesic identity (Σ₀=Σ₁),
   linearity in V, Wasserstein-unitarity (the check quoted in the paper
   appendix: within 0.5% with 200 Euler steps), and consistency between
   ``transport``/``transport_batch``/``transport_operator``.
4. **Features, graphs and datasets**: feature dimensions, KNN / fully
   connected edge-construction invariants, dedup, seed determinism, and PyG
   batching.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilbnet.statistical_bundle import (  # noqa: E402
    DistanceType,
    FeatureType,
    GraphType,
    LeviCivitaTransport,
    StatisticalBundleGraphConfig,
    StatisticalBundleGraphDataset,
    create_graph_structure,
    extract_features,
    matrix_sqrt,
    matrix_sqrt_inv,
    sample_tangent_vector,
    sample_uniform_spd,
    solve_lyapunov,
    sym_to_vec,
    vec_to_sym,
    wasserstein_distance,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _seed_each_test():
    """Reset torch RNG at the start of every test for determinism."""
    torch.manual_seed(12345)


@pytest.fixture
def n() -> int:
    return 3


@pytest.fixture
def m(n) -> int:
    return n * (n + 1) // 2


@pytest.fixture
def Sigma(n) -> torch.Tensor:
    return sample_uniform_spd(n, 1)[0]


@pytest.fixture
def Sigma_pair(n):
    A, B = sample_uniform_spd(n, 2)
    return A, B


@pytest.fixture
def V_at(Sigma) -> torch.Tensor:
    return sample_tangent_vector(Sigma, normalize=False)


# ─── 1. Linear algebra utilities ─────────────────────────────────────────────


class TestLinearAlgebra:
    def test_matrix_sqrt_squared_recovers(self, Sigma):
        S = matrix_sqrt(Sigma)
        assert torch.allclose(S @ S, Sigma, atol=1e-5)

    def test_matrix_sqrt_inv_inverts(self, Sigma):
        S = matrix_sqrt(Sigma)
        S_inv = matrix_sqrt_inv(Sigma)
        eye = torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
        assert torch.allclose(S @ S_inv, eye, atol=1e-5)

    def test_lyapunov_residual(self, Sigma, V_at):
        X = solve_lyapunov(Sigma, V_at)
        residual = X @ Sigma + Sigma @ X - V_at
        assert torch.norm(residual) < 1e-5

    def test_sym_vec_roundtrip(self, n, V_at):
        v = sym_to_vec(V_at)
        V_back = vec_to_sym(v, n)
        assert v.shape[-1] == n * (n + 1) // 2
        assert torch.allclose(V_back, V_at, atol=1e-7)

    def test_sym_vec_frobenius_isometry(self, n):
        # ⟨vec(U), vec(V)⟩ should equal Tr(UᵀV) (= Frobenius inner product).
        U = sample_tangent_vector(sample_uniform_spd(n, 1)[0], normalize=False)
        V = sample_tangent_vector(sample_uniform_spd(n, 1)[0], normalize=False)
        ip_vec = sym_to_vec(U) @ sym_to_vec(V)
        ip_frob = torch.einsum("ij,ij->", U, V)
        assert torch.allclose(ip_vec, ip_frob, atol=1e-5)

    def test_sym_vec_batched(self, n):
        S = sample_uniform_spd(n, 4)  # (4, n, n)
        v = sym_to_vec(S)
        S_back = vec_to_sym(v, n)
        assert v.shape == (4, n * (n + 1) // 2)
        assert torch.allclose(S_back, S, atol=1e-7)


# ─── 2. Distances on the SPD manifold ────────────────────────────────────────


@pytest.mark.parametrize("dist_fn", [wasserstein_distance])
class TestDistances:
    def test_self_distance_zero(self, Sigma, dist_fn):
        # Float32 cancellation: W₂²(Σ,Σ) sits at ~1e-6, so sqrt → ~1e-3.
        assert dist_fn(Sigma, Sigma).item() < 5e-3

    def test_symmetry(self, Sigma_pair, dist_fn):
        A, B = Sigma_pair
        d_AB = dist_fn(A, B).item()
        d_BA = dist_fn(B, A).item()
        assert abs(d_AB - d_BA) < 1e-4

    def test_nonnegative(self, Sigma_pair, dist_fn):
        A, B = Sigma_pair
        assert dist_fn(A, B).item() >= 0.0

    def test_triangle_inequality(self, n, dist_fn):
        # Probabilistic check: 20 random triples should satisfy d(A,C) ≤ d(A,B)+d(B,C).
        for _ in range(20):
            A, B, C = sample_uniform_spd(n, 3)
            ab = dist_fn(A, B).item()
            bc = dist_fn(B, C).item()
            ac = dist_fn(A, C).item()
            # Allow a small tolerance for float32 noise.
            assert ac <= ab + bc + 1e-4, f"AC={ac}, AB+BC={ab + bc}"


def test_wasserstein_matches_closed_form_at_diagonals():
    # For diagonal Σ₁=diag(a), Σ₂=diag(b), W₂² = Σ_i (√aᵢ - √bᵢ)².
    a = torch.tensor([1.0, 4.0, 9.0])
    b = torch.tensor([2.0, 5.0, 8.0])
    A = torch.diag(a)
    B = torch.diag(b)
    expected_sq = torch.sum((torch.sqrt(a) - torch.sqrt(b)) ** 2)
    assert torch.allclose(
        wasserstein_distance(A, B) ** 2, expected_sq, atol=1e-4
    )


# ─── 3. Levi-Civita parallel transport ───────────────────────────────────────


def _w_norm_sq(Sigma: torch.Tensor, V: torch.Tensor) -> float:
    """Wasserstein squared norm of V at Σ: W_Σ(V,V) = ½ Tr(L_Σ[V] · V)."""
    L = solve_lyapunov(Sigma, V)
    return float(0.5 * torch.einsum("ij,ij->", L, V))


class TestLeviCivitaTransport:
    def test_trivial_geodesic_returns_input(self, Sigma, V_at):
        V_t = LeviCivitaTransport.transport(V_at, Sigma, Sigma, n_steps=20)
        assert torch.allclose(V_t, V_at, atol=1e-4)

    def test_w_unitary(self, Sigma_pair):
        # Paper appendix: LC transport is W-unitary, W_{Σ₁}(τV, τV) = W_{Σ₀}(V, V),
        # verified to within 0.5% with 200 Euler steps.
        A, B = Sigma_pair
        V = sample_tangent_vector(A, normalize=False)
        V_AB = LeviCivitaTransport.transport(V, A, B, n_steps=200)
        rel_err = abs(_w_norm_sq(A, V) - _w_norm_sq(B, V_AB)) / max(_w_norm_sq(A, V), 1e-9)
        assert rel_err < 5e-3, f"W-norm relative error {rel_err}"

    def test_linearity(self, Sigma_pair):
        A, B = Sigma_pair
        U = sample_tangent_vector(A, normalize=False)
        V = sample_tangent_vector(A, normalize=False)
        a, b = 0.7, -1.3
        lhs = LeviCivitaTransport.transport(a * U + b * V, A, B, n_steps=100)
        rhs = a * LeviCivitaTransport.transport(U, A, B, n_steps=100) + b * LeviCivitaTransport.transport(V, A, B, n_steps=100)
        assert torch.allclose(lhs, rhs, atol=2e-3)

    def test_transport_batch_matches_serial(self, Sigma_pair):
        A, B = Sigma_pair
        Vs = torch.stack([sample_tangent_vector(A, normalize=False) for _ in range(4)])
        out_batch = LeviCivitaTransport.transport_batch(Vs, A, B, n_steps=50)
        for k in range(4):
            out_single = LeviCivitaTransport.transport(Vs[k], A, B, n_steps=50)
            assert torch.allclose(out_batch[k], out_single, atol=1e-5)

    def test_transport_operator_consistency(self, Sigma_pair, m):
        A, B = Sigma_pair
        T_op = LeviCivitaTransport.transport_operator(A, B, n_steps=50)
        assert T_op.shape == (m, m)
        # T_op @ vec(V) ≈ vec(τ(V))
        V = sample_tangent_vector(A, normalize=False)
        out_op = T_op @ sym_to_vec(V)
        out_direct = sym_to_vec(LeviCivitaTransport.transport(V, A, B, n_steps=50))
        assert torch.allclose(out_op, out_direct, atol=1e-5)

    def test_transport_operator_trivial_is_identity(self, Sigma, m):
        T_op = LeviCivitaTransport.transport_operator(Sigma, Sigma, n_steps=20)
        eye = torch.eye(m, dtype=T_op.dtype)
        assert torch.allclose(T_op, eye, atol=1e-4)


# ─── 4. Node features ────────────────────────────────────────────────────────


class TestFeatures:
    def test_extract_features_dimension(self, n):
        Sigma = sample_uniform_spd(n, 1)[0]
        V = sample_tangent_vector(Sigma, normalize=False)
        m = n * (n + 1) // 2

        feats_cov = extract_features(Sigma, None, FeatureType.COVARIANCE_VEC)
        assert feats_cov.shape == (m,)

        feats_log = extract_features(Sigma, None, FeatureType.LOG_EIGENVALUES)
        assert feats_log.shape == (n,)

        feats_chol = extract_features(Sigma, None, FeatureType.CHOLESKY_VEC)
        assert feats_chol.shape == (m,)

        feats_tan = extract_features(Sigma, V, FeatureType.TANGENT_VEC)
        assert feats_tan.shape == (m,)
        assert torch.allclose(feats_tan, sym_to_vec(V))

# ─── 5. Graph construction ───────────────────────────────────────────────────


def _basic_config(**overrides):
    base = dict(
        n=2,
        n_nodes=12,
        feature_type=FeatureType.TANGENT_VEC,
        graph_type=GraphType.KNN,
        distance_type=DistanceType.WASSERSTEIN,
        k_neighbors=3,
        include_fiber=True,
        edge_weight_sigma=0.5,
    )
    base.update(overrides)
    return StatisticalBundleGraphConfig(**base)


class TestGraphConstruction:
    def test_knn_no_self_loops(self):
        cfg = _basic_config(graph_type=GraphType.KNN, k_neighbors=3, n_nodes=10)
        cov = sample_uniform_spd(cfg.n, cfg.n_nodes)
        from hilbnet.statistical_bundle import compute_distance_matrix

        D = compute_distance_matrix(cov, cfg.distance_type)
        edge_index, edge_attr = create_graph_structure(D, cfg)
        assert (edge_index[0] != edge_index[1]).all()
        assert edge_index.dtype == torch.long
        assert edge_attr.shape[0] == edge_index.shape[1]

    def test_knn_undirected(self):
        cfg = _basic_config(graph_type=GraphType.KNN, k_neighbors=3, n_nodes=10)
        cov = sample_uniform_spd(cfg.n, cfg.n_nodes)
        from hilbnet.statistical_bundle import compute_distance_matrix

        D = compute_distance_matrix(cov, cfg.distance_type)
        edge_index, _ = create_graph_structure(D, cfg)
        edge_set = {(int(s), int(t)) for s, t in edge_index.T.tolist()}
        for s, t in list(edge_set):
            assert (t, s) in edge_set

    def test_full_graph_edge_count(self):
        cfg = _basic_config(graph_type=GraphType.FULLY_CONNECTED, n_nodes=8)
        cov = sample_uniform_spd(cfg.n, cfg.n_nodes)
        from hilbnet.statistical_bundle import compute_distance_matrix

        D = compute_distance_matrix(cov, cfg.distance_type)
        edge_index, _ = create_graph_structure(D, cfg)
        # N*(N-1) directed, no self-loops.
        assert edge_index.shape[1] == 8 * 7

    def test_no_duplicate_edges(self):
        cfg = _basic_config(graph_type=GraphType.KNN, k_neighbors=3, n_nodes=10)
        cov = sample_uniform_spd(cfg.n, cfg.n_nodes)
        from hilbnet.statistical_bundle import compute_distance_matrix

        D = compute_distance_matrix(cov, cfg.distance_type)
        edge_index, _ = create_graph_structure(D, cfg)
        ids = (edge_index[0] * cfg.n_nodes + edge_index[1]).tolist()
        assert len(ids) == len(set(ids))

    def test_edge_weight_decreasing_in_distance(self):
        cfg = _basic_config(graph_type=GraphType.FULLY_CONNECTED, n_nodes=6)
        cov = sample_uniform_spd(cfg.n, cfg.n_nodes)
        from hilbnet.statistical_bundle import compute_distance_matrix

        D = compute_distance_matrix(cov, cfg.distance_type)
        edge_index, edge_attr = create_graph_structure(D, cfg)
        # Sort edges by distance, weights should descend.
        dists = D[edge_index[0], edge_index[1]]
        order = torch.argsort(dists)
        sorted_w = edge_attr[order, 0]
        diffs = sorted_w[1:] - sorted_w[:-1]
        assert (diffs <= 1e-6).all()


# ─── 6. Datasets ─────────────────────────────────────────────────────────────


class TestDatasets:
    def test_seed_determinism_main(self):
        cfg = _basic_config()
        ds1 = StatisticalBundleGraphDataset(cfg, n_graphs=4, seed=7)
        ds2 = StatisticalBundleGraphDataset(cfg, n_graphs=4, seed=7)
        for g1, g2 in zip(ds1.data_list, ds2.data_list):
            assert torch.allclose(g1.x, g2.x)
            assert torch.equal(g1.edge_index, g2.edge_index)
            assert torch.allclose(g1.edge_attr, g2.edge_attr)
            assert torch.allclose(g1.covariances, g2.covariances)

    def test_different_seeds_diverge(self):
        cfg = _basic_config()
        ds1 = StatisticalBundleGraphDataset(cfg, n_graphs=2, seed=1)
        ds2 = StatisticalBundleGraphDataset(cfg, n_graphs=2, seed=2)
        # Covariances should differ across runs with different seeds.
        diff = (ds1.data_list[0].covariances - ds2.data_list[0].covariances).abs().max()
        assert diff > 1e-3

    def test_pyg_dataloader_batches(self):
        cfg = _basic_config(n_nodes=6)
        ds = StatisticalBundleGraphDataset(cfg, n_graphs=8, seed=0)
        from torch_geometric.loader import DataLoader

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.num_graphs == 4
        assert batch.x.shape[0] == 4 * cfg.n_nodes
        assert batch.edge_index.max().item() < 4 * cfg.n_nodes

    def test_no_fiber_no_tangent(self):
        cfg = _basic_config(include_fiber=False, feature_type=FeatureType.COVARIANCE_VEC)
        ds = StatisticalBundleGraphDataset(cfg, n_graphs=2, seed=0)
        for g in ds.data_list:
            # PyG drops keys whose value is None, so the attribute is simply absent.
            assert getattr(g, "tangent_vectors", None) is None
