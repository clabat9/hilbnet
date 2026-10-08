"""Tests for ``hilbnet.bundle_validation``.

Coverage spans the four layers of the module:

1. **Wasserstein-Gram + Cholesky rescaling** — algebraic identities tying
   ``gram_matrix_w`` and ``cholesky_rescale_factor`` to the Wasserstein
   bilinear form on the fiber.
2. **Closed-form LC transports** — paper invariants in both raw and
   rescaled coordinates: W-unitarity (paper appendix, Levi-Civita transport)
   in raw basis; Frobenius-orthogonality in rescaled basis.
3. **Sheaf-Laplacian assembly + spectrum** — built from the closed-form
   transports via ``hilbnet.utils.build_sheaf_laplacian_from_transport``.
4. **Restricted-class projections** — circulant / identity, plus
   :class:`BundleTransportModule` dispatch.
5. **Experiment-script smoke tests** for transport recovery (Table 1) and
   spectral stability (Figure 3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilbnet.bundle_validation import (  # noqa: E402
    BundleTransportModule,
    assemble_sheaf_laplacian,
    cholesky_factors_per_node,
    cholesky_rescale_factor,
    closed_form_lc_transports,
    eigenvalues_smallest,
    gram_matrix_w,
    project_to_circulant,
    project_to_identity,
    projection_distance,
)
from hilbnet.statistical_bundle import (  # noqa: E402
    DistanceType,
    FeatureType,
    GraphType,
    LeviCivitaTransport,
    StatisticalBundleGraphConfig,
    StatisticalBundleGraphDataset,
    sample_tangent_vector,
    sample_uniform_spd,
    solve_lyapunov,
    sym_to_vec,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _seed_each_test():
    torch.manual_seed(7919)


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
def small_graph(n):
    cfg = StatisticalBundleGraphConfig(
        n=n,
        n_nodes=6,
        feature_type=FeatureType.TANGENT_VEC,
        graph_type=GraphType.KNN,
        distance_type=DistanceType.WASSERSTEIN,
        k_neighbors=2,
        include_fiber=True,
    )
    ds = StatisticalBundleGraphDataset(cfg, n_graphs=1, seed=0)
    return ds.data_list[0]


# ─── 1. Wasserstein-Gram and Cholesky rescaling ──────────────────────────────


class TestGramRescale:
    def test_gram_matrix_symmetric_pd(self, Sigma):
        G = gram_matrix_w(Sigma)
        assert torch.allclose(G, G.T, atol=1e-5)
        eigs = torch.linalg.eigvalsh(G)
        assert eigs.min().item() > 1e-6

    def test_gram_matches_w_inner_product(self, Sigma):
        # ⟨vec(U), G vec(V)⟩ should equal ½ Tr(L_Σ[U] V).
        U = sample_tangent_vector(Sigma, normalize=False)
        V = sample_tangent_vector(Sigma, normalize=False)
        G = gram_matrix_w(Sigma)
        lhs = sym_to_vec(U) @ G @ sym_to_vec(V)
        rhs = 0.5 * torch.einsum("ij,ji->", solve_lyapunov(Sigma, U), V)
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_cholesky_rescale_factor_correct(self, Sigma):
        G = gram_matrix_w(Sigma)
        R = cholesky_rescale_factor(Sigma)
        assert torch.allclose(R.T @ R, G, atol=1e-4)

    def test_rescale_makes_w_into_frobenius(self, Sigma):
        # ‖R u‖² == u^T G u == W_Σ(U, U)
        U = sample_tangent_vector(Sigma, normalize=False)
        u = sym_to_vec(U)
        R = cholesky_rescale_factor(Sigma)
        rescaled = R @ u
        # Wasserstein norm via solve_lyapunov:
        L_U = solve_lyapunov(Sigma, U)
        w_norm_sq = 0.5 * torch.einsum("ij,ji->", L_U, U)
        assert torch.allclose(rescaled.dot(rescaled), w_norm_sq, atol=1e-4)

# ─── 2. Closed-form Levi-Civita transports ───────────────────────────────────


class TestClosedFormLC:
    def test_unrescaled_lc_is_w_unitary_per_edge(self, small_graph):
        T_raw = closed_form_lc_transports(
            small_graph.covariances, small_graph.edge_index,
            n_steps=80, rescale=False,
        )
        E = small_graph.edge_index.shape[1]
        for e in range(E):
            src = int(small_graph.edge_index[0, e].item())
            dst = int(small_graph.edge_index[1, e].item())
            if src == dst:
                continue
            G_src = gram_matrix_w(small_graph.covariances[src])
            G_dst = gram_matrix_w(small_graph.covariances[dst])
            # T^T G_dst T == G_src  (W-unitarity in coordinates)
            lhs = T_raw[e].T @ G_dst @ T_raw[e]
            rel = (lhs - G_src).norm() / G_src.norm()
            assert rel < 1e-2, f"edge {e}: rel err {rel}"

    def test_rescaled_lc_is_frobenius_orthogonal_per_edge(self, small_graph, m):
        T_resc = closed_form_lc_transports(
            small_graph.covariances, small_graph.edge_index,
            n_steps=80, rescale=True,
        )
        E = small_graph.edge_index.shape[1]
        eye = torch.eye(m)
        for e in range(E):
            src = int(small_graph.edge_index[0, e].item())
            dst = int(small_graph.edge_index[1, e].item())
            if src == dst:
                continue
            err = (T_resc[e].T @ T_resc[e] - eye).norm()
            assert err < 1e-2, f"edge {e}: ||T^T T - I|| = {err}"

    def test_rescaled_lc_consistent_with_raw_via_R(self, small_graph):
        T_raw = closed_form_lc_transports(
            small_graph.covariances, small_graph.edge_index,
            n_steps=50, rescale=False,
        )
        T_resc = closed_form_lc_transports(
            small_graph.covariances, small_graph.edge_index,
            n_steps=50, rescale=True,
        )
        Rs = cholesky_factors_per_node(small_graph.covariances)
        R_inv = torch.linalg.inv(Rs)
        E = small_graph.edge_index.shape[1]
        for e in range(E):
            src = int(small_graph.edge_index[0, e].item())
            dst = int(small_graph.edge_index[1, e].item())
            if src == dst:
                continue
            expected = Rs[dst] @ T_raw[e] @ R_inv[src]
            assert torch.allclose(T_resc[e], expected, atol=1e-5)

    def test_rescaled_action_matches_direct_transport(self, small_graph):
        # Apply the rescaled operator to a random rescaled tangent and check it
        # agrees with direct LC transport followed by rescaling.
        T_resc = closed_form_lc_transports(
            small_graph.covariances, small_graph.edge_index,
            n_steps=80, rescale=True,
        )
        Rs = cholesky_factors_per_node(small_graph.covariances)
        E = small_graph.edge_index.shape[1]
        for e in range(E):
            src = int(small_graph.edge_index[0, e].item())
            dst = int(small_graph.edge_index[1, e].item())
            if src == dst:
                continue
            V = sample_tangent_vector(small_graph.covariances[src], normalize=False)
            transported = LeviCivitaTransport.transport(
                V, small_graph.covariances[src], small_graph.covariances[dst], n_steps=80,
            )
            ref_rescaled = Rs[dst] @ sym_to_vec(transported)
            actual = T_resc[e] @ (Rs[src] @ sym_to_vec(V))
            assert torch.allclose(actual, ref_rescaled, atol=2e-3)


# ─── 3. Sheaf-Laplacian assembly + spectrum ─────────────────────────────────


class TestSheafLaplacian:
    def test_sheaf_laplacian_psd(self, small_graph):
        L = assemble_sheaf_laplacian(
            small_graph.covariances, small_graph.edge_index,
            edge_weights=small_graph.edge_attr.squeeze(-1),
            n_steps=40, rescale=True,
        )
        # Symmetric (assume_orthogonal=True path is symmetric).
        assert torch.allclose(L, L.T, atol=1e-4)
        eigs = torch.linalg.eigvalsh(L)
        assert eigs.min().item() > -1e-4

    def test_eigenvalues_smallest_matches_eigvalsh(self, small_graph):
        L = assemble_sheaf_laplacian(
            small_graph.covariances, small_graph.edge_index,
            edge_weights=small_graph.edge_attr.squeeze(-1),
            n_steps=20, rescale=True,
        )
        k = 5
        out = eigenvalues_smallest(L, k)
        ref = torch.linalg.eigvalsh(L)[:k]
        assert torch.allclose(out, ref, atol=1e-6)

    def test_sheaf_laplacian_kernel_when_curvature_zero(self, n):
        # Build a tiny graph with all Σ_i = I_n. Then LC transport is trivial
        # (Sigma=I means W = ½ Frobenius up to the 1/2 factor; T^LC = I), and
        # the sheaf Laplacian reduces to (degree-graph-Laplacian) ⊗ I_m.
        # Constant sections ought to be in the kernel.
        m = n * (n + 1) // 2
        N = 4
        cov = torch.eye(n).unsqueeze(0).expand(N, n, n).contiguous()
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 0, 3, 2], [1, 0, 3, 2, 2, 3, 0, 1]],
            dtype=torch.long,
        )
        L = assemble_sheaf_laplacian(cov, edge_index, n_steps=10, rescale=True)
        # Constant section: same vector at every node.
        v0 = torch.randn(m)
        f = v0.unsqueeze(0).expand(N, m).reshape(-1)
        Lf = L @ f
        assert Lf.norm() < 1e-3


# ─── 4. Restricted-class projections ────────────────────────────────────────


class TestProjections:
    def test_project_to_circulant_is_circulant(self):
        E, T = 4, 6
        U = torch.linalg.qr(torch.randn(E, T, T))[0]
        proj = project_to_circulant(U)
        # First-row determines a circulant matrix; check P[i, j] == c[(i-j) mod T].
        for e in range(E):
            c = proj[e, :, 0]  # first column
            for i in range(T):
                for j in range(T):
                    assert torch.allclose(proj[e, i, j], c[(i - j) % T], atol=1e-5)

    def test_project_to_circulant_minimizes_residual(self):
        # The projection's Frobenius residual should be ≤ that of the identity
        # (a particular circulant) — necessary condition for "closest".
        E, T = 4, 8
        U = torch.linalg.qr(torch.randn(E, T, T))[0]
        proj = project_to_circulant(U)
        eye = torch.eye(T).unsqueeze(0).expand_as(U)
        res_proj = (U - proj).pow(2).sum().item()
        res_eye = (U - eye).pow(2).sum().item()
        assert res_proj <= res_eye + 1e-4

    def test_project_to_circulant_idempotent_when_in_class(self):
        # Construct a circulant orthogonal directly via the codebase's builder
        # (which pins DC + Nyquist to +1, so we must stay inside that subgroup
        # to expect exact idempotency).
        from hilbnet.circulant_transport import build_circulant_transport_maps
        E, T = 3, 7  # odd T to avoid Nyquist pinning
        num_free = (T - 1) // 2
        phases = torch.randn(E, num_free)
        P = build_circulant_transport_maps(phases, T)
        proj = project_to_circulant(P)
        assert (proj - P).norm() < 1e-3

    def test_project_to_identity_returns_eye(self):
        E, T = 3, 4
        target = torch.randn(E, T, T)
        proj = project_to_identity(target)
        eye = torch.eye(T).unsqueeze(0).expand(E, T, T)
        assert torch.allclose(proj, eye)

    def test_projection_distance_zero_when_equal(self):
        E, T = 5, 6
        target = torch.randn(E, T, T)
        d = projection_distance(target, target)
        assert d.item() < 1e-10

    def test_projection_distance_normalization(self):
        # Single-edge case: ‖A - B‖_F² / (1·T·T) = MSE per element.
        T = 4
        A = torch.zeros(1, T, T)
        B = torch.eye(T).unsqueeze(0)  # ‖A-B‖_F² = T (only diagonal differs by 1)
        d = projection_distance(A, B)
        assert torch.allclose(d, torch.tensor(1.0 / T))

# ─── 5. BundleTransportModule ────────────────────────────────────────────────


def _toy_edge_index(N: int = 5) -> torch.Tensor:
    src, dst = [], []
    for i in range(N):
        j = (i + 1) % N
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


class TestBundleTransportModule:
    @pytest.mark.parametrize("variant", ["direct", "circulant", "frozen_id"])
    def test_dispatch_and_shape(self, variant):
        T = 6
        edge_index = _toy_edge_index(N=4)
        E = edge_index.shape[1]
        kwargs = dict(num_reflections=T, num_bands=None)
        if variant == "circulant":
            kwargs["num_bands"] = (T - 1) // 2
        m = BundleTransportModule(variant, edge_index=edge_index, stalk_dim=T, **kwargs)
        out = m()
        assert out.shape == (E, T, T)

    def test_frozen_id_no_parameters(self):
        T = 5
        edge_index = _toy_edge_index(N=4)
        m = BundleTransportModule("frozen_id", edge_index=edge_index, stalk_dim=T)
        params = list(m.parameters())
        assert len(params) == 0
        out = m()
        eye = torch.eye(T).unsqueeze(0).expand_as(out)
        assert torch.allclose(out, eye)

    def test_circulant_at_init_orthogonal(self):
        T = 6
        edge_index = _toy_edge_index(N=4)
        m = BundleTransportModule(
            "circulant", edge_index=edge_index, stalk_dim=T,
            transport_init="identity_plus_noise",
        )
        out = m()
        eye = torch.eye(T).unsqueeze(0).expand_as(out)
        err = (out.transpose(-1, -2) @ out - eye).norm()
        assert err < 1e-3

    def test_direct_at_init_orthogonal(self):
        # Householder products are orthogonal by construction regardless of the
        # vectors' magnitudes. (init="identity_plus_noise" is named for its
        # non-zero gradient property, not for being near I: all-zero vectors
        # would get zero gradient.)
        T = 5
        edge_index = _toy_edge_index(N=4)
        m = BundleTransportModule(
            "direct", edge_index=edge_index, stalk_dim=T,
            num_reflections=T, transport_init="identity_plus_noise",
        )
        out = m()
        eye = torch.eye(T).unsqueeze(0).expand_as(out)
        err = (out.transpose(-1, -2) @ out - eye).norm()
        # householder_eps=1e-8 adds a small non-orthogonality when ||v|| is small;
        # 1e-2 is comfortably below "structurally non-orthogonal".
        assert err < 1e-2

    def test_trainable_params_exist_for_learnable_variants(self):
        T = 5
        edge_index = _toy_edge_index(N=4)
        for variant in ("direct", "circulant"):
            m = BundleTransportModule(
                variant, edge_index=edge_index, stalk_dim=T, num_reflections=T,
            )
            assert sum(p.numel() for p in m.parameters() if p.requires_grad) > 0


# ─── 6. End-to-end smoke tests of the experiment scripts ─────────────────────


class TestOperatorConvergenceSmoke:
    """Tiny in-process smoke for ``scripts/bundle_operator_convergence.py``.

    Exercises ``_sweep_seed`` directly so we don't pay subprocess cost; checks
    that spectral L2 to the reference decreases monotonically as N grows.
    """

    def test_sweep_seed_decreases_spectral_l2(self):
        from scripts.bundle_operator_convergence import _sweep_seed
        cfg = dict(
            n_dim=2,
            n_grid=[10, 20, 40],
            n_max=80,
            top_k=5,
            graph_type="knn",
            distance_type="wasserstein",
            k_neighbors=3,
            edge_weight_sigma=1.0,
            n_transport_steps=15,
            rescale=True,
        )
        rows = _sweep_seed(cfg, seed=0, device=torch.device("cpu"))
        assert len(rows) == len(cfg["n_grid"])
        spectral = [r["spectral_l2"] for r in rows]
        # spectral_l2 should drop by at least 2× from smallest N to largest N.
        assert spectral[-1] < spectral[0] * 0.5, (
            f"spectral_l2 did not decrease meaningfully: {spectral}"
        )
        section = [r["section_norm"] for r in rows]
        assert section[-1] < section[0], (
            f"section_norm did not decrease: {section}"
        )


class TestTransportRecoverySmoke:
    """Tiny in-process smoke for ``scripts/bundle_transport_recovery.py``.

    Validates the plateau structure behind Table 1: direct (free O(m)) reaches
    near zero, frozen-id matches its analytical projection distance.
    """

    def _cfg(self):
        return dict(
            n_dim=2,
            n_nodes=8,
            graph_type="knn",
            distance_type="wasserstein",
            k_neighbors=3,
            edge_weight_sigma=1.0,
            n_transport_steps=15,
            rescale=True,
            num_reflections=8,
            num_bands=None,
            transport_init="identity_plus_noise",
            epochs=1500,
            batch_size=64,
            lr=5.0e-3,
            weight_decay=0.0,
            log_every=10_000,            # silence per-epoch printing
        )

    def test_direct_converges_to_zero(self):
        from scripts.bundle_transport_recovery import _build_graph, _train_variant
        from hilbnet.bundle_validation import closed_form_lc_transports
        cfg = self._cfg()
        graph = _build_graph(cfg, n_nodes=cfg["n_nodes"], seed=0)
        T_gt = closed_form_lc_transports(
            graph.covariances, graph.edge_index,
            n_steps=cfg["n_transport_steps"], rescale=True,
        )
        out = _train_variant(
            "direct", T_gt, graph.edge_index, cfg,
            device=torch.device("cpu"), seed=0,
        )
        # Free Householder with m=3 + num_reflections=8 should drive loss to near zero.
        assert out["best_loss"] < 1e-4, f"direct best_loss={out['best_loss']}"
        # Deterministic closed-form counterpart should converge too, and can
        # never go below the class's floor (0 for the free class).
        assert out["best_frob"] < 1e-4, f"direct best_frob={out['best_frob']}"
        assert out["best_frob"] >= 0.0

    def test_frozen_id_matches_projection_theory(self):
        from scripts.bundle_transport_recovery import _build_graph, _project, _train_variant
        from hilbnet.bundle_validation import closed_form_lc_transports, projection_distance
        cfg = self._cfg()
        graph = _build_graph(cfg, n_nodes=cfg["n_nodes"], seed=0)
        T_gt = closed_form_lc_transports(
            graph.covariances, graph.edge_index,
            n_steps=cfg["n_transport_steps"], rescale=True,
        )
        # frozen_id has zero parameters; "training" reduces to a single
        # batch-evaluation of the loss.
        out = _train_variant(
            "frozen_id", T_gt, graph.edge_index, cfg,
            device=torch.device("cpu"), seed=0,
        )
        T_proj = _project(T_gt, "frozen_id", cfg)
        m = cfg["n_dim"] * (cfg["n_dim"] + 1) // 2
        theory = float(projection_distance(T_gt, T_proj).item()) * m
        # Single-batch estimator concentrates around the theory value.
        rel_err = abs(out["best_loss"] - theory) / max(theory, 1e-9)
        assert rel_err < 0.20, (
            f"frozen_id empirical={out['best_loss']} theory={theory} rel={rel_err}"
        )
        # The deterministic metric is the same formula as the theory plateau
        # evaluated at T_pred = I = T_proj, so it must match to float precision
        # — and in general can never fall below the plateau.
        rel_frob = abs(out["best_frob"] - theory) / max(theory, 1e-9)
        assert rel_frob < 1e-5, (
            f"frozen_id best_frob={out['best_frob']} theory={theory} rel={rel_frob}"
        )
        assert out["best_frob"] >= theory - 1e-9


class TestTrainingLossConsistency:
    """The empirical edge-MSE loss with random isotropic V should equal
    ``m · projection_distance(T_pred, T_target)`` in expectation. Verify on a
    fixed pair of (T_pred, T_target)."""

    def test_loss_equals_projection_distance(self):
        torch.manual_seed(123)
        E, T = 6, 5
        T_target = torch.linalg.qr(torch.randn(E, T, T))[0]
        T_pred = torch.linalg.qr(torch.randn(E, T, T))[0]

        # Simulate the transport-recovery training loss: random isotropic V_in per edge.
        B = 4096
        V_src = torch.randn(B, E, T)
        target = torch.einsum("emn,ben->bem", T_target, V_src)
        pred = torch.einsum("emn,ben->bem", T_pred, V_src)
        empirical = (pred - target).pow(2).mean().item()

        # Theoretical: under V ~ N(0, I), E[‖(P-Q) V‖²] / T = ‖P-Q‖_F² / (E·T·T) · T
        # The script's loss is `mean over (B, E, T)`, so:
        #   E[loss] = ‖T_pred - T_target‖_F² / (E · T)
        # projection_distance returns ‖.‖_F² / (E · T · T), so empirical ≈ proj * T.
        proj = projection_distance(T_pred, T_target).item()
        theoretical = proj * T

        # 4096 samples → standard error ~ 5%. Allow 10% for safety.
        assert abs(empirical - theoretical) / max(theoretical, 1e-9) < 0.10


# ─── 7. JSON persistence + plotting smoke ────────────────────────────────────


class TestJsonAndPlotting:
    """Round-trip tests for the JSON output of both scripts plus a smoke
    test for the plotting helpers in ``scripts/bundle_make_figures.py``.
    """

    def _operator_smoke_cfg(self) -> dict:
        return dict(
            n_dim_grid=[2],
            n_max_by_n_dim={2: 40},
            n_grid=[10, 20],
            top_k=4,
            graph_type="knn",
            distance_type="wasserstein",
            k_neighbors=3,
            edge_weight_sigma=1.0,
            n_transport_steps=10,
            rescale=True,
            seeds=[0],
        )

    def _transport_smoke_cfg(self) -> dict:
        return dict(
            n_dim=2,
            n_grid=[8, 16],
            graph_type="knn",
            distance_type="wasserstein",
            k_neighbors=3,
            edge_weight_sigma=1.0,
            n_transport_steps=10,
            rescale=True,
            variants=["direct", "frozen_id"],     # skip circulant to keep it fast
            num_reflections=4,
            num_bands=None,
            transport_init="identity_plus_noise",
            epochs=200,
            batch_size=32,
            lr=5.0e-3,
            weight_decay=0.0,
            log_every=10_000,
            seeds=[0],
        )

    def test_operator_convergence_writes_json(self, tmp_path):
        from scripts.bundle_operator_convergence import (
            _resolve_n_dim_grid,
            _sweep_seed,
            _dump_json,
        )
        cfg = self._operator_smoke_cfg()
        all_rows = []
        for seed in cfg["seeds"]:
            for n_dim, n_max in _resolve_n_dim_grid(cfg):
                sub_cfg = {**cfg, "n_dim": n_dim, "n_max": n_max}
                all_rows.extend(_sweep_seed(sub_cfg, seed, torch.device("cpu")))

        out_path = tmp_path / "operator.json"
        _dump_json(out_path, experiment="operator_convergence",
                   config_path="<inline>", cfg=cfg, rows=all_rows)

        with open(out_path) as f:
            payload = json.load(f)
        assert payload["experiment"] == "operator_convergence"
        # 1 seed × 1 n_dim × 2 N values
        assert len(payload["rows"]) == 2
        for r in payload["rows"]:
            for key in ("seed", "n_dim", "n_nodes",
                        "spectral_l2", "spectral_relmax", "section_norm"):
                assert key in r, f"missing key {key} in row {r}"

    def test_transport_recovery_writes_json(self, tmp_path):
        from scripts.bundle_transport_recovery import (
            _build_graph,
            _project,
            _train_variant,
            _resolve_n_grid,
            _dump_json,
        )
        from hilbnet.bundle_validation import (
            closed_form_lc_transports, projection_distance,
        )
        cfg = self._transport_smoke_cfg()
        n = cfg["n_dim"]
        m = n * (n + 1) // 2
        all_rows = []
        for seed in cfg["seeds"]:
            for n_nodes in _resolve_n_grid(cfg):
                graph = _build_graph(cfg, n_nodes=n_nodes, seed=seed)
                T_gt = closed_form_lc_transports(
                    graph.covariances, graph.edge_index,
                    n_steps=cfg["n_transport_steps"], rescale=True,
                )
                sub_cfg = {**cfg, "n_nodes": n_nodes}
                for variant in cfg["variants"]:
                    T_proj = _project(T_gt, variant, cfg)
                    theory = float(projection_distance(T_gt, T_proj).item()) * m
                    out = _train_variant(
                        variant, T_gt, graph.edge_index, sub_cfg,
                        torch.device("cpu"), seed=seed,
                    )
                    all_rows.append({
                        "seed": int(seed),
                        "n_dim": int(n),
                        "n_nodes": int(n_nodes),
                        "stalk_dim_m": int(m),
                        "num_edges": int(graph.edge_index.shape[1]),
                        "variant": variant,
                        "n_params": int(out["n_params"]),
                        "empirical": float(out["best_loss"]),
                        "final_loss": float(out["final_loss"]),
                        "empirical_frob": float(out["best_frob"]),
                        "final_frob": float(out["final_frob"]),
                        "theory": theory,
                    })

        out_path = tmp_path / "transport.json"
        _dump_json(out_path, experiment="transport_recovery",
                   config_path="<inline>", cfg=cfg, rows=all_rows)

        with open(out_path) as f:
            payload = json.load(f)
        assert payload["experiment"] == "transport_recovery"
        # 1 seed × 2 N × 2 variants
        assert len(payload["rows"]) == 4
        for r in payload["rows"]:
            for key in ("seed", "n_dim", "n_nodes", "variant",
                        "empirical", "empirical_frob", "theory"):
                assert key in r, f"missing key {key} in row {r}"
            # The deterministic metric never dips below the analytical floor.
            assert r["empirical_frob"] >= r["theory"] - 1e-9, r

    def test_make_figures_smoke(self, tmp_path):
        from scripts.bundle_make_figures import (
            fig_operator_convergence,
            _setup_paper_style,
        )
        _setup_paper_style()

        op_rows = [
            {"seed": s, "n_dim": d, "n_nodes": N,
             "spectral_l2": 1.0 / N, "spectral_relmax": 2.0 / N,
             "section_norm": 0.5 / N}
            for s in (0, 1) for d in (2, 3) for N in (10, 50, 250)
        ]
        op_pdf = tmp_path / "operator.pdf"
        fig_operator_convergence(op_rows, op_pdf)
        assert op_pdf.exists() and op_pdf.stat().st_size > 1024
        # TrueType embedding (pdf.fonttype 42): no Type 3 fonts in the figure.
        assert b"/Type3" not in op_pdf.read_bytes()
