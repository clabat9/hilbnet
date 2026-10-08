"""Tests that tie the code to the numbers and definitions in the paper.

1. The sheaf Laplacian behind Figure 3 uses the paper's orientation
   (midpoint construction).
2. Transport-recovery runs do not depend on Python's per-process hash salt.
3. The Theory column of Table 1 is recomputed exactly from the code.
4. Every Params cell of Table 2 is the learnable-parameter count of the
   shipped traffic configs.
5. The shipped reference results aggregate to the paper's Table 1 and Table 2.
6. The configs match the paper's hyperparameter tables and the configs
   recorded inside the reference results.
7. Figure 3 renders from the shipped reference results.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hilbnet.bundle_validation import (  # noqa: E402
    BundleTransportModule,
    assemble_sheaf_laplacian,
    closed_form_lc_transports,
    projection_distance,
)
from hilbnet.statistical_bundle import (  # noqa: E402
    matrix_sqrt,
    matrix_sqrt_inv,
    sample_uniform_spd,
)

PAPER_RESULTS = ROOT / "results" / "paper"
CONFIGS = ROOT / "scripts" / "configs"
TRAFFIC_ARCHS = ("circulant", "free", "frozen_id", "mlp_fiber", "stgnn_conv")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── 1. Orientation of the Figure 3 operator ─────────────────────────────────


class TestLaplacianOrientation:
    def _pair(self, n: int) -> torch.Tensor:
        """Two random SPD matrices, shape (2, n, n)."""
        torch.manual_seed(0)
        return sample_uniform_spd(n, 2)

    def test_block_uses_transport_from_dst_to_src(self):
        n, w = 3, 0.7
        m = n * (n + 1) // 2
        cov = self._pair(n)
        L = assemble_sheaf_laplacian(cov, torch.tensor([[0], [1]]),
                                     edge_weights=torch.tensor([w]), n_steps=50)
        block_01 = L.reshape(2, m, 2, m).permute(0, 2, 1, 3)[0, 1]
        P_1to0 = closed_form_lc_transports(cov, torch.tensor([[1], [0]]), n_steps=50)[0]
        P_0to1 = closed_form_lc_transports(cov, torch.tensor([[0], [1]]), n_steps=50)[0]
        assert torch.allclose(block_01, -w * P_1to0, atol=1e-6)
        # The check is meaningful: the two orientations differ for this pair.
        assert (P_1to0 - P_0to1).norm() > 1e-2

        # Quadratic form: f^T L f = w (|f0|² − 2 f0·P_{1→0} f1 + |f1|²)
        # (assume_orthogonal=True puts w·I on both diagonal blocks).
        f = torch.randn(2, m)
        energy = f.reshape(-1) @ L @ f.reshape(-1)
        expected = w * (f[0].dot(f[0]) - 2 * f[0].dot(P_1to0 @ f[1]) + f[1].dot(f[1]))
        assert torch.allclose(energy, expected, rtol=1e-4, atol=1e-5)

    def test_block_matches_midpoint_construction(self):
        # Paper: restriction maps transport each endpoint to the midpoint m_e of
        # the Wasserstein geodesic, so block (0, 1) = -F_0^T F_1.
        n, steps = 3, 100
        m = n * (n + 1) // 2
        S0, S1 = self._pair(n)
        A, A_inv = matrix_sqrt(S0), matrix_sqrt_inv(S0)
        T_ot = A_inv @ matrix_sqrt(A @ S1 @ A) @ A_inv     # T Σ0 T = Σ1
        M = 0.5 * (torch.eye(n) + T_ot)
        S_mid = M @ S0 @ M.T                                  # geodesic midpoint
        F = closed_form_lc_transports(torch.stack([S0, S1, S_mid]),
                                      torch.tensor([[0, 1], [2, 2]]), n_steps=steps)
        midpoint_block = -(F[0].T @ F[1])

        L = assemble_sheaf_laplacian(torch.stack([S0, S1]), torch.tensor([[0], [1]]),
                                     n_steps=steps)
        block_01 = L.reshape(2, m, 2, m).permute(0, 2, 1, 3)[0, 1]
        assert (block_01 - midpoint_block).norm() < 1e-3


# ─── 2. Seeds do not depend on the Python hash salt ──────────────────────────


_TINY_RUN = textwrap.dedent("""
    import json, sys, torch
    sys.path.insert(0, {root!r})
    from scripts.bundle_transport_recovery import _build_graph, _train_variant
    from hilbnet.bundle_validation import closed_form_lc_transports
    cfg = dict(n_dim=2, n_nodes=8, graph_type="knn", distance_type="wasserstein",
               k_neighbors=3, edge_weight_sigma=1.0, n_transport_steps=10, rescale=True,
               num_reflections=4, num_bands=None, transport_init="identity_plus_noise",
               epochs=30, batch_size=16, lr=5e-3, weight_decay=0.0, log_every=10**9)
    g = _build_graph(cfg, n_nodes=8, seed=0)
    T = closed_form_lc_transports(g.covariances, g.edge_index, n_steps=10)
    out = {{v: _train_variant(v, T, g.edge_index, cfg, device=torch.device("cpu"), seed=0)
           for v in ("direct", "circulant")}}
    print("RESULT " + json.dumps({{v: [o["final_loss"], o["final_frob"]] for v, o in out.items()}}))
""")


def _tiny_run(hash_seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", _TINY_RUN.format(root=str(ROOT))],
                          capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_transport_recovery_seed_is_process_independent():
    assert _tiny_run("1") == _tiny_run("2")


# ─── 3. Table 1, Theory column ───────────────────────────────────────────────


def test_table1_theory_recomputed_from_code():
    """n=16 (all seeds and variants): edges, parameters and Theory match the
    shipped reference results. The other n values use the same code path."""
    from scripts.bundle_transport_recovery import _build_graph, _project

    cfg = _load_yaml(CONFIGS / "synthetic" / "transport_recovery_paper.yaml")
    ref = _load_json(PAPER_RESULTS / "transport_recovery.json")
    rows = {(r["seed"], r["n_nodes"], r["variant"]): r for r in ref["rows"]}
    n = cfg["n_dim"]
    m = n * (n + 1) // 2
    N = 16
    for seed in cfg["seeds"]:
        g = _build_graph(cfg, n_nodes=N, seed=seed * 1009 + N)
        T_gt = closed_form_lc_transports(g.covariances, g.edge_index,
                                         n_steps=cfg["n_transport_steps"],
                                         rescale=cfg["rescale"])
        for variant in cfg["variants"]:
            r = rows[(seed, N, variant)]
            assert g.edge_index.shape[1] == r["num_edges"]
            theory = float(projection_distance(T_gt, _project(T_gt, variant, cfg)).item()) * m
            assert theory == pytest.approx(r["theory"], rel=1e-5, abs=1e-12)
            module = BundleTransportModule(variant, edge_index=g.edge_index, stalk_dim=m,
                                           num_reflections=cfg["num_reflections"],
                                           num_bands=cfg["num_bands"])
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            assert n_params == r["n_params"]


# ─── 4. Table 2, Params column ───────────────────────────────────────────────


TABLE2_PARAMS = {
    # dataset: (n_nodes, undirected edges E, {arch: learnable params})
    "metr-la": (207, 1180, {"mlp_fiber": 5212, "stgnn_conv": 8908, "frozen_id": 5756,
                            "circulant": 11656, "free": 119036}),
    "pems-bay": (325, 1922, {"mlp_fiber": 5212, "stgnn_conv": 8908, "frozen_id": 5756,
                             "circulant": 15366, "free": 190268}),
}


def _random_graph(n_nodes: int, n_edges: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    pairs = set()
    while len(pairs) < n_edges:
        i, j = torch.randint(0, n_nodes, (2,), generator=gen).tolist()
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    edge_index = torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
    return edge_index, torch.rand(n_edges, generator=gen) + 0.1


@pytest.mark.parametrize("dataset", sorted(TABLE2_PARAMS))
def test_table2_params_match_configs(dataset):
    from scripts.traffic_eval import build_forecasting_model

    n_nodes, n_edges, expected = TABLE2_PARAMS[dataset]
    edge_index, edge_attr = _random_graph(n_nodes, n_edges)
    for arch in TRAFFIC_ARCHS:
        cfg = _load_yaml(CONFIGS / "traffic" / f"{arch}.yaml")
        model, _ = build_forecasting_model(cfg, n_nodes, 12, 12, edge_index, edge_attr)
        learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert learnable == expected[arch], (dataset, arch, learnable)


# ─── 5. Shipped reference results reproduce the paper's tables ───────────────


# Table 1: (mantissa, ±mantissa, exponent) for Det., and (mantissa, exponent) for Theory.
TABLE1 = {
    "direct": {16: ((1.71, 0.17, -7), (0.0, 0)), 32: ((1.28, 0.57, -7), (0.0, 0)),
               64: ((1.82, 0.47, -7), (0.0, 0)), 128: ((2.22, 0.25, -7), (0.0, 0)),
               256: ((1.87, 0.35, -7), (0.0, 0))},
    "circulant": {16: ((1.84, 0.29, -2), (1.84, -2)), 32: ((1.18, 0.14, -2), (1.18, -2)),
                  64: ((1.03, 0.14, -2), (1.03, -2)), 128: ((8.94, 0.71, -3), (8.93, -3)),
                  256: ((7.80, 0.12, -3), (7.80, -3))},
    "frozen_id": {16: ((2.30, 0.31, -2), (2.30, -2)), 32: ((1.46, 0.18, -2), (1.46, -2)),
                  64: ((1.29, 0.16, -2), (1.29, -2)), 128: ((1.11, 0.09, -2), (1.11, -2)),
                  256: ((9.67, 0.18, -3), (9.67, -3))},
}


def _matches_printed(value: float, printed: float, exponent: int) -> bool:
    """True if ``value`` rounds to ``printed × 10^exponent`` at two decimals."""
    return abs(value / 10.0 ** exponent - printed) <= 0.005 + 1e-9


def test_reference_results_match_table1():
    ref = _load_json(PAPER_RESULTS / "transport_recovery.json")
    assert len(ref["rows"]) == 45
    for variant, by_n in TABLE1.items():
        for n_nodes, ((det_m, det_s, det_e), (th_m, th_e)) in by_n.items():
            rows = [r for r in ref["rows"] if r["variant"] == variant and r["n_nodes"] == n_nodes]
            assert sorted(r["seed"] for r in rows) == [0, 1, 2]
            det = np.array([r["empirical_frob"] for r in rows])
            theory = np.array([r["theory"] for r in rows])
            assert _matches_printed(det.mean(), det_m, det_e), (variant, n_nodes, det.mean())
            assert _matches_printed(det.std(ddof=1), det_s, det_e), (variant, n_nodes, det.std(ddof=1))
            assert _matches_printed(theory.mean(), th_m, th_e), (variant, n_nodes, theory.mean())
            # Det. never falls below Theory (up to float32 rounding).
            assert (det >= theory - 1e-8).all()


# Table 2: {arch: [(MAE, RMSE, MAPE%) means at horizons 3, 6, 12]} and the
# matching sample standard deviations, as printed in the paper.
TABLE2 = {
    "metr-la": {
        "mlp_fiber": ([(3.131, 6.074, 8.271), (3.775, 7.496, 10.626), (4.690, 9.184, 14.341)],
                      [(0.004, 0.007, 0.021), (0.005, 0.013, 0.046), (0.011, 0.014, 0.097)]),
        "stgnn_conv": ([(3.453, 6.709, 9.241), (4.160, 8.158, 11.916), (5.277, 10.102, 16.006)],
                       [(0.080, 0.220, 0.279), (0.117, 0.184, 0.491), (0.093, 0.128, 0.662)]),
        "frozen_id": ([(3.092, 5.920, 8.218), (3.713, 7.312, 10.520), (4.608, 8.991, 14.166)],
                      [(0.007, 0.013, 0.065), (0.010, 0.020, 0.061), (0.034, 0.043, 0.240)]),
        "circulant": ([(2.939, 5.630, 7.908), (3.409, 6.765, 9.844), (4.059, 8.149, 12.471)],
                      [(0.021, 0.061, 0.067), (0.032, 0.092, 0.125), (0.049, 0.114, 0.195)]),
        "free": ([(2.923, 5.586, 7.808), (3.372, 6.732, 9.507), (3.938, 8.042, 11.642)],
                 [(0.013, 0.048, 0.083), (0.023, 0.066, 0.096), (0.030, 0.101, 0.136)]),
    },
    "pems-bay": {
        "mlp_fiber": ([(1.459, 3.145, 3.026), (1.942, 4.378, 4.385), (2.513, 5.658, 6.209)],
                      [(0.003, 0.017, 0.020), (0.004, 0.017, 0.045), (0.004, 0.015, 0.038)]),
        "stgnn_conv": ([(1.400, 2.980, 2.924), (1.850, 4.162, 4.222), (2.388, 5.395, 5.924)],
                       [(0.002, 0.016, 0.003), (0.002, 0.009, 0.009), (0.004, 0.023, 0.032)]),
        "frozen_id": ([(1.439, 3.077, 2.985), (1.901, 4.264, 4.319), (2.446, 5.495, 6.020)],
                      [(0.003, 0.022, 0.007), (0.006, 0.021, 0.015), (0.007, 0.019, 0.032)]),
        "circulant": ([(1.413, 2.971, 2.982), (1.806, 3.950, 4.162), (2.211, 4.866, 5.386)],
                      [(0.002, 0.018, 0.015), (0.003, 0.015, 0.037), (0.014, 0.048, 0.047)]),
        "free": ([(1.417, 2.969, 3.058), (1.793, 3.958, 4.127), (2.181, 4.873, 5.214)],
                 [(0.002, 0.015, 0.013), (0.005, 0.026, 0.037), (0.003, 0.019, 0.045)]),
    },
}


@pytest.mark.parametrize("dataset", sorted(TABLE2))
def test_reference_results_match_table2(dataset):
    ref = _load_json(PAPER_RESULTS / f"{dataset}.json")
    assert ref["dataset"] == dataset
    assert ref["horizons"] == [3, 6, 12]
    assert len(ref["rows"]) == 25
    for arch, (means, stds) in TABLE2[dataset].items():
        rows = [r for r in ref["rows"] if r["arch"] == arch]
        assert sorted(r["seed"] for r in rows) == [0, 1, 2, 3, 4]
        assert {r["params"] for r in rows} == {TABLE2_PARAMS[dataset][2][arch]}
        for h, mean_triplet, std_triplet in zip((3, 6, 12), means, stds):
            for metric, scale, mean_p, std_p in zip(("mae", "rmse", "mape"), (1, 1, 100),
                                                    mean_triplet, std_triplet):
                vals = scale * np.array([r["metrics"][f"{metric}_{h}"] for r in rows])
                assert abs(vals.mean() - mean_p) <= 0.0005 + 1e-9, (arch, metric, h, vals.mean())
                assert abs(vals.std(ddof=1) - std_p) <= 0.0005 + 1e-9, (arch, metric, h, vals.std(ddof=1))


# ─── 6. Configs match the paper's hyperparameter tables ──────────────────────


def test_traffic_configs_match_paper_hyperparameters():
    # Paper, appendix table "Traffic forecasting hyperparameters".
    for arch in TRAFFIC_ARCHS:
        cfg = _load_yaml(CONFIGS / "traffic" / f"{arch}.yaml")
        assert cfg["window_in"] == 12 and cfg["window_out"] == 12
        assert cfg["add_time_of_day"] is True
        assert cfg["threshold_k"] == 0.1
        assert cfg["features"] == [2, 16, 32]
        assert cfg["batch_size"] == 32
        assert cfg["epochs"] == 150 and cfg["patience"] == 20
        assert cfg["weight_decay"] == 0.0 and cfg["dropout"] == 0.0
        assert cfg["clip_grad_norm"] == 5.0
        assert cfg["kappa"] == ([1, 1] if arch == "mlp_fiber" else [2, 2])
        assert cfg["lr"] == (1e-3 if arch == "stgnn_conv" else 5e-3)
    free = _load_yaml(CONFIGS / "traffic" / "free.yaml")
    assert free["transport_param"] == "direct" and free["num_reflections"] == 8
    assert free["reg_param"] == 0.01
    circ = _load_yaml(CONFIGS / "traffic" / "circulant.yaml")
    assert circ["transport_param"] == "circulant" and circ["num_bands"] is None
    assert circ["reg_param"] == 0.01
    frozen = _load_yaml(CONFIGS / "traffic" / "frozen_id.yaml")
    assert frozen["transport_init"] == "identity" and frozen["transport_lr"] == 0


def test_synthetic_configs_match_paper_hyperparameters():
    # Paper, appendix table "Synthetic experiments: hyperparameters".
    tr = _load_yaml(CONFIGS / "synthetic" / "transport_recovery_paper.yaml")
    assert tr["n_dim"] == 4 and tr["n_grid"] == [16, 32, 64, 128, 256]
    assert tr["graph_type"] == "knn" and tr["k_neighbors"] == 8
    assert tr["distance_type"] == "wasserstein"
    assert tr["seeds"] == [0, 1, 2]
    assert tr["epochs"] == 5000 and tr["patience"] == 600
    assert tr["lr"] == 5e-3 and tr["batch_size"] == 256
    assert tr["num_reflections"] == 16 and tr["n_transport_steps"] == 50
    assert tr["variants"] == ["direct", "circulant", "frozen_id"]
    op = _load_yaml(CONFIGS / "synthetic" / "operator_convergence_paper.yaml")
    assert op["n_dim_grid"] == [2, 3, 4]
    assert op["n_grid"] == [50, 100, 200, 400, 800]
    assert op["n_max_by_n_dim"] == {2: 4000, 3: 2000, 4: 1000}
    assert op["graph_type"] == "knn" and op["k_neighbors"] == 8
    assert op["top_k"] == 32 and op["seeds"] == [0, 1, 2]
    assert op["n_transport_steps"] == 50


@pytest.mark.parametrize("name", ["transport_recovery", "operator_convergence"])
def test_reference_results_were_produced_by_the_shipped_configs(name):
    ref = _load_json(PAPER_RESULTS / f"{name}.json")
    shipped = _load_yaml(CONFIGS / "synthetic" / f"{name}_paper.yaml")
    # JSON turns integer dict keys into strings; compare in JSON form.
    assert ref["config"] == json.loads(json.dumps(shipped))
    assert ref["config_path"] == f"scripts/configs/synthetic/{name}_paper.yaml"


# ─── 7. Figure 3 renders from the shipped reference ──────────────────────────


def test_figure3_renders_from_reference(tmp_path, monkeypatch):
    import scripts.bundle_make_figures as figs

    closed = []
    monkeypatch.setattr(figs.plt, "close", lambda fig: closed.append(fig))
    monkeypatch.setattr(sys, "argv", ["bundle_make_figures.py", "--operator",
                                      str(PAPER_RESULTS / "operator_convergence.json"),
                                      "--out_dir", str(tmp_path)])
    figs.main()
    pdf = tmp_path / "operator_convergence.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1024
    assert b"/Type3" not in pdf.read_bytes()
    legend = [t.get_text() for t in closed[0].axes[0].get_legend().get_texts()]
    assert legend == ["$d=3$", "$d=6$", "$d=10$"]
