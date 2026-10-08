"""End-to-end test of ``scripts/traffic_table_runner.py`` on a tiny synthetic dataset.

The real loader is replaced by a fake one (6 nodes on a ring, a few dozen
windows), so the test runs on CPU in seconds without the METR-LA files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.traffic_table_runner as runner  # noqa: E402

N_NODES = 6
ARCH_ORDER = ["circulant", "frozen_id", "free", "mlp_fiber", "stgnn_conv"]
# Learnable parameters on a ring with E = 6 undirected edges: the backbone has
# 5,756; circulant adds 5 phases per edge, free O(T) adds 8 x 12 = 96 per edge.
EXPECTED_PARAMS = {"circulant": 5756 + 5 * 6, "frozen_id": 5756, "free": 5756 + 96 * 6,
                   "mlp_fiber": 5212, "stgnn_conv": 8908}


def _fake_split(split, **kwargs):
    gen = torch.Generator().manual_seed({"train": 0, "val": 1, "test": 2}[split])
    windows = []
    for _ in range({"train": 40, "val": 8, "test": 8}[split]):
        y = 50.0 + 10.0 * torch.randn(N_NODES, 12, generator=gen)
        windows.append(Data(x=torch.randn(N_NODES, 12, 2, generator=gen), y=y,
                            mask=(y > 0).float(), num_nodes=N_NODES))
    ring = torch.arange(N_NODES)
    edge_index = torch.stack([ring, (ring + 1) % N_NODES])
    edge_attr = torch.full((N_NODES,), 0.5)
    return windows, edge_index, edge_attr, {"mean": 50.0, "std": 10.0}


def _paper_results_digest() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "results" / "paper").glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def _run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(runner, "load_traffic_split", _fake_split)
    monkeypatch.setattr(sys, "argv", ["traffic_table_runner.py", *args])
    runner.main()


def test_runner_trains_all_five_models(tmp_path, monkeypatch):
    before = _paper_results_digest()
    out = tmp_path / "table.json"
    _run(monkeypatch, "--seeds", "0", "--epochs", "1", "--device", "cpu",
         "--output_json", str(out))
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "metr-la"
    assert payload["horizons"] == [3, 6, 12]
    assert [r["arch"] for r in payload["rows"]] == ARCH_ORDER
    metric_keys = {f"{m}_{h}" for m in ("mae", "rmse", "mape") for h in (3, 6, 12)}
    for row in payload["rows"]:
        assert row["seed"] == 0
        assert row["params"] == EXPECTED_PARAMS[row["arch"]]
        assert set(row["metrics"]) == metric_keys
        assert all(np.isfinite(v) for v in row["metrics"].values())
    # The shipped reference results are never touched by a run.
    assert _paper_results_digest() == before


def test_runner_resumes_without_retraining(tmp_path, monkeypatch):
    out = tmp_path / "table.json"
    _run(monkeypatch, "--archs", "mlp_fiber", "--seeds", "0", "--epochs", "1",
         "--device", "cpu", "--output_json", str(out))
    first = out.read_text()

    def _no_training(*args, **kwargs):
        raise AssertionError("a completed (arch, seed) run was trained again")

    monkeypatch.setattr(runner, "train_and_forecast", _no_training)
    _run(monkeypatch, "--archs", "mlp_fiber", "--seeds", "0", "--epochs", "1",
         "--device", "cpu", "--output_json", str(out))
    assert out.read_text() == first


def test_short_runs_never_write_the_full_run_file(tmp_path, monkeypatch):
    # Point the runner at a scratch copy of the repo layout.
    shutil.copytree(ROOT / "scripts" / "configs", tmp_path / "scripts" / "configs")
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    _run(monkeypatch, "--archs", "mlp_fiber", "--seeds", "0", "--epochs", "1",
         "--device", "cpu")
    assert (tmp_path / "results" / "traffic" / "metr-la_table_epochs1.json").exists()
    assert not (tmp_path / "results" / "traffic" / "metr-la_table.json").exists()
