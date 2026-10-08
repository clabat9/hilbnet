"""Checks on the real METR-LA / PEMS-BAY files. Skipped unless the data is in
``data/traffic/`` (see the README for the download steps).

They pin the graph and the splits used in the paper: node and edge counts,
the 70/10/20 split, the number of 12-in / 12-out windows per split, and the
learnable parameter counts of Table 2 on the real graphs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "traffic"

CASES = {
    "metr-la": dict(
        files=("metr-la.h5", "distances_la_2012.csv", "graph_sensor_ids.txt"),
        n_nodes=207, n_edges=1180, n_steps=34272, windows=(23967, 3405, 6831),
        params={"mlp_fiber": 5212, "stgnn_conv": 8908, "frozen_id": 5756,
                "circulant": 11656, "free": 119036},
    ),
    "pems-bay": dict(
        files=("pems-bay.h5", "distances_bay_2017.csv", "graph_sensor_ids_bay.txt"),
        n_nodes=325, n_edges=1922, n_steps=52116, windows=(36458, 5189, 10400),
        params={"mlp_fiber": 5212, "stgnn_conv": 8908, "frozen_id": 5756,
                "circulant": 15366, "free": 190268},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_graph_splits_and_params(name):
    case = CASES[name]
    folder = DATA / name
    if not all((folder / f).exists() for f in case["files"]):
        pytest.skip(f"{name} data not found in {folder} (see the README, section 'Table 2: traffic forecasting')")

    from hilbnet.traffic_loader import (_build_adjacency, _read_sensor_ids,
                                        _read_speed_table, _temporal_split_indices)
    from scripts.traffic_eval import build_forecasting_model

    h5, distances, sensor_ids = (folder / f for f in case["files"])
    ids = _read_sensor_ids(sensor_ids)
    assert len(ids) == case["n_nodes"]
    edge_index, edge_attr = _build_adjacency(distances, ids, threshold_k=0.1)
    assert edge_index.shape[1] == case["n_edges"]

    speeds, _ = _read_speed_table(h5)
    assert speeds.shape == (case["n_steps"], case["n_nodes"])
    i_train, i_val = _temporal_split_indices(case["n_steps"], 0.7, 0.1)
    split_lengths = (i_train, i_val - i_train, case["n_steps"] - i_val)
    # Stride-1 windows of 12 input + 12 output steps inside each split.
    assert tuple(length - 24 + 1 for length in split_lengths) == case["windows"]

    for arch, expected in case["params"].items():
        with open(ROOT / "scripts" / "configs" / "traffic" / f"{arch}.yaml") as f:
            cfg = yaml.safe_load(f)
        model, _ = build_forecasting_model(cfg, case["n_nodes"], 12, 12, edge_index, edge_attr)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected, arch
