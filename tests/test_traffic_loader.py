"""Unit tests for hilbnet/traffic_loader.py.

Pure-function tests (split arithmetic, adjacency, windowing) run unmocked.
The end-to-end loader test mocks the speed-table and sensor-id reads and
writes a tiny distances CSV to a tmp dir, so no downloaded dataset is required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilbnet import traffic_loader  # noqa: E402


# ─── Pure helpers ────────────────────────────────────────────────────────────


class TestTemporalSplitIndices(unittest.TestCase):
    def test_seventy_ten_twenty(self):
        i_train, i_val = traffic_loader._temporal_split_indices(
            n_steps=1000, train_frac=0.7, val_frac=0.1,
        )
        self.assertEqual(i_train, 700)
        self.assertEqual(i_val, 800)

    def test_no_overlap_full_coverage(self):
        n = 34272  # METR-LA-sized
        i_train, i_val = traffic_loader._temporal_split_indices(n, 0.7, 0.1)
        # Train [0, i_train), val [i_train, i_val), test [i_val, n).
        self.assertLess(i_train, i_val)
        self.assertLess(i_val, n)
        self.assertEqual(i_train + (i_val - i_train) + (n - i_val), n)

    def test_invalid_fractions_raise(self):
        with self.assertRaises(ValueError):
            traffic_loader._temporal_split_indices(100, 0.7, 0.4)  # sum ≥ 1
        with self.assertRaises(ValueError):
            traffic_loader._temporal_split_indices(100, -0.1, 0.1)
        with self.assertRaises(ValueError):
            traffic_loader._temporal_split_indices(100, 0.5, -0.1)


class TestMakeWindows(unittest.TestCase):
    def test_window_count_matches_arithmetic(self):
        # T=50, window_in=12, window_out=12 → 50 - 24 + 1 = 27 windows
        speeds_norm = np.zeros((50, 5), dtype=np.float32)
        speeds_raw = np.ones((50, 5), dtype=np.float32) * 30.0  # all > 0
        windows = traffic_loader._make_windows(
            speeds_norm, speeds_raw, time_of_day=None,
            window_in=12, window_out=12, n_nodes=5,
        )
        self.assertEqual(len(windows), 27)

    def test_x_y_shapes_no_tod(self):
        speeds_norm = np.zeros((40, 7), dtype=np.float32)
        speeds_raw = np.ones((40, 7), dtype=np.float32) * 30.0
        windows = traffic_loader._make_windows(
            speeds_norm, speeds_raw, time_of_day=None,
            window_in=12, window_out=6, n_nodes=7,
        )
        w = windows[0]
        self.assertEqual(tuple(w.x.shape), (7, 12, 1))    # F_in=1
        self.assertEqual(tuple(w.y.shape), (7, 6))
        self.assertEqual(tuple(w.mask.shape), (7, 6))
        self.assertEqual(w.x.dtype, torch.float32)
        self.assertEqual(w.y.dtype, torch.float32)

    def test_x_shape_with_tod(self):
        T, N = 40, 4
        speeds_norm = np.zeros((T, N), dtype=np.float32)
        speeds_raw = np.ones((T, N), dtype=np.float32) * 30.0
        tod = np.linspace(0, 1, T, dtype=np.float32, endpoint=False)
        windows = traffic_loader._make_windows(
            speeds_norm, speeds_raw, time_of_day=tod,
            window_in=8, window_out=4, n_nodes=N,
        )
        w = windows[0]
        self.assertEqual(tuple(w.x.shape), (N, 8, 2))     # F_in=2
        # Time-of-day should be the same value across nodes within a timestep.
        np.testing.assert_allclose(
            w.x[:, :, 1].numpy(),
            np.broadcast_to(tod[:8], (N, 8)),
            atol=1e-6,
        )

    def test_mask_zero_where_missing(self):
        # Inject zeros (missing) into raw targets; mask must mirror that.
        T, N = 30, 3
        speeds_norm = np.zeros((T, N), dtype=np.float32)
        speeds_raw = np.ones((T, N), dtype=np.float32) * 30.0
        speeds_raw[20, 1] = 0.0  # one missing entry in target window of first window
        windows = traffic_loader._make_windows(
            speeds_norm, speeds_raw, time_of_day=None,
            window_in=12, window_out=12, n_nodes=N,
        )
        # First window: input = [0, 12), output = [12, 24). Position 20 → t_out=8.
        w0 = windows[0]
        self.assertEqual(w0.mask[1, 8].item(), 0.0)
        # Other positions all 1.
        self.assertEqual(w0.mask.sum().item(), N * 12 - 1)

    def test_too_short_returns_empty(self):
        speeds_norm = np.zeros((10, 5), dtype=np.float32)
        speeds_raw = np.zeros((10, 5), dtype=np.float32)
        windows = traffic_loader._make_windows(
            speeds_norm, speeds_raw, time_of_day=None,
            window_in=12, window_out=12, n_nodes=5,
        )
        self.assertEqual(windows, [])


class TestBuildAdjacency(unittest.TestCase):
    """`_build_adjacency` reads a CSV; write a tiny one to a tmpfile."""

    def _run(self, edges, sensor_ids, threshold_k=0.1):
        import tempfile
        import pandas as pd
        df = pd.DataFrame(edges, columns=["from", "to", "cost"])
        df["from"] = df["from"].astype(str)
        df["to"] = df["to"].astype(str)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            df.to_csv(fh, index=False)
            tmp_path = Path(fh.name)
        try:
            return traffic_loader._build_adjacency(
                tmp_path, sensor_ids, threshold_k=threshold_k,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_symmetric_no_self_loops(self):
        # 4 sensors, a few directed edges. After symmetrization edges are undirected.
        edges = [
            ("a", "b", 100.0),
            ("b", "a", 100.0),
            ("b", "c", 200.0),
            ("c", "d", 50.0),  # very short → high weight
        ]
        edge_index, edge_attr = self._run(edges, sensor_ids=["a", "b", "c", "d"])
        # No self-loops.
        self.assertTrue((edge_index[0] != edge_index[1]).all().item())
        # Returned edges are upper-triangle (i < j), so symmetric by construction.
        self.assertTrue((edge_index[0] < edge_index[1]).all().item())
        # Edge weights in (0, 1].
        self.assertTrue((edge_attr > 0).all().item())
        self.assertTrue((edge_attr <= 1.0 + 1e-6).all().item())

    def test_threshold_drops_far_edges(self):
        # Two edges with very different costs; high threshold should drop the far one.
        edges = [
            ("a", "b", 1.0),     # very close → kept
            ("a", "c", 1e6),     # far → tiny weight → dropped if κ > 0.1
        ]
        edge_index, edge_attr = self._run(edges, sensor_ids=["a", "b", "c"], threshold_k=0.1)
        # Only the (a, b) edge should survive — index pair (0, 1).
        self.assertEqual(edge_index.shape[1], 1)
        self.assertEqual(int(edge_index[0]), 0)
        self.assertEqual(int(edge_index[1]), 1)


class TestReadSensorIds(unittest.TestCase):
    def test_one_per_line(self):
        with patch("pathlib.Path.read_text", return_value="a\nb\nc\n"):
            ids = traffic_loader._read_sensor_ids(Path("/fake.txt"))
        self.assertEqual(ids, ["a", "b", "c"])

    def test_comma_separated_single_line(self):
        with patch("pathlib.Path.read_text", return_value="a,b,c"):
            ids = traffic_loader._read_sensor_ids(Path("/fake.txt"))
        self.assertEqual(ids, ["a", "b", "c"])


# ─── End-to-end loader (all I/O mocked) ──────────────────────────────────────


def _make_mock_speed_table(n_steps: int, n_sensors: int):
    """Synthesize a (speeds, timestamps) pair for the speed-table mock."""
    rng = np.random.RandomState(0)
    speeds = rng.uniform(20, 70, size=(n_steps, n_sensors)).astype(np.float32)
    # Inject some zeros (missing) so the mask logic is exercised.
    speeds[rng.rand(n_steps, n_sensors) < 0.01] = 0.0
    # 5-minute-spaced timestamps starting from a fixed midnight.
    start = np.datetime64("2026-01-01T00:00:00", "ns")
    step = np.timedelta64(5, "m").astype("timedelta64[ns]")
    timestamps = start + step * np.arange(n_steps)
    return speeds, timestamps


def _make_mock_distances_df(n_sensors: int):
    """Synthesize a `from,to,cost` DataFrame compatible with `_build_adjacency`."""
    import pandas as pd

    rng = np.random.RandomState(0)
    rows = []
    for i in range(n_sensors):
        for j in range(n_sensors):
            if i == j:
                continue
            cost = float(rng.uniform(50, 1000))
            rows.append((str(i), str(j), cost))
    df = pd.DataFrame(rows, columns=["from", "to", "cost"])
    return df


class TestLoadTrafficSplit(unittest.TestCase):
    """End-to-end shape contract — speed table, sensor ids + Path.exists are
    mocked; the distances CSV is a real tmpfile in a tmp ``data_dir``."""

    def _patches(self, n_steps: int, n_sensors: int):
        speeds, timestamps = _make_mock_speed_table(n_steps, n_sensors)

        sensor_ids_text = "\n".join(str(i) for i in range(n_sensors))

        # Patch the I/O entry points and Path.exists. The speed-table reader
        # uses h5py directly (not pandas.read_hdf) for legacy-format compat,
        # so we mock the helper rather than the underlying h5py call.
        return [
            patch("hilbnet.traffic_loader._read_speed_table",
                  return_value=(speeds, timestamps)),
            patch("pathlib.Path.read_text", return_value=sensor_ids_text),
            patch("pathlib.Path.exists", return_value=True),
        ]

    def _load(self, split, n_steps=200, n_sensors=10, **kwargs):
        import tempfile
        ctxs = self._patches(n_steps, n_sensors)
        name = kwargs.get("name", "metr-la")
        with tempfile.TemporaryDirectory() as tmp:
            # `_build_adjacency` sniffs the CSV header with a bare open(), which
            # a pandas.read_csv mock does not intercept — so the distances file
            # must exist on disk (otherwise this only passes where data/ exists).
            _make_mock_distances_df(n_sensors).to_csv(
                Path(tmp) / traffic_loader._DCRNN_FILES[name]["distances"],
                index=False,
            )
            for c in ctxs:
                c.start()
            try:
                return traffic_loader.load_traffic_split(
                    split=split, data_dir=Path(tmp), **kwargs,
                )
            finally:
                for c in ctxs:
                    c.stop()

    def test_train_split_shapes(self):
        windows, edge_index, _edge_attr, scaler = self._load(
            "train", n_steps=200, n_sensors=10,
        )
        # Train slice = first 70% = 140 timesteps. With window_in=12, window_out=12:
        # 140 - 24 + 1 = 117 windows.
        self.assertEqual(len(windows), 117)
        w = windows[0]
        self.assertEqual(tuple(w.x.shape), (10, 12, 2))   # F_in=2 (default)
        self.assertEqual(tuple(w.y.shape), (10, 12))
        self.assertEqual(tuple(w.mask.shape), (10, 12))
        self.assertEqual(int(w.num_nodes), 10)

        # Edge index is a [2, E] LongTensor with i<j.
        self.assertEqual(edge_index.dim(), 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertTrue((edge_index[0] < edge_index[1]).all().item())

        # Scaler computed on TRAIN, present, finite.
        self.assertIn("mean", scaler)
        self.assertIn("std", scaler)
        self.assertTrue(np.isfinite(scaler["mean"]))
        self.assertGreater(scaler["std"], 0.0)

    def test_val_and_test_use_train_scaler(self):
        # Verifying numerically that the same scaler comes back regardless of split.
        _, _, _, scaler_train = self._load("train", n_steps=300, n_sensors=8)
        _, _, _, scaler_val = self._load("val", n_steps=300, n_sensors=8)
        _, _, _, scaler_test = self._load("test", n_steps=300, n_sensors=8)
        self.assertAlmostEqual(scaler_train["mean"], scaler_val["mean"], places=4)
        self.assertAlmostEqual(scaler_train["mean"], scaler_test["mean"], places=4)
        self.assertAlmostEqual(scaler_train["std"], scaler_val["std"], places=4)

    def test_split_temporal_disjointness(self):
        # All three splits' timestamps should be contiguous and non-overlapping.
        # Easiest check: window counts add up sensibly relative to the underlying
        # timestep counts (modulo windowing boundary).
        n_steps = 1000
        train_w, _, _, _ = self._load("train", n_steps=n_steps, n_sensors=5)
        val_w, _, _, _ = self._load("val", n_steps=n_steps, n_sensors=5)
        test_w, _, _, _ = self._load("test", n_steps=n_steps, n_sensors=5)
        # 70/10/20 → 700/100/200 timesteps; window_in+window_out=24:
        # 700-24+1=677 train, 100-24+1=77 val, 200-24+1=177 test.
        self.assertEqual(len(train_w), 677)
        self.assertEqual(len(val_w), 77)
        self.assertEqual(len(test_w), 177)

    def test_add_time_of_day_off_gives_F_in_one(self):
        windows, _, _, _ = self._load(
            "train", n_steps=100, n_sensors=4, add_time_of_day=False,
        )
        self.assertEqual(tuple(windows[0].x.shape), (4, 12, 1))

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            self._load("invalid_split", n_steps=100, n_sensors=4)


if __name__ == "__main__":
    unittest.main()
