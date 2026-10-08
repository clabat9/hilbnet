"""Traffic forecasting loader for HilbNet — METR-LA / PEMS-BAY.

Standard DCRNN-format datasets, three files per dataset:
    {name}.h5                  — HDF5 traffic speeds, shape [T_total, n_sensors].
    distances_{name}.csv       — CSV with columns (from, to, cost) where cost is
                                 sensor-to-sensor travel time in seconds.
    graph_sensor_ids.txt       — sensor IDs in the canonical column order, one per line.

Conventions adopted from DCRNN's `data/model/dcrnn_la.yaml`:
    - 12-step input → 12-step output (1 hour at TR=5 min).
    - 70 / 10 / 20 contiguous temporal split.
    - Input dim = 2 (speed + time-of-day fraction in [0, 1]); output dim = 1 (speed).
    - Adjacency: thresholded Gaussian kernel ``W_ij = exp(-d_ij² / σ²)`` with
      σ = std(distances) and threshold κ = 0.1.

Documented deviation from DCRNN: HilbNet's sheaf-Laplacian construction
requires an undirected graph. We symmetrize the DCRNN adjacency via
``A_sym = max(A, A.T)``. DCRNN's canonical setup uses asymmetric dual
random-walk filters; comparing in-codebase HilbNet vs STGNNConv (both symmetric)
is apples-to-apples; comparing to published DCRNN/GraphWaveNet numbers
needs a symmetrization footnote.

Returned tensors per window:
    x:    float32, shape [N, T_in, F_in]   — speeds (z-scored on train) and
                                              optional time-of-day in [0, 1].
    y:    float32, shape [N, T_out]        — target speeds in *raw* units;
                                              de-normalize predictions before metrics.
    mask: float32, shape [N, T_out]        — 1 where target is present (raw > 0),
                                              0 where missing (METR-LA encodes
                                              missing as 0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAFFIC_DATA = _REPO_ROOT / "data" / "traffic"


# Canonical DCRNN file names per dataset (the distances CSV name doesn't follow
# a clean pattern — "la_2012" / "bay_2017" — so we map explicitly).
_DCRNN_FILES = {
    "metr-la": {
        "h5": "metr-la.h5",
        "distances": "distances_la_2012.csv",
        "sensor_ids": "graph_sensor_ids.txt",
    },
    "pems-bay": {
        "h5": "pems-bay.h5",
        "distances": "distances_bay_2017.csv",
        "sensor_ids": "graph_sensor_ids_bay.txt",
    },
}


# Public DCRNN GitHub raw URLs for the distances CSV (the only "free" small
# text file actually hosted in the DCRNN repo for both datasets — the
# graph_sensor_ids file is only published for METR-LA, and even that is
# trivially recoverable from the h5 axis0 dataset). The h5 traffic data is
# too large for direct GitHub hosting (~62 MB / 136 MB) and is distributed
# via Google Drive — we leave h5 download manual, since (a) gdown adds a
# dependency, (b) Google-Drive direct download is fragile (rotating tokens /
# bandwidth caps), and (c) typical users already have the h5 from a prior
# DCRNN setup.
_DCRNN_TEXT_URLS = {
    "metr-la": {
        "distances_la_2012.csv":
            "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
    },
    "pems-bay": {
        "distances_bay_2017.csv":
            "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_bay_2017.csv",
    },
}


def _ensure_dcrnn_text_files(name: str, data_dir: Path) -> None:
    """Auto-fetch the small DCRNN distances CSV into ``data_dir`` if missing.
    Sensor-ids txt is handled separately (extracted from h5 axis0; see
    ``_ensure_sensor_ids_file``). The h5 traffic data must still be
    downloaded manually (see error message in ``load_traffic_split``).
    """
    import urllib.request

    urls = _DCRNN_TEXT_URLS.get(name, {})
    for fname, url in urls.items():
        target = data_dir / fname
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {fname} from {url} ...")
        urllib.request.urlretrieve(url, target)


def _dcrnn_h5_root_group(f) -> str:
    """Return the name of the single top-level group inside a DCRNN-format
    h5 file. METR-LA's distribution uses ``df`` (the pandas DataFrame name);
    PEMS-BAY's uses ``speed``. Both files have exactly one top-level group.
    """
    keys = list(f.keys())
    if len(keys) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level group in DCRNN-format h5; "
            f"found {keys}."
        )
    return keys[0]


def _ensure_sensor_ids_file(h5_path: Path, sensor_ids_path: Path) -> None:
    """If the canonical ``sensor_ids_path`` is missing, write it by reading
    ``<group>/axis0`` from the DCRNN-format h5 file. The axis0 dataset stores
    the sensor IDs in canonical column order — i.e., the same ordering that
    the accompanying ``graph_sensor_ids*.txt`` files use when published — so
    a file written from this source is interchangeable with the official one.
    Sensor IDs may be bytes (METR-LA) or int64 (PEMS-BAY); both are converted
    to plain strings. No-op if the txt is already present.
    """
    if sensor_ids_path.exists():
        return
    import h5py

    if not h5_path.exists():
        return  # caller will raise FileNotFoundError on h5 below
    with h5py.File(str(h5_path), "r") as f:
        root = _dcrnn_h5_root_group(f)
        raw = f[f"{root}/axis0"][:]
    ids = [(s.decode("utf-8") if isinstance(s, bytes) else str(s)) for s in raw]
    sensor_ids_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_ids_path.write_text("\n".join(ids) + "\n")
    print(f"  Wrote {len(ids)} sensor IDs to {sensor_ids_path.name} "
          f"(extracted from {h5_path.name} axis0).")


def _read_speed_table(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read DCRNN-format speed HDF5 → (speeds [T, N], timestamps [T]).

    The METR-LA / PEMS-BAY h5 files were written with pandas 0.15.2 / PyTables
    2.1 in the legacy "frame" format. Modern pandas (2.x+) `read_hdf` raises
    ``TypeError: a bytes-like object is required, not 'str'`` on these files
    because attribute encoding changed. We read the underlying datasets
    directly with h5py — the structure is simple:

        <root>/block0_values  : float64 [T_total, n_sensors]   speed matrix
        <root>/axis1          : int64   [T_total]              timestamps (ns)
        <root>/axis0          : bytes/int64 [n_sensors]        sensor-id strings

    METR-LA uses ``<root> = df``; PEMS-BAY uses ``<root> = speed``; we
    auto-discover via ``_dcrnn_h5_root_group``.
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        root = _dcrnn_h5_root_group(f)
        speeds = f[f"{root}/block0_values"][:].astype(np.float32)   # [T, N]
        ts_int = f[f"{root}/axis1"][:]                              # int64 ns
    timestamps = ts_int.astype("datetime64[ns]")
    return speeds, timestamps


def _build_adjacency(
    distances_csv: Path, sensor_ids: List[str], threshold_k: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DCRNN-format Gaussian-kernel adjacency, symmetrized for sheaf Laplacian.

    Returns:
        edge_index: LongTensor [2, E] of (source, dest) pairs (E counts each
            undirected edge once; pair u<v).
        edge_attr:  FloatTensor [E] of edge weights (Gaussian-kernel values).
    """
    import pandas as pd

    n = len(sensor_ids)
    sensor_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        dist[i, i] = 0.0

    # METR-LA's distances CSV ships with a "from,to,cost" header row;
    # PEMS-BAY's ships header-less. Detect by sniffing the first cell.
    with open(distances_csv) as fh:
        first_field = fh.readline().split(",", 1)[0].strip()
    has_header = first_field.lower() == "from"
    if has_header:
        df = pd.read_csv(distances_csv, dtype={"from": str, "to": str})
    else:
        df = pd.read_csv(
            distances_csv, header=None, names=["from", "to", "cost"],
            dtype={"from": str, "to": str},
        )
    for src_id, dst_id, cost in zip(df["from"], df["to"], df["cost"]):
        if src_id in sensor_to_idx and dst_id in sensor_to_idx:
            i, j = sensor_to_idx[src_id], sensor_to_idx[dst_id]
            dist[i, j] = float(cost)

    # Gaussian kernel with σ = std of finite, off-diagonal distances.
    finite = dist[np.isfinite(dist) & (dist > 0)]
    if finite.size == 0:
        raise RuntimeError(f"No finite positive distances in {distances_csv}.")
    sigma = float(finite.std())
    W = np.exp(-(dist ** 2) / (sigma ** 2))
    W[dist == np.inf] = 0.0
    np.fill_diagonal(W, 0.0)

    # Threshold + symmetrize. DCRNN uses asymmetric dual random walk; HilbNet's
    # sheaf Laplacian construction needs a symmetric undirected graph.
    W[W < threshold_k] = 0.0
    W = np.maximum(W, W.T)

    # Build undirected edge list (i < j once).
    iu, ju = np.triu_indices(n, k=1)
    keep = W[iu, ju] > 0
    src, dst, w = iu[keep], ju[keep], W[iu, ju][keep]
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = torch.from_numpy(w.astype(np.float32))
    return edge_index, edge_attr


def _temporal_split_indices(
    n_steps: int, train_frac: float, val_frac: float,
) -> Tuple[int, int]:
    """Return (i_train_end, i_val_end) such that
        train = [0, i_train_end),  val = [i_train_end, i_val_end),  test = [i_val_end, n_steps).
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError(
            f"Need train_frac + val_frac < 1 and both in (0, 1). "
            f"Got train_frac={train_frac}, val_frac={val_frac}."
        )
    i_train = int(round(train_frac * n_steps))
    i_val = int(round((train_frac + val_frac) * n_steps))
    return i_train, i_val


def _time_of_day(timestamps: np.ndarray) -> np.ndarray:
    """Time-of-day fraction in [0, 1) for each timestamp. Shape [T]."""
    # numpy datetime64[ns] → seconds since midnight, normalized to [0, 1)
    ts_seconds = (timestamps.astype("datetime64[s]")
                  - timestamps.astype("datetime64[D]")
                  ).astype(np.int64)  # seconds since midnight
    return (ts_seconds / 86400.0).astype(np.float32)


def _make_windows(
    speeds_norm: np.ndarray,         # [T, N], z-scored
    speeds_raw: np.ndarray,          # [T, N], original (for masked targets)
    time_of_day: Optional[np.ndarray],  # [T] or None
    window_in: int,
    window_out: int,
    n_nodes: int,
) -> List[Data]:
    """Slide overlapping (input, output) windows. Stride = 1 (DCRNN convention).

    Each window:
        x: [N, window_in, F_in], where F_in is 1 (speed) or 2 (speed + tod).
        y: [N, window_out] in raw units.
        mask: [N, window_out] — 1 where raw > 0, 0 elsewhere.
    """
    T = speeds_norm.shape[0]
    if T < window_in + window_out:
        return []

    out: List[Data] = []
    for s in range(T - window_in - window_out + 1):
        in_slice = slice(s, s + window_in)
        out_slice = slice(s + window_in, s + window_in + window_out)

        x_speed = speeds_norm[in_slice]  # [T_in, N]
        if time_of_day is not None:
            tod = np.broadcast_to(time_of_day[in_slice, None], (window_in, n_nodes))  # [T_in, N]
            x = np.stack([x_speed.T, tod.T], axis=-1)  # [N, T_in, 2]
        else:
            x = x_speed.T[..., None]  # [N, T_in, 1]
        x = np.ascontiguousarray(x, dtype=np.float32)

        y_raw = speeds_raw[out_slice].T  # [N, T_out]
        mask = (y_raw > 0).astype(np.float32)
        y = np.ascontiguousarray(y_raw, dtype=np.float32)

        out.append(Data(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            mask=torch.from_numpy(mask),
            num_nodes=n_nodes,
        ))
    return out


def _read_sensor_ids(path: Path) -> List[str]:
    text = path.read_text().strip()
    # File can be one-per-line or comma-separated; handle both.
    if "," in text and "\n" not in text.strip():
        return [s.strip() for s in text.split(",") if s.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_traffic_split(
    split: str,
    *,
    name: str = "metr-la",
    data_dir: Optional[Path] = None,
    window_in: int = 12,
    window_out: int = 12,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    add_time_of_day: bool = True,
    threshold_k: float = 0.1,
) -> Tuple[List[Data], torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Load a temporal split of a DCRNN-format traffic dataset.

    Args:
        split: 'train' / 'val' / 'test'.
        name: dataset short name; resolves to ``{name}.h5`` and
            ``distances_{name}.csv`` (and ``graph_sensor_ids.txt``) under
            ``data_dir``. Default 'metr-la'; use 'pems-bay' for PEMS-BAY.
        data_dir: directory containing the three files. Default
            ``<repo>/data/traffic/{name}/``.
        window_in / window_out: input / forecast lengths (default 12 / 12).
        train_frac / val_frac: contiguous temporal split. Test = remainder.
        add_time_of_day: if True (default), append time-of-day fraction as a
            second feature → F_in=2. DCRNN canonical input dim.
        threshold_k: Gaussian-kernel adjacency threshold (DCRNN
            ``normalized_k=0.1``).

    Returns:
        windows: list of ``Data`` objects with x / y / mask (see module docstring).
        edge_index: [2, E] LongTensor of undirected edges (i < j) — same for
            all splits, returned here for convenience.
        edge_attr: [E] FloatTensor of edge weights.
        scaler_meta: {'mean': float, 'std': float} computed on TRAIN split only.

    Notes:
        - Predictions must be de-normalized via ``pred * std + mean`` before
          metric computation (standard DCRNN convention).
        - Mask is 1 where the raw (un-normalized) target speed > 0; the loss
          should ignore masked-out positions to avoid penalizing the model
          for missing-data zeros.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train' / 'val' / 'test', got {split!r}.")

    if name not in _DCRNN_FILES:
        raise ValueError(
            f"Unknown dataset name {name!r}. Supported: {sorted(_DCRNN_FILES)}."
        )
    files = _DCRNN_FILES[name]

    if data_dir is None:
        data_dir = DEFAULT_TRAFFIC_DATA / name
    data_dir = Path(data_dir)

    # Auto-fetch the distances CSV from the DCRNN GitHub repo if missing,
    # and recover sensor IDs from the h5 file's ``df/axis0`` dataset if the
    # canonical txt is missing. The h5 traffic data must still be downloaded
    # manually — see the FileNotFoundError below.
    _ensure_dcrnn_text_files(name, data_dir)
    _ensure_sensor_ids_file(data_dir / files["h5"], data_dir / files["sensor_ids"])

    h5_path = data_dir / files["h5"]
    distances_csv = data_dir / files["distances"]
    sensor_ids_path = data_dir / files["sensor_ids"]
    # Fall back to the dataset-agnostic sensor-ids name (the METR-LA distribution
    # ships it as `graph_sensor_ids.txt` rather than the prefixed form).
    if not sensor_ids_path.exists():
        alt = data_dir / "graph_sensor_ids.txt"
        if alt.exists():
            sensor_ids_path = alt

    for p in (h5_path, distances_csv, sensor_ids_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing traffic data file: {p}. Download {files['h5']} from the "
                f"Google Drive folder linked under 'Data Preparation' at "
                f"https://github.com/liyaguang/DCRNN and put it in {data_dir}/. "
                f"The distances CSV is downloaded on first use (needs network "
                f"access) and the sensor IDs are read from the .h5 file."
            )

    print(f"  Loading {name} (split={split}, window_in={window_in}, "
          f"window_out={window_out}, F_in={2 if add_time_of_day else 1})...")

    sensor_ids = _read_sensor_ids(sensor_ids_path)
    n_nodes = len(sensor_ids)

    speeds, timestamps = _read_speed_table(h5_path)
    if speeds.shape[1] != n_nodes:
        raise RuntimeError(
            f"Speed table has {speeds.shape[1]} columns; "
            f"graph_sensor_ids.txt has {n_nodes} entries."
        )

    # Train-only scaler. DCRNN computes mean/std over the train slice only.
    n_total = speeds.shape[0]
    i_train_end, i_val_end = _temporal_split_indices(n_total, train_frac, val_frac)
    train_speeds = speeds[:i_train_end]
    train_finite = train_speeds[train_speeds != 0]   # zeros = missing → exclude
    if train_finite.size == 0:
        raise RuntimeError("Train split has no non-missing speed values.")
    mean = float(train_finite.mean())
    std = float(train_finite.std())
    if std == 0.0:
        raise RuntimeError("Train split has zero variance in speeds.")

    # Normalize the *full* series with train scaler (val/test use same mean/std).
    speeds_norm = (speeds - mean) / std

    # Slice the appropriate split.
    if split == "train":
        sl = slice(0, i_train_end)
    elif split == "val":
        sl = slice(i_train_end, i_val_end)
    else:
        sl = slice(i_val_end, n_total)

    speeds_norm_split = speeds_norm[sl]
    speeds_raw_split = speeds[sl]
    timestamps_split = timestamps[sl]

    tod = _time_of_day(timestamps_split) if add_time_of_day else None

    windows = _make_windows(
        speeds_norm=speeds_norm_split,
        speeds_raw=speeds_raw_split,
        time_of_day=tod,
        window_in=window_in,
        window_out=window_out,
        n_nodes=n_nodes,
    )
    print(f"  Built {len(windows)} windows from {speeds_norm_split.shape[0]} "
          f"timesteps ({split}).")

    edge_index, edge_attr = _build_adjacency(
        distances_csv, sensor_ids, threshold_k=threshold_k,
    )
    print(f"  Adjacency: {n_nodes} nodes, {edge_index.shape[1]} undirected edges "
          f"(thresholded Gaussian kernel, κ={threshold_k}, symmetrized).")

    scaler_meta = {"mean": mean, "std": std}
    return windows, edge_index, edge_attr, scaler_meta
