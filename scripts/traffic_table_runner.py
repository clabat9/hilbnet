"""Train and evaluate the five traffic-forecasting models of Table 2 on METR-LA or PEMS-BAY.

Runs every architecture for every seed (default: 5 seeds). The dataset is
loaded once and reused across architectures. Progress is saved after each
(architecture, seed) run, so an interrupted run resumes where it stopped.

Architectures (configs in scripts/configs/traffic/):
    1. HilbNet, circulant transports        circulant.yaml
    2. HilbNet, frozen identity transports  frozen_id.yaml
    3. HilbNet, free O(T) transports        free.yaml
    4. MLP fiber baseline                   mlp_fiber.yaml  (HilbNet with kappa=[1, 1])
    5. Spatiotemporal graph baseline        stgnn_conv.yaml

Usage:
    # METR-LA, 5 architectures x 5 seeds (25 runs):
    python scripts/traffic_table_runner.py --dataset metr-la

    # PEMS-BAY:
    python scripts/traffic_table_runner.py --dataset pems-bay

    # Subset of architectures:
    python scripts/traffic_table_runner.py --archs circulant free

    # One-epoch smoke test (writes results/traffic/<dataset>_table_epochs1.json):
    python scripts/traffic_table_runner.py --seeds 0 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from scripts._config_utils import load_config  # noqa: E402
from scripts.traffic_eval import (  # noqa: E402
    train_and_forecast,
    _print_forecast_table,
)
from hilbnet.traffic_loader import load_traffic_split  # noqa: E402


# ─── The 5 paper-table architectures ─────────────────────────────────────────
# (short_name, config_relpath, model_label_for_dispatch)
#
# All 5 configs apply to BOTH METR-LA and PEMS-BAY — the runner overrides the
# dataset via the `--dataset` CLI flag. `model_label_for_dispatch` matches
# cfg["model"] which build_forecasting_model reads.
ARCH_TABLE: List[tuple[str, str, str]] = [
    ("circulant",   "scripts/configs/traffic/circulant.yaml",   "hilbnet"),
    ("frozen_id",   "scripts/configs/traffic/frozen_id.yaml",   "hilbnet"),
    ("free",        "scripts/configs/traffic/free.yaml",        "hilbnet"),
    ("mlp_fiber",   "scripts/configs/traffic/mlp_fiber.yaml",   "hilbnet"),
    ("stgnn_conv",  "scripts/configs/traffic/stgnn_conv.yaml",  "stgnn_conv"),
]


def _select_archs(filter_names: List[str] | None) -> List[tuple[str, str, str]]:
    if not filter_names:
        return list(ARCH_TABLE)
    by_short = {short: row for row in ARCH_TABLE for short in [row[0]]}
    missing = [n for n in filter_names if n not in by_short]
    if missing:
        raise SystemExit(
            f"Unknown arch name(s): {missing}. "
            f"Valid: {sorted(by_short)}."
        )
    return [by_short[n] for n in filter_names]


# ─── Resumable JSON state ────────────────────────────────────────────────────
#
# The runner persists one row per completed (arch, seed) to ``output_json``
# *immediately* after that run finishes. On startup, the runner loads the
# existing JSON (if any), builds a `done` set of (arch, seed) pairs already
# in the file, and skips them. So if the process crashes / is killed mid-run,
# re-launching the same command picks up exactly where it left off.
#
# Schema:
#   {
#     "dataset": "metr-la",
#     "horizons": [3, 6, 12],
#     "rows": [
#       {"arch": "circulant", "seed": 0, "params": 11656,
#        "metrics": {"mae_3": 2.95, "rmse_3": 5.6, "mape_3": 0.079, ...}},
#       ...
#     ]
#   }


def _load_state(path: Path) -> Tuple[List[dict], Set[Tuple[str, int]]]:
    """Read existing rows + (arch, seed) pairs already completed."""
    if not path.exists():
        return [], set()
    with open(path) as f:
        payload = json.load(f)
    rows = list(payload.get("rows", []))
    done = {(str(r["arch"]), int(r["seed"])) for r in rows}
    return rows, done


def _save_state(path: Path, dataset: str, horizons: List[int],
                rows: List[dict]) -> None:
    """Atomically rewrite the JSON state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "horizons": list(horizons), "rows": rows}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _replay_into_memory(rows: List[dict],
                        results: Dict[str, List[Dict[str, float]]],
                        param_counts: Dict[str, int]) -> None:
    """Populate the in-memory result dicts from previously-saved rows.

    Lets the final summary table include both freshly-run and resumed rows.
    """
    for r in rows:
        arch = str(r["arch"])
        results.setdefault(arch, []).append(dict(r["metrics"]))
        if arch not in param_counts and "params" in r:
            param_counts[arch] = int(r["params"])


def main():
    # Force line-buffered stdout so progress is visible when piped to `tee`
    # or redirected to a log file (Python switches to block-buffered ~4-8 KB
    # buffering on non-TTY stdout by default).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archs", type=str, nargs="+", default=None,
                        help="Subset of arch short-names to run "
                             "(e.g. 'circulant free'). Default: all 5.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to run for each arch. Default: 0..4 (5 seeds).")
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12],
                        help="Forecast horizons in 5-min steps. Default: 3 6 12 (=15/30/60 min).")
    parser.add_argument("--no_val", action="store_true",
                        help="Skip val loop / early stopping (FINAL-EPOCH eval only).")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        help="Dataset name passed to load_traffic_split.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: cpu / mps / cuda. Default: mps, then cuda, then cpu.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the configs' epoch budget (e.g. 1 for a smoke test). "
                             "The default output then becomes "
                             "results/traffic/<dataset>_table_epochs<k>.json, so short runs "
                             "never mix with a full run.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to the resumable progress file. Each "
                             "completed (arch, seed) is appended immediately. "
                             "Re-running with the same path skips already-"
                             "completed rows. Default: "
                             "results/traffic/<dataset>_table.json "
                             "(or <dataset>_table_epochs<k>.json with --epochs).")
    parser.add_argument("--no_json", action="store_true",
                        help="Disable progressive JSON output (no resume).")
    args = parser.parse_args()

    archs = _select_archs(args.archs)
    use_val = not args.no_val

    # ── Load the dataset once (shared across all archs) ────────────────────
    # Window sizes and graph threshold come from the first arch's config (all
    # five configs agree on them).
    primary_cfg = load_config(_ROOT / archs[0][1])
    loader_kwargs = dict(
        name=args.dataset,
        window_in=primary_cfg.get("window_in", 12),
        window_out=primary_cfg.get("window_out", 12),
        train_frac=primary_cfg.get("train_frac", 0.7),
        val_frac=primary_cfg.get("val_frac", 0.1),
        add_time_of_day=primary_cfg.get("add_time_of_day", True),
        threshold_k=primary_cfg.get("threshold_k", 0.1),
    )
    train_w, edge_index, edge_attr, scaler = load_traffic_split("train", **loader_kwargs)
    val_w, _, _, _ = load_traffic_split("val", **loader_kwargs)
    test_w, _, _, _ = load_traffic_split("test", **loader_kwargs)
    n_nodes = int(edge_index.max().item()) + 1
    t_in = primary_cfg.get("window_in", 12)
    t_out = primary_cfg.get("window_out", 12)
    f_in = train_w[0].x.shape[-1]

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── Resumable JSON state ───────────────────────────────────────────────
    if args.no_json:
        json_path = None
        rows: List[dict] = []
        done: Set[Tuple[str, int]] = set()
    else:
        suffix = "" if args.epochs is None else f"_epochs{args.epochs}"
        json_path = (Path(args.output_json) if args.output_json
                     else _ROOT / "results" / "traffic" / f"{args.dataset}_table{suffix}.json")
        rows, done = _load_state(json_path)

    print("=" * 72)
    print("  PAPER-TABLE MULTI-SEED RUN")
    print(f"  Dataset: {args.dataset}  N={n_nodes}  T_in={t_in}  T_out={t_out}  F_in={f_in}")
    print(f"  Archs ({len(archs)}): {[a[0] for a in archs]}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Horizons: {args.horizons} steps ({[h*5 for h in args.horizons]} min)")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs if args.epochs is not None else 'from config'}")
    print(f"  Progress JSON: {json_path or '<disabled>'}"
          f"{f'  (resuming, {len(done)} rows already done)' if done else ''}")
    print("=" * 72)

    # ── Run each (arch, seed) sequentially ─────────────────────────────────
    results: Dict[str, List[Dict[str, float]]] = {}
    param_counts: Dict[str, int] = {}
    _replay_into_memory(rows, results, param_counts)
    total_runs = len(archs) * len(args.seeds)
    run_idx = 0

    for short_name, cfg_path, _model_label in archs:
        cfg = load_config(_ROOT / cfg_path)
        if args.epochs is not None:
            cfg = {**cfg, "epochs": args.epochs}
        # `model_name` is the row label in the printed table and the JSON; we
        # use the short name rather than e.g. "hilbnet" three times.
        model_name = short_name
        results.setdefault(model_name, [])

        print(f"\n{'='*72}")
        print(f"  ARCH: {short_name.upper()}  features={cfg['features']}  kappa={cfg['kappa']}")
        print(f"  Config: {cfg_path}")
        print(f"{'='*72}")

        for seed in args.seeds:
            run_idx += 1
            if (short_name, seed) in done:
                print(f"\n>>>>> [{run_idx}/{total_runs}] {short_name} | seed={seed} "
                      "— already in JSON, skipping <<<<<")
                continue
            print(f"\n>>>>> [{run_idx}/{total_runs}] {short_name} | seed={seed} <<<<<")
            metrics = train_and_forecast(
                cfg, seed, device,
                train_windows=train_w, val_windows=val_w, test_windows=test_w,
                edge_index=edge_index, edge_attr=edge_attr, scaler=scaler,
                n_nodes=n_nodes, t_in=t_in, t_out=t_out, f_in=f_in,
                horizons=args.horizons, use_val=use_val,
                model_name=model_name, param_counts=param_counts,
            )
            results[model_name].append(metrics)

            # Persist progress immediately so a crash doesn't lose this row.
            if json_path is not None:
                rows.append({
                    "arch": short_name,
                    "seed": int(seed),
                    "params": int(param_counts.get(model_name, 0)),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                })
                done.add((short_name, int(seed)))
                _save_state(json_path, args.dataset, args.horizons, rows)

    _print_forecast_table(results, param_counts, args.horizons)
    if json_path is not None:
        print(f"\n  [json] {len(rows)} total rows in {json_path}")


if __name__ == "__main__":
    main()
