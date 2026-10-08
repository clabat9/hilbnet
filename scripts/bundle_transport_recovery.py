"""Synthetic transport recovery — learn the closed-form Levi-Civita transport
via direct edge-MSE, sweep across HilbNet's transport variants (free /
circulant / frozen-id) and check that each variant's training plateau matches
its analytical projection distance from the LC operator.

This is the experiment that gives empirical teeth to the paper's "subgroup
restriction = connection class" claim. With Cholesky-rescaled fibers the LC
operator is Frobenius-orthogonal, so the free Householder hypothesis class
contains it; restricted variants converge to their respective Frobenius-
closest projections.

Loss (per training step, ``B`` fresh isotropic tangents per node)::

    loss = mean_b mean_e mean_t  (T_pred[e] V_in[b, src(e), :] - T_gt[e] V_in[b, src(e), :])²

Under random isotropic V_in this concentrates on
``‖T_pred − T_gt‖_F² / (E · m²) · m   =   projection_distance(T_pred, T_gt) · m``.
The empirical plateau and ``projection_distance(T_gt, T_proj_of_class) * m``
should agree to optimization-slack tolerance.

Two empirical statistics are reported per run:

- ``best_loss`` (JSON field ``empirical``): min over epochs of the per-epoch
  Monte-Carlo loss above. As a min-statistic of a noisy unbiased estimate it
  is downward-biased and can land slightly *below* the analytical plateau.
- ``best_frob`` (JSON field ``empirical_frob``, the "Det." column of Table 1):
  min over epochs of the deterministic closed-form value
  ``projection_distance(T_pred, T_gt) · m`` at that epoch's parameters. Every
  in-class operator satisfies ``projection_distance(T_pred, T_gt) >=
  projection_distance(T_proj_of_class, T_gt)``, so this statistic is >= the
  analytical plateau (``theory``) by construction.

Configs accept either a single ``n_nodes`` (legacy / smoke configs) or
``n_grid`` (paper configs); the loop iterates each ``n_nodes`` and produces
one (variant, N, seed) row per training run.

Usage:
    python scripts/bundle_transport_recovery.py \\
        --config scripts/configs/synthetic/transport_recovery_paper.yaml

    # Smoke test:
    python scripts/bundle_transport_recovery.py \\
        --config scripts/configs/synthetic/transport_recovery_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from hilbnet.bundle_validation import (  # noqa: E402
    BundleTransportModule,
    closed_form_lc_transports,
    project_to_circulant,
    project_to_identity,
    projection_distance,
)
from hilbnet.statistical_bundle import (  # noqa: E402
    DistanceType,
    FeatureType,
    GraphType,
    StatisticalBundleGraphConfig,
    StatisticalBundleGraphDataset,
)
from scripts._config_utils import load_config  # noqa: E402


# ─── Bundle-graph builder (shared with bundle_operator_convergence.py) ───────


def _build_graph(cfg: dict, n_nodes: int, seed: int):
    """One synthetic-bundle graph, deterministic in (cfg, n_nodes, seed)."""
    bcfg = StatisticalBundleGraphConfig(
        n=cfg["n_dim"],
        n_nodes=n_nodes,
        feature_type=FeatureType.TANGENT_VEC,
        graph_type=GraphType(cfg["graph_type"]),
        distance_type=DistanceType(cfg["distance_type"]),
        k_neighbors=cfg.get("k_neighbors", 8),
        epsilon=cfg.get("epsilon", 1.0),
        include_fiber=False,
        edge_weight_sigma=cfg.get("edge_weight_sigma", 1.0),
    )
    ds = StatisticalBundleGraphDataset(bcfg, n_graphs=1, seed=seed)
    return ds.data_list[0]


# ─── n_grid resolution ───────────────────────────────────────────────────────


def _resolve_n_grid(cfg: dict) -> List[int]:
    """Return the list of ``n_nodes`` values to sweep.

    Accepts either ``n_grid: [...]`` (paper) or scalar ``n_nodes`` (smoke/legacy).
    """
    if "n_grid" in cfg and cfg["n_grid"] is not None:
        return [int(x) for x in cfg["n_grid"]]
    return [int(cfg["n_nodes"])]


# ─── Analytical plateau (Theory column of Table 1) ───────────────────────────


def _project(T_gt: torch.Tensor, variant: str, cfg: dict) -> torch.Tensor:
    """Closest in-class operator to ``T_gt`` for each variant.

    Returns a tensor with the same shape as ``T_gt``.
    """
    if variant == "direct":
        # Free Householder with num_reflections >= m spans O(m); the closest
        # in-class operator is T_gt itself (up to optimizer slack at fit time).
        return T_gt.clone()
    if variant == "circulant":
        return project_to_circulant(T_gt, num_bands=cfg.get("num_bands"))
    if variant == "frozen_id":
        return project_to_identity(T_gt)
    raise ValueError(f"unknown variant: {variant!r}")


# ─── Training one variant ────────────────────────────────────────────────────


def _train_variant(
    variant: str,
    T_gt: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: dict,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Train ``BundleTransportModule(variant)`` and return final-loss metrics.

    Reads ``n_nodes`` from ``cfg`` so the outer N-sweep can inject it via
    ``{**cfg, "n_nodes": N}`` without changing the signature (kept stable for
    the in-process smoke tests).
    """
    n = int(cfg["n_dim"])
    m = n * (n + 1) // 2
    n_nodes = int(cfg["n_nodes"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    log_every = int(cfg.get("log_every", 50))
    # Early stopping on best loss: 0/None disables. Counter resets on any
    # strict improvement; we break once `patience` epochs go by with no new best.
    patience = int(cfg.get("patience", 0) or 0)

    # crc32 instead of hash(): Python salts str hashes per process, which made
    # this seed (and every run's init + probe draws) change from run to run.
    torch.manual_seed(seed * 1009 + zlib.crc32(variant.encode()) % (10**6))

    model = BundleTransportModule(
        transport_param_type=variant,
        edge_index=edge_index.to(device),
        stalk_dim=m,
        num_reflections=int(cfg["num_reflections"]),
        num_bands=cfg.get("num_bands"),
        transport_init=str(cfg.get("transport_init", "identity_plus_noise")),
    ).to(device)

    src = edge_index[0].long().to(device)
    T_gt_dev = T_gt.to(device)
    learnable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in learnable)

    if n_params == 0:
        # Frozen-id: no training. Evaluate the constant identity loss.
        with torch.no_grad():
            V = torch.randn(batch_size, n_nodes, m, device=device)
            tgt = torch.einsum("emn,ben->bem", T_gt_dev, V[:, src, :])
            T_pred = model.transport_maps.to(device)
            pred = torch.einsum("emn,ben->bem", T_pred, V[:, src, :])
            final_loss = float((pred - tgt).pow(2).mean().item())
            frob = float(projection_distance(T_pred, T_gt_dev).item()) * m
        return {"variant": variant, "n_params": 0,
                "final_loss": final_loss, "best_loss": final_loss,
                "final_frob": frob, "best_frob": frob,
                "stopped_epoch": 0}

    optimizer = torch.optim.Adam(
        learnable,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    best_loss = math.inf
    final_loss = math.inf
    best_frob = math.inf
    final_frob = math.inf
    epochs_since_improvement = 0
    stopped_epoch = epochs
    for epoch in range(epochs):
        V = torch.randn(batch_size, n_nodes, m, device=device)
        V_src = V[:, src, :]
        T_pred = model.transport_maps
        with torch.no_grad():
            tgt = torch.einsum("emn,ben->bem", T_gt_dev, V_src)
        pred = torch.einsum("emn,ben->bem", T_pred, V_src)
        loss = (pred - tgt).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        # Deterministic closed-form counterpart of the MC loss at this epoch's
        # parameters; E[loss] = frob_val exactly, but frob_val has no sampling
        # noise and therefore never dips below the analytical plateau.
        with torch.no_grad():
            frob_val = float(
                projection_distance(T_pred.detach(), T_gt_dev).item()) * m
        if loss_val < best_loss:
            best_loss = loss_val
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        final_loss = loss_val
        best_frob = min(best_frob, frob_val)
        final_frob = frob_val

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"    [{variant:>10s}] epoch={epoch:>5d}/{epochs}  "
                  f"loss={loss_val:.4e}  best={best_loss:.4e}  "
                  f"frob={frob_val:.4e}", flush=True)

        if patience > 0 and epochs_since_improvement >= patience:
            stopped_epoch = epoch + 1
            print(f"    [{variant:>10s}] early stop at epoch={stopped_epoch}/{epochs}  "
                  f"(no improvement in {patience} epochs, best={best_loss:.4e})",
                  flush=True)
            break

    return {
        "variant": variant,
        "n_params": int(n_params),
        "final_loss": final_loss,
        "best_loss": best_loss,
        "final_frob": final_frob,
        "best_frob": best_frob,
        "stopped_epoch": int(stopped_epoch),
    }


# ─── Reporting ───────────────────────────────────────────────────────────────


def _sample_std(x: np.ndarray) -> float:
    """Sample standard deviation (ddof=1), as reported in the paper; NaN for one seed."""
    return float(x.std(ddof=1)) if x.size > 1 else float("nan")


def _sd(x: np.ndarray) -> str:
    """Formatted sample std; "-" when there is a single seed."""
    s = _sample_std(x)
    return f"{s:>8.2e}" if math.isfinite(s) else f"{'-':>8s}"


def _print_summary(rows: List[Dict[str, object]]) -> None:
    """Per-(variant, n_nodes) table with mean ± sample std across seeds."""
    if not rows:
        return
    variants = sorted({r["variant"] for r in rows})
    Ns = sorted({int(r["n_nodes"]) for r in rows})

    for N in Ns:
        sub = [r for r in rows if int(r["n_nodes"]) == N]
        print("\n" + "=" * 120)
        print(f"  TRANSPORT-RECOVERY SUMMARY  n_nodes={N}")
        print("  MC = best Monte-Carlo loss across training (min-statistic, "
              "downward-biased); Frob = best deterministic")
        print("  projection_distance(T_pred, T_gt) * m across training "
              "(>= Theory by construction); Theory = "
              "projection_distance(T_gt, T_proj) * m")
        print("=" * 120)
        header = (f"{'variant':>11s} | {'n_params':>9s} | "
                  f"{'MC empirical':>22s} | {'Frob (det.)':>22s} | "
                  f"{'theory':>22s} | {'gap (f - t)':>22s}")
        print(header)
        print("-" * len(header))
        for v in variants:
            vrows = [r for r in sub if r["variant"] == v]
            if not vrows:
                continue
            emp = np.array([r["best_loss"] for r in vrows], dtype=np.float64)
            frob = np.array([r["empirical_frob"] for r in vrows], dtype=np.float64)
            th = np.array([r["theory_plateau"] for r in vrows], dtype=np.float64)
            gap_f = frob - th
            np_ = vrows[0]["n_params"]
            print(f"{v:>11s} | {np_:>9d} | "
                  f"{emp.mean():>10.3e} ± {_sd(emp)} | "
                  f"{frob.mean():>10.3e} ± {_sd(frob)} | "
                  f"{th.mean():>10.3e} ± {_sd(th)} | "
                  f"{gap_f.mean():>+10.3e} ± {_sd(gap_f)}")
        print("=" * 120)


# ─── JSON output ─────────────────────────────────────────────────────────────


def _dump_json(path: Path, *, experiment: str, config_path: str, cfg: dict,
               rows: List[Dict]) -> None:
    """Write the JSON results file: ``{experiment, config_path, config, rows}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "config_path": config_path,
        "config": cfg,
        "rows": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  [json] wrote {len(rows)} rows to {path}")


def _default_output_json_path(config_path: str) -> Path:
    """Default output: ``results/synthetic/<config_stem>.json``."""
    stem = Path(config_path).stem
    return _ROOT / "results" / "synthetic" / f"{stem}.json"


# ─── Device selection ────────────────────────────────────────────────────────


def _resolve_device(arg: Optional[str]) -> torch.device:
    """Default preference: MPS, then CUDA, then CPU."""
    if arg is not None:
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config path (see scripts/configs/synthetic/).")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Override the config's variant list "
                             "(direct/circulant/frozen_id).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the config's seed list.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: cpu / mps / cuda. Default: mps→cuda→cpu.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path for the JSON results dump. "
                             "Default: results/synthetic/<config_stem>.json")
    parser.add_argument("--no_json", action="store_true",
                        help="Disable JSON output (still prints summary to stdout).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = list(args.seeds) if args.seeds is not None else list(cfg["seeds"])
    variants = list(args.variants) if args.variants is not None else list(cfg["variants"])
    n_grid = _resolve_n_grid(cfg)
    device = _resolve_device(args.device)

    print("=" * 96)
    print("  SYNTHETIC TRANSPORT RECOVERY")
    print(f"  Config: {args.config}")
    print(f"  n_dim: {cfg['n_dim']}  m={cfg['n_dim']*(cfg['n_dim']+1)//2}")
    print(f"  N grid: {n_grid}  k_neighbors: {cfg.get('k_neighbors')}")
    print(f"  Variants: {variants}")
    print(f"  epochs: {cfg['epochs']}  batch_size: {cfg['batch_size']}  lr: {cfg['lr']}")
    print(f"  num_reflections: {cfg.get('num_reflections')}")
    print(f"  Cholesky rescale: {cfg.get('rescale', True)}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print("=" * 96)

    n = int(cfg["n_dim"])
    m = n * (n + 1) // 2

    all_rows: List[Dict[str, object]] = []
    total_runs = len(seeds) * len(n_grid) * len(variants)
    run_idx = 0
    for seed in seeds:
        for n_nodes in n_grid:
            torch.manual_seed(seed * 1009 + n_nodes)
            np.random.seed((seed * 1009 + n_nodes) % (2 ** 31))

            # Build the graph + closed-form ground truth once per (seed, N).
            graph = _build_graph(cfg, n_nodes=n_nodes, seed=seed * 1009 + n_nodes)
            T_gt = closed_form_lc_transports(
                graph.covariances,
                graph.edge_index,
                n_steps=int(cfg["n_transport_steps"]),
                rescale=bool(cfg.get("rescale", True)),
            )
            num_edges = int(graph.edge_index.shape[1])

            # Inject n_nodes into the cfg passed to _train_variant (keeps the
            # function's signature stable for in-process smoke tests).
            sub_cfg = {**cfg, "n_nodes": n_nodes}

            for variant in variants:
                run_idx += 1
                print(
                    f"\n>>>>> [{run_idx}/{total_runs}]  seed={seed}  "
                    f"n_nodes={n_nodes}  variant={variant}  <<<<<", flush=True)

                # Analytical plateau: project T_gt onto the variant's class.
                T_proj = _project(T_gt, variant, cfg)
                # Loss is mean over (B, E, m) of squared error on `T_pred V`
                # where V ~ N(0, I_m) per node. Under that draw,
                # E[loss] = ‖T_pred - T_gt‖_F² / (E · m), and
                # projection_distance returns ‖.‖_F² / (E · m²), so
                # E[loss] = projection_distance(...) · m.
                theory_plateau = float(projection_distance(T_gt, T_proj).item()) * m

                train_metrics = _train_variant(
                    variant=variant,
                    T_gt=T_gt,
                    edge_index=graph.edge_index,
                    cfg=sub_cfg,
                    device=device,
                    seed=seed,
                )

                row = {
                    "seed": int(seed),
                    "n_dim": int(n),
                    "n_nodes": int(n_nodes),
                    "stalk_dim_m": int(m),
                    "num_edges": num_edges,
                    "variant": variant,
                    "n_params": int(train_metrics["n_params"]),
                    "empirical": float(train_metrics["best_loss"]),
                    "final_loss": float(train_metrics["final_loss"]),
                    "empirical_frob": float(train_metrics["best_frob"]),
                    "final_frob": float(train_metrics["final_frob"]),
                    "theory": theory_plateau,
                    "stopped_epoch": int(train_metrics.get("stopped_epoch",
                                                           int(cfg["epochs"]))),
                    # Aliases read by _print_summary.
                    "best_loss": float(train_metrics["best_loss"]),
                    "theory_plateau": theory_plateau,
                    "gap": float(train_metrics["best_loss"]) - theory_plateau,
                    "gap_frob": float(train_metrics["best_frob"]) - theory_plateau,
                }
                all_rows.append(row)
                print(f"    -> empirical={row['empirical']:.4e}  "
                      f"frob={row['empirical_frob']:.4e}  "
                      f"theory={theory_plateau:.4e}  gap={row['gap']:+.4e}  "
                      f"gap_frob={row['gap_frob']:+.4e}", flush=True)

    _print_summary(all_rows)

    if not args.no_json:
        out_path = (
            Path(args.output_json) if args.output_json
            else _default_output_json_path(args.config)
        )
        _dump_json(
            out_path,
            experiment="transport_recovery",
            config_path=args.config,
            cfg=cfg,
            rows=all_rows,
        )


if __name__ == "__main__":
    main()
