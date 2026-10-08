"""Spectral stability of the discrete sheaf Laplacian (paper Figure 3) on the
synthetic statistical bundle over Sym++(n) with the Levi-Civita connection of
the Wasserstein metric.

No learning. Closed-form Levi-Civita transports are plugged into HilbNet's
``build_sheaf_laplacian_from_transport`` (``hilbnet/utils.py``); we sweep the
sample size ``n_nodes`` and the manifold dimension ``n_dim`` and measure how
the resulting discrete operator approaches a reference operator computed at
``n_max`` samples. Metrics:

* spectral: bottom-``top_k`` eigenvalues of ``L_N`` vs those of the reference
  operator ``L_{n_max}``, reported as an L2 distance (``spectral_l2``) and a
  relative max-eigenvalue error (``spectral_relmax``). These are the two
  panels of Figure 3.
* section action: ``‖L_N f_N‖`` for a smooth synthetic test section
  ``f(Σ) = sym_to_vec(Σ - I)`` (rescaled), ``√N``-normalized so the scale is
  comparable across sample sizes (``section_norm``; recorded, not plotted).

Configs accept either a single ``n_dim``/``n_max`` (smoke config) or
``n_dim_grid`` + ``n_max_by_n_dim`` (paper config); the loop iterates each
``n_dim`` and produces one curve per dimension.

Usage:
    python scripts/bundle_operator_convergence.py \\
        --config scripts/configs/synthetic/operator_convergence_paper.yaml

    # Smoke test:
    python scripts/bundle_operator_convergence.py \\
        --config scripts/configs/synthetic/operator_convergence_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from hilbnet.bundle_validation import (  # noqa: E402
    assemble_sheaf_laplacian,
    cholesky_factors_per_node,
    eigenvalues_smallest,
)
from hilbnet.statistical_bundle import (  # noqa: E402
    DistanceType,
    FeatureType,
    GraphType,
    StatisticalBundleGraphConfig,
    StatisticalBundleGraphDataset,
    sym_to_vec,
)
from scripts._config_utils import load_config  # noqa: E402


# ─── Bundle-graph builder ────────────────────────────────────────────────────


def _build_graph(cfg: dict, n_nodes: int, seed: int):
    """One synthetic-bundle graph, deterministic in (cfg, n_nodes, seed)."""
    bcfg = StatisticalBundleGraphConfig(
        n=cfg["n_dim"],
        n_nodes=n_nodes,
        feature_type=FeatureType.TANGENT_VEC,         # cheapest — we don't use x
        graph_type=GraphType(cfg["graph_type"]),
        distance_type=DistanceType(cfg["distance_type"]),
        k_neighbors=cfg.get("k_neighbors", 8),
        epsilon=cfg.get("epsilon", 1.0),
        include_fiber=False,                          # we don't need tangents here
        edge_weight_sigma=cfg.get("edge_weight_sigma", 1.0),
    )
    ds = StatisticalBundleGraphDataset(bcfg, n_graphs=1, seed=seed)
    return ds.data_list[0]


# ─── Smooth test section (section_norm metric) ───────────────────────────────


def _build_test_section(graph, rescale: bool) -> torch.Tensor:
    """Smooth synthetic section ``f(Σ) = sym_to_vec(Σ - I)``.

    In rescaled coordinates we apply ``R_Σ`` per node so the section lives in
    the same basis as the (rescaled) sheaf Laplacian.
    """
    cov = graph.covariances
    n = cov.shape[-1]
    eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
    raw = sym_to_vec(cov - eye)                           # (N, m)
    if not rescale:
        return raw.flatten()
    R = cholesky_factors_per_node(cov)                    # (N, m, m)
    return torch.einsum("imn,in->im", R, raw).flatten()   # (N*m,)


# ─── n_dim grid resolution ───────────────────────────────────────────────────


def _resolve_n_dim_grid(cfg: dict) -> List[Tuple[int, int]]:
    """Return a list of ``(n_dim, n_max)`` pairs from the config.

    Two accepted shapes:
      * paper: ``n_dim_grid: [...]`` + ``n_max_by_n_dim: {n: n_max, ...}``
      * legacy/smoke: scalar ``n_dim`` + scalar ``n_max``
    """
    if "n_dim_grid" in cfg:
        n_dim_list = [int(x) for x in cfg["n_dim_grid"]]
        n_max_map = cfg.get("n_max_by_n_dim", {})
        # YAML loads keys as ints when the YAML uses int keys (default) — be defensive.
        n_max_map = {int(k): int(v) for k, v in n_max_map.items()}
        default_n_max = cfg.get("n_max")
        out: List[Tuple[int, int]] = []
        for n_dim in n_dim_list:
            if n_dim in n_max_map:
                out.append((n_dim, n_max_map[n_dim]))
            elif default_n_max is not None:
                out.append((n_dim, int(default_n_max)))
            else:
                raise ValueError(
                    f"n_dim={n_dim} has no n_max: provide n_max_by_n_dim[{n_dim}] or top-level n_max."
                )
        return out
    return [(int(cfg["n_dim"]), int(cfg["n_max"]))]


# ─── Per-(seed, n_dim) sweep ─────────────────────────────────────────────────


def _eigvalsh_safe(L: torch.Tensor, device: torch.device, k: int) -> torch.Tensor:
    """``eigenvalues_smallest`` with a CPU bounce when device is MPS.

    MPS support for ``torch.linalg.eigvalsh`` on large dense matrices is
    patchy; CUDA handles it natively, CPU is always safe.
    """
    if device.type == "mps":
        return eigenvalues_smallest(L.cpu(), k=k)
    return eigenvalues_smallest(L, k=k).cpu()


def _sweep_seed(
    cfg: dict,
    seed: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    """One seed × one n_dim: build reference + sweep ``n_grid``.

    The config here is expected to be already specialized for a single
    ``n_dim`` / ``n_max`` (see :func:`_resolve_n_dim_grid` callers). The
    returned rows include ``n_dim`` and ``n_ref`` for downstream grouping.
    """
    rescale = bool(cfg.get("rescale", True))
    n_steps = int(cfg["n_transport_steps"])
    top_k = int(cfg["top_k"])
    n_dim = int(cfg["n_dim"])
    n_max = int(cfg["n_max"])

    # Reference (large N) — distinct seed offset to avoid sample reuse.
    ref_seed = seed * 1009 + n_dim * 7919
    print(f"  [seed={seed} n_dim={n_dim}] building reference at n_max={n_max} ...", flush=True)
    ref_graph = _build_graph(cfg, n_nodes=n_max, seed=ref_seed)
    L_ref = assemble_sheaf_laplacian(
        ref_graph.covariances.to(device),
        ref_graph.edge_index.to(device),
        edge_weights=ref_graph.edge_attr.squeeze(-1).to(device),
        n_steps=n_steps,
        rescale=rescale,
    )
    eigs_ref = _eigvalsh_safe(L_ref, device, k=top_k)
    eigs_ref_max = float(eigs_ref.abs().max().item())
    print(f"  [seed={seed} n_dim={n_dim}] reference top-{top_k} eigenvalue range "
          f"[{eigs_ref.min().item():.4f}, {eigs_ref.max().item():.4f}]", flush=True)
    del L_ref

    rows: List[Dict[str, float]] = []
    for N in cfg["n_grid"]:
        N = int(N)
        graph = _build_graph(cfg, n_nodes=N, seed=ref_seed + N)
        L_N = assemble_sheaf_laplacian(
            graph.covariances.to(device),
            graph.edge_index.to(device),
            edge_weights=graph.edge_attr.squeeze(-1).to(device),
            n_steps=n_steps,
            rescale=rescale,
        )

        # Spectral metrics (the two panels of Figure 3).
        eigs_N = _eigvalsh_safe(L_N, device, k=top_k)
        spectral_l2 = float(((eigs_N - eigs_ref) ** 2).sum().sqrt().item() / max(top_k, 1))
        spectral_relmax = float(
            (eigs_N - eigs_ref).abs().max().item() / max(eigs_ref_max, 1e-9)
        )

        # Section action.
        f_N = _build_test_section(graph, rescale=rescale).to(device)
        section_norm = float((L_N @ f_N).norm().item() / math.sqrt(max(N, 1)))

        row = {
            "seed": int(seed),
            "n_dim": n_dim,
            "n_nodes": N,
            "n_ref": n_max,
            "top_k": top_k,
            "spectral_l2": spectral_l2,
            "spectral_relmax": spectral_relmax,
            "section_norm": section_norm,
        }
        rows.append(row)

        print(f"  [seed={seed} n_dim={n_dim}]  N={N:>5d}  "
              f"spectral_l2={spectral_l2:.4e}  "
              f"spectral_relmax={spectral_relmax:.4e}  "
              f"section_norm={section_norm:.4e}", flush=True)

        del L_N, f_N

    return rows


# ─── Reporting ───────────────────────────────────────────────────────────────


def _loglog_slope(xs: List[int], ys: List[float]) -> Optional[float]:
    """Least-squares slope of log y vs log x. Returns None if any y is non-positive."""
    if any(y <= 0 for y in ys) or len(xs) < 2:
        return None
    log_x = np.log(np.asarray(xs, dtype=np.float64))
    log_y = np.log(np.asarray(ys, dtype=np.float64))
    slope, _ = np.polyfit(log_x, log_y, 1)
    return float(slope)


def _sample_std(vals: List[float]) -> float:
    """Sample standard deviation (ddof=1), as reported in the paper; NaN for one seed."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _sd(vals: List[float]) -> str:
    """Formatted sample std; "-" when there is a single seed."""
    s = _sample_std(vals)
    return f"{s:>8.2e}" if math.isfinite(s) else f"{'-':>8s}"


def _print_summary(all_rows: List[Dict[str, float]]) -> None:
    """Per-n_dim table with mean ± sample std across seeds, plus log-log slopes."""
    if not all_rows:
        return
    metric_keys = ["spectral_l2", "spectral_relmax", "section_norm"]
    n_dims = sorted({int(r["n_dim"]) for r in all_rows})

    for n_dim in n_dims:
        sub = [r for r in all_rows if int(r["n_dim"]) == n_dim]
        seeds = sorted({int(r["seed"]) for r in sub})
        Ns = sorted({int(r["n_nodes"]) for r in sub})

        print("\n" + "=" * 80)
        print(f"  OPERATOR / SPECTRAL CONVERGENCE  n_dim={n_dim}  m={n_dim*(n_dim+1)//2}")
        print(f"  Seeds: {seeds}    N grid: {Ns}")
        print("=" * 80)
        header = f"{'N':>6s}"
        for k in metric_keys:
            header += f" | {k:>22s}"
        print(header)
        print("-" * len(header))

        for N in Ns:
            line = f"{N:>6d}"
            for k in metric_keys:
                vals = [r[k] for r in sub if int(r["n_nodes"]) == N]
                line += f" | {np.mean(vals):>10.4e} ± {_sd(vals)}"
            print(line)

        print("\n  Log-log slopes (mean across seeds, lower is better — should be < 0):")
        for k in metric_keys:
            per_seed = []
            for s in seeds:
                ys = [
                    r[k] for r in sorted(
                        [r for r in sub if int(r["seed"]) == s],
                        key=lambda r: int(r["n_nodes"]),
                    )
                ]
                per_seed.append(_loglog_slope(Ns, ys))
            valid = [v for v in per_seed if v is not None]
            if valid:
                print(f"    {k:>22s}:  mean slope = {np.mean(valid):+.3f}  "
                      f"(per seed: {[f'{v:+.3f}' if v is not None else 'NA' for v in per_seed]})")
            else:
                print(f"    {k:>22s}:  slope undefined (non-positive values).")
        print("=" * 80)


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
    n_dim_pairs = _resolve_n_dim_grid(cfg)
    device = _resolve_device(args.device)

    print("=" * 80)
    print("  BUNDLE OPERATOR / SPECTRAL CONVERGENCE")
    print(f"  Config: {args.config}")
    print(f"  n_dim_grid: {[d for d, _ in n_dim_pairs]}  "
          f"n_max_by_n_dim: {{{', '.join(f'{d}: {m}' for d, m in n_dim_pairs)}}}")
    print(f"  N grid: {cfg['n_grid']}")
    print(f"  top_k eigenvalues: {cfg['top_k']}")
    print(f"  Cholesky rescale: {cfg.get('rescale', True)}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print("=" * 80)

    all_rows: List[Dict] = []
    for s in seeds:
        for n_dim, n_max in n_dim_pairs:
            torch.manual_seed(s * 1009 + n_dim * 7919)
            np.random.seed((s * 1009 + n_dim * 7919) % (2 ** 31))

            sub_cfg = {**cfg, "n_dim": n_dim, "n_max": n_max}
            rows = _sweep_seed(sub_cfg, s, device)
            all_rows.extend(rows)

    _print_summary(all_rows)

    if not args.no_json:
        out_path = (
            Path(args.output_json) if args.output_json
            else _default_output_json_path(args.config)
        )
        _dump_json(
            out_path,
            experiment="operator_convergence",
            config_path=args.config,
            cfg=cfg,
            rows=all_rows,
        )


if __name__ == "__main__":
    main()
