"""Render Figure 3 (spectral stability) from the operator-convergence JSON.

Reads the JSON written by ``scripts/bundle_operator_convergence.py`` and saves
``operator_convergence.pdf``: two side-by-side log-log panels, ``spectral_l2``
and ``spectral_relmax`` vs ``n_nodes``, one curve per fiber dimension
``d = n(n+1)/2``, with a mean ± sample-std band across seeds.

Usage:
    python scripts/bundle_make_figures.py \\
        --operator results/paper/operator_convergence.json --out_dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend; no display required.
import matplotlib.pyplot as plt  # noqa: E402


# ─── Style ───────────────────────────────────────────────────────────────────


def _setup_paper_style() -> None:
    """Compact serif look matching common LaTeX paper styles."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Embed TrueType fonts (no Type 3 fonts in the PDF).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ─── JSON loading + grouping ─────────────────────────────────────────────────


def _load_rows(json_path: Path) -> Tuple[Dict, List[Dict]]:
    with open(json_path) as f:
        payload = json.load(f)
    return payload, payload["rows"]


def _sample_std(vals: List[float]) -> float:
    """Sample standard deviation (ddof=1); 0 (no band) for a single seed."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _operator_curves(rows: List[Dict], metric: str
                     ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group operator-convergence rows by n_dim.

    Returns ``{n_dim: (n_nodes, mean, std)}`` arrays for the chosen metric.
    """
    by_n_dim: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_n_dim[int(r["n_dim"])][int(r["n_nodes"])].append(float(r[metric]))

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for n_dim, by_N in by_n_dim.items():
        Ns = np.array(sorted(by_N.keys()), dtype=np.int64)
        means = np.array([np.mean(by_N[int(N)]) for N in Ns], dtype=np.float64)
        stds = np.array([_sample_std(by_N[int(N)]) for N in Ns], dtype=np.float64)
        out[n_dim] = (Ns, means, stds)
    return out


# ─── Figure 3 ────────────────────────────────────────────────────────────────


def fig_operator_convergence(rows: List[Dict], out_pdf: Path) -> None:
    """Two-panel loglog figure: spectral_l2 (panel A), spectral_relmax (panel B).

    One curve per fiber dimension ``d`` in each panel; mean across seeds as
    solid line, sample std as filled band.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)

    # Sort n_dim values and assign sequential viridis colors so the dimension
    # ordering is visually clear.
    n_dims = sorted({int(r["n_dim"]) for r in rows})
    cmap = matplotlib.colormaps["viridis"]
    colors = {n_dim: cmap(i / max(len(n_dims) - 1, 1)) for i, n_dim in enumerate(n_dims)}

    metric_specs = [
        ("spectral_l2",     "spectral $L_2$ distance"),
        ("spectral_relmax", "spectral rel-max error"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        curves = _operator_curves(rows, metric)
        for n_dim in n_dims:
            Ns, means, stds = curves[n_dim]
            color = colors[n_dim]
            d = n_dim * (n_dim + 1) // 2  # fiber dimension of Sym(n_dim)
            label = f"$d={d}$"
            # Clamp std band away from non-positive values so log-y stays valid.
            lo = np.maximum(means - stds, means * 1e-3)
            hi = means + stds
            ax.plot(Ns, means, color=color, label=label, marker="o", markersize=3)
            ax.fill_between(Ns, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\rm nodes}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--operator", type=str, required=True,
                        help="JSON written by bundle_operator_convergence.py "
                             "(e.g. results/paper/operator_convergence.json).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for the PDF. Default: figures/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "figures"
    _setup_paper_style()

    op_path = Path(args.operator)
    if not op_path.exists():
        raise SystemExit(f"--operator JSON not found: {op_path}")
    _, rows = _load_rows(op_path)
    out_pdf = out_dir / "operator_convergence.pdf"
    fig_operator_convergence(rows, out_pdf)
    print(f"  [fig] wrote {out_pdf}  ({len(rows)} rows from {op_path})")


if __name__ == "__main__":
    main()
