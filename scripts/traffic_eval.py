"""Traffic-forecasting training and evaluation (METR-LA / PEMS-BAY).

Library module used by ``scripts/traffic_table_runner.py``: it builds a model
from a YAML config, trains it for one seed with validation-based early stopping,
and evaluates it on the test split. Metrics are MAE / RMSE / MAPE at the
requested horizons (3 / 6 / 12 steps = 15 / 30 / 60 min), de-normalized to raw
mph and computed only on non-missing targets.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Allow importing hilbnet/* (sibling to scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilbnet.forecasters import (  # noqa: E402
    HilbNetForecaster,
    STGNNConvForecaster,
    masked_mae,
)


# ─── Model construction ──────────────────────────────────────────────────────


def build_forecasting_model(
    cfg: dict,
    n_nodes: int,
    t_in: int,
    t_out: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Construct a forecaster + Adam optimizer from a YAML config dict."""
    model_type = cfg["model"]
    features = cfg["features"]
    kappa = cfg.get("kappa")
    dropout = cfg.get("dropout", 0.0)
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)

    if model_type == "hilbnet":
        model = HilbNetForecaster(
            n_nodes=n_nodes,
            time_steps_in=t_in,
            time_steps_out=t_out,
            edge_index=edge_index,
            in_features=features,
            activation=nn.ReLU(),
            kappa=kappa,
            dropout=dropout,
            edge_weights=edge_attr,
            reg_param=cfg.get("reg_param", 0.0),
            kernel=cfg.get("kernel", "rbf"),
            kernel_param=cfg.get("kernel_param", 1.0),
            transport_init=cfg.get("transport_init", "identity_plus_noise"),
            transport_param_type=cfg.get("transport_param", "circulant"),
            num_householder_reflections=cfg.get("num_reflections", 8),
            filter_init=cfg.get("filter_init", "xavier"),
            filter_init_scale=cfg.get("filter_init_scale", 1e-2),
            num_bands=cfg.get("num_bands", None),
        )
        # Separate learning rate for the transport parameters; transport_lr == 0
        # freezes them (frozen-identity variant).
        transport_lr = cfg.get("transport_lr", lr)
        if transport_lr == 0:
            for p in model.transport_parameters():
                p.requires_grad_(False)
        transport_params = [p for p in model.transport_parameters() if p.requires_grad]
        transport_ids = {id(p) for p in transport_params}
        other_params = [p for p in model.parameters() if id(p) not in transport_ids]
        param_groups = []
        if transport_params:
            param_groups.append({"params": transport_params, "lr": transport_lr})
        param_groups.append({"params": other_params, "lr": lr})
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif model_type == "stgnn_conv":
        model = STGNNConvForecaster(
            n_nodes=n_nodes,
            time_steps_in=t_in,
            time_steps_out=t_out,
            edge_index=edge_index,
            in_features=features,
            activation=nn.ReLU(),
            kappa=kappa,
            dropout=dropout,
            edge_weights=edge_attr,
            kernel_size=cfg.get("kernel_size", 3),
            filter_init=cfg.get("filter_init", "xavier"),
            filter_init_scale=cfg.get("filter_init_scale", 1e-2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return model, optimizer


# ─── Batch packing + eval ────────────────────────────────────────────────────


def _stack_batch(
    batch, n_nodes: int, t_in: int, t_out: int, f_in: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape a PyG Batch of forecasting Data into dense [B, N, ...] tensors."""
    bsz = batch.num_graphs
    x = batch.x.reshape(bsz, n_nodes, t_in, f_in)
    y = batch.y.reshape(bsz, n_nodes, t_out)
    mask = batch.mask.reshape(bsz, n_nodes, t_out)
    return x, y, mask


def _eval_forecast_loader(
    model: nn.Module,
    loader: DataLoader,
    n_nodes: int,
    t_in: int,
    t_out: int,
    f_in: int,
    scaler_mean: float,
    scaler_std: float,
    device: torch.device,
    horizons: List[int],
) -> Dict[str, float]:
    """Run model on the loader, accumulate per-horizon MAE/RMSE/MAPE in raw units."""
    model.eval()
    sums = {f"{m}_{h}_num": 0.0 for h in horizons for m in ("mae", "rmse", "mape")}
    sums.update({f"{h}_denom": 0.0 for h in horizons})
    with torch.no_grad():
        for batch in loader:
            x, y_raw, mask = _stack_batch(batch.to(device), n_nodes, t_in, t_out, f_in)
            pred_norm = model(x)
            pred_raw = pred_norm * scaler_std + scaler_mean
            for h in horizons:
                idx = h - 1
                m = mask[..., idx]
                denom = m.sum().item()
                diff = (pred_raw[..., idx] - y_raw[..., idx]) * m
                sums[f"mae_{h}_num"] += diff.abs().sum().item()
                sums[f"rmse_{h}_num"] += (diff ** 2).sum().item()
                # MAPE: |pred-y|/|y| only where mask=1 (y > 0)
                y_safe = y_raw[..., idx].clamp(min=1e-6)
                sums[f"mape_{h}_num"] += ((diff.abs() / y_safe) * m).sum().item()
                sums[f"{h}_denom"] += denom
    out = {}
    for h in horizons:
        denom = max(sums[f"{h}_denom"], 1.0)
        out[f"mae_{h}"] = sums[f"mae_{h}_num"] / denom
        out[f"rmse_{h}"] = (sums[f"rmse_{h}_num"] / denom) ** 0.5
        out[f"mape_{h}"] = sums[f"mape_{h}_num"] / denom
    return out


# ─── Train + eval one (config, seed) ─────────────────────────────────────────


def train_and_forecast(
    cfg: dict,
    seed: int,
    device: torch.device,
    *,
    train_windows,
    val_windows,
    test_windows,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    scaler: dict,
    n_nodes: int,
    t_in: int,
    t_out: int,
    f_in: int,
    horizons: List[int],
    use_val: bool,
    model_name: str,
    param_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Train one config × seed and return the test metrics of the best-validation epoch.

    If ``param_counts`` is given, the model's learnable parameter count
    (``requires_grad=True``) is stored under ``model_name``.
    """
    print(f"  >> Starting {model_name} | seed={seed} — building model…")

    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_windows, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_windows, batch_size=batch_size, shuffle=False) if use_val else None
    test_loader = DataLoader(test_windows, batch_size=batch_size, shuffle=False)

    model, optimizer = build_forecasting_model(
        cfg, n_nodes, t_in, t_out, edge_index, edge_attr,
    )
    model.to(device)

    if param_counts is not None and model_name not in param_counts:
        param_counts[model_name] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    epochs = cfg.get("epochs", 80)
    patience = cfg.get("patience", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    reg_param = cfg.get("reg_param", 0.0)
    clip_grad_norm = cfg.get("clip_grad_norm", None)
    scaler_mean, scaler_std = scaler["mean"], scaler["std"]

    best_val_mae = float("inf")
    best_val_state = None
    epochs_without_improvement = 0

    print(f"  >> Training {model_name} | seed={seed} — {epochs} epochs"
          f"{f' (patience={patience})' if use_val else ''}, "
          f"{len(train_windows)} train / "
          f"{len(val_windows) if use_val else 0} val / {len(test_windows)} test, "
          f"{len(train_loader)} batches/epoch")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x, y_raw, mask = _stack_batch(batch.to(device), n_nodes, t_in, t_out, f_in)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(x)
            pred_raw = pred_norm * scaler_std + scaler_mean
            mae = masked_mae(pred_raw, y_raw, mask)
            penalty = model.kernel_penalty(x) if reg_param > 0 else torch.zeros((), device=device)
            loss = mae + reg_param * penalty
            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)

        # Val every epoch (cheap on METR-LA val = 3405 windows).
        val_mae = None
        if use_val:
            val_metrics = _eval_forecast_loader(
                model, val_loader, n_nodes, t_in, t_out, f_in,
                scaler_mean, scaler_std, device, horizons,
            )
            # Use mean MAE across reported horizons as the early-stopping criterion.
            val_mae = sum(val_metrics[f"mae_{h}"] for h in horizons) / len(horizons)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Print every 5 epochs (forecasting runs are typically shorter than
        # classification at the same wall-clock; finer log granularity helps
        # diagnose convergence).
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            val_str = f" val_mae(avg)={val_mae:.4f}" if val_mae is not None else ""
            best_str = f" best_val_mae={best_val_mae:.4f}" if use_val else ""
            print(f"    [{model_name}] seed={seed} epoch {epoch+1}/{epochs} "
                  f"loss={avg_loss:.4f}{val_str}{best_str}")

        if use_val and epochs_without_improvement >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs).")
            break

    # Final eval: load best-val weights, evaluate on test.
    if use_val and best_val_state is not None:
        model.load_state_dict(best_val_state)
    metrics = _eval_forecast_loader(
        model, test_loader, n_nodes, t_in, t_out, f_in,
        scaler_mean, scaler_std, device, horizons,
    )
    print(f"  >> Done. {model_name} | seed={seed} test: " + " ".join(
        f"MAE@{h}={metrics[f'mae_{h}']:.3f}" for h in horizons
    ))

    return metrics


# ─── Pretty-print results table ──────────────────────────────────────────────


def _aggregate(seed_metrics: List[Dict[str, float]], horizons: List[int]) -> Dict[str, Tuple[float, float]]:
    """Mean ± sample std (ddof=1, as in the paper) for each metric across seeds."""
    out = {}
    for h in horizons:
        for m in ("mae", "rmse", "mape"):
            key = f"{m}_{h}"
            vals = [seed_metrics[s][key] for s in range(len(seed_metrics))]
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            out[key] = (float(np.mean(vals)), std)
    return out


def _print_forecast_table(
    results: Dict[str, List[Dict[str, float]]],
    param_counts: Dict[str, int],
    horizons: List[int],
) -> None:
    title_line = "=" * 84
    headers = [f"@{h*5}min" for h in horizons]
    for metric_label, metric_key in [("MAE (mph)", "mae"), ("RMSE (mph)", "rmse"), ("MAPE (%)", "mape")]:
        # MAPE is stored as a fraction in the JSON; print it in percent as in Table 2.
        scale = 100.0 if metric_key == "mape" else 1.0
        print()
        print(title_line)
        print(f"  FINAL TEST {metric_label}: mean ± sample std over seeds")
        print(title_line)
        print(f"{'Model':<32} {'Params':>10} | " + " | ".join(f"{h:>14}" for h in headers))
        print("-" * 32 + " " + "-" * 10 + "-+-" + "-+-".join("-" * 14 for _ in headers))
        for model_name, seed_metrics in results.items():
            agg = _aggregate(seed_metrics, horizons)
            params = param_counts.get(model_name, 0)
            cells = []
            for h in horizons:
                mean, std = agg[f"{metric_key}_{h}"]
                std_str = f"{scale * std:5.3f}" if np.isfinite(std) else f"{'-':>5s}"
                cells.append(f"{scale * mean:6.3f}±{std_str}")
            row = f"{model_name:<32} {params:>10,d} | " + " | ".join(cells)
            print(row)
        print(title_line)
