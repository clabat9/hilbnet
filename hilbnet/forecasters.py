"""Forecasting models: HilbNet and the spatiotemporal graph baseline (STGNNConv).

Both classes consume input shape ``[B, N, T_in, F_in]`` and produce
``[B, N, T_out]`` per-node multi-step forecasts. They run in z-scored
input space; the training code (``scripts/traffic_eval.py``) is
responsible for de-normalizing predictions before MAE/RMSE/MAPE metrics.

The regression head is a single ``nn.Linear(T_in * F_last, T_out)`` shared
across nodes, the standard DCRNN-style readout. There is no spatial pooling,
since forecasting needs a per-node output.

Readout-parity invariant: HilbNetForecaster and STGNNConvForecaster differ only
in their layer stack (``HilbertConvLayer`` vs ``SpatioTemporalConvLayer``).
The head, dropout, and forward pipeline are identical, so the comparison
isolates the layer choice. Enforced by
``tests/test_forecasters.py::TestReadoutParityHilbNetVsSTGNNConv``.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hilbnet._transport_setup import setup_transport
from hilbnet.layers import HilbertConvLayer, SpatioTemporalConvLayer
from hilbnet.utils import canonicalize_edge_index, default_edge_weights


class PairwiseKernelLoss(nn.Module):
    """Differentiable kernel-similarity penalty on transported edge signals."""

    def __init__(self, kernel: str = "rbf", kernel_param: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if kernel not in {"rbf", "cosine", "linear"}:
            raise ValueError("kernel must be one of {'rbf', 'cosine', 'linear'}")
        self.kernel = kernel
        self.kernel_param = float(kernel_param)
        self.eps = float(eps)

    def forward(self, x_u: torch.Tensor, transported_x_v: torch.Tensor) -> torch.Tensor:
        if self.kernel == "rbf":
            diff_sq = (x_u - transported_x_v).pow(2).mean(dim=-1)
            sim = torch.exp(-diff_sq / max(self.kernel_param, self.eps))
            return (1.0 - sim).mean()
        if self.kernel == "cosine":
            sim = F.cosine_similarity(x_u, transported_x_v, dim=-1, eps=self.eps)
            return (1.0 - sim).mean()
        sim = (x_u * transported_x_v).sum(dim=-1)
        return (-sim / x_u.shape[-1]).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Shared head construction (regression: per-node, multi-step).
# ─────────────────────────────────────────────────────────────────────────────


def _make_regression_head(
    f_last: int, t_in: int, t_out: int,
    dropout: float, dtype: torch.dtype,
) -> nn.Module:
    """Per-node Linear from concatenated [T_in, F_last] features to T_out steps.

    The same projection is applied to every node (parallelism via reshape +
    matmul, not per-node parameters). Standard DCRNN-style readout. ``Dropout``
    sits on the flattened input to the head.
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(t_in * f_last, t_out, dtype=dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HilbNetForecaster
# ─────────────────────────────────────────────────────────────────────────────


class HilbNetForecaster(nn.Module):
    """Sheaf-network forecaster: sheaf convolution layers plus a per-node linear head.

    Forward:
        x: [B, N, T_in, F_in]   (z-scored speeds + optional time-of-day)
        →   layer stack ([B, N, T_in, F_last])
        →   reshape to [B, N, T_in * F_last]
        →   Linear → [B, N, T_out]
    """

    def __init__(
        self,
        n_nodes: int,
        time_steps_in: int,
        time_steps_out: int,
        edge_index: torch.Tensor,
        in_features: Iterable[int],
        activation: nn.Module,
        kappa: Iterable[int],
        dropout: float = 0.0,
        edge_weights: Optional[torch.Tensor] = None,
        reg_param: float = 0.0,
        kernel: str = "rbf",
        kernel_param: float = 1.0,
        transport_init: str = "identity",
        transport_param_type: str = "direct",
        num_householder_reflections: Optional[int] = None,
        householder_eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        filter_init: str = "xavier",
        filter_init_scale: float = 1e-2,
        num_bands: Optional[int] = None,
    ):
        super().__init__()
        features = list(in_features)
        kappas = list(kappa)
        if len(features) < 2:
            raise ValueError("in_features must contain at least input and output dimensions.")
        if len(kappas) != len(features) - 1:
            raise ValueError("kappa must provide one value per convolution layer.")

        oriented = canonicalize_edge_index(edge_index, num_nodes=n_nodes, edge_weights=edge_weights)
        self.edge_index = oriented.edge_index
        self.n_nodes = int(n_nodes)
        self.time_steps_in = int(time_steps_in)
        self.time_steps_out = int(time_steps_out)
        # `time_steps` is the stalk dim (= T_in for forecasting; transports
        # operate on input-window-length stalks). Transport-setup, kernel-loss,
        # and other utilities expect this name.
        self.time_steps = int(time_steps_in)
        self.reg_param = float(reg_param)
        self.dtype_ = dtype
        self.householder_eps = float(householder_eps)
        self.transport_param_type = transport_param_type
        self.num_householder_reflections = (
            self.time_steps_in if num_householder_reflections is None
            else int(num_householder_reflections)
        )
        if self.num_householder_reflections < 1:
            raise ValueError("num_householder_reflections must be >= 1.")

        # Symmetric edge-weight normalization D^{-1/2} W D^{-1/2}.
        raw_weights = default_edge_weights(
            self.edge_index, oriented.edge_weights,
            device=self.edge_index.device, dtype=dtype,
        )
        src, dst = self.edge_index[0], self.edge_index[1]
        deg = torch.zeros(n_nodes, device=raw_weights.device, dtype=dtype)
        deg.scatter_add_(0, src, raw_weights)
        deg.scatter_add_(0, dst, raw_weights)
        deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
        normalized_weights = raw_weights * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        self.register_buffer("edge_weights", normalized_weights)

        self.num_bands = num_bands
        setup_transport(
            self,
            kappas=kappas,
            edge_index=self.edge_index,
            time_steps=self.time_steps_in,
            transport_param_type=transport_param_type,
            transport_init=transport_init,
            num_householder_reflections=self.num_householder_reflections,
            num_bands=num_bands,
            dtype=dtype,
        )
        self.kernel_loss = PairwiseKernelLoss(kernel=kernel, kernel_param=kernel_param)

        # No internal LayerNorm — the input is already z-scored by the loader's
        # train-only StandardScaler, and DCRNN-style baselines don't add an
        # internal norm. (Adding one over [F_in=2] mixed with time-of-day
        # would also blunt the tod signal.)
        self.norms = nn.ModuleList([nn.Identity() for _ in features[:-1]])

        self.sheaf_layers = nn.ModuleList([
            HilbertConvLayer(
                in_channels=features[i],
                out_channels=features[i + 1],
                kappa=kappas[i],
                activation=activation,
                filter_init=filter_init,
                filter_init_scale=filter_init_scale,
            )
            for i in range(len(features) - 1)
        ])

        self.head = _make_regression_head(
            f_last=features[-1],
            t_in=self.time_steps_in,
            t_out=self.time_steps_out,
            dropout=dropout,
            dtype=dtype,
        )

    # ── transport accessors so the test fixtures
    #    that introspect `transport_maps` / `transport_parameters` work ──

    @property
    def transport_maps(self) -> Optional[torch.Tensor]:
        if not self._uses_transport:
            return None
        from hilbnet.utils import build_householder_transport_maps
        if self.transport_param_type == "circulant":
            return self.transport_param_module()
        # direct
        vectors = self.householder_vectors
        return build_householder_transport_maps(vectors, eps=self.householder_eps)

    def transport_parameters(self) -> List[nn.Parameter]:
        if not self._uses_transport:
            return []
        if self.transport_param_type == "circulant":
            return list(self.transport_param_module.parameters())
        return [self.householder_vectors]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B, N, T_in, F_in]; got {tuple(x.shape)}."
            )
        transport_maps = self.transport_maps
        if transport_maps is None:
            ref = self.sheaf_layers[0].weight
            x = x.to(dtype=ref.dtype, device=ref.device)
            ref_device = ref.device
        else:
            x = x.to(dtype=transport_maps.dtype, device=transport_maps.device)
            ref_device = transport_maps.device

        z = x
        for norm, layer in zip(self.norms, self.sheaf_layers):
            z = norm(z)
            z = layer(
                z,
                transport_maps=transport_maps,
                edge_index=self.edge_index.to(ref_device),
                edge_weights=self.edge_weights.to(ref_device),
                assume_orthogonal=True,
            )
        # z: [B, N, T_in, F_last]
        B, N, T_in, F_last = z.shape
        z = z.reshape(B, N, T_in * F_last)
        out = self.head(z)  # [B, N, T_out]
        return out

    def kernel_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth-section penalty on the speed channel only (channel 0).

        Multi-feature inputs (speed + time-of-day) need to be reduced to the
        single-channel form the penalty was derived for. We use the speed
        channel since the regularizer is meant to encourage transport-matched
        speed signals across edges, not periodic time-of-day.
        """
        if x.ndim == 4:
            x = x[..., 0]  # [B, N, T_in]
        if x.ndim != 3:
            raise ValueError(
                f"kernel_penalty expects [B, N, T] or [B, N, T, F]; got {tuple(x.shape)}."
            )
        if not self._uses_transport:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        transport_maps = self.transport_maps
        x = x.to(device=transport_maps.device, dtype=transport_maps.dtype)
        if self.edge_index.shape[1] == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        src = self.edge_index[0].to(transport_maps.device)
        dst = self.edge_index[1].to(transport_maps.device)
        x_u = x[:, src, :]
        x_v = x[:, dst, :]
        transported_x_v = torch.matmul(
            transport_maps.unsqueeze(0), x_v.unsqueeze(-1)
        ).squeeze(-1)
        return self.kernel_loss(
            x_u.reshape(-1, self.time_steps_in),
            transported_x_v.reshape(-1, self.time_steps_in),
        )


# ─────────────────────────────────────────────────────────────────────────────
# STGNNConvForecaster
# ─────────────────────────────────────────────────────────────────────────────


class STGNNConvForecaster(nn.Module):
    """Minimal CNN-GCN forecasting baseline: SpatioTemporalConvLayer + Linear head.

    Differs from HilbNetForecaster only in the layer stack (no transport,
    explicit per-node temporal Conv1d). Readout, head, and forward shape
    contract are identical — readout-parity invariant.

    A minimal spatiotemporal-GNN baseline: plain Conv1d → ReLU, polynomial of
    the unnormalized combinatorial Laplacian L=D-A (kappa=2 by default), no
    residuals, no LayerNorm — a "PolyGCN+TCN" architecture rather than the
    full Yu-2018 STGCN sandwich. Used as the "spatiotemporal graph baseline"
    row in the paper's Table 2.
    """

    def __init__(
        self,
        n_nodes: int,
        time_steps_in: int,
        time_steps_out: int,
        edge_index: torch.Tensor,
        in_features: Iterable[int],
        activation: nn.Module,
        kappa: Iterable[int],
        dropout: float = 0.0,
        edge_weights: Optional[torch.Tensor] = None,
        kernel_size: int = 3,
        dtype: torch.dtype = torch.float32,
        filter_init: str = "xavier",
        filter_init_scale: float = 1e-2,
    ):
        super().__init__()
        features = list(in_features)
        kappas = list(kappa)
        if len(features) < 2:
            raise ValueError("in_features must contain at least input and output dimensions.")
        if len(kappas) != len(features) - 1:
            raise ValueError("kappa must provide one value per convolution layer.")

        oriented = canonicalize_edge_index(edge_index, num_nodes=n_nodes, edge_weights=edge_weights)
        self.edge_index = oriented.edge_index
        self.n_nodes = int(n_nodes)
        self.time_steps_in = int(time_steps_in)
        self.time_steps_out = int(time_steps_out)
        self.dtype_ = dtype

        raw_weights = default_edge_weights(
            self.edge_index, oriented.edge_weights,
            device=self.edge_index.device, dtype=dtype,
        )
        src, dst = self.edge_index[0], self.edge_index[1]
        deg = torch.zeros(n_nodes, device=raw_weights.device, dtype=dtype)
        deg.scatter_add_(0, src, raw_weights)
        deg.scatter_add_(0, dst, raw_weights)
        deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
        normalized_weights = raw_weights * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        self.register_buffer("edge_weights", normalized_weights)

        self.norms = nn.ModuleList([nn.Identity() for _ in features[:-1]])

        self.blocks = nn.ModuleList([
            SpatioTemporalConvLayer(
                in_channels=features[i],
                out_channels=features[i + 1],
                kappa=kappas[i],
                kernel_size=kernel_size,
                activation=activation,
                filter_init=filter_init,
                filter_init_scale=filter_init_scale,
            )
            for i in range(len(features) - 1)
        ])

        self.head = _make_regression_head(
            f_last=features[-1],
            t_in=self.time_steps_in,
            t_out=self.time_steps_out,
            dropout=dropout,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B, N, T_in, F_in]; got {tuple(x.shape)}."
            )
        z = x.to(dtype=self.edge_weights.dtype, device=self.edge_weights.device)
        for norm, block in zip(self.norms, self.blocks):
            z = norm(z)
            z = block(z, self.edge_index.to(z.device), self.edge_weights.to(z.device))
        # z: [B, N, T_in, F_last]
        B, N, T_in, F_last = z.shape
        z = z.reshape(B, N, T_in * F_last)
        out = self.head(z)
        return out

    def kernel_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """No transport in STGNNConv; penalty is zero by construction."""
        return torch.zeros((), device=x.device, dtype=x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper (kept here so train loops can import once and use uniformly)
# ─────────────────────────────────────────────────────────────────────────────


def masked_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean absolute error. Mask is 1 where target is valid, 0 elsewhere.

    All three tensors must be in the same units (e.g. all z-scored or all raw).
    """
    if mask.dtype != pred.dtype:
        mask = mask.to(pred.dtype)
    abs_err = (pred - target).abs() * mask
    return abs_err.sum() / mask.sum().clamp(min=1.0)
