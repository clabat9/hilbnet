from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hilbnet.utils import apply_sheaf_laplacian_from_transport


class HilbertConvLayer(nn.Module):
    """Polynomial filtering with an edge-wise sheaf Laplacian application.

    Input shape: [B, N, T, F_in]
    Operator action is applied edge-wise from one transport per oriented edge.
    Output shape: [B, N, T, F_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kappa: int,
        activation: nn.Module | None = None,
        filter_init: str = "xavier",
        filter_init_scale: float = 1e-2,
    ):
        super().__init__()
        if kappa < 1:
            raise ValueError("kappa must be >= 1")
        if filter_init not in ("xavier", "near_identity"):
            raise ValueError("filter_init must be 'xavier' or 'near_identity'.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kappa = int(kappa)
        self.activation = activation if activation is not None else nn.Identity()
        self.filter_init = filter_init
        self.filter_init_scale = float(filter_init_scale)
        self.weight = nn.Parameter(torch.empty(self.kappa, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # "near_identity": θ₀ gets standard xavier (pass-through channel mix), θ_k (k≥1)
        # get small random (scale ≪ 1) so the k=0 term dominates at init but gradient
        # still flows through higher-order Laplacian terms (θ_k enters linearly in L^k x).
        if self.filter_init == "near_identity":
            nn.init.xavier_uniform_(self.weight[0])
            for k in range(1, self.kappa):
                nn.init.normal_(self.weight[k], mean=0.0, std=self.filter_init_scale)
        else:
            for k in range(self.kappa):
                nn.init.xavier_uniform_(self.weight[k])
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        transport_maps: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
        assume_orthogonal: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, N, T, F].")

        propagated = x
        out = torch.einsum('bntf,fg->bntg', propagated, self.weight[0])
        for k in range(1, self.kappa):
            propagated = apply_sheaf_laplacian_from_transport(
                propagated,
                transport_maps=transport_maps,
                edge_index=edge_index,
                edge_weights=edge_weights,
                assume_orthogonal=assume_orthogonal,
            )
            out = out + torch.einsum('bntf,fg->bntg', propagated, self.weight[k])
        out = out + self.bias.view(1, 1, 1, -1)
        return self.activation(out)


# ---------------------------------------------------------------------------
# Standard graph Laplacian application (identity transport)
# ---------------------------------------------------------------------------

def apply_graph_laplacian(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply L x = D x − A x on [B, N, T, F] with oriented edge_index [2, E]."""
    B, N, T, F = x.shape
    src, dst = edge_index[0].long(), edge_index[1].long()
    E = src.shape[0]
    w = edge_weights.to(device=x.device, dtype=x.dtype) if edge_weights is not None \
        else torch.ones(E, device=x.device, dtype=x.dtype)

    w_view = w.view(1, -1, 1, 1)

    # Scatter adjacency: accumulate in [N, B, T, F] for contiguous index_add_
    out_t = torch.zeros(N, B, T, F, device=x.device, dtype=x.dtype)
    out_t.index_add_(0, dst, (w_view * x[:, src]).permute(1, 0, 2, 3))
    out_t.index_add_(0, src, (w_view * x[:, dst]).permute(1, 0, 2, 3))
    Ax = out_t.permute(1, 0, 2, 3)  # [B, N, T, F]

    deg = torch.zeros(N, device=x.device, dtype=x.dtype)
    deg.scatter_add_(0, src, w)
    deg.scatter_add_(0, dst, w)
    return deg.view(1, N, 1, 1) * x - Ax


# ---------------------------------------------------------------------------
# Spatiotemporal GNN layer: 1D temporal conv + spatial polynomial GCN
# ---------------------------------------------------------------------------

class SpatioTemporalConvLayer(nn.Module):
    """Temporal 1D conv (per node) followed by a polynomial graph filter (per time step).

    Equivalent to: Conv1d along T on X ∈ R^{N×T}, then poly-GCN across N.

    Input:  [B, N, T, F_in]
    Output: [B, N, T, F_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kappa: int,
        kernel_size: int = 15,
        activation: nn.Module | None = None,
        filter_init: str = "xavier",
        filter_init_scale: float = 1e-2,
    ):
        super().__init__()
        if filter_init not in ("xavier", "near_identity"):
            raise ValueError("filter_init must be 'xavier' or 'near_identity'.")
        self.temporal_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.kappa = int(kappa)
        self.spatial_weight = nn.Parameter(torch.empty(self.kappa, out_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.activation = activation if activation is not None else nn.Identity()
        self.filter_init = filter_init
        self.filter_init_scale = float(filter_init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.temporal_conv.weight)
        if self.temporal_conv.bias is not None:
            nn.init.zeros_(self.temporal_conv.bias)
        if self.filter_init == "near_identity":
            nn.init.xavier_uniform_(self.spatial_weight[0])
            for k in range(1, self.kappa):
                nn.init.normal_(self.spatial_weight[k], mean=0.0, std=self.filter_init_scale)
        else:
            for k in range(self.kappa):
                nn.init.xavier_uniform_(self.spatial_weight[k])
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, T, F_in = x.shape

        # Temporal 1D conv per node
        z = x.reshape(B * N, T, F_in).permute(0, 2, 1)   # [B*N, F_in, T]
        z = self.temporal_conv(z)                           # [B*N, F_out, T]
        z = z.permute(0, 2, 1).reshape(B, N, T, -1)       # [B, N, T, F_out]

        # Spatial polynomial filter on graph Laplacian
        propagated = z
        out = torch.einsum('bntf,fg->bntg', propagated, self.spatial_weight[0])
        for k in range(1, self.kappa):
            propagated = apply_graph_laplacian(propagated, edge_index, edge_weights)
            out = out + torch.einsum('bntf,fg->bntg', propagated, self.spatial_weight[k])
        out = out + self.bias.view(1, 1, 1, -1)
        return self.activation(out)
