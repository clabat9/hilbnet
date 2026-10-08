from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class OrientedEdges:
    """Canonical undirected edges oriented by increasing node index."""

    edge_index: torch.Tensor  # shape [2, E], with edge_index[0, e] < edge_index[1, e]
    edge_weights: Optional[torch.Tensor] = None

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])



def canonicalize_edge_index(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    reduce: str = "mean",
) -> OrientedEdges:
    """Return unique undirected edges oriented as u < v."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.numel() == 0:
        empty_w = None if edge_weights is None else edge_index.new_zeros((0,), dtype=torch.float32)
        return OrientedEdges(edge_index=edge_index.new_zeros((2, 0)), edge_weights=empty_w)

    edges = edge_index.long()
    u = torch.minimum(edges[0], edges[1])
    v = torch.maximum(edges[0], edges[1])
    mask = u != v
    u = u[mask]
    v = v[mask]

    pairs = torch.stack([u, v], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)

    if num_nodes is not None and unique_pairs.numel() > 0:
        if unique_pairs.min() < 0 or unique_pairs.max() >= num_nodes:
            raise ValueError("edge_index contains invalid node ids.")

    out_weights = None
    if edge_weights is not None:
        ew = edge_weights.reshape(-1)
        if ew.shape[0] != edges.shape[1]:
            raise ValueError("edge_weights must have one value per input edge.")
        ew = ew[mask].to(dtype=torch.float32)
        out_weights = torch.zeros(unique_pairs.shape[0], dtype=torch.float32, device=ew.device)
        counts = torch.zeros_like(out_weights)
        out_weights.scatter_add_(0, inverse, ew)
        counts.scatter_add_(0, inverse, torch.ones_like(ew))
        if reduce == "mean":
            out_weights = out_weights / counts.clamp_min(1)
        elif reduce == "sum":
            pass
        else:
            raise ValueError("reduce must be 'mean' or 'sum'.")

    return OrientedEdges(edge_index=unique_pairs.t().contiguous(), edge_weights=out_weights)



def default_edge_weights(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor], *, device, dtype) -> torch.Tensor:
    if edge_weights is None:
        return torch.ones(edge_index.shape[1], device=device, dtype=dtype)
    weights = edge_weights.to(device=device, dtype=dtype)
    if weights.ndim != 1 or weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights must have shape [E].")
    return weights


def build_householder_transport_maps(householder_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build orthogonal transport maps from Householder vectors.

    For each oriented edge e, this returns

        P_e = H_{e,1} H_{e,2} ... H_{e,R}

    where each reflector is

        H_{e,r} = I - 2 v_{e,r} v_{e,r}^T / (||v_{e,r}||_2^2 + eps).

    A zero vector yields the identity reflector, which provides an exact
    identity initialization.
    """
    if householder_vectors.ndim != 3:
        raise ValueError("householder_vectors must have shape [E, R, T].")

    num_edges, _, stalk_dim = householder_vectors.shape
    eye = torch.eye(stalk_dim, device=householder_vectors.device, dtype=householder_vectors.dtype)
    transport_maps = eye.unsqueeze(0).repeat(num_edges, 1, 1)

    for reflection_idx in range(householder_vectors.shape[1]):
        vectors = householder_vectors[:, reflection_idx, :]
        denom = vectors.pow(2).sum(dim=-1, keepdim=True).unsqueeze(-1) + eps
        outer = vectors.unsqueeze(-1) * vectors.unsqueeze(-2)
        reflection = eye.unsqueeze(0) - 2.0 * outer / denom
        transport_maps = torch.matmul(transport_maps, reflection)

    return transport_maps



def apply_sheaf_laplacian_from_transport(
    x: torch.Tensor,
    transport_maps: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    assume_orthogonal: bool = False,
) -> torch.Tensor:
    """Apply the transport-parameterized sheaf Laplacian without materializing [NT, NT].

    Parameters
    ----------
    x:
        Node signals with shape [B, N, T, F].
    transport_maps:
        Edge transports with shape [E, T, T].
    edge_index:
        Canonical oriented edges [2, E].
    edge_weights:
        Optional weights [E].
    assume_orthogonal:
        If True, use the orthogonal-transport identity P_e^T P_e = I.

    Returns
    -------
    torch.Tensor
        The Laplacian action Lx with the same shape as x.
    """
    if x.ndim != 4:
        raise ValueError("x must have shape [B, N, T, F].")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if transport_maps.ndim != 3:
        raise ValueError("transport_maps must have shape [E, T, T].")

    batch_size, num_nodes, stalk_dim, num_features = x.shape
    num_edges = edge_index.shape[1]
    if transport_maps.shape != (num_edges, stalk_dim, stalk_dim):
        raise ValueError("transport_maps must have shape [E, T, T] compatible with x.")

    # Allocate output in [N, B, T, F] order so index_add_ on dim=0 operates on a
    # contiguous tensor. MPS silently no-ops index_add_ on non-contiguous views,
    # so we must not permute a [B,N,T,F] zeros tensor and then write into it.
    out_t = torch.zeros(
        num_nodes, batch_size, stalk_dim, num_features,
        device=x.device, dtype=x.dtype,
    )  # [N, B, T, F]
    if num_edges == 0:
        return out_t.permute(1, 0, 2, 3)

    src = edge_index[0].long().to(x.device)
    dst = edge_index[1].long().to(x.device)
    weights = default_edge_weights(edge_index, edge_weights, device=x.device, dtype=x.dtype)

    x_u = x[:, src, :, :]  # [B, E, T, F]
    x_v = x[:, dst, :, :]  # [B, E, T, F]

    P = transport_maps.to(device=x.device, dtype=x.dtype)
    Pt = P.transpose(-1, -2)

    Pxv = torch.matmul(P.unsqueeze(0), x_v)
    Ptxu = torch.matmul(Pt.unsqueeze(0), x_u)

    weighted_xu = weights.view(1, num_edges, 1, 1) * x_u
    weighted_xv = weights.view(1, num_edges, 1, 1) * x_v
    weighted_Pxv = weights.view(1, num_edges, 1, 1) * Pxv
    weighted_Ptxu = weights.view(1, num_edges, 1, 1) * Ptxu

    contrib_src = weighted_xu - weighted_Pxv
    if assume_orthogonal:
        contrib_dst = -weighted_Ptxu + weighted_xv
    else:
        PtPxv = torch.matmul(Pt.unsqueeze(0), Pxv)
        weighted_PtPxv = weights.view(1, num_edges, 1, 1) * PtPxv
        contrib_dst = -weighted_Ptxu + weighted_PtPxv

    # Permute contributions to [E, B, T, F] and scatter into contiguous out_t.
    out_t.index_add_(0, src, contrib_src.permute(1, 0, 2, 3).contiguous())
    out_t.index_add_(0, dst, contrib_dst.permute(1, 0, 2, 3).contiguous())
    return out_t.permute(1, 0, 2, 3)  # [B, N, T, F]



def build_sheaf_laplacian_from_transport(
    transport_maps: torch.Tensor,
    edge_index: torch.Tensor,
    stalk_dim: int,
    edge_weights: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    return_blocks: bool = False,
    assume_orthogonal: bool = False,
) -> torch.Tensor:
    """Build a genuine sheaf Laplacian from one learnable transport per oriented edge."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if transport_maps.ndim != 3:
        raise ValueError("transport_maps must have shape [E, T, T].")
    num_edges = edge_index.shape[1]
    if transport_maps.shape[0] != num_edges:
        raise ValueError("transport_maps and edge_index disagree on the number of edges.")
    if transport_maps.shape[1] != stalk_dim or transport_maps.shape[2] != stalk_dim:
        raise ValueError("transport_maps must have shape [E, T, T] with T=stalk_dim.")

    device = transport_maps.device
    dtype = transport_maps.dtype
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if num_edges > 0 else 0

    weights = default_edge_weights(edge_index, edge_weights, device=device, dtype=dtype)
    blocks = torch.zeros((num_nodes, num_nodes, stalk_dim, stalk_dim), device=device, dtype=dtype)
    eye = torch.eye(stalk_dim, device=device, dtype=dtype)

    if num_edges == 0:
        return blocks if return_blocks else blocks.permute(0, 2, 1, 3).reshape(num_nodes * stalk_dim, num_nodes * stalk_dim)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    P = transport_maps
    Pt = P.transpose(1, 2)
    PtP = None if assume_orthogonal else Pt @ P

    for e in range(num_edges):
        u = int(src[e].item())
        v = int(dst[e].item())
        w = weights[e]
        blocks[u, u] += w * eye
        blocks[u, v] += -w * P[e]
        blocks[v, u] += -w * Pt[e]
        blocks[v, v] += w * (eye if assume_orthogonal else PtP[e])

    if return_blocks:
        return blocks
    return blocks.permute(0, 2, 1, 3).reshape(num_nodes * stalk_dim, num_nodes * stalk_dim)
