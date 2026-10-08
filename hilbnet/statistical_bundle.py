#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Bundle Graph Dataset (PyTorch Geometric)

A PyTorch Geometric Dataset where:
- Nodes: Points Σ on the base manifold Sym++(n) sampled uniformly
- Edges: Connectivity based on geodesic (Wasserstein) distance
- Node Features: Distribution features (covariances, tangent vectors, etc.)

Mathematical Framework
======================

Base Manifold:
    M = Sym++(n) - positive definite matrices (covariances of N(0,Σ))

Graph Structure:
    - Nodes are uniformly sampled covariance matrices {Σ₁, ..., Σ_N}
    - Edges connect nodes within geodesic distance threshold (k-NN or ε-ball)
    - Edge weights: exp(-d²(Σᵢ, Σⱼ)/σ²) or inverse distance

Node Features (options):
    1. Vectorized covariance: vec(Σ) ∈ ℝ^{n(n+1)/2}
    2. Tangent vector: vec(V) for fiber element
    3. Log-eigenvalues: log(λ₁), ..., log(λₙ)
    4. Cholesky factors: vec(L) where Σ = LL^T

Statistical Bundle Fiber:
    At each node Σ, the fiber is T_Σ M ≅ Sym(n)
    Score function: s_V(y) = ½(Tr(Σ⁻¹V) - y^T Σ⁻¹VΣ⁻¹y)

Geodesic Distance (Wasserstein):
    W₂²(Σ₀, Σ₁) = Tr(Σ₀) + Tr(Σ₁) - 2Tr((Σ₀^{1/2}Σ₁Σ₀^{1/2})^{1/2})

References:
    Malagò, Montrucchio, Pistone (2018) arXiv:1801.09269v4
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import torch
from torch_geometric.data import Data, Dataset


# =============================================================================
# Enums and Configuration
# =============================================================================

class FeatureType(Enum):
    """Type of node features to compute."""
    COVARIANCE_VEC = "covariance_vec"       # Vectorized Σ
    LOG_EIGENVALUES = "log_eigenvalues"     # log(λ₁), ..., log(λₙ)
    CHOLESKY_VEC = "cholesky_vec"           # Vectorized Cholesky factor
    TANGENT_VEC = "tangent_vec"             # Fiber element (tangent vector)


class GraphType(Enum):
    """Type of graph connectivity."""
    KNN = "knn"                             # k-nearest neighbors
    EPSILON_BALL = "epsilon_ball"           # ε-neighborhood
    FULLY_CONNECTED = "fully_connected"     # Complete graph


class DistanceType(Enum):
    """Geodesic distance on base manifold."""
    WASSERSTEIN = "wasserstein"             # W₂ distance


@dataclass
class StatisticalBundleGraphConfig:
    """Configuration for the Statistical Bundle Graph Dataset."""

    n: int = 2                              # Dimension of ℝⁿ (covariance is n×n)
    n_nodes: int = 50                       # Number of nodes (sampled Σ's)
    feature_type: FeatureType = FeatureType.TANGENT_VEC
    graph_type: GraphType = GraphType.KNN
    distance_type: DistanceType = DistanceType.WASSERSTEIN
    k_neighbors: int = 5                    # For KNN graph
    epsilon: float = 1.0                    # For ε-ball graph
    include_fiber: bool = True              # Include tangent vectors
    edge_weight_sigma: float = 1.0          # For Gaussian edge weights


# =============================================================================
# Linear Algebra Utilities
# =============================================================================

def solve_lyapunov(Sigma: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Solve Lyapunov equation XΣ + ΣX = V."""
    Sigma_np = Sigma.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    X_np = la.solve_sylvester(Sigma_np, Sigma_np, V_np)
    return torch.from_numpy(X_np).to(Sigma.device, Sigma.dtype)


def _eigh_mps_safe(A: torch.Tensor):
    """eigh with CPU fallback on MPS (where eigh is not implemented)."""
    if A.device.type == "mps":
        eigvals, eigvecs = torch.linalg.eigh(A.cpu())
        return eigvals.to(A.device), eigvecs.to(A.device)
    return torch.linalg.eigh(A)


def matrix_sqrt(A: torch.Tensor) -> torch.Tensor:
    """Matrix square root via eigendecomposition."""
    eigvals, eigvecs = _eigh_mps_safe(A)
    eigvals = torch.clamp(eigvals, min=1e-10)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T


def matrix_sqrt_inv(A: torch.Tensor) -> torch.Tensor:
    """Inverse matrix square root."""
    eigvals, eigvecs = _eigh_mps_safe(A)
    eigvals = torch.clamp(eigvals, min=1e-10)
    return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T


# =============================================================================
# Symmetric Matrix Utilities
# =============================================================================

def sym_to_vec(S: torch.Tensor) -> torch.Tensor:
    """Convert symmetric matrix (..., n, n) to vector (..., m) where m = n(n+1)/2."""
    n = S.shape[-1]
    m = n * (n + 1) // 2
    batch_shape = S.shape[:-2]
    v = torch.zeros(*batch_shape, m, device=S.device, dtype=S.dtype)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                v[..., idx] = S[..., i, i]
            else:
                v[..., idx] = S[..., i, j] * np.sqrt(2)
            idx += 1
    return v


def vec_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    """Convert vector (..., m) to symmetric matrix (..., n, n)."""
    batch_shape = v.shape[:-1]
    S = torch.zeros(*batch_shape, n, n, device=v.device, dtype=v.dtype)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                S[..., i, i] = v[..., idx]
            else:
                S[..., i, j] = v[..., idx] / np.sqrt(2)
                S[..., j, i] = v[..., idx] / np.sqrt(2)
            idx += 1
    return S


# =============================================================================
# Distance Functions on SPD Manifold
# =============================================================================

def wasserstein_distance(Sigma1: torch.Tensor, Sigma2: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein-2 distance between N(0, Σ₁) and N(0, Σ₂).

    W₂²(Σ₁, Σ₂) = Tr(Σ₁) + Tr(Σ₂) - 2Tr((Σ₁^{1/2}Σ₂Σ₁^{1/2})^{1/2})
    """
    Sigma1_sqrt = matrix_sqrt(Sigma1)
    inner = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt
    inner_sqrt = matrix_sqrt(inner)
    W2_sq = torch.trace(Sigma1) + torch.trace(Sigma2) - 2 * torch.trace(inner_sqrt)
    return torch.sqrt(torch.clamp(W2_sq, min=0))


def compute_distance(Sigma1: torch.Tensor, Sigma2: torch.Tensor,
                     distance_type: DistanceType) -> torch.Tensor:
    """Compute distance between two covariance matrices."""
    if distance_type == DistanceType.WASSERSTEIN:
        return wasserstein_distance(Sigma1, Sigma2)
    raise ValueError(f"Unknown distance type: {distance_type}")


# =============================================================================
# Sampling on SPD Manifold
# =============================================================================

def sample_uniform_spd(n: int, size: int, eigenvalue_range: Tuple[float, float] = (0.5, 2.0),
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Sample uniformly from a region of SPD manifold.

    Uses eigenvalue decomposition: Σ = QΛQ^T with uniform eigenvalues.
    """
    samples = []
    low, high = eigenvalue_range

    for _ in range(size):
        # Random orthogonal matrix (Haar measure)
        A = torch.randn(n, n, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A)

        # Random eigenvalues (log-uniform for scale invariance)
        log_eigvals = torch.rand(n, device=device, dtype=dtype) * (np.log(high) - np.log(low)) + np.log(low)
        eigvals = torch.exp(log_eigvals)

        # Construct SPD matrix
        Sigma = Q @ torch.diag(eigvals) @ Q.T
        samples.append(Sigma)

    return torch.stack(samples)


def sample_tangent_vector(Sigma: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Sample a random tangent vector at Σ."""
    n = Sigma.shape[0]
    m = n * (n + 1) // 2

    # Random symmetric matrix
    v = torch.randn(m, device=Sigma.device, dtype=Sigma.dtype)
    V = vec_to_sym(v, n)

    if normalize:
        # Normalize w.r.t. Wasserstein metric
        L_V = solve_lyapunov(Sigma, V)
        norm_sq = torch.trace(L_V @ Sigma @ L_V)
        if norm_sq > 1e-8:
            V = V / torch.sqrt(norm_sq)

    return V


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(Sigma: torch.Tensor, V: Optional[torch.Tensor],
                     feature_type: FeatureType) -> torch.Tensor:
    """
    Extract node features from a covariance matrix and optional tangent vector.
    """
    n = Sigma.shape[0]
    features = []

    if feature_type == FeatureType.COVARIANCE_VEC:
        features.append(sym_to_vec(Sigma))

    elif feature_type == FeatureType.LOG_EIGENVALUES:
        eigvals = torch.linalg.eigvalsh(Sigma)
        features.append(torch.log(eigvals))

    elif feature_type == FeatureType.CHOLESKY_VEC:
        L = torch.linalg.cholesky(Sigma)
        # Vectorize lower triangular part
        mask = torch.tril(torch.ones(n, n, device=Sigma.device, dtype=torch.bool))
        features.append(L[mask])

    elif feature_type == FeatureType.TANGENT_VEC:
        if V is None:
            V = torch.zeros(n, n, device=Sigma.device, dtype=Sigma.dtype)
        features.append(sym_to_vec(V))

    return torch.cat(features)


# =============================================================================
# Graph Construction
# =============================================================================

def compute_distance_matrix(covariances: torch.Tensor,
                            distance_type: DistanceType) -> torch.Tensor:
    """Compute pairwise distance matrix for N covariance matrices."""
    N = covariances.shape[0]
    D = torch.zeros(N, N, device=covariances.device, dtype=covariances.dtype)

    for i in range(N):
        for j in range(i + 1, N):
            d = compute_distance(covariances[i], covariances[j], distance_type)
            D[i, j] = d
            D[j, i] = d

    return D


def create_graph_structure(distance_matrix: torch.Tensor,
                           config: StatisticalBundleGraphConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create edge_index and edge_attr from distance matrix.

    Returns:
        edge_index: (2, num_edges) connectivity (undirected, deduplicated).
        edge_attr:  (num_edges, 1) Gaussian-kernel edge weights.
    """
    N = distance_matrix.shape[0]
    device = distance_matrix.device
    dtype = distance_matrix.dtype

    if config.graph_type == GraphType.KNN:
        # Mask self on the diagonal then take the k smallest per row in one pass.
        masked = distance_matrix.clone()
        masked.fill_diagonal_(float('inf'))
        k = min(config.k_neighbors, N - 1) if N > 1 else 0
        if k == 0:
            src = torch.empty(0, dtype=torch.long, device=device)
            dst = torch.empty(0, dtype=torch.long, device=device)
        else:
            _, knn_idx = torch.topk(masked, k, dim=1, largest=False)  # (N, k)
            src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            dst = knn_idx.reshape(-1)

    elif config.graph_type == GraphType.EPSILON_BALL:
        eye = torch.eye(N, dtype=torch.bool, device=device)
        mask = (distance_matrix < config.epsilon) & ~eye
        src, dst = torch.nonzero(mask, as_tuple=True)

    elif config.graph_type == GraphType.FULLY_CONNECTED:
        eye = torch.eye(N, dtype=torch.bool, device=device)
        mask = ~eye
        src, dst = torch.nonzero(mask, as_tuple=True)

    else:
        raise ValueError(f"Unknown graph type: {config.graph_type}")

    if src.numel() == 0:
        # Fallback: self-loops so downstream code never sees an empty edge_index.
        src = torch.arange(N, device=device)
        dst = torch.arange(N, device=device)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.ones(N, 1, device=device, dtype=dtype)
        return edge_index, edge_attr

    edge_index = torch.stack([src, dst], dim=0)

    # Symmetrize then deduplicate. Both ops are vectorized.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = _unique_edges(edge_index, num_nodes=N)

    # Edge weights from the Gaussian kernel of the geodesic distance.
    d = distance_matrix[edge_index[0], edge_index[1]]
    edge_attr = torch.exp(-d ** 2 / (2.0 * config.edge_weight_sigma ** 2)).unsqueeze(1)

    return edge_index, edge_attr


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return ``edge_index`` with duplicates removed; output is sorted lexicographically."""
    edge_ids = edge_index[0] * num_nodes + edge_index[1]
    unique_ids = torch.unique(edge_ids, sorted=True)
    src = unique_ids // num_nodes
    dst = unique_ids % num_nodes
    return torch.stack([src.to(edge_index.dtype), dst.to(edge_index.dtype)], dim=0)


# =============================================================================
# Parallel Transport
# =============================================================================

class LeviCivitaTransport:
    """Levi-Civita parallel transport in the statistical bundle."""

    @staticmethod
    def transport(V: torch.Tensor, Sigma0: torch.Tensor,
                  Sigma1: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """Transport ``V`` from ``Σ₀`` to ``Σ₁`` along the Wasserstein geodesic.

        Uses the explicit Euler scheme on the parallel-transport ODE
        ``dU/dt + Γ(Σ(t); Σ̇(t), U(t)) = 0`` along
        ``Σ(t) = M(t) Σ₀ M(t)ᵀ``, ``M(t) = (1-t) I + t T``,
        where ``T = Σ₀^{-1/2} (Σ₀^{1/2} Σ₁ Σ₀^{1/2})^{1/2} Σ₀^{-1/2}`` solves
        ``T Σ₀ T = Σ₁``.
        """
        return LeviCivitaTransport.transport_batch(
            V.unsqueeze(0), Sigma0, Sigma1, n_steps=n_steps
        ).squeeze(0)

    @staticmethod
    def transport_batch(Vs: torch.Tensor, Sigma0: torch.Tensor,
                        Sigma1: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """Transport a batch of tangent vectors along the same geodesic.

        Args:
            Vs: ``(B, n, n)`` symmetric tangent vectors at ``Σ₀``.
            Sigma0, Sigma1: ``(n, n)`` SPD endpoints.
            n_steps: number of explicit-Euler steps.

        Returns:
            ``(B, n, n)`` transported tangent vectors at ``Σ₁``.

        Sharing the geodesic across the batch amortizes the per-step Lyapunov
        solve for ``Σ̇(t)`` and the eigendecompositions for ``T``.
        """
        if Vs.dim() == 2:
            Vs = Vs.unsqueeze(0)
        B, n, _ = Vs.shape
        device, dtype = Sigma0.device, Sigma0.dtype
        dt = 1.0 / n_steps

        Sigma0_sqrt = matrix_sqrt(Sigma0)
        Sigma0_inv_sqrt = matrix_sqrt_inv(Sigma0)
        inner = Sigma0_sqrt @ Sigma1 @ Sigma0_sqrt
        inner_sqrt = matrix_sqrt(inner)
        T = Sigma0_inv_sqrt @ inner_sqrt @ Sigma0_inv_sqrt

        I = torch.eye(n, device=device, dtype=dtype)
        dM = T - I
        Us = Vs.clone()

        for k in range(n_steps):
            t = k * dt
            M = (1 - t) * I + t * T
            Sigma_t = M @ Sigma0 @ M.T
            Sigma_dot = dM @ Sigma0 @ M.T + M @ Sigma0 @ dM.T

            # Lyapunov solves that depend only on Σ_t (shared across batch).
            L_X = solve_lyapunov(Sigma_t, Sigma_dot)
            for b in range(B):
                Yb = Us[b]
                L_Y = solve_lyapunov(Sigma_t, Yb)
                # Christoffel Γ(Σ_t; Σ̇, U_b), symmetrized.
                R = Sigma_t @ L_Y @ L_X + L_Y @ L_X @ Sigma_t - L_X @ Yb - L_Y @ Sigma_dot
                Gamma = 0.5 * (R + R.T)
                Us[b] = Yb - dt * Gamma

        return Us

    @staticmethod
    def transport_operator(Sigma0: torch.Tensor, Sigma1: torch.Tensor,
                            n_steps: int = 50) -> torch.Tensor:
        """Materialize Levi-Civita parallel transport as a linear map.

        Returns the ``(m, m)`` matrix ``T_op`` such that, for any ``V ∈ Sym(n)``
        with vectorization ``v = sym_to_vec(V)``,
        ``T_op @ v = sym_to_vec(transport(V, Σ₀, Σ₁))``.

        ``m = n(n+1)/2``. The columns are obtained by transporting the
        Frobenius-orthonormal basis ``{E_a}`` of ``Sym(n)`` (the basis induced
        by :func:`vec_to_sym`).
        """
        n = Sigma0.shape[0]
        m = n * (n + 1) // 2
        device, dtype = Sigma0.device, Sigma0.dtype

        # Stack the m basis matrices as a (m, n, n) tensor.
        basis_vecs = torch.eye(m, device=device, dtype=dtype)  # rows = basis vectors
        basis_mats = vec_to_sym(basis_vecs, n)                  # (m, n, n)

        transported = LeviCivitaTransport.transport_batch(
            basis_mats, Sigma0, Sigma1, n_steps=n_steps
        )  # (m, n, n)

        # Each column of T_op is the vec of the transported basis matrix.
        cols = sym_to_vec(transported)  # (m, m): rows index basis, cols index vec dim
        return cols.T.contiguous()


# =============================================================================
# Main Dataset Class
# =============================================================================

class StatisticalBundleGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for the Statistical Bundle.

    Each sample is a graph where:
        - Nodes: Points Σᵢ ∈ Sym++(n) sampled uniformly on the manifold
        - Edges: Connectivity based on geodesic distance
        - Node features: Distribution features (covariance, tangent vectors)

    Args:
        config: StatisticalBundleGraphConfig.
        n_graphs: Number of graph samples to generate.
        seed: Optional integer seed. When provided, calls
            ``torch.manual_seed(seed)`` before sampling.
    """

    def __init__(
        self,
        config: StatisticalBundleGraphConfig,
        n_graphs: int = 100,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.n = config.n
        self.m = self.n * (self.n + 1) // 2
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        self.n_graphs = n_graphs
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        # Generate graphs
        self.data_list = self._generate_graphs()

        # Store shared graph structure for the manifold
        self._compute_manifold_structure()

        self._print_summary()

    def _generate_graphs(self) -> List[Data]:
        """Generate graph samples."""
        graphs = []

        for i in range(self.n_graphs):
            # Sample nodes (covariance matrices) uniformly
            covariances = sample_uniform_spd(
                self.n, self.config.n_nodes,
                device=self.device, dtype=self.dtype
            )

            # Sample tangent vectors if needed
            if self.config.include_fiber:
                tangent_vectors = torch.stack([
                    sample_tangent_vector(Sigma) for Sigma in covariances
                ])
            else:
                tangent_vectors = None

            # Compute distance matrix
            dist_matrix = compute_distance_matrix(covariances, self.config.distance_type)

            # Create graph structure
            edge_index, edge_attr = create_graph_structure(dist_matrix, self.config)

            # Extract node features
            features = []
            for j in range(self.config.n_nodes):
                V = tangent_vectors[j] if tangent_vectors is not None else None
                feat = extract_features(covariances[j], V, self.config.feature_type)
                features.append(feat)

            x = torch.stack(features)

            data_kwargs = dict(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=self.config.n_nodes,
                covariances=covariances,
                tangent_vectors=tangent_vectors,
                distance_matrix=dist_matrix,
            )

            graphs.append(Data(**data_kwargs))

        return graphs

    def _compute_manifold_structure(self):
        """Compute and store manifold-level statistics."""
        # Aggregate statistics across all graphs
        all_distances = []
        for g in self.data_list:
            # Upper triangular distances
            D = g.distance_matrix
            mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
            all_distances.append(D[mask])

        all_distances = torch.cat(all_distances)
        self.mean_distance = all_distances.mean()
        self.std_distance = all_distances.std()
        self.max_distance = all_distances.max()

    def _print_summary(self):
        """Print dataset summary."""
        print(f"\n{'='*60}")
        print("Statistical Bundle Graph Dataset (PyG)")
        print(f"{'='*60}")
        print(f"Base manifold: Sym++({self.n})")
        print(f"Number of graphs: {len(self.data_list)}")
        print(f"Nodes per graph: {self.config.n_nodes}")
        print(f"Feature type: {self.config.feature_type.value}")
        print(f"Graph type: {self.config.graph_type.value}")
        print(f"Distance type: {self.config.distance_type.value}")
        if self.data_list:
            print(f"Node feature dimension: {self.data_list[0].x.shape[1]}")
            print(f"Edges per graph: {self.data_list[0].edge_index.shape[1]}")
        print(f"Mean geodesic distance: {self.mean_distance:.4f}")
        print(f"Max geodesic distance: {self.max_distance:.4f}")
        print(f"{'='*60}\n")

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """Get a graph sample."""
        return self.data_list[idx]
