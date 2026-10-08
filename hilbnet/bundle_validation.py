"""Closed-form utilities and a thin trainable-transport head for the synthetic
statistical-bundle theorem-validation experiments.

This module is the bridge between

  * the synthetic data + closed-form Levi-Civita transport in
    :mod:`hilbnet.statistical_bundle`, and
  * HilbNet's existing sheaf-Laplacian assembly + transport variants in
    :mod:`hilbnet.utils`, :mod:`hilbnet.circulant_transport`,
    and :mod:`hilbnet._transport_setup`.

Conventions
-----------
Fiber elements ``V ∈ Sym(n)`` are vectorized with :func:`hilbnet.statistical_bundle.sym_to_vec`
into ``ℝ^m`` with ``m = n(n+1)/2``. The basis is Frobenius-orthonormal:
diagonals are ``e_ii`` and off-diagonals are ``(e_ij + e_ji)/√2``. The
Wasserstein inner product at ``Σ`` written in this basis has Gram matrix
``G_Σ`` (computed by :func:`gram_matrix_w`); ``G_Σ ≠ I`` when ``Σ ≠ I``.

Cholesky rescaling
~~~~~~~~~~~~~~~~~~
With ``G_Σ = R_Σᵀ R_Σ`` (upper-triangular Cholesky), the rescaled coordinate
``ũ = R_Σ u`` satisfies ``‖ũ‖₂² = u^T G_Σ u = W_Σ(U, U)``. In rescaled
coordinates the Levi-Civita transport becomes Frobenius-orthogonal
(``T̃_e = R_{Σ_dst} T^{LC}_e R_{Σ_src}^{-1} ∈ O(m)``), which matches HilbNet's
hypothesis class. We default to rescaled coordinates throughout the
validation experiments.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hilbnet._transport_setup import setup_transport
from hilbnet.circulant_transport import (
    build_circulant_transport_maps,
    procrustes_to_circulant_phases,
)
from hilbnet.statistical_bundle import (
    LeviCivitaTransport,
    solve_lyapunov,
    vec_to_sym,
)
from hilbnet.utils import (
    build_householder_transport_maps,
    build_sheaf_laplacian_from_transport,
)


# =============================================================================
# Wasserstein-Gram matrix and Cholesky rescaling
# =============================================================================

def gram_matrix_w(Sigma: torch.Tensor) -> torch.Tensor:
    """Wasserstein-Gram matrix of ``Sym(n)`` at ``Σ`` in the ``sym_to_vec`` basis.

    Returns ``G ∈ ℝ^{m×m}`` (symmetric positive definite) such that
    ``W_Σ(U, V) = sym_to_vec(U)^T G sym_to_vec(V)`` for all ``U, V ∈ Sym(n)``.

    The Wasserstein bilinear form is ``W_Σ(U, V) = ½ Tr(L_Σ[U] V)`` where
    ``L_Σ[U]`` solves the Lyapunov equation ``L_Σ[U] Σ + Σ L_Σ[U] = U``.
    """
    n = Sigma.shape[-1]
    m = n * (n + 1) // 2
    device, dtype = Sigma.device, Sigma.dtype

    # Frobenius-orthonormal basis of Sym(n) via vec_to_sym(e_a, n).
    basis_vecs = torch.eye(m, device=device, dtype=dtype)
    basis_mats = vec_to_sym(basis_vecs, n)              # (m, n, n)

    # Lyapunov solutions L_a := L_Σ[E_a] for each basis vector.
    L_basis = torch.empty_like(basis_mats)              # (m, n, n)
    for a in range(m):
        L_basis[a] = solve_lyapunov(Sigma, basis_mats[a])

    # G[a, b] = ½ Tr(L_a · E_b).
    G = 0.5 * torch.einsum("aij,bji->ab", L_basis, basis_mats)
    # Symmetrize to absorb numerical asymmetry from the per-basis solver.
    G = 0.5 * (G + G.T)
    return G


def cholesky_rescale_factor(Sigma: torch.Tensor) -> torch.Tensor:
    """Return the Cholesky factor ``R`` of the W-Gram matrix at ``Σ``.

    ``R`` is upper-triangular with ``Rᵀ R = G_Σ``. Then for any ``U ∈ Sym(n)``
    with ``u = sym_to_vec(U)``, we have ``‖R u‖₂² = u^T G_Σ u = W_Σ(U, U)``.
    """
    G = gram_matrix_w(Sigma)
    # `cholesky_ex` is not implemented on MPS as of PyTorch 2.x; bounce to CPU.
    orig_device = G.device
    G_cpu = G.cpu() if orig_device.type == "mps" else G
    L, info = torch.linalg.cholesky_ex(G_cpu, upper=False)
    if int(info.item()) != 0:
        raise RuntimeError(
            "Cholesky failed on Wasserstein-Gram matrix; Σ may be ill-conditioned."
        )
    R = L.transpose(-1, -2).contiguous()
    return R.to(orig_device) if orig_device.type == "mps" else R


def cholesky_factors_per_node(covariances: torch.Tensor) -> torch.Tensor:
    """Stack ``cholesky_rescale_factor`` across nodes; returns ``(N, m, m)``."""
    N = covariances.shape[0]
    R = []
    for i in range(N):
        R.append(cholesky_rescale_factor(covariances[i]))
    return torch.stack(R, dim=0)


# =============================================================================
# Closed-form Levi-Civita transports (per edge)
# =============================================================================

def closed_form_lc_transports(
    covariances: torch.Tensor,
    edge_index: torch.Tensor,
    n_steps: int = 50,
    rescale: bool = True,
) -> torch.Tensor:
    """Per-edge Levi-Civita transport operators in ``sym_to_vec`` coordinates.

    Args:
        covariances: ``(N, n, n)`` SPD endpoints.
        edge_index: ``(2, E)`` directed edges.
        n_steps: Euler steps for the parallel-transport ODE.
        rescale: when True, returns ``T̃_e = R_dst @ T^{LC}_e @ R_src^{-1}``,
            which is Frobenius-orthogonal (W-orthogonal in raw coords becomes
            Euclidean-orthogonal in rescaled coords).

    Returns:
        ``(E, m, m)`` linear maps such that
        ``T_e @ sym_to_vec(V) = sym_to_vec(τ^{LC}(V; Σ_src → Σ_dst))``
        when ``rescale=False``; when ``rescale=True`` they act on rescaled
        coordinates ``ũ = R_Σ u``.
    """
    E = edge_index.shape[1]
    n = covariances.shape[-1]
    m = n * (n + 1) // 2
    orig_device, dtype = covariances.device, covariances.dtype

    # MPS lacks eigh / cholesky_ex / inv (PyTorch 2.x); the LC transport ODE
    # invokes these per edge. Bounce the entire assembly to CPU and move the
    # final transport tensor back. One-time cost at graph build, no inner-loop
    # MPS↔CPU pingpong.
    work_device = torch.device("cpu") if orig_device.type == "mps" else orig_device
    cov_w = covariances.to(work_device)
    edge_w = edge_index.to(work_device)

    if rescale:
        R = cholesky_factors_per_node(cov_w)  # (N, m, m)
        R_inv = torch.linalg.inv(R)

    out = torch.empty(E, m, m, device=work_device, dtype=dtype)
    for e in range(E):
        src = int(edge_w[0, e].item())
        dst = int(edge_w[1, e].item())
        if src == dst:
            out[e] = torch.eye(m, device=work_device, dtype=dtype)
            continue
        T_e = LeviCivitaTransport.transport_operator(
            cov_w[src], cov_w[dst], n_steps=n_steps
        )
        if rescale:
            out[e] = R[dst] @ T_e @ R_inv[src]
        else:
            out[e] = T_e
    return out.to(orig_device)


# =============================================================================
# Sheaf-Laplacian assembly with closed-form transports
# =============================================================================

def assemble_sheaf_laplacian(
    covariances: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    rescale: bool = True,
) -> torch.Tensor:
    """Build the discrete sheaf Laplacian ``L_N`` from closed-form LC transports.

    Orientation: for an edge ``e = (i, j)`` the off-diagonal block ``(i, j)`` is
    ``-w_e P_{j→i}``, with ``P_{j→i}`` the Levi-Civita transport from ``x_j`` to
    ``x_i``, so that ``f^T L_N f = Σ_e w_e ‖f_i − P_{j→i} f_j‖²``. This is the
    block of the paper's midpoint construction, which transports ``x_j → m_e → x_i``
    along the geodesic.

    Args:
        covariances: ``(N, n, n)``.
        edge_index: ``(2, E)``.
        edge_weights: ``(E,)`` or ``None`` (defaults to all ones).
        n_steps: Euler steps for the parallel-transport ODE.
        rescale: use Cholesky-rescaled coordinates so ``L_N`` is symmetric PSD
            and the LC operator is Frobenius-orthogonal (``assume_orthogonal=True``).

    Returns:
        ``(N·m, N·m)`` dense matrix.
    """
    # build_sheaf_laplacian_from_transport puts -w_e · transport_maps[e] in block
    # (src, dst), so each edge needs the transport from dst to src: compute the
    # closed-form transports along the reversed edges.
    transport_maps = closed_form_lc_transports(
        covariances, edge_index.flip(0), n_steps=n_steps, rescale=rescale
    )
    n = covariances.shape[-1]
    m = n * (n + 1) // 2
    return build_sheaf_laplacian_from_transport(
        transport_maps,
        edge_index,
        stalk_dim=m,
        edge_weights=edge_weights,
        num_nodes=int(covariances.shape[0]),
        return_blocks=False,
        assume_orthogonal=rescale,
    )


def eigenvalues_smallest(L: torch.Tensor, k: int) -> torch.Tensor:
    """Return the ``k`` smallest eigenvalues of a symmetric matrix ``L``."""
    eigvals = torch.linalg.eigvalsh(L)  # ascending
    if k >= eigvals.numel():
        return eigvals
    return eigvals[:k]


# =============================================================================
# Restricted-class projections (analytical Theory plateau of Table 1)
# =============================================================================

def project_to_circulant(transport_maps: torch.Tensor,
                         num_bands: Optional[int] = None) -> torch.Tensor:
    """Frobenius-closest circulant-orthogonal approximation per edge.

    Uses :func:`hilbnet.circulant_transport.procrustes_to_circulant_phases`
    (closed-form: average wrapped diagonals, FFT, project phases to the unit
    circle, optionally band-aggregate via circular mean).

    Args:
        transport_maps: ``(E, T, T)``.
        num_bands: optional band restriction; ``None`` = full ``floor((T-1)/2)`` phases.

    Returns:
        ``(E, T, T)`` circulant-orthogonal matrices.
    """
    if transport_maps.ndim != 3:
        raise ValueError("transport_maps must have shape [E, T, T].")
    T = transport_maps.shape[-1]
    phases, ba, _ = procrustes_to_circulant_phases(
        transport_maps, T=T, num_bands=num_bands
    )
    return build_circulant_transport_maps(phases, T, band_assignment=ba)


def project_to_identity(transport_maps: torch.Tensor) -> torch.Tensor:
    """Trivial projection: returns ``I`` of the appropriate shape per edge."""
    if transport_maps.ndim != 3:
        raise ValueError("transport_maps must have shape [E, T, T].")
    E, T, _ = transport_maps.shape
    eye = torch.eye(T, device=transport_maps.device, dtype=transport_maps.dtype)
    return eye.unsqueeze(0).expand(E, T, T).contiguous()


def projection_distance(transport_maps: torch.Tensor,
                        projected: torch.Tensor) -> torch.Tensor:
    """Mean-squared Frobenius residual between two ``(E, T, T)`` tensors.

    Specifically returns
    ``(1/(E·T²)) · Σ_e ‖transport_maps[e] - projected[e]‖_F²``.
    The transport-recovery training loss (``scripts/bundle_transport_recovery.py``)
    is the per-element MSE ``mean_{b,e,t} ((T_pred − T_gt) V_src)²`` with
    ``V ~ N(0, I_T)``, whose expectation is ``T · projection_distance(T_pred, T_gt)``.
    """
    if transport_maps.shape != projected.shape:
        raise ValueError("transport_maps and projected must have the same shape.")
    diff = transport_maps - projected
    E, T, _ = diff.shape
    return diff.pow(2).sum() / (E * T * T)


# =============================================================================
# Trainable transport head used by the transport-recovery experiment
# =============================================================================

class BundleTransportModule(nn.Module):
    """Standalone trainable head exposing ``transport_maps: (E, T, T)``.

    Used by the synthetic transport-recovery experiment to train one of the
    three paper variants directly against an edge-level MSE loss.

    Variants:
        * ``"direct"`` — free Householder ``O(T)``.
        * ``"circulant"`` — banded Fourier-phase circulant subgroup.
        * ``"frozen_id"`` — non-learnable identity per edge (GCN baseline).

    The first two are dispatched through
    :func:`hilbnet._transport_setup.setup_transport`. ``frozen_id`` is handled
    locally (no parameters).
    """

    SUPPORTED = ("direct", "circulant", "frozen_id")

    def __init__(
        self,
        transport_param_type: str,
        edge_index: torch.Tensor,
        stalk_dim: int,
        num_reflections: int = 8,
        num_bands: Optional[int] = None,
        transport_init: str = "identity_plus_noise",
        householder_eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if transport_param_type not in self.SUPPORTED:
            raise ValueError(
                f"transport_param_type must be one of {self.SUPPORTED}, "
                f"got {transport_param_type!r}."
            )
        self.transport_param_type = transport_param_type
        self.householder_eps = householder_eps
        self.stalk_dim = int(stalk_dim)
        self._dtype = dtype
        self.register_buffer("edge_index", edge_index.long().clone())
        self.num_edges = int(edge_index.shape[1])

        if transport_param_type == "frozen_id":
            # No learnable parameters; identity transport per edge.
            self._uses_transport = False
            self.transport_param_module = None
            return

        # Dispatch through the canonical codebase entry point. setup_transport
        # mutates `self`, installing self._uses_transport, self.transport_param_module,
        # and (for "direct") self.householder_vectors as nn.Parameter.
        setup_transport(
            self,
            kappas=[2],  # > 1 so setup_transport actually allocates the transport
            edge_index=self.edge_index,
            time_steps=self.stalk_dim,
            transport_param_type=transport_param_type,
            transport_init=transport_init,
            num_householder_reflections=num_reflections,
            num_bands=num_bands,
            dtype=dtype,
        )

    @property
    def transport_maps(self) -> torch.Tensor:
        if self.transport_param_type == "frozen_id":
            eye = torch.eye(self.stalk_dim, device=self.edge_index.device, dtype=self._dtype)
            return eye.unsqueeze(0).expand(self.num_edges, self.stalk_dim, self.stalk_dim).contiguous()

        if self.transport_param_type == "circulant":
            return self.transport_param_module()
        # "direct"
        vectors = self.householder_vectors
        return build_householder_transport_maps(vectors, eps=self.householder_eps)

    def forward(self) -> torch.Tensor:
        return self.transport_maps
