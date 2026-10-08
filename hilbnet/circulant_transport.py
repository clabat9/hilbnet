"""Time-translation-invariant (circulant) transport map parameterization.

Restricts each edge transport P_e to the subgroup of SO(T) commuting with the
cyclic time-shift operator.  By the spectral theorem this is the class of real
circulant orthogonal matrices, parameterized by frequency-domain phases.

For real signals, conjugate symmetry of the spectrum + DC/Nyquist pinned to +1
(stay in the identity component of O(T)) leaves a torus T^{floor((T-1)/2)} of
free phases per edge.  Optionally, phases can be constrained to be piecewise
constant on K linearly-partitioned Fourier-bin "bands", giving K params per edge.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _band_assignment(num_free_phases: int, num_bands: int) -> torch.Tensor:
    """Linear partition of free-frequency indices {0,..,num_free_phases-1} into
    num_bands contiguous groups.  Returns a LongTensor of shape [num_free_phases]
    mapping each free-frequency index to its band index in [0, num_bands)."""
    if num_bands <= 0 or num_bands > num_free_phases:
        raise ValueError(
            f"num_bands must be in [1, {num_free_phases}], got {num_bands}."
        )
    # Split [0, num_free_phases) into `num_bands` nearly-equal contiguous blocks.
    boundaries = torch.linspace(0, num_free_phases, num_bands + 1).round().long()
    assignment = torch.zeros(num_free_phases, dtype=torch.long)
    for b in range(num_bands):
        lo, hi = int(boundaries[b].item()), int(boundaries[b + 1].item())
        assignment[lo:hi] = b
    return assignment


def build_circulant_transport_maps(
    phases: torch.Tensor,
    T: int,
    band_assignment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build [E, T, T] real circulant orthogonal matrices from phase parameters.

    Args:
        phases: shape [E, P] where P = num_free_phases if band_assignment is None,
                else P = num_bands.  Phases are in radians.
        T: stalk dimension.
        band_assignment: optional LongTensor [num_free_phases] mapping each
                         free-frequency index to a band; when provided, per-frequency
                         phases are recovered as phases[:, band_assignment].

    Returns:
        P: [E, T, T] real orthogonal matrices, each circulant: P[e, i, j] depends
           only on (i - j) mod T.
    """
    if phases.ndim != 2:
        raise ValueError(f"phases must be 2-D [E, P], got shape {tuple(phases.shape)}.")

    num_free = (T - 1) // 2  # number of independent-phase positive frequencies

    # Expand banded phases to per-frequency if needed.
    if band_assignment is not None:
        if band_assignment.numel() != num_free:
            raise ValueError(
                f"band_assignment has length {band_assignment.numel()} but "
                f"expected {num_free} = floor((T-1)/2)."
            )
        # phases: [E, num_bands] -> [E, num_free]
        phi = phases[:, band_assignment.to(phases.device)]
    else:
        if phases.shape[1] != num_free:
            raise ValueError(
                f"phases.shape[1]={phases.shape[1]} but expected {num_free} = "
                f"floor((T-1)/2) when band_assignment is None."
            )
        phi = phases

    # Compute the circulant's first column directly in real arithmetic.  This avoids
    # complex tensors and FFT ops, both of which have patchy MPS support.
    #
    # With DC (and, for even T, Nyquist) pinned to +1, and conjugate-symmetric unit-
    # modulus entries d[k] = e^{i phi_k} for k = 1..num_free, the inverse DFT is real:
    #   T even:  c[n] = (1/T) [1 + (-1)^n + 2 Σ_{k=1..num_free} cos(phi_k + 2π k n / T)]
    #   T odd:   c[n] = (1/T) [1          + 2 Σ_{k=1..num_free} cos(phi_k + 2π k n / T)]
    dtype = phi.dtype
    device = phi.device
    k_idx = torch.arange(1, num_free + 1, device=device, dtype=dtype)          # [num_free]
    n_idx = torch.arange(T, device=device, dtype=dtype)                         # [T]
    basis_angles = (2.0 * torch.pi / T) * k_idx.unsqueeze(1) * n_idx.unsqueeze(0)  # [num_free, T]
    total_angles = phi.unsqueeze(2) + basis_angles.unsqueeze(0)                 # [E, num_free, T]
    c_pos = 2.0 * torch.cos(total_angles).sum(dim=1)                            # [E, T]
    dc_term = torch.ones(T, device=device, dtype=dtype)
    if T % 2 == 0:
        sign_alt = 1.0 - 2.0 * (torch.arange(T, device=device) % 2).to(dtype)   # +1 / -1
        dc_term = dc_term + sign_alt
    first_col = (c_pos + dc_term.unsqueeze(0)) / T                              # [E, T]

    # Build [E, T, T] via advanced indexing: P[e, i, j] = first_col[e, (i - j) mod T].
    i = torch.arange(T, device=first_col.device)
    j = torch.arange(T, device=first_col.device)
    idx = (i.unsqueeze(1) - j.unsqueeze(0)) % T  # [T, T]
    P = first_col[:, idx]  # [E, T, T]
    return P


def procrustes_to_circulant_phases(
    targets: torch.Tensor,
    T: int,
    num_bands: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project orthogonal target matrices (e.g. Levi-Civita transports) onto
    circulant-orthogonal ones via diagonal-shift averaging + unit-modulus normalization.

    Approach (two-step Frobenius projection):
      1. Best circulant approx of each target: first column c_e[k] = mean over i of
         targets[e, (i + k) mod T, i]  (average along wrapped diagonals).
      2. Project onto orthogonal circulants: take fft(c_e), discard magnitude
         (set |.| = 1), keep only the phases of the independent positive frequencies.
         DC and (for even T) Nyquist are pinned to +1.

    If num_bands is set, aggregate per-frequency phases within each linearly-
    partitioned band via the circular mean (angle of sum of complex exponentials).

    Args:
        targets: [E, T, T] orthogonal matrices (unconstrained).
        T: stalk dimension.
        num_bands: if given, reduce from floor((T-1)/2) free phases to num_bands.

    Returns:
        phases: [E, num_bands or floor((T-1)/2)] initial phase angles in radians.
        band_assignment: [floor((T-1)/2)] LongTensor (or None when num_bands is None).
        residual_frob: average Frobenius error ||targets - reconstructed_circulant||_F
                       over edges (scientific diagnostic).
    """
    if targets.ndim != 3 or targets.shape[-1] != T or targets.shape[-2] != T:
        raise ValueError(
            f"targets must have shape [E, T, T] with T={T}, got {tuple(targets.shape)}."
        )
    E = targets.shape[0]
    num_free = (T - 1) // 2

    # Step 1: extract Frobenius-nearest circulant first-columns by averaging
    # along the T wrapped diagonals.  Vectorized: for each k, c[:, k] is the mean
    # over i of targets[:, (i + k) mod T, i].
    i = torch.arange(T, device=targets.device)
    k = torch.arange(T, device=targets.device)
    row_idx = (i.unsqueeze(0) + k.unsqueeze(1)) % T  # [T(=k), T(=i)]
    col_idx = i.unsqueeze(0).expand(T, T)              # [T(=k), T(=i)]
    # targets[e, row_idx[k,i], col_idx[k,i]] -> [E, T(=k), T(=i)], then mean over i.
    diag_samples = targets[:, row_idx, col_idx]        # [E, T, T]
    first_col = diag_samples.mean(dim=-1)              # [E, T]

    # Step 2: fft -> eigenvalues, extract phases of independent positive freqs.
    fft_dtype = torch.complex64 if first_col.dtype == torch.float32 else torch.complex128
    eigvals = torch.fft.fft(first_col.to(fft_dtype))   # [E, T] complex
    phases_full = torch.angle(eigvals[:, 1 : 1 + num_free])  # [E, num_free]
    phases_full = phases_full.to(first_col.dtype)

    # Reconstruct for residual diagnostic.
    if num_bands is not None:
        ba = _band_assignment(num_free, num_bands).to(targets.device)
        # Aggregate within bands via circular mean.
        complex_per_freq = torch.exp(1j * phases_full.to(fft_dtype))  # [E, num_free]
        band_sum = torch.zeros(E, num_bands, dtype=fft_dtype, device=targets.device)
        band_sum.index_add_(1, ba, complex_per_freq)
        phases = torch.angle(band_sum).to(first_col.dtype)  # [E, num_bands]
        reconstructed = build_circulant_transport_maps(phases, T, band_assignment=ba)
    else:
        ba = None
        phases = phases_full
        reconstructed = build_circulant_transport_maps(phases, T, band_assignment=None)

    residual = (targets - reconstructed).pow(2).sum(dim=(1, 2)).sqrt()  # [E]
    residual_frob = float(residual.mean().item())

    return phases, ba, residual_frob


class CirculantTransportParam(nn.Module):
    """Learnable per-edge circulant-orthogonal transport maps.

    Parameters are phase angles (radians) in a torus T^K, where K is either
    floor((T-1)/2) (full resolution) or num_bands (piecewise-constant on linearly-
    partitioned Fourier bins).

    forward() returns [E, T, T] real orthogonal circulant matrices directly
    (bypassing Householder composition).
    """

    def __init__(
        self,
        num_edges: int,
        stalk_dim: int,
        num_bands: Optional[int] = None,
        init: str = "identity_plus_noise",
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if init not in ("identity", "identity_plus_noise"):
            raise ValueError("init must be 'identity' or 'identity_plus_noise'.")
        self.num_edges = int(num_edges)
        self.stalk_dim = int(stalk_dim)
        self.num_bands = num_bands

        num_free = (stalk_dim - 1) // 2
        if num_free < 1:
            raise ValueError(
                f"stalk_dim={stalk_dim} leaves zero free phases; circulant "
                f"transports are trivial for T < 3."
            )

        if num_bands is not None:
            ba = _band_assignment(num_free, num_bands)
            self.register_buffer("band_assignment", ba)
            num_param = int(num_bands)
        else:
            self.band_assignment = None  # type: ignore[assignment]
            num_param = num_free
        self.num_free_phases = num_free
        self.num_param = num_param

        if init == "identity":
            phi_init = torch.zeros(num_edges, num_param, dtype=dtype)
        else:  # identity_plus_noise
            phi_init = init_scale * torch.randn(num_edges, num_param, dtype=dtype)
        self.phases = nn.Parameter(phi_init)

    def forward(self) -> torch.Tensor:
        """Compute [E, T, T] real circulant orthogonal transport maps."""
        ba = self.band_assignment if self.num_bands is not None else None
        return build_circulant_transport_maps(self.phases, self.stalk_dim, band_assignment=ba)
