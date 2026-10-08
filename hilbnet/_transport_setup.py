"""Transport parameterization shared by ``HilbNetForecaster`` and ``BundleTransportModule``.

The ``setup_transport`` function configures these attributes on the host module:

    self._uses_transport          (bool)
    self.transport_param_module   (None | CirculantTransportParam)
    self.householder_vectors      (only set in the 'direct' branch; not present otherwise)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from hilbnet.circulant_transport import CirculantTransportParam


def init_householder_vectors(
    transport_init: str,
    num_edges: int,
    num_householder_reflections: int,
    time_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initial Householder reflector vectors for the 'direct' transport.

    ``"identity"`` returns zeros (so each reflection is the identity). Zero
    vectors get zero gradient through the Householder map, so use
    ``"identity_plus_noise"`` for trainable runs; ``"identity"`` is fine for
    *frozen* (transport_lr=0) runs.
    """
    shape = (num_edges, num_householder_reflections, time_steps)
    if transport_init == "identity":
        return torch.zeros(shape, dtype=dtype)
    if transport_init == "identity_plus_noise":
        return 1e-2 * torch.randn(shape, dtype=dtype)
    raise ValueError("transport_init must be 'identity' or 'identity_plus_noise'.")


def setup_transport(
    module: nn.Module,
    *,
    kappas: List[int],
    edge_index: torch.Tensor,
    time_steps: int,
    transport_param_type: str,
    transport_init: str,
    num_householder_reflections: int,
    num_bands: Optional[int],
    dtype: torch.dtype,
) -> None:
    """Configure transport parameterization on ``module``.

    Sets ``module._uses_transport``, ``module.transport_param_module``, and
    (for ``transport_param_type='direct'``) ``module.householder_vectors``.

    When ``max(kappas) == 1`` every layer reduces to a per-(node, timestep)
    pointwise channel mix (out = W_0 · x); the sheaf Laplacian is never
    applied. We skip transport allocation entirely so the reported param
    count is honest for the kappa=1 ablation (the "MLP fiber" baseline).
    """
    module._uses_transport = max(kappas) > 1
    num_edges = edge_index.shape[1]

    if not module._uses_transport:
        module.transport_param_module = None
        return

    if transport_param_type == "circulant":
        module.transport_param_module = CirculantTransportParam(
            num_edges=num_edges,
            stalk_dim=time_steps,
            num_bands=num_bands,
            init=transport_init,
            init_scale=1e-2,
            dtype=dtype,
        )
    else:  # "direct"
        module.transport_param_module = None
        module.householder_vectors = nn.Parameter(
            init_householder_vectors(
                transport_init, num_edges,
                num_householder_reflections, time_steps, dtype=dtype,
            )
        )
