"""
PyTorch utilities for LCAL computations.

This module provides utility functions for converting between NumPy arrays
and PyTorch tensors, and common tensor operations used in LCAL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Minimum value for numerical stability (avoid log(0))
EPS = 1e-10

__all__ = [
    "EPS",
    "to_tensor",
    "logit_softmax",
    "elastic_demand",
    "compute_production",
    "safe_log",
]


def to_tensor(
    arr: NDArray | Tensor,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Convert numpy array to PyTorch tensor.

    Parameters
    ----------
    arr : ndarray or Tensor
        Input array.
    dtype : torch.dtype, default torch.float64
        Data type for the tensor.
    device : torch.device or str, optional
        Device to place tensor on.

    Returns
    -------
    Tensor
        PyTorch tensor.
    """
    if isinstance(arr, Tensor):
        tensor = arr.to(dtype=dtype)
    else:
        tensor = torch.tensor(arr, dtype=dtype)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def logit_softmax(
    utilities: Tensor,
    attractors: Tensor,
    beta: float | Tensor,
    dim: int = -1,
) -> Tensor:
    """
    Compute softmax probabilities with attractors (logit model).

    P_j = A_j * exp(-beta * U_j) / sum_k(A_k * exp(-beta * U_k))

    Uses log-space computation for numerical stability:
    logits = log(A) - beta * U
    P = softmax(logits)

    Parameters
    ----------
    utilities : Tensor
        Utility values.
    attractors : Tensor
        Attractor values (must be positive).
    beta : float or Tensor
        Dispersion parameter.
    dim : int, default -1
        Dimension to normalize over.

    Returns
    -------
    Tensor
        Probability distribution.
    """
    # Clamp attractors to avoid log(0)
    logits = torch.log(attractors.clamp(min=EPS)) - beta * utilities
    return F.softmax(logits, dim=dim)


def elastic_demand(
    U: Tensor,
    demin: Tensor,
    demax: Tensor,
    delta: Tensor,
) -> Tensor:
    """
    Compute elastic demand function.

    a = demin + (demax - demin) * exp(-delta * U)

    Parameters
    ----------
    U : Tensor
        Utility (price + shadow price).
    demin : Tensor
        Minimum demand coefficient.
    demax : Tensor
        Maximum demand coefficient.
    delta : Tensor
        Demand elasticity parameter.

    Returns
    -------
    Tensor
        Demand function values.
    """
    gap = demax - demin
    return demin + gap * torch.exp(-delta * U)


def compute_production(
    demands: Tensor,
    probabilities: Tensor,
) -> Tensor:
    """
    Compute production from demands and location probabilities.

    X_j = sum_i(D_i * Pr_{ij})

    Parameters
    ----------
    demands : Tensor of shape (n_zones,) or (n_sectors, n_zones)
        Demand by zone.
    probabilities : Tensor of shape (n_zones, n_zones) or (n_sectors, n_zones, n_zones)
        Location probabilities Pr[i, j] = prob demand from i satisfied in j.

    Returns
    -------
    Tensor
        Production by zone.
    """
    if demands.dim() == 1:
        # Single sector: D @ Pr
        return torch.einsum("i,ij->j", demands, probabilities)
    else:
        # Multiple sectors: D[n] @ Pr[n] for each n
        return torch.einsum("ni,nij->nj", demands, probabilities)


def safe_log(x: Tensor) -> Tensor:
    """Compute log with numerical stability using clamp."""
    return torch.log(x.clamp(min=EPS))
