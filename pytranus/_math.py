"""
Pure NumPy implementations of derivative computations for LCAL calibration.

This module replaces the Cython DX.pyx implementation with vectorized NumPy operations.
The functions compute derivatives of production with respect to shadow prices and other
parameters used in the land-use calibration optimization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_DX_h(
    Pr: NDArray[np.float64],
    lamda: NDArray[np.float64],
    beta: NDArray[np.float64],
    U: NDArray[np.float64],
    alpha: NDArray[np.float64],
    gap: NDArray[np.float64],
    delta: NDArray[np.float64],
    X_0: NDArray[np.float64],
    n_sectors: int,
    n_zones: int,
    genflux_sectors: NDArray[np.intp],
) -> NDArray[np.float64]:
    """
    Compute derivative of production with respect to housing shadow prices.

    This is a vectorized NumPy implementation of the Cython cython_DX_h function.
    Computes ∂X/∂h for land-use sectors.

    Parameters
    ----------
    Pr : ndarray of shape (n_sectors, n_zones, n_zones)
        Localization probabilities Pr[n, i, j] = probability that demand from zone i
        for sector n is satisfied in zone j.
    lamda : ndarray of shape (n_sectors,)
        Marginal utility parameters for each sector.
    beta : ndarray of shape (n_sectors,)
        Dispersion parameters for localization choice.
    U : ndarray of shape (n_sectors, n_zones)
        Utilities for each sector and zone.
    alpha : ndarray of shape (n_sectors, n_sectors, n_zones)
        Demand functions α[m, n, i] = demand of sector m for sector n in zone i.
    gap : ndarray of shape (n_sectors, n_sectors)
        Demand gap (demax - demin) for elastic demand.
    delta : ndarray of shape (n_sectors, n_sectors)
        Dispersion parameter in demand function.
    X_0 : ndarray of shape (n_sectors, n_zones)
        Base year production.
    n_sectors : int
        Number of economic sectors.
    n_zones : int
        Number of zones.
    genflux_sectors : ndarray of int
        Indices of sectors that generate flux (transportable sectors).

    Returns
    -------
    DX_h : ndarray of shape (n_sectors, n_zones, n_sectors, n_zones)
        Derivative matrix where DX_h[n, j, n, k] = ∂X_n,j / ∂h_n,k
    """
    DX_h = np.zeros((n_sectors, n_zones, n_sectors, n_zones), dtype=np.float64)

    for n in genflux_sectors:
        # Precompute terms for this sector
        aux2 = lamda[n] * beta[n]

        for j in range(n_zones):
            for k in range(n_zones):
                sum_mi = 0.0

                for m in range(n_sectors):
                    # Skip if no consumption of n by m
                    if alpha[m, n, 0] == 0:
                        continue

                    delta_mn = delta[m, n]

                    if gap[m, n] != 0:  # Elastic demand case
                        aux1 = gap[m, n] * delta_mn
                        for i in range(n_zones):
                            exp_term = np.exp(-delta_mn * U[n, i])
                            if k == j:
                                # Diagonal term
                                sum_mi += (
                                    -aux1 * exp_term * Pr[n, i, k] * Pr[n, i, j]
                                    + alpha[m, n, i]
                                    * (-aux2 * (Pr[n, i, j] - Pr[n, i, j] ** 2))
                                ) * X_0[m, i]
                            else:
                                # Off-diagonal term
                                sum_mi += (
                                    -aux1 * exp_term * Pr[n, i, k] * Pr[n, i, j]
                                    + alpha[m, n, i] * (aux2 * Pr[n, i, j] * Pr[n, i, k])
                                ) * X_0[m, i]
                    else:
                        # Non-elastic demand case (demin == demax)
                        for i in range(n_zones):
                            if k == j:
                                sum_mi += (
                                    alpha[m, n, i]
                                    * (-aux2 * (Pr[n, i, j] - Pr[n, i, j] ** 2))
                                ) * X_0[m, i]
                            else:
                                sum_mi += (
                                    alpha[m, n, i] * (aux2 * Pr[n, i, j] * Pr[n, i, k])
                                ) * X_0[m, i]

                DX_h[n, j, n, k] = sum_mi

    return DX_h


def compute_DX_n(
    DX: NDArray[np.float64],
    n_sectors: int,
    n_zones: int,
    beta: float,
    lamda: float,
    D_n: NDArray[np.float64],
    Pr_n: NDArray[np.float64],
    U_n: NDArray[np.float64],
    logit: bool = True,
) -> NDArray[np.float64]:
    """
    Compute derivative of production for transportable sector n.

    This is a vectorized NumPy implementation of the Cython cython_DX_n function.
    Computes ∂X_n / ∂(p+h)_n for a transportable sector.

    Parameters
    ----------
    DX : ndarray of shape (n_zones, n_zones)
        Output array to store the derivative matrix.
    n_sectors : int
        Number of sectors (unused, kept for API compatibility).
    n_zones : int
        Number of zones.
    beta : float
        Dispersion parameter for sector n.
    lamda : float
        Marginal utility parameter for sector n.
    D_n : ndarray of shape (n_zones,)
        Demands for sector n by zone.
    Pr_n : ndarray of shape (n_zones, n_zones)
        Localization probabilities for sector n.
    U_n : ndarray of shape (n_zones, n_zones)
        Utilities for sector n.
    logit : bool, default True
        If True, use logit model; if False, use powit (power) model.

    Returns
    -------
    DX : ndarray of shape (n_zones, n_zones)
        Derivative matrix where DX[j, k] = ∂X_n,j / ∂(p+h)_n,k

    Notes
    -----
    This function modifies DX in-place and also returns it.
    The derivative is computed as:

    For logit model:
        ∂X_j/∂φ_k = Σ_i D_i * λ * β * (Pr_{ij} * Pr_{ik} - δ_{jk} * Pr_{ij})

    For powit model:
        ∂X_j/∂φ_k = Σ_i D_i * (λ * β / U_{ij}) * (Pr_{ij} * Pr_{ik} - δ_{jk} * Pr_{ij})

    where δ_{jk} is the Kronecker delta.
    """
    coef = lamda * beta

    if logit:
        # Vectorized computation for logit model
        # Pr_n[i, j] * Pr_n[i, k] summed over i, weighted by D_n[i]
        # For off-diagonal: coef * Pr[i,j] * Pr[i,k] * D[i]
        # For diagonal: -coef * (Pr[i,j] - Pr[i,j]^2) * D[i]

        # Off-diagonal terms: DX[j, k] = coef * sum_i(D_i * Pr_ij * Pr_ik)
        # Shape: (n_zones, n_zones)
        weighted_Pr = Pr_n * D_n[:, np.newaxis]  # (n_zones, n_zones)
        DX[:, :] = coef * (weighted_Pr.T @ Pr_n)

        # Diagonal correction: subtract 2 * coef * sum_i(D_i * Pr_ij * Pr_ij)
        # and add -coef * sum_i(D_i * Pr_ij)
        diag_term = coef * np.sum(D_n[:, np.newaxis] * Pr_n * (1 - Pr_n), axis=0)
        np.fill_diagonal(DX, -diag_term)

    else:
        # Powit model - need to divide by U_n
        # Avoid division by zero
        U_safe = np.where(U_n != 0, U_n, 1.0)

        for j in range(n_zones):
            for k in range(n_zones):
                sum_i = 0.0
                for i in range(n_zones):
                    if U_n[i, j] != 0:
                        if k == j:
                            sum_i += (
                                -(coef / U_n[i, j])
                                * (Pr_n[i, j] - Pr_n[i, j] ** 2)
                                * D_n[i]
                            )
                        else:
                            sum_i += (
                                (coef / U_n[i, j]) * Pr_n[i, j] * Pr_n[i, k] * D_n[i]
                            )
                DX[j, k] = sum_i

    return DX


def compute_DX_n_vectorized(
    n_zones: int,
    beta: float,
    lamda: float,
    D_n: NDArray[np.float64],
    Pr_n: NDArray[np.float64],
    logit: bool = True,
) -> NDArray[np.float64]:
    """
    Fully vectorized computation of derivative for transportable sector.

    This is an alternative implementation that creates a new array instead of
    modifying one in place. It may be more efficient for some use cases.

    Parameters
    ----------
    n_zones : int
        Number of zones.
    beta : float
        Dispersion parameter.
    lamda : float
        Marginal utility parameter.
    D_n : ndarray of shape (n_zones,)
        Demands by zone.
    Pr_n : ndarray of shape (n_zones, n_zones)
        Localization probabilities.
    logit : bool, default True
        If True, use logit model.

    Returns
    -------
    DX : ndarray of shape (n_zones, n_zones)
        Derivative matrix.
    """
    coef = lamda * beta

    if logit:
        # D_n weighted probabilities: (n_zones, n_zones)
        weighted_Pr = Pr_n * D_n[:, np.newaxis]

        # Off-diagonal: coef * sum_i(D_i * Pr_ij * Pr_ik) = coef * Pr.T @ (D * Pr)
        DX = coef * (weighted_Pr.T @ Pr_n)

        # Diagonal: -coef * sum_i(D_i * Pr_ij * (1 - Pr_ij))
        diag_term = coef * np.sum(D_n[:, np.newaxis] * Pr_n * (1 - Pr_n), axis=0)
        np.fill_diagonal(DX, -diag_term)

        return DX
    else:
        raise NotImplementedError("Powit model not yet vectorized")
