"""
PyTorch-based LCAL - Land Use Calibration Module for Tranus.

This module provides a PyTorch reimplementation of the LCAL calibration,
using automatic differentiation and GPU acceleration for optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from pytranus.modules import LCALModel
from pytranus.params import load_params

if TYPE_CHECKING:
    from pytranus.config import TranusConfig

logger = logging.getLogger(__name__)


def train(
    model: LCALModel,
    max_steps: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Train the LCAL model using standard PyTorch training loop.

    Parameters
    ----------
    model : LCALModel
        The model to train.
    max_steps : int, default 1000
        Maximum training steps.
    lr : float, default 0.1
        Learning rate.
    tol : float, default 1e-6
        Stop when loss < tol.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    dict
        Training history with 'loss' and 'steps'.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "steps": 0}

    for step in range(max_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history["loss"].append(loss_val)
        history["steps"] = step + 1

        if verbose and step % 100 == 0:
            print(f"  Step {step}: loss = {loss_val:.6f}")

        if loss_val < tol:
            if verbose:
                print(f"  Converged at step {step} with loss = {loss_val:.6f}")
            break

    return history


def calibrate(
    config: TranusConfig,
    device: str | None = None,
    max_steps: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = True,
) -> tuple[LCALModel, dict]:
    """
    Convenience function for LCAL calibration.

    Parameters
    ----------
    config : TranusConfig
        Tranus configuration.
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps').
    max_steps : int, default 1000
        Maximum training steps.
    lr : float, default 0.1
        Learning rate.
    tol : float, default 1e-6
        Convergence tolerance.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    tuple
        (model, stats) - trained model and goodness of fit statistics.

    Example
    -------
    >>> from pytranus import calibrate, TranusConfig
    >>> config = TranusConfig(...)
    >>> model, stats = calibrate(config, device='cuda')
    >>> print(f"Housing R²: {stats['housing']['r_squared']:.4f}")
    >>> h = model.h  # shadow prices
    """
    model = LCALModel.from_config(config, device=device)

    if verbose:
        print(f"Training LCAL model on {model._device}...")

    train(model, max_steps=max_steps, lr=lr, tol=tol, verbose=verbose)

    stats = model.goodness_of_fit()

    if verbose:
        print(f"\nResults:")
        print(f"  Housing R²: {stats['housing']['r_squared']:.4f}")

    return model, stats


# Backwards compatibility alias
class Lcal:
    """
    Legacy wrapper for LCALModel.

    For new code, use LCALModel directly with standard PyTorch training:

    >>> model = LCALModel.from_config(config, device='cuda')
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    >>> for step in range(1000):
    ...     optimizer.zero_grad()
    ...     loss = model()
    ...     loss.backward()
    ...     optimizer.step()
    >>> h = model.h  # shadow prices
    """

    def __init__(
        self,
        config: TranusConfig,
        normalize: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = config
        self.param = load_params(config, normalize)
        self.model = LCALModel(self.param, device=device)

        self.n_sectors = self.model.n_sectors
        self.n_zones = self.model.n_zones
        self.housing_sectors = self.model.housing_sectors.cpu().numpy().tolist()
        self.genflux_sectors = self.model.genflux_sectors.cpu().numpy().tolist()
        self.device = self.model._device
        self.dtype = self.model._dtype

        # Aliases for backwards compatibility
        self.X_target = self.model.X_target
        self.X = self.model.X_0.clone()
        self.h = self.model.h
        self.D = torch.zeros_like(self.model.X_0)

    def calc_sp_housing(self, max_steps: int = 1000, lr: float = 0.1, verbose: bool = True, **kwargs) -> None:
        """Train housing sectors."""
        optimizer = torch.optim.Adam(self.model.housing_model.parameters(), lr=lr)

        for step in range(max_steps):
            optimizer.zero_grad()
            loss = self.model.housing_model()
            loss.backward()
            optimizer.step()

            if verbose and step % 100 == 0:
                print(f"  Step {step}: housing loss = {loss.item():.6f}")

            if loss.item() < 1e-6:
                break

        # Update state
        hs = self.housing_sectors
        with torch.no_grad():
            self.model.h.data[hs, :] = self.model.housing_model.h.data
            self.X[hs, :] = self.model.housing_model.production()

            U_ni = self.model.price + self.model.h
            _, _, self.D = self.model.compute_demands(U_ni)

        if verbose:
            stats = self.model.goodness_of_fit()
            print(f"  Housing R²: {stats['housing']['r_squared']:.4f}")

    def compute_shadow_prices(
        self,
        ph0: Tensor | None = None,
        max_steps: int = 1000,
        lr: float = 0.1,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Compute shadow prices using standard training loop.

        Returns
        -------
        tuple
            (p, h, converged) - prices, shadow prices, convergence flag.
        """
        # Train full model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for step in range(max_steps):
            optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            optimizer.step()

            if verbose and step % 100 == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

            if loss.item() < 1e-6:
                break

        # Get results
        h = self.model.h.detach()
        p = self.model.price.clone()

        # Update state
        with torch.no_grad():
            self.X = self.model.production()
            U_ni = self.model.price + self.model.h
            _, _, self.D = self.model.compute_demands(U_ni)

        converged = 1 if loss.item() < 1e-4 else 0

        return p, h, converged

    def goodness_of_fit(self) -> dict:
        """Get goodness of fit statistics."""
        return self.model.goodness_of_fit()

    def to_numpy(self) -> dict:
        """Convert results to numpy."""
        return self.model.to_numpy()

    def reset(self) -> None:
        """Reset model parameters."""
        with torch.no_grad():
            self.model.h.zero_()
            self.model.housing_model.h.zero_()


# Make LcalTorch an alias
LcalTorch = Lcal
