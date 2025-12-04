"""
PyTorch nn.Module implementations for LCAL computations.

This module provides differentiable building blocks for the LCAL model,
enabling automatic differentiation and GPU acceleration.

Best practices used:
- F.softmax for numerically stable softmax computation
- torch.clamp for safe logarithms (avoid log(0))
- torch.einsum for efficient tensor contractions
- Vectorized operations where possible (avoid Python loops)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytranus.torch_utils import EPS

if TYPE_CHECKING:
    from pytranus.config import TranusConfig


def _safe_log(x: Tensor) -> Tensor:
    """Compute log with numerical stability using clamp."""
    return torch.log(x.clamp(min=EPS))


class DemandFunction(nn.Module):
    """
    Elastic demand function module.

    Computes: a^{mn}_i = a^{mn}_min + (a^{mn}_max - a^{mn}_min) * exp(-delta^{mn} * U^n_i)
    """

    def __init__(self, demin: Tensor, demax: Tensor, delta: Tensor) -> None:
        super().__init__()
        self.register_buffer("demin", demin)
        self.register_buffer("demax", demax)
        self.register_buffer("delta", delta)

    def forward(self, U_ni: Tensor) -> Tensor:
        """Compute demand functions a[m, n, i]."""
        U = U_ni.unsqueeze(0)
        demin = self.demin.unsqueeze(-1)
        demax = self.demax.unsqueeze(-1)
        delta = self.delta.unsqueeze(-1)
        gap = demax - demin
        return demin + gap * torch.exp(-delta * U)


class SubstitutionProbability(nn.Module):
    """Substitution probability module (logit-based)."""

    def __init__(self, sigma: Tensor, omega: Tensor, Kn: Tensor, attractor: Tensor) -> None:
        super().__init__()
        self.register_buffer("sigma", sigma)
        self.register_buffer("omega", omega)
        self.register_buffer("Kn", Kn.float())
        self.register_buffer("attractor", attractor)

    def forward(self, U_ni: Tensor, a_mni: Tensor) -> Tensor:
        """Compute substitution probabilities S[m, n, i]."""
        n_sectors = U_ni.shape[0]
        n_zones = U_ni.shape[1]

        # U_tilde[m,n,i] = omega[m,n] * a[m,n,i] * U[n,i]
        omega = self.omega.unsqueeze(-1)
        U_tilde = omega * a_mni * U_ni.unsqueeze(0)

        # Start with identity (S=1 for all pairs)
        S = torch.ones(n_sectors, n_sectors, n_zones, dtype=U_ni.dtype, device=U_ni.device)

        # Find sectors with substitution choices
        sectors_with_choices = self.Kn.sum(dim=1).nonzero(as_tuple=True)[0]

        # Compute logit probabilities for each sector with choices
        for m in sectors_with_choices:
            choices = self.Kn[m].nonzero(as_tuple=True)[0]
            # logits = log(A) - sigma * U_tilde
            logits = (
                _safe_log(self.attractor[choices, :])
                - self.sigma[m] * U_tilde[m, choices, :]
            )
            # F.softmax handles numerical stability internally
            S[m, choices, :] = F.softmax(logits, dim=0)

        return S


class LocationProbability(nn.Module):
    """Location choice probability module (logit-based)."""

    def __init__(
        self,
        beta: Tensor,
        lamda: Tensor,
        t_nij: Tensor,
        A_ni: Tensor,
        housing_sectors: Tensor,
        genflux_sectors: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("beta", beta)
        self.register_buffer("lamda", lamda)
        self.register_buffer("t_nij", t_nij)
        self.register_buffer("A_ni", A_ni)
        self.register_buffer("housing_sectors", housing_sectors)
        self.register_buffer("genflux_sectors", genflux_sectors)

        n_sectors = beta.shape[0]
        n_zones = t_nij.shape[1]
        self.n_sectors = n_sectors
        self.n_zones = n_zones
        self.register_buffer("eye", torch.eye(n_zones, dtype=beta.dtype))

        # Pre-compute log attractors for efficiency
        self.register_buffer("log_A_ni", _safe_log(A_ni))

    def forward(self, ph: Tensor) -> Tensor:
        """Compute location probabilities Pr[n, i, j]."""
        Pr = self.eye.unsqueeze(0).expand(self.n_sectors, -1, -1).clone()

        # U_nij = lambda_n * ph_j + t_nij  (destination utility)
        U_nij = self.lamda[:, None, None] * ph[:, None, :] + self.t_nij

        for n in self.genflux_sectors:
            if n in self.housing_sectors:
                continue
            # logits = log(A_j) - beta * U_ij
            logits = self.log_A_ni[n].unsqueeze(0) - self.beta[n] * U_nij[n]
            Pr[n] = F.softmax(logits, dim=-1)

        return Pr

    def forward_single(self, ph_n: Tensor, n: int) -> Tensor:
        """Compute location probabilities for a single sector."""
        if n in self.housing_sectors or n not in self.genflux_sectors:
            return self.eye
        U_nij = self.lamda[n] * ph_n.unsqueeze(0) + self.t_nij[n]
        logits = self.log_A_ni[n].unsqueeze(0) - self.beta[n] * U_nij
        return F.softmax(logits, dim=-1)


class TotalDemand(nn.Module):
    """Total demand computation module."""

    def __init__(self, exog_demand: Tensor, exog_prod: Tensor) -> None:
        super().__init__()
        self.register_buffer("exog_demand", exog_demand)
        self.register_buffer("exog_prod", exog_prod)

    def forward(self, a_mni: Tensor, S_mni: Tensor, X_0: Tensor) -> Tensor:
        """Compute total demand D[n, i]."""
        X_total = X_0 + self.exog_prod
        induced = torch.einsum("mni,mni,mi->ni", a_mni, S_mni, X_total)
        return self.exog_demand + induced


class HousingModel(nn.Module):
    """
    Housing sector model with learnable shadow prices.

    This is the main module for housing calibration. Shadow prices `h` are
    nn.Parameters that we optimize to match observed production.
    """

    def __init__(
        self,
        housing_sectors: Tensor,
        demin: Tensor,
        demax: Tensor,
        delta: Tensor,
        sigma: Tensor,
        omega: Tensor,
        Kn: Tensor,
        attractor: Tensor,
        exog_demand: Tensor,
        price: Tensor,
        X_0: Tensor,
        X_target: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("housing_sectors", housing_sectors)
        self.register_buffer("demin", demin)
        self.register_buffer("demax", demax)
        self.register_buffer("delta", delta)
        self.register_buffer("sigma", sigma)
        self.register_buffer("omega", omega)
        self.register_buffer("Kn", Kn.float())
        self.register_buffer("attractor", attractor)
        self.register_buffer("exog_demand", exog_demand)
        self.register_buffer("price", price)
        self.register_buffer("X_0", X_0)
        self.register_buffer("X_target", X_target)

        n_housing = len(housing_sectors)
        n_zones = price.shape[1]

        # Shadow prices are the learnable parameters!
        self.h = nn.Parameter(torch.zeros(n_housing, n_zones, dtype=price.dtype))

    def forward(self) -> Tensor:
        """
        Compute MSE loss between predicted and target production.

        Returns
        -------
        Tensor (scalar)
            MSE loss.
        """
        X_pred = self.production()
        return F.mse_loss(X_pred, self.X_target)

    def production(self) -> Tensor:
        """Compute predicted production for housing sectors."""
        U_n = self.price + self.h

        # Demand function: a = demin + (demax - demin) * exp(-delta * U)
        gap = self.demax - self.demin
        a_mn = self.demin.unsqueeze(-1) + gap.unsqueeze(-1) * torch.exp(
            -self.delta.unsqueeze(-1) * U_n.unsqueeze(0)
        )

        # Substitution utility
        omega = self.omega.unsqueeze(-1)
        U_tilde = omega * a_mn * U_n.unsqueeze(0)

        # Substitution probabilities (logit model)
        S_mn = torch.ones_like(a_mn)
        sectors_with_choices = self.Kn.sum(dim=1).nonzero(as_tuple=True)[0]

        for m in sectors_with_choices:
            choices = self.Kn[m].nonzero(as_tuple=True)[0]
            # logits = log(A) - sigma * U_tilde
            logits = (
                _safe_log(self.attractor[choices, :])
                - self.sigma[m] * U_tilde[m, choices, :]
            )
            S_mn[m, choices, :] = F.softmax(logits, dim=0)

        # Production: exog + einsum over (m,i) of X_0[m,i] * a[m,n,i] * S[m,n,i]
        return self.exog_demand + torch.einsum("mi,mni->ni", self.X_0, a_mn * S_mn)


class TransportableModel(nn.Module):
    """
    Transportable sector model with learnable combined price phi.

    For each transportable sector, phi = lambda * (p + h) is the learnable parameter.
    """

    def __init__(
        self,
        sector_idx: int,
        beta: Tensor,
        lamda: Tensor,
        t_nij: Tensor,
        A_ni: Tensor,
        D_n: Tensor,
        X_target: Tensor,
    ) -> None:
        super().__init__()
        self.sector_idx = sector_idx
        self.register_buffer("beta", beta)
        self.register_buffer("lamda", lamda)
        self.register_buffer("t_nij", t_nij)
        self.register_buffer("D_n", D_n)
        self.register_buffer("X_target", X_target)

        # Pre-compute log attractors
        self.register_buffer("log_A_ni", _safe_log(A_ni))

        n_zones = t_nij.shape[0]
        # phi = lambda * (p + h) is the learnable parameter
        self.phi = nn.Parameter(torch.zeros(n_zones, dtype=beta.dtype))

    def forward(self) -> Tensor:
        """Compute MSE loss between predicted and target production."""
        X_pred = self.production()
        return F.mse_loss(X_pred, self.X_target)

    def production(self) -> Tensor:
        """Compute predicted production."""
        # U_ij = lambda * phi_j + t_ij
        U_nij = self.lamda * self.phi.unsqueeze(0) + self.t_nij
        # logits = log(A_j) - beta * U_ij
        logits = self.log_A_ni.unsqueeze(0) - self.beta * U_nij
        Pr_n = F.softmax(logits, dim=-1)
        return torch.einsum("i,ij->j", self.D_n, Pr_n)

    def get_location_probs(self) -> Tensor:
        """Get location probabilities."""
        U_nij = self.lamda * self.phi.unsqueeze(0) + self.t_nij
        logits = self.log_A_ni.unsqueeze(0) - self.beta * U_nij
        return F.softmax(logits, dim=-1)


class LCALModel(nn.Module):
    """
    Complete LCAL calibration model.

    Shadow prices `h` are nn.Parameters. Call forward() to get the loss,
    then backprop and step your optimizer.

    Example
    -------
    >>> model = LCALModel.from_config(config, device='cuda')
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>>
    >>> for step in range(1000):
    ...     optimizer.zero_grad()
    ...     loss = model()
    ...     loss.backward()
    ...     optimizer.step()
    ...     if loss.item() < 1e-6:
    ...         break
    >>>
    >>> h = model.h  # optimized shadow prices
    """

    def __init__(
        self,
        params,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.n_sectors = params.n_sectors
        self.n_zones = params.n_zones

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device

        # MPS doesn't support float64
        if device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float64
        self._dtype = dtype

        def to_tensor(arr):
            return torch.tensor(arr, dtype=dtype, device=device)

        # Store indices
        self.register_buffer("housing_sectors", to_tensor(params.housing_sectors).long())
        self.n_housing = len(params.housing_sectors)

        # Identify transportable sectors
        genflux = []
        for n in range(params.n_sectors):
            aux = params.demin[:, n] + params.demax[:, n]
            if params.beta[n] != 0 and aux.any():
                genflux.append(n)
        self.register_buffer("genflux_sectors", torch.tensor(genflux, dtype=torch.long, device=device))

        # Store parameters as buffers
        self.register_buffer("beta", to_tensor(params.beta))
        self.register_buffer("lamda", to_tensor(params.lamda))
        self.register_buffer("sigma", to_tensor(params.sigma))
        self.register_buffer("alpha", to_tensor(params.alpha))
        self.register_buffer("demin", to_tensor(params.demin))
        self.register_buffer("demax", to_tensor(params.demax))
        self.register_buffer("delta", to_tensor(params.delta))
        self.register_buffer("omega", to_tensor(params.omega))
        self.register_buffer("Kn", to_tensor(params.Kn))
        self.register_buffer("bkn", to_tensor(params.bkn))
        self.register_buffer("t_nij", to_tensor(params.t_nij))
        self.register_buffer("tm_nij", to_tensor(params.tm_nij))
        self.register_buffer("exog_prod", to_tensor(params.exog_prod))
        self.register_buffer("indu_prod", to_tensor(params.indu_prod))
        self.register_buffer("exog_demand", to_tensor(params.exog_demand))
        self.register_buffer("price", to_tensor(params.price))
        self.register_buffer("value_added", to_tensor(params.value_added))
        self.register_buffer("attractor", to_tensor(params.attractor))

        X_0 = to_tensor(params.exog_prod + params.indu_prod)
        self.register_buffer("X_0", X_0)
        self.register_buffer("X_target", to_tensor(params.indu_prod))

        A_ni = (self.bkn @ X_0) ** self.alpha[:, None] * self.attractor
        self.register_buffer("A_ni", A_ni)

        # Shadow prices - THE learnable parameters
        self.h = nn.Parameter(torch.zeros(params.n_sectors, params.n_zones, dtype=dtype, device=device))

        # Submodules for forward computation
        self.demand_fn = DemandFunction(self.demin, self.demax, self.delta)
        self.substitution = SubstitutionProbability(self.sigma, self.omega, self.Kn, self.attractor)
        self.location_prob = LocationProbability(
            self.beta, self.lamda, self.t_nij, self.A_ni,
            self.housing_sectors, self.genflux_sectors
        )
        self.total_demand = TotalDemand(self.exog_demand, self.exog_prod)

        # Housing submodel for phase 1
        hs = params.housing_sectors
        self.housing_model = HousingModel(
            housing_sectors=self.housing_sectors,
            demin=to_tensor(params.demin[:, hs]),
            demax=to_tensor(params.demax[:, hs]),
            delta=to_tensor(params.delta[:, hs]),
            sigma=self.sigma,
            omega=to_tensor(params.omega[:, hs]),
            Kn=to_tensor(params.Kn[:, hs]),
            attractor=to_tensor(params.attractor[hs, :]),
            exog_demand=to_tensor(params.exog_demand[hs, :]),
            price=to_tensor(params.price[hs, :]),
            X_0=self.X_0,
            X_target=to_tensor(params.indu_prod[hs, :]),
        )

    @classmethod
    def from_config(
        cls,
        config: TranusConfig,
        normalize: bool = True,
        device: torch.device | str | None = None,
    ) -> "LCALModel":
        """Create model from TranusConfig."""
        from pytranus.params import load_params
        params = load_params(config, normalize)
        return cls(params, device=device)

    def forward(self) -> Tensor:
        """
        Compute total calibration loss.

        Returns MSE between predicted and observed production for all sectors.
        """
        X_pred = self.production()
        return F.mse_loss(X_pred, self.X_target)

    def production(self) -> Tensor:
        """Compute predicted production from current shadow prices."""
        U_ni = self.price + self.h
        a_mni = self.demand_fn(U_ni)
        S_mni = self.substitution(U_ni, a_mni)
        D = self.total_demand(a_mni, S_mni, self.X_0)
        Pr = self.location_prob(U_ni)
        return torch.einsum("ni,nij->nj", D, Pr)

    def housing_loss(self) -> Tensor:
        """Compute loss for housing sectors only."""
        return self.housing_model()

    def housing_production(self, h_housing: Tensor | None = None) -> Tensor:
        """Compute production for housing sectors."""
        if h_housing is None:
            h_housing = self.h[self.housing_sectors, :]
        return self.housing_model.production()

    def transportable_production(self, phi_n: Tensor, n: int, D_n: Tensor) -> Tensor:
        """Compute production for a transportable sector given phi and D."""
        # U_ij = lambda * phi_j + t_ij
        U_nij = self.lamda[n] * phi_n.unsqueeze(0) + self.t_nij[n]
        # logits = log(A_j) - beta * U_ij
        logits = _safe_log(self.A_ni[n]).unsqueeze(0) - self.beta[n] * U_nij
        Pr_n = F.softmax(logits, dim=-1)
        return torch.einsum("i,ij->j", D_n, Pr_n)

    def compute_demands(self, U_ni: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute demand functions, substitution, and total demands."""
        a_mni = self.demand_fn(U_ni)
        S_mni = self.substitution(U_ni, a_mni)
        D = self.total_demand(a_mni, S_mni, self.X_0)
        return a_mni, S_mni, D

    def goodness_of_fit(self) -> dict:
        """Compute R², MSE, MAE statistics."""
        with torch.no_grad():
            X_pred = self.production()

            stats = {}

            # Housing
            hs = self.housing_sectors
            target_h = self.X_target[hs, :]
            pred_h = X_pred[hs, :]
            ss_err = torch.sum((pred_h - target_h) ** 2)
            ss_tot = torch.sum((target_h - target_h.mean()) ** 2)

            stats["housing"] = {
                "r_squared": (1 - ss_err / ss_tot).item(),
                "mse": F.mse_loss(pred_h, target_h).item(),
                "mae": F.l1_loss(pred_h, target_h).item(),
            }

            # Per sector
            for n in self.genflux_sectors.tolist():
                target_n = self.X_target[n, :]
                pred_n = X_pred[n, :]
                ss_err = torch.sum((pred_n - target_n) ** 2)
                ss_tot = torch.sum((target_n - target_n.mean()) ** 2)

                stats[f"sector_{n}"] = {
                    "r_squared": (1 - ss_err / ss_tot).item(),
                    "mse": F.mse_loss(pred_n, target_n).item(),
                    "mae": F.l1_loss(pred_n, target_n).item(),
                }

            return stats

    def to_numpy(self) -> dict:
        """Convert results to numpy arrays."""
        with torch.no_grad():
            return {
                "h": self.h.cpu().numpy(),
                "X": self.production().cpu().numpy(),
                "X_target": self.X_target.cpu().numpy(),
            }


class GeneralLCALModel(nn.Module):
    """
    General LCAL calibration model (Equations 3.3-3.4 from the paper).

    This model solves the FULL calibration problem without the housing sector
    simplification (Pr = I). All sectors are treated uniformly with full
    location choice probabilities.

    The general equilibrium system:
        X_n,j = Σ_i D_n,i × Pr_n,i,j(p, h)     # Production
        D_n,i = f(a, S, X)                      # Demand
        Pr_n,i,j = softmax_j(log(A_j) - β × U_ij)  # Location choice

    With PyTorch autodiff, we jointly optimize h for all sectors to minimize:
        L(h) = Σ_n Σ_j (X_n,j(h) - X^obs_n,j)²

    Example
    -------
    >>> model = GeneralLCALModel.from_config(config, device='cuda')
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>>
    >>> for step in range(1000):
    ...     optimizer.zero_grad()
    ...     loss = model()
    ...     loss.backward()
    ...     optimizer.step()
    >>>
    >>> h = model.h  # optimized shadow prices for ALL sectors
    """

    def __init__(
        self,
        params,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.n_sectors = params.n_sectors
        self.n_zones = params.n_zones

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device

        # MPS doesn't support float64
        if device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float64
        self._dtype = dtype

        def to_tensor(arr):
            return torch.tensor(arr, dtype=dtype, device=device)

        # Store indices (but we treat ALL sectors the same way)
        self.register_buffer("housing_sectors", to_tensor(params.housing_sectors).long())
        self.n_housing = len(params.housing_sectors)

        # ALL sectors with non-zero beta participate in location choice
        genflux = []
        for n in range(params.n_sectors):
            aux = params.demin[:, n] + params.demax[:, n]
            if params.beta[n] != 0 and aux.any():
                genflux.append(n)
        self.register_buffer("genflux_sectors", torch.tensor(genflux, dtype=torch.long, device=device))

        # Store parameters as buffers
        self.register_buffer("beta", to_tensor(params.beta))
        self.register_buffer("lamda", to_tensor(params.lamda))
        self.register_buffer("sigma", to_tensor(params.sigma))
        self.register_buffer("alpha", to_tensor(params.alpha))
        self.register_buffer("demin", to_tensor(params.demin))
        self.register_buffer("demax", to_tensor(params.demax))
        self.register_buffer("delta", to_tensor(params.delta))
        self.register_buffer("omega", to_tensor(params.omega))
        self.register_buffer("Kn", to_tensor(params.Kn))
        self.register_buffer("bkn", to_tensor(params.bkn))
        self.register_buffer("t_nij", to_tensor(params.t_nij))
        self.register_buffer("tm_nij", to_tensor(params.tm_nij))
        self.register_buffer("exog_prod", to_tensor(params.exog_prod))
        self.register_buffer("indu_prod", to_tensor(params.indu_prod))
        self.register_buffer("exog_demand", to_tensor(params.exog_demand))
        self.register_buffer("price", to_tensor(params.price))
        self.register_buffer("value_added", to_tensor(params.value_added))
        self.register_buffer("attractor", to_tensor(params.attractor))

        X_0 = to_tensor(params.exog_prod + params.indu_prod)
        self.register_buffer("X_0", X_0)
        self.register_buffer("X_target", to_tensor(params.indu_prod))

        # Location attractors
        A_ni = (self.bkn @ X_0) ** self.alpha[:, None] * self.attractor
        self.register_buffer("A_ni", A_ni)

        # Identity matrix for sectors without location choice
        self.register_buffer("eye", torch.eye(params.n_zones, dtype=dtype, device=device))

        # Shadow prices - learnable for ALL sectors
        self.h = nn.Parameter(torch.zeros(params.n_sectors, params.n_zones, dtype=dtype, device=device))

        # Submodules
        self.demand_fn = DemandFunction(self.demin, self.demax, self.delta)
        self.substitution = SubstitutionProbability(self.sigma, self.omega, self.Kn, self.attractor)
        self.total_demand = TotalDemand(self.exog_demand, self.exog_prod)

    @classmethod
    def from_config(
        cls,
        config: "TranusConfig",
        normalize: bool = True,
        device: torch.device | str | None = None,
    ) -> "GeneralLCALModel":
        """Create model from TranusConfig."""
        from pytranus.params import load_params
        params = load_params(config, normalize)
        return cls(params, device=device)

    def forward(self) -> Tensor:
        """
        Compute calibration loss for the general problem.

        Returns MSE between predicted and observed production for all sectors.
        This is the objective function for equations 3.3-3.4.
        """
        X_pred = self.production()
        return F.mse_loss(X_pred, self.X_target)

    def location_probabilities(self, U_ni: Tensor) -> Tensor:
        """
        Compute location probabilities for ALL sectors (no housing special case).

        This implements the general location choice model where every sector
        with beta != 0 has proper location choice probabilities.

        Parameters
        ----------
        U_ni : Tensor of shape (n_sectors, n_zones)
            Total utility U = p + h

        Returns
        -------
        Tensor of shape (n_sectors, n_zones, n_zones)
            Location probabilities Pr[n, i, j]
        """
        n_sectors = self.n_sectors

        # Initialize all to identity (for sectors with beta=0)
        Pr = self.eye.unsqueeze(0).expand(n_sectors, -1, -1).clone()

        # U_nij = lambda_n * U_n,j + t_n,i,j
        # For location choice: destination utility matters
        U_nij = self.lamda[:, None, None] * U_ni[:, None, :] + self.t_nij

        # Compute for ALL sectors with beta != 0
        for n in range(n_sectors):
            if self.beta[n] == 0:
                continue  # Keep Pr = I for sectors without location choice

            # logits = log(A_j) - beta * U_ij
            logits = _safe_log(self.A_ni[n]).unsqueeze(0) - self.beta[n] * U_nij[n]
            Pr[n] = F.softmax(logits, dim=-1)

        return Pr

    def production(self) -> Tensor:
        """
        Compute predicted production using full location choice model.

        This implements equations 3.3-3.4 where:
        - Demand D depends on utilities U = p + h
        - Location probabilities Pr depend on U
        - Production X = D @ Pr

        The entire computation is differentiable w.r.t. h.
        """
        U_ni = self.price + self.h

        # Demand functions: a[m,n,i] = f(U_n,i)
        a_mni = self.demand_fn(U_ni)

        # Substitution probabilities: S[m,n,i]
        S_mni = self.substitution(U_ni, a_mni)

        # Total demand: D[n,i] = exog + Σ_m(a × S × X_0)
        D = self.total_demand(a_mni, S_mni, self.X_0)

        # Location probabilities for ALL sectors (general case)
        Pr = self.location_probabilities(U_ni)

        # Production: X[n,j] = Σ_i D[n,i] × Pr[n,i,j]
        return torch.einsum("ni,nij->nj", D, Pr)

    def goodness_of_fit(self) -> dict:
        """Compute R², MSE, MAE statistics."""
        with torch.no_grad():
            X_pred = self.production()
            stats = {}

            # Overall
            ss_err_total = torch.sum((X_pred - self.X_target) ** 2)
            ss_tot_total = torch.sum((self.X_target - self.X_target.mean()) ** 2)

            stats["overall"] = {
                "r_squared": (1 - ss_err_total / ss_tot_total).item(),
                "mse": F.mse_loss(X_pred, self.X_target).item(),
                "mae": F.l1_loss(X_pred, self.X_target).item(),
            }

            # Housing sectors
            hs = self.housing_sectors
            target_h = self.X_target[hs, :]
            pred_h = X_pred[hs, :]
            ss_err = torch.sum((pred_h - target_h) ** 2)
            ss_tot = torch.sum((target_h - target_h.mean()) ** 2)

            stats["housing"] = {
                "r_squared": (1 - ss_err / ss_tot).item(),
                "mse": F.mse_loss(pred_h, target_h).item(),
                "mae": F.l1_loss(pred_h, target_h).item(),
            }

            # Transportable sectors
            for n in self.genflux_sectors.tolist():
                if n in self.housing_sectors.tolist():
                    continue
                target_n = self.X_target[n, :]
                pred_n = X_pred[n, :]
                ss_err = torch.sum((pred_n - target_n) ** 2)
                ss_tot = torch.sum((target_n - target_n.mean()) ** 2)

                stats[f"sector_{n}"] = {
                    "r_squared": (1 - ss_err / ss_tot).item(),
                    "mse": F.mse_loss(pred_n, target_n).item(),
                    "mae": F.l1_loss(pred_n, target_n).item(),
                }

            return stats

    def to_numpy(self) -> dict:
        """Convert results to numpy arrays."""
        with torch.no_grad():
            return {
                "h": self.h.cpu().numpy(),
                "X": self.production().cpu().numpy(),
                "X_target": self.X_target.cpu().numpy(),
            }
