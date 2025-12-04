"""
LCAL - Land Use Calibration Module for Tranus.

This module implements the Python Tranus Land Use Calibration (LCAL) system,
reformulating the calibration process as an optimization problem for semi-automatic
parameter calibration.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import leastsq, minimize

from pytranus._math import compute_DX_n
from pytranus.params import LcalParams, load_params

if TYPE_CHECKING:
    from pytranus.config import TranusConfig

logger = logging.getLogger(__name__)


class Lcal:
    """
    LCAL - Land Use Calibration class.

    This class is the core of the Tranus Land Use module. It computes all variables
    related to land use and finds the prices and shadow prices for calibration.

    Parameters
    ----------
    config : TranusConfig
        Configuration object for the Tranus project.
    normalize : bool, default True
        Whether to normalize input parameters.

    Attributes
    ----------
    param : LcalParams
        Container for all LCAL parameters loaded from input files.
    X : ndarray of shape (n_sectors, n_zones)
        Induced production matrix.
    h : ndarray of shape (n_sectors, n_zones)
        Shadow prices.
    p : ndarray of shape (n_sectors, n_zones)
        Prices.

    Examples
    --------
    >>> from pytranus import TranusConfig, Lcal
    >>> config = TranusConfig(bin_path, working_dir, project_id, scenario_id)
    >>> lcal = Lcal(config)
    >>> p, h, conv, lamda_opt = lcal.compute_shadow_prices(ph0)
    """

    def __init__(self, config: TranusConfig, normalize: bool = True) -> None:
        """Initialize the LCAL calibration object."""
        self.config = config
        self.param = load_params(config, normalize)

        n_sectors = self.param.n_sectors
        n_zones = self.param.n_zones

        # Production and price arrays
        self.X: NDArray[np.float64] = np.zeros((n_sectors, n_zones))
        self.h: NDArray[np.float64] = np.zeros((n_sectors, n_zones))
        self.p: NDArray[np.float64] = np.zeros((n_sectors, n_zones))

        # Model state
        self.a: NDArray[np.float64] | int = 0
        self.S: NDArray[np.float64] | int = 0
        self.Pr: NDArray[np.float64] | int = 0
        self.U_nij: NDArray[np.float64] | int = 0
        self.D: NDArray[np.float64] = np.zeros((n_sectors, n_zones))

        # Model configuration
        self.logit: bool = True
        self.use_multiprocessing: bool = True

        # Base year production
        self.X_0: NDArray[np.float64] = self.param.exog_prod + self.param.indu_prod

        # Attractors for location probability
        self.A_ni: NDArray[np.float64] = (
            np.dot(self.param.bkn, self.X_0) ** self.param.alpha[:, np.newaxis]
        ) * self.param.attractor

        # Housing sectors
        self.housing_sectors: NDArray[np.intp] = self.param.housing_sectors
        n_housing = len(self.housing_sectors)
        self.h_housing: NDArray[np.float64] = np.zeros((n_housing, n_zones))
        self.X_housing: NDArray[np.float64] = np.zeros((n_housing, n_zones))

        # Sectors that generate flux
        self.genflux_sectors: NDArray[np.intp] = np.array(
            [n for n in range(n_sectors) if self._generates_flux(n)], dtype=np.intp
        )

        # State flag
        self._non_transportable_done: bool = False

        # Backward compatibility
        self.tranus_config = config

        logger.debug("LCAL instance created")

    def reset(self, X: NDArray[np.float64] | None = None) -> None:
        """Reset the LCAL parameters."""
        self.param = load_params(self.config, normalize=False)
        self.a = 0
        self.S = 0
        if X is not None:
            self.X_0 = X
        self._non_transportable_done = False

    def _generates_flux(self, n: int) -> bool:
        """Check if sector n generates flux."""
        beta = self.param.beta
        demax = self.param.demax
        demin = self.param.demin
        aux = demin[:, n] + demax[:, n]
        return beta[n] != 0 and aux.any()

    def calc_a(self, U_ni: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the demand functions a^{mn}_i."""
        if self._non_transportable_done:
            return self.a

        demin = self.param.demin
        demax = self.param.demax
        delta = self.param.delta

        gap_delta = demax - demin
        arg = -delta[:, :, np.newaxis] * U_ni[np.newaxis, :, :]
        a = demin[:, :, np.newaxis] + gap_delta[:, :, np.newaxis] * np.exp(arg)
        self.a = a
        return self.a

    def calc_prob_loc(
        self, h: NDArray[np.float64], p: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute localization probabilities and aggregated disutilities."""
        n_sectors = self.param.n_sectors
        n_zones = self.param.n_zones
        beta = self.param.beta
        lamda = self.param.lamda
        t_nij = self.param.t_nij

        aux0 = lamda[:, np.newaxis] * (p + h)
        U_nij = aux0[:, np.newaxis, :] + t_nij
        self.U_nij = U_nij

        A_ni = self.A_ni
        Pr_nij = np.zeros((n_sectors, n_zones, n_zones))
        U_ni = np.zeros((n_sectors, n_zones))

        for n in range(n_sectors):
            if not self._generates_flux(n):
                U_ni[n, :] = self.param.lamda[n] * (p[n, :] + h[n, :])
                Pr_nij[n, :, :] = np.eye(n_zones)
            else:
                aux = A_ni[n, np.newaxis, :] * np.exp(-beta[n] * U_nij[n, :, :])
                s = aux.sum(1)
                Pr_nij[n, :, :] = aux / s[:, np.newaxis]

        return Pr_nij, U_ni

    def calc_subst(self, U_ni: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute substitution probabilities S_mni."""
        if self._non_transportable_done:
            return self.S

        n_sectors = self.param.n_sectors
        n_zones = self.param.n_zones
        Kn = self.param.Kn
        sigma = self.param.sigma
        Attractor = self.param.attractor
        omega = self.param.omega

        S_mni = np.ones((n_sectors, n_sectors, n_zones))
        U_tilde = omega[:, :, np.newaxis] * self.calc_a(U_ni) * U_ni[np.newaxis, :, :]

        for m in Kn.sum(1).nonzero()[0]:
            subt_choices = Kn[m].nonzero()[0]
            aux = Attractor[subt_choices, :] * np.exp(-sigma[m] * U_tilde[m, subt_choices, :])
            s = aux.sum(0)
            S_mni[m, subt_choices, :] = np.where(aux == 0, 0, aux / s[np.newaxis, :])

        return S_mni

    def calc_induced_prod_housing(
        self,
        h_i: NDArray[np.float64],
        i: int,
        jac_omega: bool = False,
        jac_delta: bool = False,
        jac_sigma: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute induced production for housing sectors in zone i."""
        housing_sectors = self.housing_sectors
        n_housing = len(housing_sectors)
        p_i = self.param.price[housing_sectors, i]
        n_sectors = self.param.n_sectors
        ExogDemand = self.param.exog_demand[housing_sectors, i]
        X0 = self.X_0[:, i]

        S_mn = np.ones((n_sectors, n_housing))

        demin = self.param.demin[:, housing_sectors]
        demax = self.param.demax[:, housing_sectors]
        delta = self.param.delta[:, housing_sectors]

        U_n = p_i + h_i
        a_mn = demin + (demax - demin) * np.exp(-delta * U_n[np.newaxis, :])

        Kn = self.param.Kn[:, housing_sectors]
        sigma = self.param.sigma
        Attractor = self.param.attractor[housing_sectors, i]
        omega = self.param.omega[:, housing_sectors]

        U_tilde = omega * a_mn * U_n[np.newaxis, :]

        for m in Kn.sum(1).nonzero()[0]:
            subt_choices = Kn[m, :].nonzero()[0]
            aux = Attractor[subt_choices] * np.exp(-sigma[m] * U_tilde[m, subt_choices])
            s = aux.sum(0)
            S_mn[m, subt_choices] = np.where(aux == 0, 0, aux / s)

        if jac_delta:
            da = -U_n[np.newaxis, :] * (a_mn - demin)
            Jac_delta = np.einsum(
                "m, mn, m, mq, q, mq, mn, mq->nmq",
                X0, a_mn, -sigma, omega, U_n, da, -S_mn, S_mn
            )
            Jac_delta[range(n_housing), :, range(n_housing)] = (
                np.einsum("m, mn, mn->nm", X0, da, S_mn)
                + np.einsum("m, mn, m, mn, n, mn, mn->nm", X0, a_mn, -sigma, omega, U_n, da, S_mn - S_mn**2)
            )
            prod = ExogDemand + np.dot(X0, a_mn * S_mn)
            return prod, Jac_delta[:, self.param.substitution_sectors, :]

        if jac_sigma:
            dS = S_mn * (
                -omega * a_mn * U_n[np.newaxis, :]
                + np.einsum("ml,ml,l,ml->m", omega, a_mn, U_n, S_mn)[:, np.newaxis]
            )
            Jac_sigma = np.einsum("m, mn, mn->nm", X0, a_mn, dS)
            prod = ExogDemand + np.dot(X0, a_mn * S_mn)
            return prod, Jac_sigma[:, self.param.substitution_sectors]

        if jac_omega:
            Jac_omega = np.einsum("m,mn,mq,q,mn,mq->nmq", X0 * sigma, a_mn, a_mn, U_n, S_mn, S_mn)
            Jac_omega[range(n_housing), :, range(n_housing)] = -np.einsum(
                "m,mn,n,mn->nm", X0 * sigma, a_mn**2, U_n, S_mn - S_mn**2
            )
            prod = ExogDemand + np.dot(X0, a_mn * S_mn)
            return prod, Jac_omega[:, self.param.substitution_sectors, :]

        return ExogDemand + np.dot(X0, a_mn * S_mn)

    def residual_housing(
        self,
        h_i: NDArray[np.float64],
        i: int = 0,
        jac_omega: bool = False,
        jac_delta: bool = False,
        jac_sigma: bool = False,
        scaled: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute residual for housing sectors in zone i."""
        InduProd = self.param.indu_prod[self.housing_sectors, i]

        if scaled:
            scale = np.where(InduProd > 0, 1 / InduProd, 0)
        else:
            scale = np.ones_like(InduProd)

        if jac_delta:
            res, jac = self.calc_induced_prod_housing(h_i, i, jac_delta=True)
            return (res - InduProd) * scale, jac * scale[:, np.newaxis, np.newaxis]

        if jac_omega:
            res, jac = self.calc_induced_prod_housing(h_i, i, jac_omega=True)
            return (res - InduProd) * scale, jac * scale[:, np.newaxis, np.newaxis]

        if jac_sigma:
            res, jac = self.calc_induced_prod_housing(h_i, i, jac_sigma=True)
            return (res - InduProd) * scale, jac * scale[:, np.newaxis]

        return (self.calc_induced_prod_housing(h_i, i) - InduProd) * scale

    def calc_sp_lu(self, i: int, h0_i: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Compute optimal shadow prices for land-use sectors in zone i."""
        n_housing = len(self.housing_sectors)

        if h0_i is not None:
            if h0_i.shape != (n_housing,):
                raise ValueError(f"Shadow price shape must be ({n_housing},), got {h0_i.shape}")
            result = leastsq(self.residual_housing, h0_i, args=(i,), full_output=True, ftol=1e-12)
            return result[0]

        h0_i = np.random.random(n_housing)
        h0_i = np.where(self.param.indu_prod[self.housing_sectors, i] == 0, 0, h0_i)

        result = leastsq(self.residual_housing, h0_i, args=(i,), full_output=True, ftol=1e-12)

        if result[-1] > 4:
            if result[-1] == 5:
                result = leastsq(
                    self.residual_housing, h0_i, args=(i,), maxfev=1500, full_output=True, ftol=1e-12
                )
            logger.debug(f"Zone {i}: Status {result[-1]}, Residuals {result[2]['fvec']} - check")
        else:
            logger.debug(f"Zone {i}: Status {result[-1]}, Residuals {result[2]['fvec']}")

        if norm(result[2]["fvec"]) > 0.001:
            result = leastsq(self.residual_housing, h0_i * 0, args=(i,), full_output=True, ftol=1e-12)
            logger.debug(f"Zone {i}: Re-executing with h0=0")

        return result[0]

    def calc_sp_lu_all(self, zones: list[int] | range) -> NDArray[np.float64]:
        """Compute optimal shadow prices for all specified zones."""
        logger.debug("Computing shadow prices for housing sectors")
        ti = time()

        if self.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = list(executor.map(self.calc_sp_lu, zones))
            aux = np.array(results)
        else:
            aux = np.array([self.calc_sp_lu(z) for z in zones])

        to = time()
        logger.debug(f"Solver completed in {to - ti:.2f}s")
        return aux

    def calc_sp_housing(self) -> None:
        """Compute optimal shadow prices for all housing sectors."""
        logger.debug("Running housing shadow price optimization")

        if self._non_transportable_done:
            return

        n_zones = self.param.n_zones

        self.h_housing = self.calc_sp_lu_all(range(n_zones)).T
        self.X_housing = np.array([
            self.calc_induced_prod_housing(self.h_housing[:, i], i)
            for i in range(n_zones)
        ]).T
        self.X[self.housing_sectors, :] = self.X_housing

        self.h[self.housing_sectors, :] = self.h_housing
        self.p[self.housing_sectors, :] = self.param.price[self.housing_sectors, :]
        self.a = self.calc_a(self.h + self.p)
        self.S = self.calc_subst(self.h + self.p)

        for n in range(self.param.n_sectors):
            self.D[n, :] = (
                self.param.exog_demand[n, :]
                + (self.a[:, n, :] * self.S[:, n, :] * (self.X_0 + self.param.exog_prod)).sum(0)
            )

        self._non_transportable_done = True

        fit_error = norm(self.X_housing - self.param.indu_prod[self.housing_sectors, :])
        close_001 = np.allclose(self.X_housing, self.param.indu_prod[self.housing_sectors, :], rtol=0.001)
        close_0001 = np.allclose(self.X_housing, self.param.indu_prod[self.housing_sectors, :], rtol=0.0001)

        print(f"  Housing sectors fitting ||X-X_0|| = {fit_error}")
        print(f"  np.allclose(X, X_0, rtol=0.001) = {close_001}")
        print(f"  np.allclose(X, X_0, rtol=0.0001) = {close_0001}")

    def reshape_vec(self, vec: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reshape flat vector to (h, p) matrices."""
        n_sectors = self.param.n_sectors
        n_zones = self.param.n_zones
        h = vec[: n_sectors * n_zones].reshape((n_sectors, n_zones))
        p = vec[n_sectors * n_zones :].reshape((n_sectors, n_zones))
        return h, p

    def calc_D_n(self, n: int) -> NDArray[np.float64]:
        """Compute demands for sector n."""
        if not self._non_transportable_done:
            self.calc_sp_housing()
        return self.D[n, :]

    def calc_Pr(self, ph: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Compute localization probabilities for sector n."""
        n_zones = self.param.n_zones
        beta = self.param.beta
        lamda = self.param.lamda
        t_nij = self.param.t_nij
        A_ni = self.A_ni

        if self._generates_flux(n) and n not in self.housing_sectors:
            U_n = lamda[n] * ph[np.newaxis, :] + t_nij[n, :, :]
            if self.logit:
                aux = A_ni[n, np.newaxis, :] * np.exp(-beta[n] * U_n)
            else:
                aux = A_ni[n, np.newaxis, :] * U_n ** (-beta[n])
            s = aux.sum(1)
            return aux / s[:, np.newaxis]

        return np.eye(n_zones)

    def calc_X_n(self, ph: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Compute production X for sector n."""
        D_n = self.calc_D_n(n)
        Pr_n = self.calc_Pr(ph, n)
        return np.dot(Pr_n.T, D_n)

    def calc_X(self, ph: NDArray[np.float64]) -> None:
        """Compute production X for all sectors."""
        for n in self.genflux_sectors:
            self.X[n, :] = np.einsum("i,ij->j", self.D[n, :], self.calc_Pr(ph[n, :], n))

    def calc_DX_n(self, ph: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Compute gradient of production X for sector n."""
        n_sectors = self.param.n_sectors
        n_zones = self.param.n_zones
        beta = self.param.beta
        lamda = self.param.lamda
        t_nij = self.param.t_nij

        D_n = self.calc_D_n(n)
        Pr_n = self.calc_Pr(ph, n)
        DX = np.zeros((n_zones, n_zones))

        U_n = lamda[n] * ph[np.newaxis, :] + t_nij[n, :, :]
        compute_DX_n(DX, n_sectors, n_zones, beta[n], lamda[n], D_n, Pr_n, U_n, self.logit)

        return DX

    def res_X_n(self, ph: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Compute residual of production X for sector n."""
        return self.calc_X_n(ph, n) - self.param.indu_prod[n, :]

    def norm_res_X_n(self, ph: NDArray[np.float64], n: int) -> tuple[float, NDArray[np.float64]]:
        """Compute squared norm of residual and gradient."""
        r = self.res_X_n(ph, n)
        DX = self.calc_DX_n(ph, n)
        return norm(r) ** 2, 2 * np.dot(r, DX)

    def calc_ph_n(self, n: int, ph0: NDArray[np.float64], algo: str = "leastsq"):
        """Compute optimal ph (price + shadow price) for sector n."""
        if not self._generates_flux(n):
            print(f"Sector {n} is not transportable")
            return None

        if algo != "leastsq":
            return minimize(self.norm_res_X_n, ph0, args=(n,), jac=True, method=algo)

        result = leastsq(
            self.res_X_n, ph0, args=(n,), Dfun=self.calc_DX_n, full_output=True, ftol=1e-4, xtol=1e-4
        )

        infodict = result[2]
        ss_err = (infodict["fvec"] ** 2).sum()
        y = self.param.indu_prod[n, :]
        ss_tot = ((y - y.mean()) ** 2).sum()
        rsquared = 1 - (ss_err / ss_tot)

        logger.debug(f"  GOF: {rsquared}")
        logger.debug(f"  Error: {norm(infodict['fvec'] / self.param.indu_prod[n, :].max())}")

        return result

    def calc_ph(self, ph0: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
        """Compute optimal ph for all transportable sectors."""
        logger.debug("Computing optimal p+h")

        conv = 1
        ph = np.random.random((self.param.n_sectors, self.param.n_zones))
        status_dict = {}

        for n in self.genflux_sectors:
            logger.debug(f"  Computing h+p for sector {n}")
            result = self.calc_ph_n(n, ph0[n])
            ph[n, :] = result[0]
            status_dict[n] = result[-1]
            if status_dict[n] > 4:
                conv = 0

        return ph - ph.mean(1)[:, np.newaxis], conv

    def calc_p_linear(self, ph: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute equilibrium prices by solving linear system."""
        logger.debug("Computing linear prices")

        n_zones = self.param.n_zones
        n_sectors = self.param.n_sectors
        ValueAdded = self.param.value_added
        tm = self.param.tm_nij

        LAMBDA2 = np.zeros((n_sectors, n_zones))
        DELTA2 = np.zeros((n_sectors * n_zones, n_sectors * n_zones))

        alpha = self.a
        S = self.S

        Pr = np.zeros((n_sectors, n_zones, n_zones))
        for n in range(n_sectors):
            Pr[n, :, :] = self.calc_Pr(ph[n, :], n)
        self.Pr = Pr

        for m in range(n_sectors):
            for i in range(n_zones):
                if self._generates_flux(m):
                    LAMBDA2[m, i] = ValueAdded[m, i] + (
                        alpha[m, :, i] * S[m, :, i] * (Pr[:, i, :] * tm[:, i, :]).sum(1)
                    ).sum()
                    DELTA2[m * n_zones + i, :] = np.einsum(
                        "nj,n->nj", Pr[:, i, :], alpha[m, :, i] * S[m, :, i]
                    ).reshape(-1)

        LAMBDA2 = LAMBDA2.reshape(-1)
        DELTA2 = np.eye(n_sectors * n_zones) - DELTA2

        p = linalg.solve(DELTA2, LAMBDA2)
        logger.debug(f"Linear solve residual: {norm(np.dot(DELTA2, p) - LAMBDA2)}")

        return p.reshape((n_sectors, n_zones))

    def compute_shadow_prices(
        self, ph0: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
        """Compute optimal shadow prices for all sectors."""
        h = np.zeros((self.param.n_sectors, self.param.n_zones))

        self.calc_sp_housing()
        ph, conv = self.calc_ph(ph0)
        p = self.calc_p_linear(ph)
        self.calc_X(ph)

        p[self.housing_sectors, :] = self.param.price[self.housing_sectors, :]

        lamda_opti = self._compute_lamda(ph, p)

        for n in self.genflux_sectors:
            h[n, :] = ph[n, :] / self.param.lamda[n] - p[n, :]
            h[n, :] = h[n, :] - h[n, :].mean()
            self.h[n, :] = h[n, :]
            self.p[n, :] = p[n, :]

        return p, h, conv, lamda_opti

    def _compute_lamda(self, ph: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute optimal lambda values."""
        lamda_opti = np.zeros(self.param.n_sectors)
        for n in self.genflux_sectors:
            cov = np.cov(ph[n, :], p[n, :], bias=1)
            lamda_opti[n] = np.var(ph[n, :]) / cov[1, 0]
        return lamda_opti

    def coscon(self) -> NDArray[np.float64]:
        """Compute consumption cost."""
        consuming_sectors = self.param.demin.sum(0)
        coscon = np.einsum("nij,nij->ni", self.Pr, (self.p[:, np.newaxis, :] + self.param.tm_nij))
        coscon[consuming_sectors == 0] = 0
        return coscon

    def cospro(self) -> NDArray[np.float64]:
        """Compute production cost."""
        return np.einsum("mni, ni->mi", self.a * self.S, self.coscon()) + self.param.value_added

    def to_dataframe(self) -> pd.DataFrame:
        """Export results to DataFrame."""
        X_ser = pd.Series(self.X_housing.ravel())
        h_ser = pd.Series(self.h_housing.ravel())
        p_ser = pd.Series(self.param.price[self.housing_sectors, :].ravel())
        Indu_ser = pd.Series(self.param.indu_prod[self.housing_sectors, :].ravel())
        adjust = h_ser / p_ser * 100

        df = pd.DataFrame({
            "Production": Indu_ser,
            "Demand": X_ser,
            "Price": p_ser,
            "Adjust (%)": adjust,
            "SPrice": h_ser
        })
        return df
