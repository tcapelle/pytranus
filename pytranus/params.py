"""
LCAL parameter data structures.

This module provides dataclass-based containers for LCAL parameters,
loaded from Tranus input files (L0E, L1E, C1S).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pytranus.config import TranusConfig

logger = logging.getLogger(__name__)


def _is_float(s: str) -> bool:
    """Check if a string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _line_remove_strings(line: list[str]) -> list[str]:
    """Remove non-numeric strings from a line."""
    return [x for x in line if _is_float(x)]


def _find_section(parsed_lines: list[list[str]], section: str) -> int:
    """Find the line index of a section header."""
    for idx, line in enumerate(parsed_lines):
        if line and line[0] == section:
            return idx
    raise ValueError(f"Section {section} not found")


def _find_section_in_lines(
    lines: list[str], section: str, column_slice: tuple[int, int]
) -> int:
    """Find section header with specific column slice."""
    start, end = column_slice
    for idx, line in enumerate(lines):
        if len(line) > end and line[start:end] == section:
            return idx
    raise ValueError(f"Section {section} not found")


@dataclass
class ZoneData:
    """Zone information from Z1E file."""

    internal_zones: list[int]
    external_zones: list[int]

    @property
    def n_internal(self) -> int:
        """Number of internal zones."""
        return len(self.internal_zones)

    @property
    def n_external(self) -> int:
        """Number of external zones."""
        return len(self.external_zones)

    @property
    def n_total(self) -> int:
        """Total number of zones."""
        return self.n_internal + self.n_external

    @property
    def all_zones(self) -> list[int]:
        """All zones (internal + external)."""
        return self.internal_zones + self.external_zones


@dataclass
class SectorData:
    """Sector information from L1E file."""

    sector_ids: list[int]
    sector_names: list[str] = field(default_factory=list)

    @property
    def n_sectors(self) -> int:
        """Number of sectors."""
        return len(self.sector_ids)


@dataclass
class LcalParams:
    """
    LCAL calibration parameters.

    This dataclass holds all parameters needed for LCAL calibration,
    loaded from Tranus input files.

    Attributes
    ----------
    zones : ZoneData
        Zone information.
    sectors : SectorData
        Sector information.
    housing_sectors : ndarray
        Indices of housing (land-use) sectors.
    substitution_sectors : ndarray
        Indices of sectors with substitution.

    Production/Demand Data (from L0E):
    exog_prod : ndarray of shape (n_sectors, n_zones)
        Exogenous base-year production.
    indu_prod : ndarray of shape (n_sectors, n_zones)
        Induced base-year production.
    exog_demand : ndarray of shape (n_sectors, n_zones)
        Exogenous demand.
    price : ndarray of shape (n_sectors, n_zones)
        Base-year prices.
    value_added : ndarray of shape (n_sectors, n_zones)
        Value added.
    attractor : ndarray of shape (n_sectors, n_zones)
        Attractors W^n_i.

    Sector Parameters (from L1E):
    alpha : ndarray of shape (n_sectors,)
        Attractor exponent.
    beta : ndarray of shape (n_sectors,)
        Dispersion parameter for localization.
    lamda : ndarray of shape (n_sectors,)
        Marginal utility of price.
    theta_loc : ndarray of shape (n_sectors,)
        Localization normalization exponent.

    Demand Function Parameters (from L1E):
    demax : ndarray of shape (n_sectors, n_sectors)
        Maximum demand function value.
    demin : ndarray of shape (n_sectors, n_sectors)
        Minimum demand function value.
    delta : ndarray of shape (n_sectors, n_sectors)
        Demand function dispersion.

    Substitution Parameters (from L1E):
    sigma : ndarray of shape (n_sectors,)
        Substitution dispersion parameter.
    theta_sub : ndarray of shape (n_sectors,)
        Substitution normalization exponent.
    omega : ndarray of shape (n_sectors, n_sectors)
        Substitution weights.
    Kn : ndarray of shape (n_sectors, n_sectors)
        Substitution choice set.

    Attractor Coefficients (from L1E):
    bkn : ndarray of shape (n_sectors, n_sectors)
        Attractor weight coefficients.

    Transport Data (from C1S):
    t_nij : ndarray of shape (n_sectors, n_zones, n_zones)
        Transport disutility.
    tm_nij : ndarray of shape (n_sectors, n_zones, n_zones)
        Transport monetary cost.
    """

    zones: ZoneData
    sectors: SectorData

    # Sector classification
    housing_sectors: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    substitution_sectors: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # L0E Section 1.1 - Production/Demand data
    exog_prod: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    indu_prod: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    exog_demand: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    price: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    value_added: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    attractor: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    rmin: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    rmax: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # L1E Section 2.1 - Sector parameters
    alpha: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    beta: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lamda: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    theta_loc: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # L1E Section 2.2 - Demand function parameters
    demax: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    demin: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    delta: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # L1E Section 2.3 - Substitution parameters
    sigma: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    theta_sub: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    omega: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Kn: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # L1E Section 3.2 - Attractor coefficients
    bkn: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Transport data from C1S
    t_nij: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    tm_nij: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    @property
    def n_zones(self) -> int:
        """Number of internal zones."""
        return self.zones.n_internal

    @property
    def n_sectors(self) -> int:
        """Number of sectors."""
        return self.sectors.n_sectors

    @property
    def list_zones(self) -> list[int]:
        """List of internal zone IDs."""
        return self.zones.internal_zones

    @property
    def list_sectors(self) -> list[int]:
        """List of sector IDs."""
        return self.sectors.sector_ids

    @property
    def sectors_names(self) -> list[str]:
        """List of sector names."""
        return self.sectors.sector_names

    # Backward compatibility aliases
    @property
    def ExogProd(self) -> NDArray[np.float64]:
        return self.exog_prod

    @property
    def InduProd(self) -> NDArray[np.float64]:
        return self.indu_prod

    @property
    def ExogDemand(self) -> NDArray[np.float64]:
        return self.exog_demand

    @property
    def Price(self) -> NDArray[np.float64]:
        return self.price

    @property
    def ValueAdded(self) -> NDArray[np.float64]:
        return self.value_added

    @property
    def Attractor(self) -> NDArray[np.float64]:
        return self.attractor

    @property
    def Rmin(self) -> NDArray[np.float64]:
        return self.rmin

    @property
    def Rmax(self) -> NDArray[np.float64]:
        return self.rmax

    @property
    def alfa(self) -> NDArray[np.float64]:
        return self.alpha

    @property
    def thetaLoc(self) -> NDArray[np.float64]:
        return self.theta_loc

    @property
    def thetaSub(self) -> NDArray[np.float64]:
        return self.theta_sub

    def normalize(
        self,
        t: float | None = None,
        tm: float | None = None,
        P: float | None = None,
    ) -> None:
        """
        Normalize input variables of the utility function.

        Parameters
        ----------
        t : float, optional
            Normalization factor for transport disutility.
        tm : float, optional
            Normalization factor for transport cost.
        P : float, optional
            Normalization factor for prices.
        """
        if t is None and self.t_nij.max() > 0:
            t = 10 ** np.floor(np.log10(self.t_nij.max()))
        if tm is None and self.tm_nij.max() > 0:
            tm = 10 ** np.floor(np.log10(self.tm_nij.max()))
        if P is None and len(self.housing_sectors) > 0:
            max_price = self.price[self.housing_sectors, :].max()
            if max_price > 0:
                P = 10 ** np.floor(np.log10(max_price))

        if t is not None and t > 0:
            self.t_nij /= t
        if tm is not None and tm > 0:
            self.tm_nij /= tm
        if P is not None and P > 0:
            self.price /= P

    def __repr__(self) -> str:
        return (
            f"LcalParams(\n"
            f"  n_sectors={self.n_sectors},\n"
            f"  n_zones={self.n_zones},\n"
            f"  housing_sectors={self.housing_sectors.tolist()},\n"
            f"  substitution_sectors={self.substitution_sectors.tolist()}\n"
            f")"
        )

    def to_tensors(self, device: str | None = None, dtype=None):
        """
        Convert all parameters to PyTorch tensors.

        Parameters
        ----------
        device : str, optional
            Device for tensors ('cpu', 'cuda', 'mps').
        dtype : torch.dtype, optional
            Data type for tensors. Default is torch.float64.

        Returns
        -------
        dict
            Dictionary mapping parameter names to tensors.
        """
        import torch

        if dtype is None:
            dtype = torch.float64

        def to_tensor(arr, is_int=False):
            if is_int:
                return torch.tensor(arr, dtype=torch.long, device=device)
            return torch.tensor(arr, dtype=dtype, device=device)

        return {
            # Indices
            "housing_sectors": to_tensor(self.housing_sectors, is_int=True),
            "substitution_sectors": to_tensor(self.substitution_sectors, is_int=True),
            # Production/Demand
            "exog_prod": to_tensor(self.exog_prod),
            "indu_prod": to_tensor(self.indu_prod),
            "exog_demand": to_tensor(self.exog_demand),
            "price": to_tensor(self.price),
            "value_added": to_tensor(self.value_added),
            "attractor": to_tensor(self.attractor),
            "rmin": to_tensor(self.rmin),
            "rmax": to_tensor(self.rmax),
            # Sector parameters
            "alpha": to_tensor(self.alpha),
            "beta": to_tensor(self.beta),
            "lamda": to_tensor(self.lamda),
            "theta_loc": to_tensor(self.theta_loc),
            # Demand function
            "demax": to_tensor(self.demax),
            "demin": to_tensor(self.demin),
            "delta": to_tensor(self.delta),
            # Substitution
            "sigma": to_tensor(self.sigma),
            "theta_sub": to_tensor(self.theta_sub),
            "omega": to_tensor(self.omega),
            "Kn": to_tensor(self.Kn, is_int=True),
            # Attractors
            "bkn": to_tensor(self.bkn),
            # Transport
            "t_nij": to_tensor(self.t_nij),
            "tm_nij": to_tensor(self.tm_nij),
        }


def load_params(config: TranusConfig, normalize: bool = True) -> LcalParams:
    """
    Load LCAL parameters from Tranus input files.

    Parameters
    ----------
    config : TranusConfig
        Configuration object for the Tranus project.
    normalize : bool, default True
        Whether to normalize input parameters.

    Returns
    -------
    LcalParams
        Loaded parameters.
    """
    from pytranus.config import BinaryInterface

    print(f"  Loading Lcal parameters from: {config.working_directory}")

    # Load zone and sector info
    internal_zones, external_zones = config.get_zones()
    sector_ids = config.get_sectors()

    zones = ZoneData(internal_zones=internal_zones, external_zones=external_zones)
    sectors = SectorData(sector_ids=sector_ids)

    n_sectors = sectors.n_sectors
    n_zones = zones.n_internal

    # Initialize arrays
    params = LcalParams(
        zones=zones,
        sectors=sectors,
        exog_prod=np.zeros((n_sectors, n_zones)),
        indu_prod=np.zeros((n_sectors, n_zones)),
        exog_demand=np.zeros((n_sectors, n_zones)),
        price=np.zeros((n_sectors, n_zones)),
        value_added=np.zeros((n_sectors, n_zones)),
        attractor=np.zeros((n_sectors, n_zones)),
        rmin=np.zeros((n_sectors, n_zones)),
        rmax=np.zeros((n_sectors, n_zones)),
        alpha=np.zeros(n_sectors),
        beta=np.zeros(n_sectors),
        lamda=np.zeros(n_sectors),
        theta_loc=np.zeros(n_sectors),
        demax=np.zeros((n_sectors, n_sectors)),
        demin=np.zeros((n_sectors, n_sectors)),
        delta=np.zeros((n_sectors, n_sectors)),
        sigma=np.zeros(n_sectors),
        theta_sub=np.zeros(n_sectors),
        omega=np.zeros((n_sectors, n_sectors)),
        Kn=np.zeros((n_sectors, n_sectors), dtype=np.intp),
        bkn=np.zeros((n_sectors, n_sectors)),
        t_nij=np.zeros((n_sectors, n_zones, n_zones)),
        tm_nij=np.zeros((n_sectors, n_zones, n_zones)),
    )

    # Load from files
    _load_c1s(params, config)
    _load_l0e(params, config)
    _load_l1e(params, config)

    if normalize:
        params.normalize()

    return params


def _load_c1s(params: LcalParams, config: TranusConfig) -> None:
    """Load C1S file (transport costs and disutilities)."""
    from pytranus.config import BinaryInterface

    interface = BinaryInterface(config)
    cost_file = Path(config.working_directory) / "COST_T.MTX"

    if not cost_file.is_file():
        logger.debug(f"{cost_file}: not found!")
        logger.debug("Creating Cost Files with ./mats")
        if not interface.run_mats():
            logger.error("Generating Disutility files has failed")
            return

    path = Path(config.working_directory)
    logger.debug("Reading Activity Location Parameters File (C1S) with Mats")

    list_zones = params.list_zones
    list_sectors = params.list_sectors
    n_zones = params.n_zones

    # Read COST_T.MTX
    with open(path / "COST_T.MTX", "r") as f:
        lines = f.readlines()

    sector_line = 4
    line_idx = 9

    while line_idx < len(lines):
        n = int(lines[sector_line][:4])
        z = 0

        while True:
            param_line = (lines[line_idx][:4] + lines[line_idx][25:]).split()
            if len(param_line) == 0:
                break

            try:
                i = int(param_line[0])
                aux_z = list_zones.index(i)
            except ValueError:
                aux_z = z

            if z < n_zones:
                sector_idx = list_sectors.index(n)
                params.tm_nij[sector_idx, aux_z, :] = [
                    float(x) for x in param_line[1 : n_zones + 1]
                ]

            z += 1
            line_idx += 1
            if line_idx == len(lines):
                break

        sector_line = line_idx + 4
        line_idx += 9

    # Read DISU_T.MTX
    with open(path / "DISU_T.MTX", "r") as f:
        lines = f.readlines()

    sector_line = 4
    line_idx = 9

    while line_idx < len(lines):
        n = int(lines[sector_line][:4])
        z = 0

        while True:
            param_line = (lines[line_idx][:4] + lines[line_idx][25:]).split()
            if len(param_line) == 0:
                break

            try:
                i = int(param_line[0])
                aux_z = list_zones.index(i)
            except ValueError:
                aux_z = z

            if z < n_zones:
                sector_idx = list_sectors.index(n)
                params.t_nij[sector_idx, aux_z, :] = [
                    float(x) for x in param_line[1 : n_zones + 1]
                ]

            z += 1
            line_idx += 1
            if line_idx == len(lines):
                break

        sector_line = line_idx + 4
        line_idx += 9


def _load_l0e(params: LcalParams, config: TranusConfig) -> None:
    """Load L0E (Localization Data) file."""
    filename = Path(config.working_directory) / config.obs_file
    logger.debug(f"Reading Localization Data File (L0E): {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    parsed_lines = [line.split() for line in lines]
    end_of_section = "*-"

    list_sectors = params.list_sectors
    list_zones = params.list_zones

    # Section 1.1
    line_idx = _find_section(parsed_lines, "1.1")
    line_idx += 2

    while parsed_lines[line_idx][0][:2] != end_of_section:
        n = list_sectors.index(float(parsed_lines[line_idx][0]))
        i = list_zones.index(float(parsed_lines[line_idx][1]))

        params.exog_prod[n, i] = float(parsed_lines[line_idx][2])
        params.indu_prod[n, i] = float(parsed_lines[line_idx][3])
        params.exog_demand[n, i] = float(parsed_lines[line_idx][4])
        params.price[n, i] = float(parsed_lines[line_idx][5])
        params.value_added[n, i] = float(parsed_lines[line_idx][6])
        params.attractor[n, i] = float(parsed_lines[line_idx][7])
        line_idx += 1

    # Section 2.1 - Filter sectors
    line_idx = _find_section(parsed_lines, "2.1")
    line_idx += 2

    while parsed_lines[line_idx][0][:2] != end_of_section:
        n = list_sectors.index(float(parsed_lines[line_idx][0]))
        i = list_zones.index(float(parsed_lines[line_idx][1]))
        params.exog_demand[n, i] = float(parsed_lines[line_idx][2])
        line_idx += 1

    # Section 2.2
    line_idx = _find_section(parsed_lines, "2.2")
    line_idx += 2

    while parsed_lines[line_idx][0][:2] != end_of_section:
        n = list_sectors.index(float(parsed_lines[line_idx][0]))
        i = list_zones.index(float(parsed_lines[line_idx][1]))
        params.rmin[n, i] = float(parsed_lines[line_idx][2])
        params.rmax[n, i] = float(parsed_lines[line_idx][3])
        line_idx += 1

    # Section 3. - Housing sectors
    line_idx = _find_section(parsed_lines, "3.")
    line_idx += 2

    housing_list: list[int] = []
    while parsed_lines[line_idx][0][:2] != end_of_section:
        n = list_sectors.index(float(parsed_lines[line_idx][0]))
        i = list_zones.index(float(parsed_lines[line_idx][1]))
        if n not in housing_list:
            housing_list.append(n)
        params.rmin[n, i] = float(parsed_lines[line_idx][2])
        params.rmax[n, i] = float(parsed_lines[line_idx][3])
        line_idx += 1

    params.housing_sectors = np.array(housing_list, dtype=np.intp)


def _load_l1e(params: LcalParams, config: TranusConfig) -> None:
    """Load L1E (Activity Location Parameters) file."""
    filename = Path(config.working_directory) / config.scenario_id / config.param_file
    logger.debug(f"Reading Activity Location Parameters File (L1E): {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    end_of_section = "*-"
    list_sectors = params.list_sectors
    sector_names: list[str] = []

    # Section 2.1 - Main sector parameters
    line_idx = _find_section_in_lines(lines, "2.1", column_slice=(3, 6))
    line_idx += 3

    while lines[line_idx][:2] != end_of_section:
        param_line = _line_remove_strings(lines[line_idx].split())
        n = list_sectors.index(float(param_line[0]))

        # Extract sector name
        sector_names.append(lines[line_idx][14:33].rstrip()[:-1])

        params.alpha[n] = float(param_line[6])
        params.beta[n] = float(param_line[1])
        params.lamda[n] = float(param_line[3])
        params.theta_loc[n] = float(param_line[5])
        line_idx += 1

    params.sectors.sector_names = sector_names

    # Parse remaining sections with split lines
    parsed_lines = [line.split() for line in lines]

    # Section 2.2 - Demand functions
    line_idx = _find_section(parsed_lines, "2.2")
    line_idx += 2

    while parsed_lines[line_idx][0][:2] != end_of_section:
        m = list_sectors.index(float(parsed_lines[line_idx][0]))
        n = list_sectors.index(float(parsed_lines[line_idx][1]))
        params.demin[m, n] = float(parsed_lines[line_idx][2])
        params.demax[m, n] = float(parsed_lines[line_idx][3])
        if params.demax[m, n] == 0:
            params.demax[m, n] = params.demin[m, n]
        params.delta[m, n] = float(parsed_lines[line_idx][4])
        line_idx += 1

    # Section 2.3 - Substitution
    line_idx = _find_section(parsed_lines, "2.3")
    line_idx += 2

    n = 0
    sub_sectors_list: list[int] = []

    while parsed_lines[line_idx][0][:2] != end_of_section:
        if len(parsed_lines[line_idx]) == 5:
            n = list_sectors.index(float(parsed_lines[line_idx][0]))
            params.sigma[n] = float(parsed_lines[line_idx][1])
            params.theta_sub[n] = float(parsed_lines[line_idx][2])
            choice = list_sectors.index(float(parsed_lines[line_idx][3]))
            params.Kn[n, choice] = 1
            params.omega[n, choice] = float(parsed_lines[line_idx][4])
            sub_sectors_list.append(n)

        if len(parsed_lines[line_idx]) == 2:
            choice = list_sectors.index(int(parsed_lines[line_idx][0]))
            params.Kn[n, choice] = 1
            params.omega[n, choice] = float(parsed_lines[line_idx][1])

        line_idx += 1

    params.substitution_sectors = np.array(sub_sectors_list, dtype=np.intp)

    # Section 3.2 - Attractor coefficients
    line_idx = _find_section(parsed_lines, "3.2")
    line_idx += 2

    while parsed_lines[line_idx][0][:2] != end_of_section:
        m = list_sectors.index(float(parsed_lines[line_idx][0]))
        n = list_sectors.index(float(parsed_lines[line_idx][1]))
        params.bkn[m, n] = float(parsed_lines[line_idx][2])
        line_idx += 1


# Backward compatibility alias
LcalParam = LcalParams
