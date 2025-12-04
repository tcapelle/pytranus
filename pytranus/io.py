"""
Tranus file I/O operations.

This module provides functions and classes for reading/writing Tranus binary files,
particularly the L1S format used by LCAL.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pytranus.config import TranusConfig
    from pytranus.lcal import Lcal


def _pad_string(string: str, length: int) -> str:
    """Pad string to specified length with spaces."""
    if len(string) >= length:
        return string[:length]
    return string + " " * (length - len(string))


@dataclass
class L1sData:
    """
    Data read from an L1S file.

    Attributes
    ----------
    production : ndarray
        Production by sector and zone.
    cost_production : ndarray
        Production cost.
    price : ndarray
        Prices.
    cost_consumption : ndarray
        Consumption cost.
    utility_consumption : ndarray
        Consumption utility.
    attractor : ndarray
        Attractor values.
    adjustment : ndarray
        Shadow price adjustments.
    demand : ndarray
        Demands.
    year : int
        Year from file.
    month : int
        Month from file.
    day : int
        Day from file.
    hour : int
        Hour from file.
    minute : int
        Minute from file.
    """

    production: NDArray[np.float64]
    cost_production: NDArray[np.float64]
    price: NDArray[np.float64]
    cost_consumption: NDArray[np.float64]
    utility_consumption: NDArray[np.float64]
    attractor: NDArray[np.float64]
    adjustment: NDArray[np.float64]
    demand: NDArray[np.float64]
    year: int = 0
    month: int = 0
    day: int = 0
    hour: int = 0
    minute: int = 0


def read_l1s(file_path: str | Path, n_sectors: int, n_zones: int) -> L1sData:
    """
    Read an L1S binary file.

    Parameters
    ----------
    file_path : str or Path
        Path to the L1S file.
    n_sectors : int
        Number of sectors.
    n_zones : int
        Number of zones.

    Returns
    -------
    L1sData
        Data extracted from the file.
    """
    # Initialize output arrays
    pro = np.zeros((n_sectors, n_zones))
    cospro = np.zeros((n_sectors, n_zones))
    precio = np.zeros((n_sectors, n_zones))
    coscon = np.zeros((n_sectors, n_zones))
    utcon = np.zeros((n_sectors, n_zones))
    atrac = np.zeros((n_sectors, n_zones))
    ajuste = np.zeros((n_sectors, n_zones))
    dem = np.zeros((n_sectors, n_zones))

    with open(file_path, "rb") as f:
        data = f.read()

    offset = 0

    # Read header
    header = struct.unpack_from("<3h2i5h3s80s3s80s", data, offset)
    offset += 3 * 2 + 4 + 4 + 5 * 2 + 3 + 80 + 3 + 80

    year = header[5]
    month = header[6]
    day = header[7]
    hour = header[8]
    minute = header[9]

    # Read policy info
    num_policy = struct.unpack_from("<2i", data, offset)
    offset += 2 * 4
    npol = num_policy[0]

    for _ in range(npol):
        struct.unpack_from("<2ib5s32s", data, offset)
        offset += 2 * 4 + 1 + 5 + 32

    struct.unpack_from("<2i", data, offset)
    offset += 2 * 4

    # Read sector info
    sector_1 = struct.unpack_from("<3i", data, offset)
    offset += 3 * 4
    ns = sector_1[0]

    for _ in range(ns):
        struct.unpack_from("<2i32s?5f2i", data, offset)
        offset += 2 * 4 + 32 + 1 + 5 * 4 + 2 * 4

    struct.unpack_from("<i", data, offset)
    offset += 4

    # Read demand functions
    num_sect = struct.unpack_from("<2i", data, offset)
    offset += 2 * 4
    ns = num_sect[0]

    for _ in range(ns):
        struct.unpack_from("<i", data, offset)
        offset += 4

        for _ in range(ns):
            demand_functions = struct.unpack_from("<i12fi", data, offset)
            offset += 4 + 12 * 4 + 4
            mxsust = demand_functions[13]

            for _ in range(mxsust):
                struct.unpack_from("<i", data, offset)
                offset += 4

    struct.unpack_from("<i", data, offset)
    offset += 4

    # Read zone info
    param_nzn = struct.unpack_from("<4i", data, offset)
    offset += 4 * 4
    nzn = param_nzn[0]

    for _ in range(nzn):
        struct.unpack_from("<2i32s2i", data, offset)
        offset += 2 * 4 + 32 + 2 * 4

    struct.unpack_from("<i", data, offset)
    offset += 4

    nzn_ns = struct.unpack_from("<2i", data, offset)
    offset += 2 * 4
    nzn = nzn_ns[0]
    ns = nzn_ns[1]

    # Read L1S data
    for i in range(nzn):
        struct.unpack_from("<2i", data, offset)
        offset += 2 * 4

        for n in range(ns):
            fmt = "<idf2dfd3fd4fd3f"
            len_fmt = 4 + 8 + 4 + 2 * 8 + 4 + 8 + 3 * 4 + 8 + 4 * 4 + 8 + 3 * 4
            param_l1s = struct.unpack_from(fmt, data, offset)
            offset += len_fmt

            if n < n_sectors and i < n_zones:
                pro[n, i] = param_l1s[3]
                cospro[n, i] = param_l1s[4]
                precio[n, i] = param_l1s[6]
                dem[n, i] = param_l1s[8]
                coscon[n, i] = param_l1s[9]
                utcon[n, i] = param_l1s[10]
                atrac[n, i] = param_l1s[13]
                ajuste[n, i] = param_l1s[15]

    return L1sData(
        production=pro,
        cost_production=cospro,
        price=precio,
        cost_consumption=coscon,
        utility_consumption=utcon,
        attractor=atrac,
        adjustment=ajuste,
        demand=dem,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
    )


@dataclass
class L1sWriter:
    """
    Writer for L1S binary files.

    This class extracts parameters from Tranus input files and writes
    them along with LCAL results to an L1S file.

    Parameters
    ----------
    config : TranusConfig
        Configuration object for the Tranus project.
    """

    config: TranusConfig

    # File metadata
    file_major: int = field(default=6, init=False)
    file_minor: int = field(default=8, init=False)
    file_release: int = field(default=1, init=False)
    ifmt_l1s: int = field(default=3, init=False)

    # Zone info
    num_zones: list[int] = field(default_factory=list, init=False)
    num_zones_ext: list[int] = field(default_factory=list, init=False)
    num_sectors: list[int] = field(default_factory=list, init=False)

    # Arrays (initialized in __post_init__)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize arrays after dataclass creation."""
        self.num_zones, self.num_zones_ext = self.config.get_zones()
        self.num_sectors = self.config.get_sectors()

        self.nb_zones = len(self.num_zones)
        self.nb_zones_ext = len(self.num_zones_ext)
        self.nb_tot_zones = self.nb_zones + self.nb_zones_ext
        self.nb_sectors = len(self.num_sectors)

        # Initialize all arrays
        self._init_arrays()

    def _init_arrays(self) -> None:
        """Initialize all numpy arrays."""
        ns = self.nb_sectors
        nz = self.nb_tot_zones

        # Z1E parameters
        self.num_zon = np.zeros(nz, np.int32)
        self.nom_zon = np.empty(nz, dtype="S32")
        self.jer1 = np.zeros(nz, np.int32)
        self.jer2 = np.zeros(nz, np.int32)
        self.nzn = 0
        self.nz1 = 0
        self.nz2 = 0

        # L0E parameters
        self.xpro = np.zeros((ns, nz))
        self.probase = np.zeros((ns, nz))
        self.xdem = np.zeros((ns, nz))
        self.prebase = np.zeros((ns, nz))
        self.rmin = np.zeros((ns, nz))
        self.rmax = np.zeros((ns, nz))
        self.rmax.fill(8.99999949e09)
        self.valag = np.zeros((ns, nz))
        self.atrain = np.zeros((ns, nz))

        # L1E parameters
        self.nb_iterations = 0
        self.ns = 0
        self.nflu = 0
        self.lflu = np.zeros(ns, np.bool_)
        self.num_sec = list(self.num_sectors)
        self.nom_sec = np.empty(ns, dtype="S32")
        self.beta_1 = np.zeros(ns)
        self.beta_2 = np.zeros(ns)
        self.gama_1 = np.zeros(ns)
        self.gama_2 = np.zeros(ns)
        self.min_price_cost_ratio = np.zeros(ns)
        self.sector_type = np.zeros(ns)
        self.target_sector = np.zeros(ns)
        self.demin = np.zeros((ns, ns))
        self.demax = np.zeros((ns, ns))
        self.delas = np.zeros((ns, ns))
        self.selas = np.zeros((ns, ns))
        self.suslgsc = np.ones((ns, ns))
        self.xalfa_1 = np.zeros((ns, ns))
        self.xalfa_2 = np.zeros((ns, ns))
        self.xalfapro = np.zeros((ns, ns))
        self.xalfapre = np.zeros((ns, ns))
        self.xalfacap = np.zeros((ns, ns))
        self.alfa_1 = np.zeros((ns, ns))
        self.alfa_2 = np.zeros((ns, ns))
        self.mxsust = 256
        self.nsust = np.zeros((ns, ns, self.mxsust))

        # CTL parameters
        self.area = ""
        self.estudio = ""
        self.pol = self.config.scenario_id
        self.npol = 0
        self.i_prev_pol: list[int] = []
        self.prev_pol_type: list[int] = []
        self.nom_pol: list[str] = []
        self.desc_pol: list[str] = []
        self.i_pol = 1

        # Output parameters
        self.pro = np.zeros((ns, nz))
        self.cospro = np.zeros((ns, nz))
        self.precio = np.zeros((ns, nz))
        self.coscon = np.zeros((ns, nz))
        self.utcon = np.zeros((ns, nz))
        self.atrac = np.zeros((ns, nz))
        self.ajuste = np.zeros((ns, nz))
        self.stock = np.zeros((ns, nz))
        self.unstock = np.zeros((ns, nz))
        self.dem = np.zeros((ns, nz))

        # Time
        self.year = 0
        self.month = 0
        self.day = 0
        self.hour = 0
        self.minute = 0

        self._initialized = True

    def extract_from_files(self) -> None:
        """Extract all parameters from input files."""
        self._extract_l0e()
        self._extract_z1e()
        self._extract_l1e()
        self._extract_ctl()
        self._update_datetime()

    def _update_datetime(self) -> None:
        """Update date and time to current."""
        now = datetime.now()
        self.year = now.year
        self.month = now.month
        self.day = now.day
        self.hour = now.hour
        self.minute = now.minute

    def _find_section(self, parsed_lines: list[list[str]], section: str) -> int:
        """Find the line index of a section header."""
        for idx, line in enumerate(parsed_lines):
            if line and line[0] == section:
                return idx
        raise ValueError(f"Section {section} not found")

    def _extract_l0e(self) -> None:
        """Extract parameters from L0E file."""
        with open(self.config.L0E_filepath, "r") as f:
            lines = f.readlines()

        parsed_lines = [line.split() for line in lines]
        end_of_section = "*-"

        # Section 1.1
        line_idx = self._find_section(parsed_lines, "1.1")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = self.num_sectors.index(float(parsed_lines[line_idx][0]))
            zone = self.num_zones.index(float(parsed_lines[line_idx][1]))
            self.xpro[sector, zone] = float(parsed_lines[line_idx][2])
            self.probase[sector, zone] = float(parsed_lines[line_idx][3])
            self.xdem[sector, zone] = float(parsed_lines[line_idx][4])
            self.prebase[sector, zone] = float(parsed_lines[line_idx][5])
            self.valag[sector, zone] = float(parsed_lines[line_idx][6])
            self.atrain[sector, zone] = float(parsed_lines[line_idx][7])
            line_idx += 1

        # Section 2.1
        line_idx = self._find_section(parsed_lines, "2.1")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = self.num_sectors.index(float(parsed_lines[line_idx][0]))
            zone = self.num_zones.index(float(parsed_lines[line_idx][1]))
            self.xdem[sector, zone] = float(parsed_lines[line_idx][2])
            line_idx += 1

        # Section 2.2
        line_idx = self._find_section(parsed_lines, "2.2")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = self.num_sectors.index(float(parsed_lines[line_idx][0]))
            zone = self.num_zones.index(float(parsed_lines[line_idx][1]))
            self.rmin[sector, zone] = float(parsed_lines[line_idx][2])
            self.rmax[sector, zone] = float(parsed_lines[line_idx][3])
            self.valag[sector, zone] = float(parsed_lines[line_idx][4])
            self.atrain[sector, zone] = float(parsed_lines[line_idx][5])
            line_idx += 1

        # Section 3.
        line_idx = self._find_section(parsed_lines, "3.")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = self.num_sectors.index(float(parsed_lines[line_idx][0]))
            zone = self.num_zones.index(float(parsed_lines[line_idx][1]))
            self.rmin[sector, zone] = float(parsed_lines[line_idx][2])
            self.rmax[sector, zone] = float(parsed_lines[line_idx][3])
            line_idx += 1

        # Set rmax for external zones to 0
        for sector in range(self.nb_sectors):
            for zone in range(self.nb_zones, self.nb_tot_zones):
                self.rmax[sector, zone] = 0

    def _extract_z1e(self) -> None:
        """Extract parameters from Z1E file."""
        with open(self.config.Z1E_filepath, "r") as f:
            lines = f.readlines()

        copy_lines = list(lines)
        parsed_lines = [line.split() for line in lines]
        end_of_section = "*-"
        all_zones = self.num_zones + self.num_zones_ext

        # Section 1.0
        line_idx = self._find_section(parsed_lines, "1.0")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            zone = all_zones.index(float(parsed_lines[line_idx][0]))
            self.num_zon[zone] = int(parsed_lines[line_idx][0])
            self.nom_zon[zone] = copy_lines[line_idx].split("'")[1]
            self.nzn += 1
            self.nz1 = self.nzn
            self.nz2 = self.nzn
            self.jer1[zone] = self.nzn
            self.jer2[zone] = self.nzn
            line_idx += 1

        # Section 3.0
        line_idx = self._find_section(parsed_lines, "3.0")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            zone = all_zones.index(float(parsed_lines[line_idx][0]))
            self.num_zon[zone] = int(parsed_lines[line_idx][0])
            self.nom_zon[zone] = parsed_lines[line_idx][1].strip("'")
            self.nzn += 1
            self.jer1[zone] = self.nzn
            self.jer2[zone] = self.nzn
            line_idx += 1

    def _extract_l1e(self) -> None:
        """Extract parameters from L1E file."""
        with open(self.config.L1E_filepath, "r") as f:
            lines = f.readlines()

        copy_lines = list(lines)
        parsed_lines = [line.split() for line in lines]
        end_of_section = "*-"

        # Section 1.0
        line_idx = self._find_section(parsed_lines, "1.0")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            self.nb_iterations = int(parsed_lines[line_idx][0])
            line_idx += 1

        # Section 2.1
        line_idx = self._find_section(parsed_lines, "2.1")
        line_idx += 3

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = self.num_sectors.index(float(parsed_lines[line_idx][0]))
            self.num_sec[sector] = parsed_lines[line_idx][0]
            self.nom_sec[sector] = copy_lines[line_idx].split("'")[1]
            self.ns += 1
            self.beta_1[sector] = float(parsed_lines[line_idx][2])
            self.beta_2[sector] = float(parsed_lines[line_idx][3])
            self.gama_1[sector] = float(parsed_lines[line_idx][4])
            self.gama_2[sector] = float(parsed_lines[line_idx][5])
            if self.beta_1[sector] != 0:
                self.nflu += 1
                self.lflu[sector] = True
            else:
                self.lflu[sector] = False
            self.min_price_cost_ratio[sector] = float(parsed_lines[line_idx][8])
            self.sector_type[sector] = float(parsed_lines[line_idx][9])
            self.target_sector[sector] = 0
            line_idx += 1

        # Section 2.2
        num_sectors_float = [float(x) for x in self.num_sectors]
        line_idx = self._find_section(parsed_lines, "2.2")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector_1 = num_sectors_float.index(float(parsed_lines[line_idx][0]))
            sector_2 = num_sectors_float.index(float(parsed_lines[line_idx][1]))
            self.demin[sector_1, sector_2] = float(parsed_lines[line_idx][2])
            diff = float(parsed_lines[line_idx][3]) - float(parsed_lines[line_idx][2])
            if diff >= 0:
                self.demax[sector_1, sector_2] = diff
            self.delas[sector_1, sector_2] = float(parsed_lines[line_idx][4])
            line_idx += 1

        # Section 2.3
        line_idx = self._find_section(parsed_lines, "2.3")
        line_idx += 2

        copy_sector_1 = 0
        copy_selas = "0"
        copy_suslgsc = "0"
        list_sectors_subst: list[int] = []

        while parsed_lines[line_idx][0][:2] != end_of_section:
            if len(parsed_lines[line_idx]) == 5:
                sector_1 = num_sectors_float.index(float(parsed_lines[line_idx][0]))
                copy_sector_1 = sector_1
                sector_2 = num_sectors_float.index(float(parsed_lines[line_idx][3]))
                self.selas[sector_1, sector_2] = float(parsed_lines[line_idx][1])
                copy_selas = parsed_lines[line_idx][1]
                self.suslgsc[sector_1, sector_2] = float(parsed_lines[line_idx][2])
                copy_suslgsc = parsed_lines[line_idx][2]
                list_sectors_subst.append(sector_2)

            if len(parsed_lines[line_idx]) == 2:
                sector_2 = num_sectors_float.index(float(parsed_lines[line_idx][0]))
                self.selas[copy_sector_1, sector_2] = float(copy_selas)
                self.suslgsc[copy_sector_1, sector_2] = float(copy_suslgsc)
                list_sectors_subst.append(sector_2)

            if parsed_lines[line_idx][0] == "/":
                for value_i in list_sectors_subst:
                    copy_list = [v for v in list_sectors_subst if v != value_i]
                    for k, value_k in enumerate(copy_list):
                        self.nsust[value_i, copy_sector_1, k] = value_k + 1
                list_sectors_subst = []

            line_idx += 1

        # Section 3.1
        line_idx = self._find_section(parsed_lines, "3.1")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = num_sectors_float.index(float(parsed_lines[line_idx][0]))
            attrac_sector = num_sectors_float.index(float(parsed_lines[line_idx][1]))
            self.xalfa_1[sector, attrac_sector] = float(parsed_lines[line_idx][2])
            self.xalfa_2[sector, attrac_sector] = float(parsed_lines[line_idx][3])
            self.xalfapro[sector, attrac_sector] = float(parsed_lines[line_idx][4])
            self.xalfapre[sector, attrac_sector] = float(parsed_lines[line_idx][5])
            self.xalfacap[sector, attrac_sector] = float(parsed_lines[line_idx][6])
            line_idx += 1

        # Section 3.2
        line_idx = self._find_section(parsed_lines, "3.2")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            sector = num_sectors_float.index(float(parsed_lines[line_idx][0]))
            var = num_sectors_float.index(float(parsed_lines[line_idx][1]))
            self.alfa_1[sector, var] = float(parsed_lines[line_idx][2])
            self.alfa_2[sector, var] = float(parsed_lines[line_idx][3])
            line_idx += 1

    def _extract_ctl(self) -> None:
        """Extract parameters from CTL file."""
        with open(self.config.CTL_filepath, "r") as f:
            lines = f.readlines()

        copy_lines = list(lines)
        parsed_lines = [line.split() for line in lines]
        end_of_section = "*-"

        # Section 1.0
        line_idx = self._find_section(parsed_lines, "1.0")
        line_idx += 2

        while parsed_lines[line_idx][0][:2] != end_of_section:
            self.area = copy_lines[line_idx].split("'")[1]
            self.estudio = copy_lines[line_idx].split("'")[3]
            line_idx += 1

        # Section 2.0
        line_idx = self._find_section(parsed_lines, "2.0")
        line_idx += 2

        list_i_pol: list[int] = []
        while parsed_lines[line_idx][0][:2] != end_of_section:
            self.nom_pol.append(copy_lines[line_idx].split("'")[1])
            self.desc_pol.append(copy_lines[line_idx].split("'")[3])
            list_i_pol.append(self.i_pol)
            self.npol += 1
            self.i_pol += 1

            prev_pol = copy_lines[line_idx].split("'")[5]
            nom_pol = copy_lines[line_idx].split("'")[1]
            if prev_pol == " ":
                self.prev_pol_type.append(0)
            elif prev_pol != nom_pol:
                self.prev_pol_type.append(2)
            elif prev_pol[:2] == nom_pol[:2]:
                self.prev_pol_type.append(1)

            for i in range(len(self.nom_pol)):
                if prev_pol == " ":
                    self.i_prev_pol.append(0)
                elif prev_pol == self.nom_pol[i]:
                    self.i_prev_pol.append(i + 1)

            line_idx += 1

        for j in range(len(self.nom_pol)):
            if self.pol == self.nom_pol[j]:
                self.i_pol = list_i_pol[j]
                break

    def set_results(self, lcal: Lcal) -> None:
        """
        Extract output parameters from Lcal object.

        Parameters
        ----------
        lcal : Lcal
            LCAL calibration object with results.
        """
        n_zones = lcal.param.n_zones
        self.pro[:, :n_zones] = lcal.X
        self.cospro[:, :n_zones] = lcal.cospro()
        self.precio[:, :n_zones] = lcal.p
        self.coscon[:, :n_zones] = lcal.coscon()
        self.utcon[:, :n_zones] = lcal.p + lcal.h
        self.atrac[:, :n_zones] = lcal.A_ni
        self.ajuste[:, :n_zones] = lcal.h
        self.dem[:, :n_zones] = lcal.D

    def write(self, lcal: Lcal, output_path: str | Path | None = None) -> None:
        """
        Write L1S file with LCAL results.

        Parameters
        ----------
        lcal : Lcal
            LCAL calibration object with results.
        output_path : str or Path, optional
            Output file path. If not provided, uses config.L1S_filepath.
        """
        self.set_results(lcal)
        self._update_datetime()

        if output_path is None:
            output_path = self.config.L1S_filepath

        with open(output_path, "wb") as f:
            self._write_header(f)
            self._write_policy_info(f)
            self._write_sector_info(f)
            self._write_demand_functions(f)
            self._write_zone_info(f)
            self._write_data(f)

    def _write_header(self, f: BinaryIO) -> None:
        """Write file header."""
        header = struct.pack(
            "<3hi",
            self.file_major,
            self.file_minor,
            self.file_release,
            self.ifmt_l1s,
        )
        f.write(header)

        iterations = struct.pack("<i", self.nb_iterations)
        f.write(iterations)

        time_data = struct.pack(
            "<5h", self.day, self.month, self.year, self.hour, self.minute
        )
        f.write(time_data)

        area_estudio = struct.pack(
            "<3s80s3s80s",
            _pad_string(self.area, 3).encode(),
            _pad_string(self.estudio, 80).encode(),
            self.pol.encode(),
            b"" + b" " * 80,
        )
        f.write(area_estudio)

    def _write_policy_info(self, f: BinaryIO) -> None:
        """Write policy information."""
        npol = struct.pack("<2i", self.npol, -self.npol)
        f.write(npol)

        for i in range(self.npol):
            info_policy = struct.pack(
                "<2ib5s32s",
                i + 1,
                self.i_prev_pol[i],
                self.prev_pol_type[i],
                _pad_string(self.nom_pol[i], 5).encode(),
                _pad_string(self.desc_pol[i], 32).encode(),
            )
            f.write(info_policy)

        npol_ipol = struct.pack("<2i", self.npol, self.i_pol)
        f.write(npol_ipol)

    def _write_sector_info(self, f: BinaryIO) -> None:
        """Write sector information."""
        sector_1 = struct.pack("<3i", self.ns, -self.ns, self.nflu)
        f.write(sector_1)

        for i in range(self.ns):
            sector_2 = struct.pack(
                "<2i32s?5f2i",
                i + 1,
                int(self.num_sec[i]),
                _pad_string(str(self.nom_sec[i]), 32).encode(),
                self.lflu[i],
                self.beta_1[i],
                self.beta_2[i],
                self.gama_1[i],
                self.gama_2[i],
                self.min_price_cost_ratio[i],
                int(self.sector_type[i]),
                int(self.target_sector[i]),
            )
            f.write(sector_2)

        ns = struct.pack("<i", self.ns)
        f.write(ns)

    def _write_demand_functions(self, f: BinaryIO) -> None:
        """Write demand function data."""
        num_sect = struct.pack("<2i", self.ns, -self.ns)
        f.write(num_sect)

        for i in range(self.ns):
            index = struct.pack("<i", i + 1)
            f.write(index)

            for j in range(self.ns):
                demand_functions = struct.pack(
                    "<i12fi",
                    j + 1,
                    self.demin[j, i],
                    self.demax[j, i],
                    self.delas[j, i],
                    self.selas[j, i],
                    self.suslgsc[j, i],
                    self.xalfa_1[i, j],
                    self.xalfa_2[i, j],
                    self.xalfapro[i, j],
                    self.xalfapre[i, j],
                    self.xalfacap[i, j],
                    self.alfa_1[i, j],
                    self.alfa_2[i, j],
                    self.mxsust,
                )
                f.write(demand_functions)

                for k in range(self.mxsust):
                    nsust = struct.pack("<i", int(self.nsust[i, j, k]))
                    f.write(nsust)

        ns = struct.pack("<i", self.ns)
        f.write(ns)

    def _write_zone_info(self, f: BinaryIO) -> None:
        """Write zone information."""
        param_nzn = struct.pack(
            "<4i",
            self.nb_tot_zones,
            -self.nb_tot_zones,
            self.nz1,
            self.nz2,
        )
        f.write(param_nzn)

        for i in range(self.nb_tot_zones):
            param_zon = struct.pack(
                "<2i32s2i",
                i + 1,
                self.num_zon[i],
                _pad_string(str(self.nom_zon[i]), 32).encode(),
                self.jer1[i],
                self.jer2[i],
            )
            f.write(param_zon)

        nzn = struct.pack("<i", self.nb_tot_zones)
        f.write(nzn)

    def _write_data(self, f: BinaryIO) -> None:
        """Write main L1S data."""
        nzn_ns = struct.pack("<2i", self.nb_tot_zones, self.ns)
        f.write(nzn_ns)

        for i in range(self.nb_tot_zones):
            i_numzon = struct.pack("<2i", i + 1, self.num_zon[i])
            f.write(i_numzon)

            for n in range(self.ns):
                param_l1s = struct.pack(
                    "<idf2dfd3fd4fd3f",
                    n + 1,
                    self.xpro[n, i],
                    self.probase[n, i],
                    self.pro[n, i],
                    self.cospro[n, i],
                    self.prebase[n, i],
                    self.precio[n, i],
                    self.xdem[n, i],
                    self.dem[n, i],
                    self.coscon[n, i],
                    self.utcon[n, i],
                    self.rmin[n, i],
                    self.rmax[n, i],
                    self.atrac[n, i],
                    self.valag[n, i],
                    self.ajuste[n, i],
                    self.atrain[n, i],
                    self.stock[n, i],
                    self.unstock[n, i],
                )
                f.write(param_l1s)


# Backward compatibility aliases
class L1s:
    """Legacy L1S reader class for backward compatibility."""

    def __init__(self, file_name: str, n_sectors: int, n_zones: int) -> None:
        self.file_name = file_name
        self.n_sectors = n_sectors
        self.n_zones = n_zones
        self._data: L1sData | None = None

    def read(self) -> list[NDArray[np.float64]]:
        """Read L1S file and return arrays."""
        self._data = read_l1s(self.file_name, self.n_sectors, self.n_zones)
        return [
            self._data.production,
            self._data.cost_production,
            self._data.price,
            self._data.cost_consumption,
            self._data.utility_consumption,
            self._data.attractor,
            self._data.adjustment,
            self._data.demand,
        ]


class L1sParam:
    """Legacy L1S parameter class for backward compatibility."""

    def __init__(self, config: TranusConfig) -> None:
        self._writer = L1sWriter(config)

    @property
    def nbSectors(self) -> int:
        return self._writer.nb_sectors

    @property
    def nbTotZones(self) -> int:
        return self._writer.nb_tot_zones

    def run_parameters_extraction(self) -> None:
        """Extract parameters from files."""
        self._writer.extract_from_files()

    def write_gral1s(self, lcal: Lcal) -> None:
        """Write L1S file."""
        self._writer.write(lcal)
