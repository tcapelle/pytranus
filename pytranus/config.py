"""
Tranus configuration and binary interface.

This module provides configuration management and binary execution for Tranus.
"""

from __future__ import annotations

import glob
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranusConfig:
    """
    Configuration for a Tranus project.

    Parameters
    ----------
    tranus_bin_path : str | Path
        Path to Tranus binary executables.
    working_directory : str | Path
        Path to the working directory containing Tranus model files.
    project_id : str
        Project identifier (e.g., 'GRL').
    scenario_id : str
        Scenario identifier (e.g., '00A').

    Examples
    --------
    >>> config = TranusConfig(
    ...     tranus_bin_path="/usr/local/tranus/bin",
    ...     working_directory="/projects/my_model",
    ...     project_id="GRL",
    ...     scenario_id="00A"
    ... )
    """

    tranus_bin_path: str | Path
    working_directory: str | Path
    project_id: str
    scenario_id: str

    # Derived paths (computed in __post_init__)
    param_file: str = field(init=False)
    obs_file: str = field(init=False)
    zone_file: str = field(init=False)
    conv_factor: str = field(default="0.0001", init=False)
    n_iterations: str = field(default="250", init=False)

    # Full file paths
    CTL_filepath: str = field(init=False)
    L0E_filepath: str = field(init=False)
    Z1E_filepath: str = field(init=False)
    L1E_filepath: str = field(init=False)
    L1S_filepath: str = field(init=False)
    ScenarioPath: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths after dataclass init."""
        self.tranus_bin_path = Path(self.tranus_bin_path)
        self.working_directory = Path(self.working_directory)

        # Check if binaries exist
        lcal_path = self.tranus_bin_path / "lcal"
        if not lcal_path.exists():
            logger.error(f"Tranus binaries not found in: {self.tranus_bin_path}")

        # File names
        self.param_file = f"W_{self.project_id}{self.scenario_id}.L1E"
        self.obs_file = f"W_{self.project_id}.L0E"
        self.zone_file = f"W_{self.project_id}.Z1E"

        # Full paths
        self.CTL_filepath = str(self.working_directory / "W_TRANUS.CTL")

        # Use glob to find files
        l0e_files = glob.glob(str(self.working_directory / "W_*.L0E"))
        self.L0E_filepath = l0e_files[0] if l0e_files else ""

        z1e_files = glob.glob(str(self.working_directory / "W_*.Z1E"))
        self.Z1E_filepath = z1e_files[0] if z1e_files else ""

        self.ScenarioPath = str(self.working_directory / self.scenario_id)

        l1e_files = glob.glob(str(Path(self.ScenarioPath) / "W_*.L1E"))
        self.L1E_filepath = l1e_files[0] if l1e_files else ""

        self.L1S_filepath = str(
            Path(self.ScenarioPath) / f"pyLcal_{self.scenario_id}.L1S"
        )

    def get_zones(self) -> tuple[list[int], list[int]]:
        """
        Get zone numbering from Z1E file.

        Returns
        -------
        internal_zones : list[int]
            List of internal zone numbers.
        external_zones : list[int]
            List of external zone numbers.
        """
        zone_file_path = self.working_directory / self.zone_file
        logger.debug(f"Reading zones from file: {zone_file_path}")

        with open(zone_file_path, "r") as f:
            lines = f.readlines()

        internal_zones: list[int] = []
        external_zones: list[int] = []
        n = 5

        while n < len(lines) - 1:
            # Read internal zones
            while lines[n][1:3] != "--":
                internal_zones.append(int(lines[n].split()[0]))
                n += 1
            n += 6  # Skip second level zones

            # Read external zones
            while lines[n][1:3] != "--":
                external_zones.append(int(lines[n].split()[0]))
                n += 1

        return internal_zones, external_zones

    def get_sectors(self) -> list[int]:
        """
        Get sector numbering from L1E file.

        Returns
        -------
        sectors : list[int]
            List of sector numbers.
        """
        param_file_path = self.working_directory / self.scenario_id / self.param_file
        logger.debug(f"Reading sectors from file: {param_file_path}")

        with open(param_file_path, "r") as f:
            lines = f.readlines()

        sectors: list[int] = []
        n = 11

        while n < len(lines):
            while lines[n][1:3] != "--":
                sectors.append(int(lines[n].split()[0]))
                n += 1
            n = len(lines)  # Exit after first block

        return sectors

    # Keep old method names for backward compatibility
    def numberingZones(self) -> tuple[list[int], list[int]]:
        """Alias for get_zones() for backward compatibility."""
        return self.get_zones()

    def numberingSectors(self) -> list[int]:
        """Alias for get_sectors() for backward compatibility."""
        return self.get_sectors()


class BinaryInterface:
    """
    Interface to run Tranus binary executables.

    Parameters
    ----------
    config : TranusConfig
        Configuration object for the Tranus project.

    Examples
    --------
    >>> config = TranusConfig(...)
    >>> interface = BinaryInterface(config)
    >>> interface.run_lcal()
    """

    def __init__(self, config: TranusConfig) -> None:
        """Initialize the binary interface."""
        self.config = config
        self.tranus_has_been_run = False

        # Determine binary extension based on OS
        operating_system = platform.system()
        self._extension = ".exe" if operating_system.startswith("Windows") else ""

    def _get_binary_path(self, binary_name: str) -> Path:
        """Get the full path to a Tranus binary."""
        return Path(self.config.tranus_bin_path) / f"{binary_name}{self._extension}"

    def _check_binary_exists(self, binary_name: str) -> bool:
        """Check if a binary exists and log error if not."""
        binary_path = self._get_binary_path(binary_name)
        if not binary_path.is_file():
            logger.error(
                f"The <{binary_name}> program was not found in {self.config.tranus_bin_path}"
            )
            return False
        return True

    def _run_binary(
        self,
        binary_name: str,
        args: list[str],
        out_file: str,
        err_file: str,
    ) -> bool:
        """Run a Tranus binary with given arguments."""
        if not self._check_binary_exists(binary_name):
            return False

        program = str(self._get_binary_path(binary_name))
        working_dir = Path(self.config.working_directory)
        out_path = working_dir / out_file
        err_path = working_dir / err_file

        full_args = [program] + args

        with open(out_path, "w") as stdout_file, open(err_path, "w") as stderr_file:
            subprocess.Popen(
                full_args,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=str(working_dir),
            ).communicate()

        return True

    def run_lcal(self, freeze: bool = False, additional: bool = False) -> bool:
        """
        Run the LCAL module.

        Parameters
        ----------
        freeze : bool, default False
            If True, only land-use sectors are computed.
        additional : bool, default False
            If True, run with additional flag.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        args = [self.config.scenario_id]
        if freeze:
            args.append("-f")
        if additional:
            args.append("A")

        return self._run_binary("lcal", args, "outlcal.txt", "outlcalerr.txt")

    def run_trans(self, loop_n: int) -> bool:
        """
        Run the Trans (transport) module.

        Parameters
        ----------
        loop_n : int
            Loop number. If 0, runs with initialization flag.
        """
        args = [self.config.scenario_id]
        if loop_n == 0:
            args.extend(["-I", " "])

        return self._run_binary("trans", args, "outtrans.txt", "outtranserr.txt")

    def run_cost(self) -> bool:
        """Run the Cost module to convert transport matrices/flows to costs."""
        return self._run_binary(
            "cost", [self.config.scenario_id], "outcost.txt", "outcosterr.txt"
        )

    def run_fluj(self) -> bool:
        """Run the Fluj module to transform LCAL matrices into flows."""
        return self._run_binary(
            "fluj", [self.config.scenario_id], "outfluj.txt", "outflujerr.txt"
        )

    def run_mats(self) -> bool:
        """
        Run MATS to generate transport cost and disutility matrices.

        Generates COST_T.MTX and DISU_T.MTX files.
        """
        if not self._check_binary_exists("mats"):
            return False

        program = str(self._get_binary_path("mats"))
        working_dir = Path(self.config.working_directory)

        logger.debug("Creating Disutility matrix")

        # Generate disutility matrix
        args_disu = [
            program, self.config.scenario_id, "-S", "[k]", "-o", "DISU_T.MTX", " "
        ]
        with open(working_dir / "outmats.txt", "w") as out:
            subprocess.Popen(args_disu, stdout=out, stderr=out, cwd=str(working_dir))

        # Generate cost matrix
        args_cost = [
            program, self.config.scenario_id, "-K", "[k]", "-o", "COST_T.MTX", " "
        ]
        with open(working_dir / "outmats.txt", "w") as out:
            subprocess.Popen(args_cost, stdout=out, stderr=out, cwd=str(working_dir))

        return True

    def create_imploc(self) -> bool:
        """Create the Imploc report from LCAL."""
        return self._run_binary(
            "imploc",
            [self.config.scenario_id, "-J", "-o", "imploc_out.txt", " "],
            "outimploc.txt",
            "outimplocerr.txt",
        )

    def run_tranus(self, loop_n: int) -> bool:
        """
        Run complete Tranus simulation loop.

        Runs modules in sequence: Trans -> Cost -> Lcal -> Fluj -> Imploc

        Parameters
        ----------
        loop_n : int
            Loop number.

        Returns
        -------
        bool
            True if all modules completed successfully.
        """
        start = time.time()

        print("Running Trans...")
        self.run_trans(loop_n)
        print("Running Cost...")
        self.run_cost()
        print("Running Lcal...")
        self.run_lcal()
        print("Running Fluj...")
        self.run_fluj()
        print("Creating Imploc...")
        self.create_imploc()

        elapsed = time.time() - start
        print(f"Elapsed time: {elapsed:.2f}s")

        self.tranus_has_been_run = True
        return True
