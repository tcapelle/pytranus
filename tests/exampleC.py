"""Example C test - LCAL calibration example."""

from __future__ import annotations

import logging
from pathlib import Path
from sys import stdout

import torch

from context import pytranus  # noqa: F401
from pytranus import Lcal, TranusConfig
from pytranus.io import L1s, L1sParam

log_level = logging.DEBUG
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=log_level,
    stream=stdout,
)
torch.set_printoptions(precision=5, linewidth=210)

# Binary path - update this to your Tranus installation
BIN_PATH = "/Users/thomascapelle/Dropbox/TRAN_fortranfiles/OSX/"


def replace_L1S(t: TranusConfig) -> None:
    """Replace L1S file with newly generated one."""
    path_scn = Path(t.working_directory) / t.scenario_id
    old_file = path_scn / f"{t.project_id}{t.scenario_id}.L1S"
    new_file = path_scn / f"NEW_LCAL_{t.scenario_id}.L1S"
    if old_file.exists():
        old_file.unlink()
    if new_file.exists():
        new_file.rename(old_file)


def read_L1S(
    path_L1S: str, l1s_param: L1sParam
) -> list:
    """Read L1S file and return arrays."""
    out_L1S = L1s(path_L1S, l1s_param.nbSectors, l1s_param.nbTotZones).read()
    return [var[:, :227] for var in out_L1S]


def run_example_c() -> tuple:
    """Run ExampleC calibration."""
    scn = "03A"
    path = str(Path(__file__).parent / "ExampleC")

    t = TranusConfig(
        tranus_bin_path=BIN_PATH,
        working_directory=path,
        project_id="EXC",
        scenario_id=scn,
    )

    lcal = Lcal(t, normalize=False)

    n_sectors = lcal.param.n_sectors
    n_zones = lcal.param.n_zones

    # Run calibration
    p, h, conv = lcal.compute_shadow_prices()

    # Get goodness of fit statistics
    stats = lcal.goodness_of_fit()
    print(f"Housing R²: {stats['housing']['r_squared']:.4f}")

    l1s_param = L1sParam(t)
    path_L1S = str(Path(path) / scn / f"{t.project_id}{t.scenario_id}.L1S")

    # Convert to numpy for L1S writing
    results = lcal.to_numpy()
    p_np = results["p"]
    h_np = results["h"]

    _ = read_L1S(path_L1S, l1s_param)

    l1s_param.run_parameters_extraction()
    # Note: write_gral1s expects an object with p attribute as numpy array
    # You may need to update this function to work with the new implementation

    print("ExampleC completed successfully!")
    return lcal, p, h


if __name__ == "__main__":
    run_example_c()
