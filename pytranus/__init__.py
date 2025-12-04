"""
PyTranus - Python Tranus Land Use Calibration Module.

This package provides a Python implementation of the Tranus LCAL module,
reformulating calibration as an optimization problem for semi-automatic
parameter calibration.

Modules
-------
config : TranusConfig and BinaryInterface for project configuration
params : LcalParams dataclass for calibration parameters
io : L1S file I/O operations
lcal : Main Lcal calibration class

Examples
--------
>>> from pytranus import TranusConfig, Lcal
>>> config = TranusConfig(bin_path, working_dir, project_id, scenario_id)
>>> lcal = Lcal(config)
>>> p, h, conv, lamda = lcal.compute_shadow_prices(ph0)
"""

from pytranus.__version__ import __version__
from pytranus.config import BinaryInterface, TranusConfig
from pytranus.io import L1sData, L1sWriter, read_l1s
from pytranus.lcal import Lcal
from pytranus.params import LcalParams, load_params

__all__ = [
    # Core classes
    "Lcal",
    "TranusConfig",
    "BinaryInterface",
    # Parameters
    "LcalParams",
    "load_params",
    # I/O
    "L1sData",
    "L1sWriter",
    "read_l1s",
    # Version
    "__version__",
]
