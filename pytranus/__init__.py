"""
PyTranus - Python Tranus Land Use Calibration Module.

This package provides a PyTorch-based implementation of the Tranus LCAL module,
reformulating calibration as end-to-end differentiable optimization.

Features
--------
- Automatic differentiation (no manual Jacobians)
- GPU acceleration via CUDA or MPS
- Standard PyTorch training loop
- nn.Module architecture for extensibility

Quick Start
-----------
>>> from pytranus import LCALModel, TranusConfig
>>> config = TranusConfig(bin_path, working_dir, project_id, scenario_id)
>>> model = LCALModel.from_config(config, device='cuda')
>>>
>>> # Standard PyTorch training
>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
>>> for step in range(1000):
...     optimizer.zero_grad()
...     loss = model()
...     loss.backward()
...     optimizer.step()
>>>
>>> h = model.h  # shadow prices (nn.Parameter)
>>> stats = model.goodness_of_fit()

Convenience Function
-------------------
>>> from pytranus import calibrate
>>> model, stats = calibrate(config, device='cuda')
"""

from pytranus.__version__ import __version__
from pytranus.config import BinaryInterface, TranusConfig
from pytranus.io import L1sData, L1sWriter, read_l1s
from pytranus.lcal_torch import Lcal, LcalTorch, calibrate, train
from pytranus.modules import LCALModel
from pytranus.params import LcalParams, load_params

__all__ = [
    # Core classes
    "LCALModel",
    "Lcal",
    "LcalTorch",
    "calibrate",
    "train",
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
