# PyTranus - Python LCAL Module

A PyTorch-based implementation of the Tranus Land Use Calibration (LCAL) module.

## Overview

This project implements the [Tranus](http://www.tranus.com/tranus-english/download-install) Land Use Calibration module in Python using PyTorch. Shadow prices are `nn.Parameter`s optimized via standard PyTorch training loops.

### Features

- **Standard PyTorch API**: `h` is an `nn.Parameter`, use any optimizer
- **Automatic differentiation**: No manual Jacobian derivation
- **GPU acceleration**: CUDA and Apple Silicon (MPS) support
- **Composable**: Mix with neural networks, add regularization, etc.

### References

- [Scientific article](https://www.sciencedirect.com/science/article/pii/S0198971517302181?via%3Dihub) on the Python Tranus implementation [free version](https://inria.hal.science/hal-01396793/file/capelle_author_version_for_HAL.pdf)
- [Mathematical description](https://bitbucket-archive.softwareheritage.org/new-static/1e/1ed1b959-6f37-43ee-991b-631c36887223/attachments/GeneralDescriptionTranus.pdf) of the Tranus software

## Installation

Requires Python 3.12+.

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Dependencies

- numpy >= 1.26.0
- pandas >= 2.0.0
- torch >= 2.0.0

## Quick Start

### Standard PyTorch Training Loop

```python
import torch
from pytranus import LCALModel, TranusConfig

# Configure the project
config = TranusConfig(
    tranus_bin_path="/path/to/tranus/bin",
    working_directory="/path/to/project",
    project_id="EXC",
    scenario_id="03A",
)

# Create model - h is an nn.Parameter!
model = LCALModel.from_config(config, device='cuda')

# Standard PyTorch training
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for step in range(1000):
    optimizer.zero_grad()
    loss = model()  # forward() returns MSE loss
    loss.backward()
    optimizer.step()

    if loss.item() < 1e-6:
        break

# Access results
h = model.h  # shadow prices (nn.Parameter)
stats = model.goodness_of_fit()
print(f"Housing R²: {stats['housing']['r_squared']:.4f}")
```

### Convenience Function

```python
from pytranus import calibrate

model, stats = calibrate(config, device='cuda', max_steps=1000, lr=0.1)
```

## Package Structure

```
pytranus/
├── __init__.py      # Public API
├── config.py        # TranusConfig, BinaryInterface
├── params.py        # LcalParams, load_params()
├── io.py            # L1S file I/O
├── lcal_torch.py    # Training utilities
├── modules.py       # nn.Module implementations
└── torch_utils.py   # Tensor utilities
```

## API Reference

### Core Classes

- `LCALModel` - Main calibration model (nn.Module)
  - `h`: Shadow prices (`nn.Parameter`)
  - `forward()`: Returns MSE loss
  - `production()`: Returns predicted production
  - `goodness_of_fit()`: Returns R², MSE, MAE statistics

- `TranusConfig` - Project configuration
- `HousingModel` - Housing sector submodel
- `TransportableModel` - Transportable sector submodel

### Functions

- `calibrate(config, device, ...)` - Convenience function
- `train(model, max_steps, lr, ...)` - Training loop

### Parameters

- `LcalParams` - Dataclass containing all calibration parameters
- `load_params(config)` - Load parameters from Tranus input files

## Running Tests

```bash
pytest tests/ -v
```

## Authors

- **Thomas Capelle** - [GitHub](https://github.com/tcapelle)

## License

MIT License

## Acknowledgments

Thanks to Fausto Lo Feudo, Brian Morton, Peter Sturm, Arthur Vidard, and Tomas de la Barra.
