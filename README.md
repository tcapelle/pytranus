# PyTranus - Python LCAL Module

A Python implementation of the Tranus Land Use Calibration (LCAL) module.

## Overview

This project implements the [Tranus](http://www.tranus.com/tranus-english/download-install) Land Use Calibration module in Python. The implementation reformulates calibration as an optimization problem, enabling semi-automatic parameter calibration by minimizing a cost function.

### References

- [Scientific article](https://www.sciencedirect.com/science/article/pii/S0198971517302181?via%3Dihub) on the Python Tranus implementation
- [Mathematical description](http://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0cmFudXNtb2RlbHxneDo3YWQzYTk0OTkxN2RlN2Rj) of the Tranus software

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
- scipy >= 1.11.0

## Quick Start

```python
from pytranus import TranusConfig, Lcal

# Configure the project
config = TranusConfig(
    tranus_bin_path="/path/to/tranus/bin",
    working_directory="/path/to/project",
    project_id="EXC",
    scenario_id="03A",
)

# Run calibration
lcal = Lcal(config)
p, h, conv, lamda = lcal.compute_shadow_prices(ph0)
```

## Package Structure

```
pytranus/
├── __init__.py      # Public API
├── config.py        # TranusConfig, BinaryInterface
├── params.py        # LcalParams dataclass, load_params()
├── io.py            # L1S file I/O operations
├── lcal.py          # Main Lcal calibration class
└── _math.py         # NumPy derivative computations
```

## API Reference

### Core Classes

- `TranusConfig` - Project configuration (paths, IDs)
- `Lcal` - Main calibration class
- `BinaryInterface` - Interface to Tranus binaries

### Parameters

- `LcalParams` - Dataclass containing all calibration parameters
- `load_params(config)` - Load parameters from Tranus input files

### I/O

- `read_l1s(path, n_sectors, n_zones)` - Read L1S binary file
- `L1sWriter` - Write L1S files with calibration results
- `L1sData` - Dataclass for L1S file contents

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
