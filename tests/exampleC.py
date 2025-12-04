"""Example C test - LCAL calibration example."""

from __future__ import annotations

import logging
from pathlib import Path
from sys import stdout

import torch

from context import pytranus  # noqa: F401
from pytranus import LCALModel, TranusConfig

log_level = logging.INFO
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=log_level,
    stream=stdout,
)
torch.set_printoptions(precision=5, linewidth=210)


def run_example_c() -> tuple:
    """Run ExampleC calibration with PyTorch."""
    scn = "03A"
    path = str(Path(__file__).parent / "ExampleC")

    config = TranusConfig(
        tranus_bin_path="/dummy/path",  # Not needed for calibration
        working_directory=path,
        project_id="EXC",
        scenario_id=scn,
    )

    print(f"Loading model from {path}...")
    model = LCALModel.from_config(config, normalize=True)

    print(f"Model: {model.n_sectors} sectors, {model.n_zones} zones")
    print(f"Housing sectors: {model.housing_sectors.tolist()}")

    # Standard PyTorch training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    print("\nTraining...")
    for step in range(5000):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

        if loss.item() < 1e-6:
            print(f"  Converged at step {step}")
            break

    # Get results
    stats = model.goodness_of_fit()
    print(f"\nResults:")
    print(f"  Housing R²: {stats['housing']['r_squared']:.4f}")
    print(f"  Housing MSE: {stats['housing']['mse']:.6f}")

    # Print per-sector stats
    for key, val in stats.items():
        if key.startswith("sector_"):
            print(f"  {key} R²: {val['r_squared']:.4f}")

    h = model.h.detach()
    print(f"\nShadow prices shape: {h.shape}")
    print("ExampleC completed successfully!")

    return model, h, stats


if __name__ == "__main__":
    run_example_c()
