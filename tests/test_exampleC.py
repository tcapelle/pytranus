"""ExampleC integration tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


class TestExampleC(unittest.TestCase):
    """Tests for ExampleC scenario."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if ExampleC data exists."""
        cls.example_path = Path(__file__).parent / "ExampleC"
        cls.has_data = cls.example_path.exists()

    def test_example_data_exists(self) -> None:
        """Test that example data directory exists."""
        if not self.has_data:
            self.skipTest("ExampleC data not available")
        self.assertTrue(self.example_path.exists())

    def test_scenario_exists(self) -> None:
        """Test that 03A scenario exists."""
        if not self.has_data:
            self.skipTest("ExampleC data not available")
        scenario_path = self.example_path / "03A"
        self.assertTrue(scenario_path.exists())


class TestDerivatives(unittest.TestCase):
    """Tests for derivative computations."""

    def test_compute_DX_n_logit(self) -> None:
        """Test DX_n computation with logit model."""
        from pytranus._math import compute_DX_n

        n_zones = 5
        DX = np.zeros((n_zones, n_zones))
        D_n = np.random.random(n_zones) * 100
        Pr_n = np.random.random((n_zones, n_zones))
        Pr_n = Pr_n / Pr_n.sum(axis=1, keepdims=True)  # Normalize rows
        U_n = np.random.random((n_zones, n_zones)) + 1.0

        result = compute_DX_n(
            DX, n_sectors=10, n_zones=n_zones,
            beta=0.5, lamda=1.0, D_n=D_n, Pr_n=Pr_n, U_n=U_n, logit=True
        )

        self.assertEqual(result.shape, (n_zones, n_zones))
        # Diagonal should be negative (own-price effect)
        self.assertTrue(all(result[i, i] <= 0 for i in range(n_zones)))

    def test_compute_DX_n_vectorized(self) -> None:
        """Test vectorized DX_n computation."""
        from pytranus._math import compute_DX_n_vectorized

        n_zones = 5
        D_n = np.random.random(n_zones) * 100
        Pr_n = np.random.random((n_zones, n_zones))
        Pr_n = Pr_n / Pr_n.sum(axis=1, keepdims=True)

        result = compute_DX_n_vectorized(
            n_zones=n_zones, beta=0.5, lamda=1.0, D_n=D_n, Pr_n=Pr_n, logit=True
        )

        self.assertEqual(result.shape, (n_zones, n_zones))


if __name__ == "__main__":
    unittest.main()
