"""ExampleC integration tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch


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


class TestTorchUtilsJacobian(unittest.TestCase):
    """Tests for Jacobian computations in torch_utils."""

    def test_jacobian_production_logit(self) -> None:
        """Test Jacobian computation with logit model."""
        from pytranus.torch_utils import jacobian_production_logit

        n_zones = 5
        torch.manual_seed(42)
        D_n = torch.rand(n_zones) * 100
        Pr_n = torch.rand(n_zones, n_zones)
        Pr_n = Pr_n / Pr_n.sum(dim=1, keepdim=True)  # Normalize rows

        result = jacobian_production_logit(D_n, Pr_n, beta=0.5, lamda=1.0)

        self.assertEqual(result.shape, (n_zones, n_zones))
        # Diagonal should be negative (own-price effect)
        for i in range(n_zones):
            self.assertLessEqual(result[i, i].item(), 0)

    def test_jacobian_vs_autograd(self) -> None:
        """Test that analytical Jacobian matches autograd."""
        from pytranus.torch_utils import compute_production, jacobian_production_logit

        n_zones = 4
        beta = 0.5
        lamda = 1.0

        torch.manual_seed(42)
        D_n = torch.rand(n_zones) * 100
        phi = torch.rand(n_zones, requires_grad=True)

        # Compute probabilities from phi
        log_weights = -beta * lamda * phi
        Pr_n = torch.softmax(log_weights.unsqueeze(0).expand(n_zones, -1), dim=-1)

        # Compute production
        X = compute_production(D_n, Pr_n)

        # Autograd Jacobian
        jac_autograd = torch.zeros(n_zones, n_zones)
        for j in range(n_zones):
            if phi.grad is not None:
                phi.grad.zero_()
            X[j].backward(retain_graph=True)
            jac_autograd[j, :] = phi.grad.clone()

        # Analytical Jacobian
        jac_analytical = jacobian_production_logit(D_n, Pr_n.detach(), beta, lamda)

        np.testing.assert_array_almost_equal(
            jac_autograd.detach().numpy(), jac_analytical.numpy(), decimal=5
        )


if __name__ == "__main__":
    unittest.main()
