"""Tests for PyTorch implementation of LCAL."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

import context  # noqa: F401
from pytranus import GeneralLCALModel, LCALModel, Lcal, TranusConfig

# Path to ExampleC test data
EXAMPLE_PATH = Path(__file__).parent / "ExampleC"
HAS_EXAMPLE_DATA = EXAMPLE_PATH.exists()


class TestTorchUtils(unittest.TestCase):
    """Tests for torch_utils module."""

    def test_to_tensor(self) -> None:
        """Test numpy to tensor conversion."""
        from pytranus.torch_utils import to_tensor

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = to_tensor(arr)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertEqual(tensor.shape, (2, 2))
        np.testing.assert_array_almost_equal(arr, tensor.numpy())

    def test_logit_softmax(self) -> None:
        """Test softmax with attractors (logit model)."""
        from pytranus.torch_utils import logit_softmax

        utilities = torch.tensor([1.0, 2.0, 3.0])
        attractors = torch.tensor([1.0, 1.0, 1.0])
        beta = 1.0

        probs = logit_softmax(utilities, attractors, beta)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=6)
        # Higher utility = lower probability (negative in logit)
        self.assertGreater(probs[0].item(), probs[1].item())
        self.assertGreater(probs[1].item(), probs[2].item())

    def test_elastic_demand(self) -> None:
        """Test elastic demand function."""
        from pytranus.torch_utils import elastic_demand

        U = torch.tensor([0.0, 1.0, 2.0])
        demin = torch.tensor([0.1, 0.1, 0.1])
        demax = torch.tensor([1.0, 1.0, 1.0])
        delta = torch.tensor([1.0, 1.0, 1.0])

        a = elastic_demand(U, demin, demax, delta)

        self.assertAlmostEqual(a[0].item(), 1.0, places=6)
        self.assertGreater(a[0].item(), a[1].item())
        self.assertGreater(a[1].item(), a[2].item())

    def test_compute_production(self) -> None:
        """Test production computation."""
        from pytranus.torch_utils import compute_production

        demands = torch.tensor([100.0, 200.0, 150.0])
        probs = torch.eye(3)

        production = compute_production(demands, probs)

        np.testing.assert_array_almost_equal(
            production.numpy(), demands.numpy()
        )

    def test_safe_log(self) -> None:
        """Test safe_log handles zeros correctly."""
        from pytranus.torch_utils import safe_log

        x = torch.tensor([0.0, 1.0, 2.0])
        result = safe_log(x)

        # Should not produce -inf for zero input
        self.assertTrue(torch.isfinite(result).all())
        # log(1) should still be 0
        self.assertAlmostEqual(result[1].item(), 0.0, places=6)


class TestModules(unittest.TestCase):
    """Tests for nn.Module implementations."""

    def test_demand_function_module(self) -> None:
        """Test DemandFunction module."""
        from pytranus.modules import DemandFunction

        n_sectors = 3
        n_zones = 4

        demin = torch.rand(n_sectors, n_sectors) * 0.1
        demax = demin + torch.rand(n_sectors, n_sectors) * 0.9
        delta = torch.rand(n_sectors, n_sectors)

        module = DemandFunction(demin, demax, delta)
        U_ni = torch.rand(n_sectors, n_zones)

        a = module(U_ni)

        self.assertEqual(a.shape, (n_sectors, n_sectors, n_zones))
        self.assertTrue((a >= demin.unsqueeze(-1) - 1e-6).all())
        self.assertTrue((a <= demax.unsqueeze(-1) + 1e-6).all())

    def test_housing_model_differentiable(self) -> None:
        """Test that HousingModel is differentiable."""
        from pytranus.modules import HousingModel

        n_housing = 2
        n_sectors = 5
        n_zones = 3

        model = HousingModel(
            housing_sectors=torch.tensor([0, 1]),
            demin=torch.rand(n_sectors, n_housing),
            demax=torch.rand(n_sectors, n_housing) + 0.5,
            delta=torch.rand(n_sectors, n_housing),
            sigma=torch.zeros(n_sectors),
            omega=torch.zeros(n_sectors, n_housing),
            Kn=torch.zeros(n_sectors, n_housing, dtype=torch.long),
            attractor=torch.ones(n_housing, n_zones),
            exog_demand=torch.rand(n_housing, n_zones) * 10,
            price=torch.rand(n_housing, n_zones),
            X_0=torch.rand(n_sectors, n_zones) * 100,
            X_target=torch.rand(n_housing, n_zones) * 100,
        )

        # h is now a parameter
        self.assertIsInstance(model.h, torch.nn.Parameter)

        # Test forward returns loss
        loss = model()
        self.assertEqual(loss.dim(), 0)  # scalar

        # Test gradient computation
        loss.backward()
        self.assertIsNotNone(model.h.grad)
        self.assertEqual(model.h.grad.shape, model.h.shape)


class TestLcalIntegration(unittest.TestCase):
    """Integration tests for Lcal (PyTorch implementation)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if ExampleC data exists."""
        cls.example_path = Path(__file__).parent / "ExampleC"
        cls.has_data = cls.example_path.exists()

    def test_housing_optimization_synthetic(self) -> None:
        """Test housing optimization with synthetic data."""
        from pytranus.modules import HousingModel

        n_housing = 2
        n_sectors = 3
        n_zones = 4

        demin = torch.ones(n_sectors, n_housing) * 0.5
        demax = demin.clone()
        delta = torch.zeros(n_sectors, n_housing)

        # Create target that matches prediction at h=0
        X_0 = torch.ones(n_sectors, n_zones) * 100
        expected_prod = 100 * n_sectors * 0.5  # 150 per zone
        X_target = torch.full((n_housing, n_zones), expected_prod)

        model = HousingModel(
            housing_sectors=torch.tensor([0, 1]),
            demin=demin,
            demax=demax,
            delta=delta,
            sigma=torch.zeros(n_sectors),
            omega=torch.zeros(n_sectors, n_housing),
            Kn=torch.zeros(n_sectors, n_housing, dtype=torch.long),
            attractor=torch.ones(n_housing, n_zones),
            exog_demand=torch.zeros(n_housing, n_zones),
            price=torch.zeros(n_housing, n_zones),
            X_0=X_0,
            X_target=X_target,
        )

        # At h=0, loss should be 0
        loss = model()
        self.assertAlmostEqual(loss.item(), 0.0, places=4)


@unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
class TestExampleCCalibration(unittest.TestCase):
    """Test calibration on ExampleC data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load LCAL instance."""
        cls.config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )
        cls.lcal = Lcal(cls.config, normalize=True)

    def test_parameters_loaded(self) -> None:
        """Test that parameters are loaded correctly."""
        param = self.lcal.param

        self.assertGreater(param.n_sectors, 0)
        self.assertGreater(param.n_zones, 0)
        self.assertIsNotNone(param.housing_sectors)

    def test_housing_production_forward(self) -> None:
        """Test housing production computation."""
        model = self.lcal.model

        # h is now a parameter
        self.assertIsInstance(model.h, torch.nn.Parameter)

        # Test production
        X = model.production()
        self.assertEqual(X.shape, (model.n_sectors, model.n_zones))
        self.assertTrue((X >= 0).all())

    def test_housing_optimization_fit(self) -> None:
        """Test that housing optimization achieves good fit."""
        lcal = Lcal(self.config, normalize=True)

        lcal.calc_sp_housing(max_steps=2000, lr=0.05, verbose=False)

        stats = lcal.goodness_of_fit()
        r2_housing = stats["housing"]["r_squared"]

        print(f"\n  Housing R²: {r2_housing:.4f}")
        # With Adam optimizer, may not reach as high R² as L-BFGS
        self.assertGreater(r2_housing, 0.5, f"Housing R² too low: {r2_housing:.4f}")

    def test_full_calibration(self) -> None:
        """Test full calibration pipeline."""
        lcal = Lcal(self.config, normalize=True)

        p, h, conv = lcal.compute_shadow_prices(max_steps=500, lr=0.1, verbose=False)

        n_sectors = lcal.n_sectors
        n_zones = lcal.n_zones

        self.assertEqual(p.shape, (n_sectors, n_zones))
        self.assertEqual(h.shape, (n_sectors, n_zones))

        stats = lcal.goodness_of_fit()
        self.assertIn("housing", stats)

    def test_location_probability(self) -> None:
        """Test location probability computation."""
        model = LCALModel.from_config(self.config)

        n_sectors = model.n_sectors
        n_zones = model.n_zones

        torch.manual_seed(42)
        ph = torch.randn(n_sectors, n_zones, dtype=torch.float64) * 0.1

        for n in model.genflux_sectors.tolist():
            if n in model.housing_sectors.tolist():
                continue

            Pr = model.location_prob.forward_single(ph[n, :], n)

            self.assertEqual(Pr.shape, (n_zones, n_zones))
            row_sums = Pr.sum(dim=1)
            np.testing.assert_array_almost_equal(
                row_sums.numpy(), np.ones(n_zones), decimal=6,
                err_msg=f"Location probabilities don't sum to 1 for sector {n}"
            )

    def test_demand_functions(self) -> None:
        """Test demand function computation."""
        model = self.lcal.model

        n_sectors = model.n_sectors
        n_zones = model.n_zones

        torch.manual_seed(42)
        U = torch.randn(n_sectors, n_zones, dtype=torch.float64) * 0.1 + 1.0

        a = model.demand_fn(U)

        self.assertEqual(a.shape, (n_sectors, n_sectors, n_zones))

    def test_to_numpy_conversion(self) -> None:
        """Test conversion of results to numpy."""
        lcal = Lcal(self.config, normalize=True)

        results = lcal.to_numpy()

        self.assertIn("h", results)
        self.assertIn("X", results)
        self.assertIn("X_target", results)

        self.assertIsInstance(results["h"], np.ndarray)


class TestLCALModelAPI(unittest.TestCase):
    """Test the new PyTorch-idiomatic API."""

    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_standard_training_loop(self) -> None:
        """Test standard PyTorch training loop works."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        model = LCALModel.from_config(config)

        # h should be a parameter
        self.assertIsInstance(model.h, torch.nn.Parameter)
        self.assertTrue(model.h.requires_grad)

        # Standard training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        initial_loss = model().item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()

        final_loss = model().item()

        # Loss should decrease
        self.assertLess(final_loss, initial_loss)


class TestGeneralLCALModel(unittest.TestCase):
    """Test the general LCAL model (Equations 3.3-3.4)."""

    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_general_model_training(self) -> None:
        """Test that GeneralLCALModel can be trained."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        model = GeneralLCALModel.from_config(config)

        # h should be a parameter for ALL sectors
        self.assertIsInstance(model.h, torch.nn.Parameter)
        self.assertEqual(model.h.shape, (model.n_sectors, model.n_zones))

        # Standard training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        initial_loss = model().item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()

        final_loss = model().item()

        # Loss should decrease
        self.assertLess(final_loss, initial_loss)

    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_general_vs_special_case(self) -> None:
        """Compare GeneralLCALModel vs LCALModel (special case)."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        general = GeneralLCALModel.from_config(config)
        special = LCALModel.from_config(config)

        # Both should have same structure
        self.assertEqual(general.n_sectors, special.n_sectors)
        self.assertEqual(general.n_zones, special.n_zones)

        # At h=0, production should be similar but not identical
        # (because general model uses full location choice for housing too)
        with torch.no_grad():
            X_general = general.production()
            X_special = special.production()

        # Both should be valid production arrays
        self.assertEqual(X_general.shape, X_special.shape)
        self.assertTrue((X_general >= 0).all())
        self.assertTrue((X_special >= 0).all())


class TestDeviceSupport(unittest.TestCase):
    """Test device selection (CPU/GPU)."""

    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_cpu_device(self) -> None:
        """Test explicit CPU device."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        lcal = Lcal(config, device="cpu")

        self.assertEqual(lcal.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_cuda_device(self) -> None:
        """Test CUDA device."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        lcal = Lcal(config, device="cuda")

        self.assertEqual(lcal.device.type, "cuda")

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    @unittest.skipUnless(HAS_EXAMPLE_DATA, "ExampleC data not available")
    def test_mps_device(self) -> None:
        """Test MPS device (Apple Silicon)."""
        config = TranusConfig(
            tranus_bin_path="/dummy/path",
            working_directory=str(EXAMPLE_PATH),
            project_id="EXC",
            scenario_id="03A",
        )

        lcal = Lcal(config, device="mps")

        self.assertEqual(lcal.device.type, "mps")


if __name__ == "__main__":
    unittest.main()
