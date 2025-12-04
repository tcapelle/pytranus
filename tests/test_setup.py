"""Basic setup tests."""

from __future__ import annotations

import unittest


class TestSetup(unittest.TestCase):
    """Basic tests to verify test infrastructure works."""

    def test_import_pytranus(self) -> None:
        """Test that pytranus can be imported."""
        import pytranus
        self.assertIsNotNone(pytranus.__version__)

    def test_import_lcal(self) -> None:
        """Test that Lcal class can be imported."""
        from pytranus import Lcal
        self.assertIsNotNone(Lcal)

    def test_import_config(self) -> None:
        """Test that TranusConfig can be imported."""
        from pytranus import TranusConfig
        self.assertIsNotNone(TranusConfig)

    def test_import_derivatives(self) -> None:
        """Test that derivative functions can be imported."""
        from pytranus._math import compute_DX_n
        self.assertIsNotNone(compute_DX_n)

    def test_import_params(self) -> None:
        """Test that LcalParams can be imported."""
        from pytranus import LcalParams
        self.assertIsNotNone(LcalParams)

    def test_import_io(self) -> None:
        """Test that I/O classes can be imported."""
        from pytranus import L1sData, L1sWriter, read_l1s
        self.assertIsNotNone(L1sData)
        self.assertIsNotNone(L1sWriter)
        self.assertIsNotNone(read_l1s)


if __name__ == "__main__":
    unittest.main()
