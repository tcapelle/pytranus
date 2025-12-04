"""Test context - adds parent directory to path."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytranus  # noqa: E402, F401
