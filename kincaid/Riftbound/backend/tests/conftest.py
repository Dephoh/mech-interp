"""Pytest configuration: ensure test helpers and app are importable."""

import sys
import os

# Add tests/ dir so `from helpers import ...` works
sys.path.insert(0, os.path.dirname(__file__))
# Add backend/ dir so `from app.engine...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
