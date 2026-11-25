#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Wrapper for Network-Constrained High-Accident Risk Zone Analysis.
Delegates CLI parsing and execution to `risk_zones.cli`.
"""
from pathlib import Path
import sys

# Ensure src/ is on pythonpath when executed directly from repo root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from risk_zones.cli import main, parse_args, run_pipeline  # noqa: E402


if __name__ == "__main__":
    main()
