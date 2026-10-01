#!/usr/bin/env python3
"""Left-arm-only launcher for the existing Tool-Y force-position stream test.

Read-only unless --execute is passed. The underlying test also requires an
interactive confirmation. Keep Mission stopped while running this diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed-mm-s", type=float, default=3.0)
    parser.add_argument("--target-force-n", type=float, default=-1.0)
    args = parser.parse_args()
    if args.target_force_n >= 0:
        parser.error("left-arm Tool-Y target must be negative")
    if not 0.1 <= args.speed_mm_s <= 3.0:
        parser.error("test speed must be between 0.1 and 3.0 mm/s")
    shared = Path(__file__).with_name("left_tool_y_force_control_test.py")
    if not shared.is_file():
        parser.error(f"shared diagnostic not found: {shared}")
    argv = [
        sys.executable, str(shared), "--arm-side", "left",
        "--target-force-n", str(args.target_force_n),
        "--speed-mm-s", str(args.speed_mm_s),
    ]
    if args.execute:
        argv.append("--execute")
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
