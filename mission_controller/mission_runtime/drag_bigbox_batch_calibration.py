"""Compatibility entry point for one-Action-per-layer DragBox calibration."""

from __future__ import annotations

import sys
from typing import Sequence

from .drag_bigbox_continuous_calibration import main as continuous_main


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return continuous_main(argv)
    except KeyboardInterrupt:
        print("\n标定已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"标定失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
