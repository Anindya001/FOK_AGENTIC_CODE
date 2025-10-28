"""Launch the minimal PICP GUI."""

from __future__ import annotations

import sys

from app_ui import main as run_ui


def main() -> int:
    return run_ui()


if __name__ == "__main__":
    sys.exit(main())
