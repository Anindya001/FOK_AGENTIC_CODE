#!/usr/bin/env python3
"""Lightweight sanity checks before committing changes."""

from __future__ import annotations

import compileall
import importlib
import sys
from pathlib import Path

MODULES = [
    "app_ui.py",
    "fractional_sensitivity.py",
    "fractional_model.py",
    "fractional_uq.py",
    "fractional_prediction.py",
]

OPTIONAL_IMPORTS = [
    "matplotlib",
    "PyQt5",
]


def _check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment-specific feedback
        print(f"[warn] Could not import '{name}': {exc}")
        return False
    else:
        print(f"[ok] Imported '{name}' successfully.")
        return True


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    print(f"Project root: {project_root}")

    all_good = True

    print("\nChecking optional runtime dependencies…")
    for module in OPTIONAL_IMPORTS:
        all_good &= _check_import(module)

    print("\nCompiling key plotting and analysis modules…")
    for rel_path in MODULES:
        module_path = project_root / rel_path
        if not module_path.exists():
            print(f"[skip] {rel_path} not found; skipping.")
            continue
        print(f"Compiling {rel_path}…", end=" ")
        compiled = compileall.compile_file(str(module_path), force=True, quiet=1)
        if compiled:
            print("ok")
        else:
            print("failed")
            all_good = False

    if all_good:
        print("\nAll checks completed successfully.")
        return 0

    print("\nOne or more checks failed. See output above for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
