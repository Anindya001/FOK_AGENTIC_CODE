# FOK_AGENTIC_CODE

Fractional-order capacitor degradation prediction toolkit.

## Pre-commit sanity checks

Before creating a new commit, run the lightweight verification script to make sure
the plotting UI and Sobol analysis modules still compile:

```bash
python scripts/precommit_checks.py
```

The script attempts to import the GUI dependencies (PyQt5 and matplotlib) and
re-compiles the core analysis modules, which helps surface missing packages or
syntax errors early.
