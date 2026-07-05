"""Quickstart for comparing XGBoost, LightGBM, and CatBoost.

Examples:
    python quickstart_tree_models.py --dataset synthetic --sample-size 60000
    python quickstart_tree_models.py --dataset adult
    python -m 0_intro.run --dataset synthetic --sample-size 60000

Modular source lives under 0_intro/. See 0_intro/README.md for a reading guide.
"""

from __future__ import annotations

from importlib import import_module


def main() -> None:
    import_module("0_intro.run").main()


if __name__ == "__main__":
    main()
