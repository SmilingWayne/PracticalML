from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare common GBDT libraries.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "adult"],
        default="synthetic",
        help="Dataset to train on. Use synthetic first to avoid network setup.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60000,
        help="Rows to generate for the synthetic dataset or sample from Adult.",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=300,
        help="Number of boosting rounds/trees for each model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for saved model files.",
    )
    return parser.parse_args()
