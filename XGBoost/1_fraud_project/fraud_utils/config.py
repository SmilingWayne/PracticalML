from __future__ import annotations

from pathlib import Path


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "ieee-fraud-detection"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DEFAULT_NROWS = 20_000
DEFAULT_N_SPLITS = 3
DEFAULT_N_ESTIMATORS = 200
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_SHAP_SAMPLE_SIZE = 1_000
START_DATE = "2017-11-30"

