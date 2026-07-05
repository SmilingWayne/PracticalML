from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .features import reduce_mem_usage


@dataclass
class RawFraudData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def read_indexed_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read an IEEE-CIS CSV with TransactionID as index."""
    return pd.read_csv(path, index_col="TransactionID", nrows=nrows)


def merge_transaction_identity(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Left join identity features onto transaction rows by TransactionID."""
    return transaction.merge(identity, how="left", left_index=True, right_index=True)


def load_ieee_cis_data(
    data_dir: Path | str = DATA_DIR,
    nrows: int | None = None,
    optimize_memory: bool = True,
) -> RawFraudData:
    """Load local IEEE-CIS files and return merged train/test frames."""
    data_path = Path(data_dir)
    train_transaction = read_indexed_csv(data_path / "train_transaction.csv", nrows)
    test_transaction = read_indexed_csv(data_path / "test_transaction.csv", nrows)
    train_identity = read_indexed_csv(data_path / "train_identity.csv", nrows)
    test_identity = read_indexed_csv(data_path / "test_identity.csv", nrows)
    sample_submission = read_indexed_csv(data_path / "sample_submission.csv", nrows)

    train = merge_transaction_identity(train_transaction, train_identity)
    test = merge_transaction_identity(test_transaction, test_identity)

    if optimize_memory:
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

    return RawFraudData(train=train, test=test, sample_submission=sample_submission)

