from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml, make_classification

from .config import RANDOM_STATE


def load_dataset(dataset: str, sample_size: int) -> tuple[pd.DataFrame, pd.Series]:
    if dataset == "adult":
        try:
            bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        except TypeError:
            bunch = fetch_openml("adult", version=2, as_frame=True)
        frame = bunch.frame.copy()
        target_column = "class"
        y = frame[target_column].astype(str).str.contains(">50K").astype(int)
        x = frame.drop(columns=[target_column])
        if sample_size and sample_size < len(x):
            x = x.sample(n=sample_size, random_state=RANDOM_STATE)
            y = y.loc[x.index]
        return x.reset_index(drop=True), y.reset_index(drop=True)

    x_array, y_array = make_classification(
        n_samples=sample_size,
        n_features=40,
        n_informative=18,
        n_redundant=8,
        n_repeated=0,
        n_classes=2,
        weights=[0.89, 0.11],
        class_sep=1.2,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feature_{idx:02d}" for idx in range(x_array.shape[1])]
    return pd.DataFrame(x_array, columns=feature_names), pd.Series(y_array, name="target")
