from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class ModelResult:
    name: str
    train_seconds: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    model_path: Path
    top_features: list[tuple[str, float]]


def top_feature_importance(
    model: Any,
    feature_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    pairs = list(zip(feature_names, np.asarray(importances, dtype=float), strict=False))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:top_k]


def train_and_evaluate(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: Path,
) -> ModelResult:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started

    probability = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probability)
    pr_auc = average_precision_score(y_test, probability)
    brier = brier_score_loss(y_test, probability)

    output_dir.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        model_path = output_dir / "xgboost_model.json"
        model.save_model(model_path)
    elif name == "lightgbm":
        model_path = output_dir / "lightgbm_model.txt"
        model.booster_.save_model(model_path)
    elif name == "catboost":
        model_path = output_dir / "catboost_model.cbm"
        model.save_model(model_path)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return ModelResult(
        name=name,
        train_seconds=train_seconds,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        model_path=model_path,
        top_features=top_feature_importance(model, feature_names),
    )


def print_results(results: list[ModelResult]) -> None:
    print("\nModel comparison")
    print("-" * 88)
    print(f"{'model':<10} {'seconds':>9} {'roc_auc':>9} {'pr_auc':>9} {'brier':>9}  model_path")
    for result in results:
        print(
            f"{result.name:<10} "
            f"{result.train_seconds:>9.2f} "
            f"{result.roc_auc:>9.4f} "
            f"{result.pr_auc:>9.4f} "
            f"{result.brier_score:>9.4f}  "
            f"{result.model_path}"
        )

    for result in results:
        if not result.top_features:
            continue
        print(f"\nTop features for {result.name}:")
        for feature, importance in result.top_features:
            print(f"  {feature:<30} {importance:.4f}")
