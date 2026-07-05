from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE


@dataclass
class CVResult:
    model_name: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    fold_scores: list[dict[str, float]]
    mean_roc_auc: float
    oof_roc_auc: float
    oof_pr_auc: float
    feature_importance: pd.DataFrame
    models: list[Any]
    train_seconds: float


def build_xgboost_classifier(
    *,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float | None = None,
) -> Any:
    import xgboost as xgb

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**params)


def build_lightgbm_classifier(
    *,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=128,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        objective="binary",
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )


def run_stratified_cv(
    *,
    model_factory: Any,
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame | None = None,
    n_splits: int = 3,
    random_state: int = RANDOM_STATE,
    model_name: str = "model",
) -> CVResult:
    folds = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_predictions = np.zeros(len(x), dtype=float)
    test_predictions = np.zeros(len(x_test), dtype=float) if x_test is not None else np.array([])
    fold_scores: list[dict[str, float]] = []
    importance_frames: list[pd.DataFrame] = []
    models: list[Any] = []
    started = time.perf_counter()

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x, y), start=1):
        model = model_factory()
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(x_train, y_train)
        valid_pred = model.predict_proba(x_valid)[:, 1]
        oof_predictions[valid_idx] = valid_pred
        roc_auc = roc_auc_score(y_valid, valid_pred)
        pr_auc = average_precision_score(y_valid, valid_pred)
        fold_scores.append({"fold": float(fold_idx), "roc_auc": roc_auc, "pr_auc": pr_auc})

        if x_test is not None:
            test_predictions += model.predict_proba(x_test)[:, 1] / n_splits
        importance_frames.append(
            model_feature_importance(model, list(x.columns), fold=fold_idx)
        )
        models.append(model)

    train_seconds = time.perf_counter() - started
    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_pr_auc = average_precision_score(y, oof_predictions)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    return CVResult(
        model_name=model_name,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        fold_scores=fold_scores,
        mean_roc_auc=float(np.mean([score["roc_auc"] for score in fold_scores])),
        oof_roc_auc=oof_roc_auc,
        oof_pr_auc=oof_pr_auc,
        feature_importance=feature_importance,
        models=models,
        train_seconds=train_seconds,
    )


def model_feature_importance(
    model: Any,
    feature_names: list[str],
    *,
    fold: int | None = None,
) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None and hasattr(model, "get_booster"):
        score = model.get_booster().get_score(importance_type="gain")
        values = np.array([score.get(name, 0.0) for name in feature_names])
    if values is None:
        values = np.zeros(len(feature_names), dtype=float)

    frame = pd.DataFrame({"feature": feature_names, "importance": np.asarray(values)})
    if fold is not None:
        frame["fold"] = fold
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def summarize_cv_result(result: CVResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": result.model_name,
                "mean_roc_auc": result.mean_roc_auc,
                "oof_roc_auc": result.oof_roc_auc,
                "oof_pr_auc": result.oof_pr_auc,
                "train_seconds": result.train_seconds,
            }
        ]
    )


def save_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> Path:
    submission = sample_submission.copy()
    submission["isFraud"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.reset_index().to_csv(output_path, index=False)
    return output_path

