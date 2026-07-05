from __future__ import annotations

import importlib
from typing import Any

from .config import RANDOM_STATE


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"[skip] {module_name} is not installed.")
        return None


def build_models(estimators: int) -> dict[str, Any]:
    models: dict[str, Any] = {}

    xgboost = optional_import("xgboost")
    if xgboost is not None:
        models["xgboost"] = xgboost.XGBClassifier(
            n_estimators=estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    lightgbm = optional_import("lightgbm")
    if lightgbm is not None:
        models["lightgbm"] = lightgbm.LGBMClassifier(
            n_estimators=estimators,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    catboost = optional_import("catboost")
    if catboost is not None:
        models["catboost"] = catboost.CatBoostClassifier(
            iterations=estimators,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
        )

    return models
