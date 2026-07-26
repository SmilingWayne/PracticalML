from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def assign_feature_groups(feature_names: Iterable[str]) -> dict[str, str]:
    """Map IEEE-CIS column names into interpretable feature families."""
    return {feature: feature_group(feature) for feature in feature_names}


def feature_group(feature: str) -> str:
    if feature.startswith("uid") or "_TransactionAmt_" in feature:
        return "uid_aggregate" if feature.startswith("uid") else "amount_aggregate"
    if feature == "TransactionAmt":
        return "amount"
    if feature.startswith("card"):
        return "card"
    if feature.startswith("addr"):
        return "addr"
    if feature.startswith("C") and feature[1:].isdigit():
        return "C_count"
    if feature.startswith("D") and feature[1:].isdigit():
        return "D_time_delta"
    if feature.startswith("V") and feature[1:].isdigit():
        return "V_anonymous"
    if feature.startswith("id_"):
        return "identity"
    if "emaildomain" in feature or feature.startswith("email_"):
        return "email"
    if feature in {"DeviceInfo", "device_name", "had_id"} or feature.startswith("Device"):
        return "device"
    if feature in {"lastest_browser", "browser_family", "browser_version", "browser_fq_enc"}:
        return "browser"
    if feature.startswith("DT_"):
        return "time"
    if feature.endswith("_fq_enc"):
        return "frequency"
    return "other"


def summarize_importance_by_group(
    importance: pd.DataFrame,
    *,
    feature_col: str = "feature",
    importance_col: str = "importance",
) -> pd.DataFrame:
    frame = importance.copy()
    frame["group"] = frame[feature_col].map(feature_group)
    return (
        frame.groupby("group", as_index=False)
        .agg(
            total_importance=(importance_col, "sum"),
            mean_importance=(importance_col, "mean"),
            feature_count=(feature_col, "nunique"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )


def average_fold_importance(importance: pd.DataFrame) -> pd.DataFrame:
    return (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance(
    model: Any,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    n_repeats: int = 3,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_valid.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def sample_for_explanation(
    x: pd.DataFrame,
    *,
    sample_size: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(x) <= sample_size:
        return x.copy()
    return x.sample(n=sample_size, random_state=random_state)


def compute_shap_values(
    model: Any,
    x_sample: pd.DataFrame,
) -> tuple[Any, Any]:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    return explainer, shap_values


def dependence_candidates(
    importance: pd.DataFrame,
    *,
    top_k: int = 20,
    required_features: Iterable[str] = ("TransactionAmt",),
) -> list[str]:
    ranked = importance.sort_values("importance", ascending=False)["feature"].tolist()
    selected: list[str] = []
    for feature in list(required_features) + ranked:
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= top_k:
            break
    return selected


def two_way_binned_summary(
    frame: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    target: str,
    *,
    bins: int = 10,
    agg: Callable[[pd.Series], float] | str = "mean",
) -> pd.DataFrame:
    work = frame[[feature_a, feature_b, target]].copy()
    for feature in (feature_a, feature_b):
        if pd.api.types.is_numeric_dtype(work[feature]):
            work[f"{feature}_bin"] = pd.qcut(
                work[feature],
                q=min(bins, work[feature].nunique()),
                duplicates="drop",
            )
        else:
            top_values = work[feature].value_counts().head(bins).index
            work[f"{feature}_bin"] = work[feature].where(
                work[feature].isin(top_values),
                "other",
            )
    return (
        work.groupby([f"{feature_a}_bin", f"{feature_b}_bin"], observed=True)[target]
        .agg(agg)
        .reset_index(name=f"{target}_{agg if isinstance(agg, str) else 'agg'}")
    )


def score_auc(model: Any, x: pd.DataFrame, y: pd.Series) -> float:
    return roc_auc_score(y, model.predict_proba(x)[:, 1])

