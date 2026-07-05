from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .analysis import assign_feature_groups


def compute_permutation_synergy(
    *,
    baseline_score: float,
    single_scores: dict[str, float],
    joint_scores: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Compute whether joint permutation hurts more than individual drops."""
    rows = []
    for (feature_a, feature_b), joint_score in joint_scores.items():
        drop_a = baseline_score - single_scores[feature_a]
        drop_b = baseline_score - single_scores[feature_b]
        joint_drop = baseline_score - joint_score
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "drop_a": drop_a,
                "drop_b": drop_b,
                "joint_drop": joint_drop,
                "synergy": joint_drop - drop_a - drop_b,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("synergy", ascending=False)
        .reset_index(drop=True)
    )


def select_top_features(
    importance: pd.DataFrame,
    *,
    top_k: int,
    always_keep: Iterable[str] = (),
    feature_col: str = "feature",
    importance_col: str = "importance",
) -> list[str]:
    """Select top-k important features while preserving required columns first."""
    ranked = (
        importance.sort_values(importance_col, ascending=False)[feature_col]
        .astype(str)
        .tolist()
    )
    selected: list[str] = []
    for feature in list(always_keep) + ranked[:top_k]:
        if feature not in selected:
            selected.append(feature)
    return selected


def build_feature_group_map(feature_names: Iterable[str]) -> dict[str, list[str]]:
    groups = assign_feature_groups(feature_names)
    result: dict[str, list[str]] = {}
    for feature, group in groups.items():
        result.setdefault(group, []).append(feature)
    return result


def run_group_ablation(
    *,
    train_fn: Callable[[pd.DataFrame], Any],
    score_fn: Callable[[Any, pd.DataFrame], float],
    x_train: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    baseline_score: float,
) -> pd.DataFrame:
    """Retrain after dropping each feature group and compare the score drop."""
    rows = []
    for group, columns in feature_groups.items():
        kept = [col for col in x_train.columns if col not in set(columns)]
        if not kept:
            continue
        model = train_fn(x_train[kept])
        score = score_fn(model, x_train[kept])
        rows.append(
            {
                "group": group,
                "feature_count_removed": len(columns),
                "score": score,
                "score_drop": baseline_score - score,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("score_drop", ascending=False)
        .reset_index(drop=True)
    )


def permutation_scores_for_features(
    model: Any,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: Iterable[str],
    *,
    random_state: int = 42,
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
    scores: dict[str, float] = {}
    for feature in features:
        permuted = x_valid.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        scores[feature] = roc_auc_score(y_valid, model.predict_proba(permuted)[:, 1])
    return baseline, scores


def joint_permutation_scores(
    model: Any,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: Iterable[str],
    *,
    max_pairs: int = 100,
    random_state: int = 42,
) -> dict[tuple[str, str], float]:
    rng = np.random.default_rng(random_state)
    scores: dict[tuple[str, str], float] = {}
    for feature_a, feature_b in list(combinations(features, 2))[:max_pairs]:
        permuted = x_valid.copy()
        for feature in (feature_a, feature_b):
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        scores[(feature_a, feature_b)] = roc_auc_score(
            y_valid,
            model.predict_proba(permuted)[:, 1],
        )
    return scores


def compare_feature_sets(
    *,
    model_factory: Callable[[], Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for name, features in feature_sets.items():
        model = model_factory()
        model.fit(x_train[features], y_train)
        pred = model.predict_proba(x_valid[features])[:, 1]
        rows.append(
            {
                "feature_set": name,
                "feature_count": len(features),
                "roc_auc": roc_auc_score(y_valid, pred),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

