from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_categorical_feature(
    *,
    index: int,
    column: str,
    imputer: SimpleImputer,
    encoder: OrdinalEncoder,
    column_index: int,
) -> dict[str, Any]:
    fill_raw = imputer.statistics_[column_index]
    categories = encoder.categories_[column_index]
    encoding = {str(category): idx for idx, category in enumerate(categories)}
    fill_encoded = encoding.get(str(fill_raw), encoder.unknown_value)

    return {
        "name": column,
        "index": index,
        "kind": "categorical",
        "model_input_default": _json_value(fill_encoded),
        "fill_value_raw": _json_value(fill_raw),
        "unknown_value": _json_value(encoder.unknown_value),
        "encoding": encoding,
    }


def _build_numeric_feature(
    *,
    index: int,
    column: str,
    imputer: SimpleImputer,
    column_index: int,
) -> dict[str, Any]:
    default = imputer.statistics_[column_index]
    return {
        "name": column,
        "index": index,
        "kind": "numeric",
        "model_input_default": _json_value(default),
    }


def build_feature_schema(preprocessor: ColumnTransformer) -> dict[str, Any]:
    """Build a serving contract from a fitted ColumnTransformer."""
    feature_order = list(preprocessor.get_feature_names_out())
    features: list[dict[str, Any]] = []

    index = 0
    for name, transformer, columns in preprocessor.transformers_:
        column_list = list(columns)
        if name == "num":
            imputer = transformer.named_steps["imputer"]
            for column_index, column in enumerate(column_list):
                features.append(
                    _build_numeric_feature(
                        index=index,
                        column=column,
                        imputer=imputer,
                        column_index=column_index,
                    )
                )
                index += 1
            continue

        if name == "cat":
            imputer = transformer.named_steps["imputer"]
            encoder = transformer.named_steps["encoder"]
            for column_index, column in enumerate(column_list):
                features.append(
                    _build_categorical_feature(
                        index=index,
                        column=column,
                        imputer=imputer,
                        encoder=encoder,
                        column_index=column_index,
                    )
                )
                index += 1
            continue

        raise ValueError(f"Unsupported transformer block: {name}")

    if index != len(feature_order):
        raise RuntimeError("Feature schema length does not match preprocessor output.")

    return {
        "schema_version": "1",
        "description": (
            "Model input order after preprocessing. "
            "Java serving should assemble float[] using feature_order index-by-index."
        ),
        "feature_order": feature_order,
        "features": features,
    }


def export_feature_schema(preprocessor: ColumnTransformer, output_path: Path) -> Path:
    schema = build_feature_schema(preprocessor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
