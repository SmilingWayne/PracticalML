from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .cli import parse_args
from .config import RANDOM_STATE
from .data import load_dataset
from .evaluate import print_results, train_and_evaluate
from .feature_schema import export_feature_schema
from .models import build_models
from .preprocessing import build_preprocessor


def main() -> None:
    args = parse_args()
    x, y = load_dataset(args.dataset, args.sample_size)
    print(f"Loaded dataset={args.dataset}, rows={len(x)}, features={x.shape[1]}")
    print(f"Positive rate={y.mean():.4f}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_array = preprocessor.fit_transform(x_train)
    x_test_array = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())
    x_train_matrix = pd.DataFrame(
        x_train_array,
        columns=feature_names,
        index=x_train.index,
    )
    x_test_matrix = pd.DataFrame(
        x_test_array,
        columns=feature_names,
        index=x_test.index,
    )

    schema_path = export_feature_schema(
        preprocessor,
        args.output_dir / "feature_schema.json",
    )
    print(f"Saved feature schema -> {schema_path}")

    models = build_models(args.estimators)
    if not models:
        raise RuntimeError(
            "No GBDT library is installed. Run: pip install -r requirements.txt"
        )

    results = [
        train_and_evaluate(
            name=name,
            model=model,
            x_train=x_train_matrix,
            x_test=x_test_matrix,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=args.output_dir,
        )
        for name, model in models.items()
    ]
    print_results(results)


if __name__ == "__main__":
    main()
