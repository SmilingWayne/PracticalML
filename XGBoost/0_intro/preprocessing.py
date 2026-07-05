from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # 按 pandas dtype 自动分列：字符串/类别/布尔 → 类别列，其余 → 数值列
    categorical_columns = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [col for col in x.columns if col not in categorical_columns]

    # 数值列流水线：缺失值用训练集的中位数填充（fit 阶段统计，transform 阶段复用）
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    # 类别列流水线：先填缺失（众数），再 Ordinal 编码为整数（Private → 0, Local-gov → 1, ...）
    # Imputer: 缺失值用训练集的众数填充（fit 阶段统计，transform 阶段复用）
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",  # 测试集出现训练集未见类别 → -1
                    unknown_value=-1,
                    encoded_missing_value=-1,  # 缺失值编码后也是 -1
                ),
            ),
        ]
    )
    # 分箱
    # numeric_pipeline = Pipeline([
    #     ("imputer", SimpleImputer(strategy="median")),
    #     ("bin", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")),
    # ])


    # 合并两路流水线：数值列走 num，类别列走 cat，最终拼成模型输入矩阵
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
            # 用于不消除缺失值
            # ("num2", "passthrough", numeric_columns),
            
            
        ],
        remainder="drop",  # 未归入 num/cat 的列直接丢弃
        verbose_feature_names_out=False,  # 输出列名保持原字段名，不加 num__ / cat__ 前缀
    )
