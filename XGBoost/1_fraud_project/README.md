# IEEE-CIS Fraud Detection 学习项目

这个目录用于从 0 到 1 跑通 IEEE-CIS Fraud Detection 的树模型流程。数据放在 `ieee-fraud-detection/` 下，默认使用 `XGBoost/.venv`。

## 推荐运行顺序

1. 先打开 `fraud_xgboost_feature_engineering_and_shap.ipynb`。
2. 保持 `FULL_RUN = False`，先用 `NROWS = 20_000` 小样本跑通。
3. 再阅读原始改造版 `lightgbm-single-model-and-feature-engineering.ipynb`，对照 Kaggle 写法理解每一步。
4. 流程稳定后逐步增大 `NROWS`，最后再考虑 `FULL_RUN = True`。

## 环境

```bash
cd XGBoost
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name practicalml-xgboost --display-name "PracticalML XGBoost"
```

如果遇到 matplotlib/fontconfig cache 警告，一般不影响训练。想消除警告可以在终端里先设置一个可写缓存目录：

```bash
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
```

## 文件说明

- `fraud_utils/`：模块化 helper，承载数据读取、特征工程、建模、重要性、SHAP、消融和协同实验。
- `fraud_xgboost_feature_engineering_and_shap.ipynb`：推荐主 notebook，包含 XGBoost、SHAP、Permutation、消融、特征精简和协同关系提取。
- `lightgbm-single-model-and-feature-engineering.ipynb`：本地化改造后的原 LightGBM notebook，保留原学习路径。
- `outputs/`：提交文件、重要性表、关系候选表等输出。
- `tests/test_fraud_utils.py`：标准库 `unittest` smoke tests，不依赖 pytest。

## 分析实验怎么理解

- 消融实验：按特征家族移除并重训，回答“这一组特征整体有没有贡献”。
- 分组重要性：把单列 importance 聚合成业务特征组，避免只盯着单个强特征。
- Permutation 协同：比较联合置换损失和单列置换损失之和，提取互补或冗余候选。
- 特征精简：用 top-k、Permutation 正贡献等规则训练 smaller model，对比速度和 AUC/PR-AUC。
- SHAP：解释模型行为方向，适合看全局贡献、单样本原因和 dependence plot，但不是因果结论。

## 全量训练注意

全量 IEEE-CIS 数据约 59 万训练行、400+ 特征。原 LightGBM 5 折可能需要数小时；SHAP interaction 和全量二阶 permutation 会更慢。建议先用小样本验证流程，再逐步扩大样本和分析范围。

原 Kaggle 特征工程会用 train+test 一起做聚合统计和频次编码，适合比赛提交复现。迁移到真实业务验证时，应改成只使用训练时点之前可获得的数据，避免未来信息泄漏。

