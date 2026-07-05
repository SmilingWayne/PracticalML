# 推荐学习项目与数据集

这份列表按“今天能快速跑起来 -> 进阶特征工程 -> 排序/推荐任务”排序。建议不要一开始就钻超参，先把数据切分、特征处理、指标和模型保存跑通。

## 今天优先跑起来

### 1. 本目录 quickstart

- 入口：[`quickstart_tree_models.py`](quickstart_tree_models.py)（薄包装，转发到 `0_intro/`）
- 模块化源码：[`0_intro/`](0_intro/) — 阅读地图见 [`0_intro/README.md`](0_intro/README.md)
  - [`0_intro/run.py`](0_intro/run.py) — 全流程编排
  - [`0_intro/data.py`](0_intro/data.py) — 数据加载
  - [`0_intro/preprocessing.py`](0_intro/preprocessing.py) — 特征预处理
  - [`0_intro/models.py`](0_intro/models.py) — 三库模型定义
  - [`0_intro/evaluate.py`](0_intro/evaluate.py) — 训练、指标、保存
  - [`0_intro/cli.py`](0_intro/cli.py) — 命令行参数
- 价值：同一份数据、同一套指标，对比 XGBoost / LightGBM / CatBoost。
- 适合练习：训练耗时、AUC、PR-AUC、Brier Score、特征重要性、模型保存格式。
- 命令：

```bash
python quickstart_tree_models.py --dataset synthetic --sample-size 60000
python quickstart_tree_models.py --dataset adult
python -m 0_intro.run --dataset synthetic --sample-size 60000
```

### 2. ds-gradient-boosting-hands-on

- 地址：https://github.com/kimtth/ds-gradient-boosting-hands-on
- 数据：UCI Bank Marketing，约 45,000 行，混合类别/数值特征，正样本比例约 11%。
- 价值：同一个项目里同时比较 XGBoost、LightGBM、CatBoost，适合照着代码读。
- 建议练习：
  - 比较默认参数和轻量调参后的 PR-AUC。
  - 对类别特征分别尝试 one-hot、ordinal、CatBoost 原生类别处理。
  - 输出 gain importance，再用 permutation importance 验证。

### 3. Adult Census Income

- Hugging Face 数据页：https://huggingface.co/datasets/scikit-learn/adult-census-income
- UCI 数据页：https://archive.ics.uci.edu/dataset/2/adult
- 数据：约 48,000 行，预测收入是否超过 50K。
- 价值：类别特征多，适合练习编码、缺失值、交叉验证和 SHAP。
- 建议练习：
  - 用时间无关的普通二分类流程跑通三模型。
  - 对 `occupation`、`native-country` 等类别特征比较编码方式。
  - 看 SHAP summary plot，理解模型为什么给高分。

## 进阶特征工程

### 4. Kaggle Playground Series

- 地址：https://www.kaggle.com/competitions?searchQuery=playground
- 价值：数据干净、公开 notebook 多、适合练习从 baseline 到 leaderboard 的迭代。
- 建议练习：
  - 建立 LightGBM baseline。
  - 每次只增加一组特征，记录 CV 和 public score 变化。
  - 做 XGBoost / LightGBM / CatBoost 加权融合。

### 5. IEEE-CIS Fraud Detection

- Kaggle 地址：https://www.kaggle.com/c/ieee-fraud-detection
- 参考文章：https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- 数据：约 59 万行，400+ 特征，强类别特征，强不均衡，带时间字段。
- 价值：很接近真实业务里的大表特征工程、时间切分和泄漏风险。
- 建议练习：
  - 对比随机切分和时间切分的指标差异。
  - 构造 UID、频次、聚合统计特征。
  - 同时看 ROC-AUC 和 PR-AUC，体会不均衡任务下 accuracy 的误导。

### 6. Home Credit Default Risk

- 地址：https://www.kaggle.com/c/home-credit-default-risk
- 数据：多张关系表，需要聚合到主表。
- 价值：练习 Hive/SQL 风格的多表 join、窗口聚合、特征宽表构建。
- 建议练习：
  - 从 `bureau`、`previous_application`、`installments` 聚合出 count/mean/max/min/std。
  - 训练 LightGBM，看特征重要性中哪些表贡献最大。
  - 删除低重要性特征，观察速度和指标变化。

## 排序与推荐相关

### 7. XGBoost Learning to Rank

- 官方教程：https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
- 关键词：`XGBRanker`、`qid`、`rank:ndcg`、LambdaMART、NDCG。
- 价值：理解“一个 query 下多个候选排序”的训练方式。你的人波匹配可以把一次推荐请求、一个骑手或一个波次集合视作 group，但真实建模前要先定清楚 group 口径。

### 8. LightGBM LGBMRanker

- API 文档：https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- 关键词：`group`、`lambdarank`、`ndcg_eval_at`。
- 价值：LightGBM 排序训练速度快，适合快速尝试 Learning-to-Rank。

### 9. MSLR-WEB10K

- 数据页：https://www.microsoft.com/en-us/research/project/mslr/
- 数据：搜索排序标准数据集，带 query、document、相关性标签。
- 价值：适合专门练习 ranking 数据格式和 NDCG 指标。
- 建议练习：
  - 用 `load_svmlight_file(..., query_id=True)` 读取。
  - 同时训练 `XGBRanker` 和 `LGBMRanker`。
  - 比较二分类打分和排序目标在 NDCG@10 上的差异。

## 官方文档

- XGBoost Python API：https://xgboost.readthedocs.io/en/stable/python/python_api.html
- XGBoost 模型保存：https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
- LightGBM Python API：https://lightgbm.readthedocs.io/en/latest/Python-API.html
- CatBoost Python 文档：https://catboost.ai/docs/en/concepts/python-reference_catboost
- SHAP 文档：https://shap.readthedocs.io/
- scikit-learn permutation importance：https://scikit-learn.org/stable/modules/permutation_importance.html
- scikit-learn calibration：https://scikit-learn.org/stable/modules/calibration.html

## 建议学习顺序

1. 跑本目录 quickstart，确认三模型训练、评估、保存全流程。
2. 读 `ds-gradient-boosting-hands-on`，学习成熟项目结构。
3. 用 Adult 或 Bank Marketing 手写一次特征处理和 CV。
4. 进入 Kaggle Playground，练习迭代和对照实验。
5. 选择 IEEE-CIS Fraud 或 Home Credit，练习时间切分、多表聚合和泄漏排查。
6. 最后看 XGBoost / LightGBM Learning-to-Rank，把二分类意愿模型升级为排序模型的候选方案。
