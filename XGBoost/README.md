# XGBoost / LightGBM / CatBoost 树模型学习包

这份材料面向一个外卖配送里的“人波匹配”问题：系统每隔一段时间召回一批高单量、高聚合度、高顺路度的波次，再召回附近骑手，之后需要给 `Wave * Rider` 对打分，并把若干波次组成一次推荐给骑手。这里的树模型不应该直接替代所有派单策略，而是更适合成为现有 `WaveRiderScore` 的数据驱动分量。

## 1. 骑手接波意愿模型怎么训练

最朴素、最容易上线的第一版，是把问题做成二分类：

- 一行样本：`推荐时刻 t` 下的一个 `wave_id * rider_id`。
- 特征：只允许使用 `t` 时刻线上已经知道的信息，例如波次承托比、波次单量、取送点 AOI、骑手到取点距离、骑手当前状态、历史 AOI 熟悉度、顺路度、餐段、天气、供需强度等。
- 标签：骑手是否最终接受这个波次。第一版可以定义为 `accept_this_wave = 1/0`；如果一次推荐里有多个波次，需要明确没有被展示、被展示但未抢到、被展示且主动拒绝、被展示但其它波次被接走这几类样本是否进入训练。
- 切分：按时间切分，而不是随机切分。例如用前 3 周训练、后 3 天验证、最后 1 天测试，避免把未来供需状态、骑手行为泄漏到训练集里。
- 评估：离线不要只看 accuracy。更应该看 `AUC`、`PR-AUC`、`LogLoss`、分桶校准、Top-K 命中率，以及接入匹配策略后的模拟收益。

60 多万行、40 个左右特征，对这三类 GBDT 模型都不算大。现代多核 CPU 上，几百棵树通常是几十秒以内到 1-2 分钟量级；具体取决于树深、迭代轮数、类别特征处理、交叉验证次数和是否做超参搜索。单次训练一般不难，真正耗时的是反复构造样本、特征回溯、验证和线上回放。

## 2. 模型权重如何保留，训练到推理要注意什么

训练完要保存的是模型文件和特征契约，而不只是 Python 对象。

- XGBoost：推荐 `model.json` 或 `model.ubj`，不要把 pickle 当长期稳定格式。
- LightGBM：推荐保存 text booster，例如 `model.txt`。
- CatBoost：推荐保存原生 `.cbm`，推理速度和兼容性通常最好。

线上推理输入必须满足同一份特征契约：

- 特征名、顺序、类型、单位完全一致，例如距离是米还是公里，时间窗是 5min 还是 15min。
- 缺失值含义一致，例如“骑手没有该 AOI 历史”不能在线上变成 0 而离线是 `NaN`。
- 类别编码一致。CatBoost 可以原生处理类别特征；XGBoost / LightGBM 如果用了编码器，编码器也必须和模型一起保存。
- 特征时间点一致。训练样本里任何聚合特征都只能用推荐时刻之前的数据。
- 线上最好记录每次推理的 feature vector、model version、score、推荐结果和后续反馈，方便做训练回流和问题排查。

模型的直接输出通常是 `p_accept(wave, rider)` 或一个单调相关的偏好分。这个分数可以进入现有加权体系、KM 匹配、精排策略或阈值策略。

## 3. 怎么选特征，怎么判断特征好不好

特征选择不要只看一个重要性指标。建议把特征分成几类看：

- 业务强先验特征：顺路度、距离、波次单量、波次承托比、骑手熟悉 AOI、骑手空闲时长、当前供需比。这些即使重要性不稳定，也应该优先保留做基线。
- 交互特征：餐段 * 区域、骑手熟悉度 * 波次复杂度、距离 * 预计收益、供需紧张度 * 推荐队列长度。
- 历史行为特征：骑手过去 N 天同餐段接波率、拒绝率、AOI 接受率、近几次推荐反馈。
- 环境特征：天气、节假日、城市、商圈、实时拥堵、订单峰值状态。

判断一个特征是否好，至少看五件事：

1. 离线增益：加入特征后，时间切分验证集上的 AUC / PR-AUC / LogLoss 是否稳定变好。
2. 线上可得：推理时是否能低延迟、稳定拿到同口径数据。
3. 泄漏风险：它是否包含了推荐之后才知道的信息，例如最终是否成团、骑手是否抢到、未来 15 分钟真实单量。
4. 稳定性：不同日期、城市、餐段下重要性和分布是否剧烈漂移。
5. 可解释性：业务方能否理解它为什么影响接波意愿，出问题时能否排查。

常见重要性方法：

- `Gain`：训练时免费得到，适合快速看模型用了哪些特征，但容易偏向高基数或强分裂特征。
- `Permutation Importance`：在验证集上打乱某个特征，看指标下降多少，更接近“这个特征对当前模型表现有多重要”。
- `SHAP`：适合解释单条预测和全局方向，例如“距离变大通常降低接波概率”。它解释的是模型行为，不是因果关系。

## 4. 模型应该返回打分还是是否分配

建议返回打分，而不是硬判断。

原因是你的系统不是单点分类，而是多波次、多骑手、多约束的组合优化。模型只看到 `Wave * Rider` 的局部接受概率，不知道全局供需、波次之间冲突、骑手之间抢波、业务兜底、承托比约束和公平性约束。更合理的结构是：

```text
召回候选 Wave/Rider
  -> 构造 WaveRider 特征
  -> 模型输出接波概率或偏好分
  -> 结合业务分、约束、抢波风险、供需策略
  -> KM/排序/组合生成 recommend
  -> 骑手反馈回流训练
```

硬判断可以作为策略层阈值，例如过滤明显不合适的 `score < 0.02` 候选，但不应该成为模型唯一输出。更好的做法是用模型分数做精排信号，再由策略层决定“给谁发、发几个、发哪组”。

## 5. 今天快速上手的项目

### 环境配置（推荐）

本目录自带独立虚拟环境，使用 **Python 3.12**，避免和 conda 全局环境的 NumPy 版本冲突：

```bash
cd XGBoost

# 首次创建（需要系统已安装 python3.12）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

之后每次使用前：

```bash
cd XGBoost
source .venv/bin/activate
python quickstart_tree_models.py
```

### 运行 quickstart

本目录的 [`quickstart_tree_models.py`](quickstart_tree_models.py) 会训练 XGBoost、LightGBM、CatBoost，并输出训练耗时、AUC、PR-AUC、Brier Score 和特征重要性。建议先跑默认合成数据确认环境，再切到 Adult Census：

```bash
python quickstart_tree_models.py --dataset synthetic --sample-size 60000
python quickstart_tree_models.py --dataset adult
```

### 模块化阅读

源码已拆到 [`0_intro/`](0_intro/) 目录，按「数据 → 预处理 → 模型 → 评估 → 编排」分文件，便于逐段阅读。阅读地图见 [`0_intro/README.md`](0_intro/README.md)。

| 模块 | 文件 | 适合什么时候看 |
|------|------|----------------|
| 编排入口 | [`0_intro/run.py`](0_intro/run.py) | 第一遍：理解整体流程 |
| 数据加载 | [`0_intro/data.py`](0_intro/data.py) | 换自己的 CSV / 业务样本 |
| 预处理 | [`0_intro/preprocessing.py`](0_intro/preprocessing.py) | 学特征编码与缺失值处理 |
| 模型定义 | [`0_intro/models.py`](0_intro/models.py) | 改 XGBoost 超参 |
| 训练评估 | [`0_intro/evaluate.py`](0_intro/evaluate.py) | 学指标与模型保存 |
| 命令行 | [`0_intro/cli.py`](0_intro/cli.py) | 调整实验参数 |

也可直接运行模块：`python -m 0_intro.run --dataset synthetic --sample-size 60000`

进一步学习材料放在 [`learning_projects.md`](learning_projects.md)。如果今天只做一件事，推荐先跑通脚本，再读 `0_intro/` 各模块，最后读 `kimtth/ds-gradient-boosting-hands-on`，挑一个 Kaggle Playground 或 UCI Bank Marketing 练习特征工程。

### 本地验证记录

在 `XGBoost/.venv`（Python 3.12.13）中执行：

```bash
cd XGBoost
source .venv/bin/activate
python quickstart_tree_models.py --dataset synthetic --sample-size 3000 --estimators 30
```

验证结果：

```text
python 3.12.13 | numpy 2.4.6 | scipy 1.18.0 | sklearn 1.9.0
xgboost 3.3.0 | lightgbm 4.6.0 | catboost 1.2.10

model        seconds   roc_auc    pr_auc     brier
xgboost         0.05    0.9101    0.7081    0.0654
lightgbm        0.22    0.9162    0.7363    0.0626
catboost        0.10    0.9232    0.7035    0.0706
```

如果你在其他 conda 环境里遇到 `NumPy 2.x` 与旧版 `scipy/sklearn` 的二进制兼容错误，优先改用本目录的 `.venv`，不要和旧 ML 环境混用。

## 6. 工程落地最需要知道的事

上线时，模型本身通常不是最大风险，最大风险是数据闭环：

- 样本定义：一次推荐多个波次时，要统一正负样本口径，避免把“没机会接”当成“主动拒绝”。
- 展示偏差：只训练历史策略展示过的候选，会学习到旧策略偏差。必要时要加位置、曝光顺序、推荐组大小等特征，或者做探索流量。
- 点时正确性：所有历史统计都必须以推荐时刻为 cutoff。
- 训练-推理一致性：离线 SQL、在线特征服务、模型输入 schema 必须一致。
- 概率校准：如果下游把分数当概率做收益计算，需要用独立验证集做 Platt / Isotonic 校准，并监控校准曲线。
- 离线到在线：AUC 提升不等于接波率提升，最终要看在线实验里的接波率、波次成交率、骑手打扰、超时、供需均衡。
- 版本管理：每个模型版本要记录训练数据时间窗、特征列表、参数、指标、模型文件 checksum 和上线实验配置。

更完整的核对清单见 [`production_checklist.md`](production_checklist.md)。
