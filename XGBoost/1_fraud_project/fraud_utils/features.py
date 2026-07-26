from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import START_DATE


EMAIL_PROVIDER_MAP = {
    "gmail": "google",
    "gmail.com": "google",
    "att.net": "att",
    "twc.com": "spectrum",
    "scranton.edu": "other",
    "optonline.net": "other",
    "hotmail.co.uk": "microsoft",
    "comcast.net": "other",
    "yahoo.com.mx": "yahoo",
    "yahoo.fr": "yahoo",
    "yahoo.es": "yahoo",
    "charter.net": "spectrum",
    "live.com": "microsoft",
    "aim.com": "aol",
    "hotmail.de": "microsoft",
    "centurylink.net": "centurylink",
    "me.com": "apple",
    "earthlink.net": "other",
    "gmx.de": "other",
    "web.de": "other",
    "cfl.rr.com": "other",
    "hotmail.com": "microsoft",
    "protonmail.com": "other",
    "hotmail.fr": "microsoft",
    "windstream.net": "other",
    "outlook.es": "microsoft",
    "yahoo.co.jp": "yahoo",
    "yahoo.de": "yahoo",
    "servicios-ta.com": "other",
    "netzero.net": "other",
    "suddenlink.net": "other",
    "roadrunner.com": "other",
    "sc.rr.com": "other",
    "live.fr": "microsoft",
    "verizon.net": "yahoo",
    "msn.com": "microsoft",
    "q.com": "centurylink",
    "prodigy.net.mx": "att",
    "frontier.com": "yahoo",
    "anonymous.com": "other",
    "rocketmail.com": "yahoo",
    "sbcglobal.net": "att",
    "frontiernet.net": "yahoo",
    "ymail.com": "yahoo",
    "outlook.com": "microsoft",
    "mail.com": "other",
    "bellsouth.net": "other",
    "embarqmail.com": "centurylink",
    "cableone.net": "other",
    "hotmail.es": "microsoft",
    "mac.com": "apple",
    "yahoo.co.uk": "yahoo",
    "netzero.com": "other",
    "yahoo.com": "yahoo",
    "live.com.mx": "microsoft",
    "ptd.net": "other",
    "cox.net": "other",
    "aol.com": "aol",
    "juno.com": "other",
    "icloud.com": "apple",
}

US_EMAIL_SUFFIXES = {"gmail", "net", "edu"}
LATEST_BROWSERS = {
    "samsung browser 7.0",
    "opera 53.0",
    "mobile safari 10.0",
    "google search application 49.0",
    "firefox 60.0",
    "edge 17.0",
    "chrome 69.0",
    "chrome 67.0 for android",
    "chrome 63.0 for android",
    "chrome 63.0 for ios",
    "chrome 64.0",
    "chrome 64.0 for android",
    "chrome 64.0 for ios",
    "chrome 65.0",
    "chrome 65.0 for android",
    "chrome 65.0 for ios",
    "chrome 66.0",
    "chrome 66.0 for android",
    "chrome 66.0 for ios",
}


@dataclass
class PreparedFraudData:
    x_train: pd.DataFrame
    y: pd.Series
    x_test: pd.DataFrame
    metadata: dict[str, Any]


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce notebook memory pressure."""
    optimized = df.copy()
    for col in optimized.columns:
        col_type = optimized[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
        c_min = optimized[col].min()
        c_max = optimized[col].max()
        if pd.api.types.is_integer_dtype(col_type):
            if np.iinfo(np.int8).min <= c_min <= c_max <= np.iinfo(np.int8).max:
                optimized[col] = optimized[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= c_min <= c_max <= np.iinfo(np.int16).max:
                optimized[col] = optimized[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= c_min <= c_max <= np.iinfo(np.int32).max:
                optimized[col] = optimized[col].astype(np.int32)
        else:
            if np.finfo(np.float32).min <= c_min <= c_max <= np.finfo(np.float32).max:
                optimized[col] = optimized[col].astype(np.float32)
    return optimized


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    drop_useless: bool = True,
    fill_value: int | float = -999,
    null_threshold: float = 0.9,
    repeated_threshold: float = 0.9,
) -> PreparedFraudData:
    """_summary_

    Args:
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_
        drop_useless (bool, optional): _description_. Defaults to True.
        fill_value (int | float, optional): _description_. Defaults to -999.
        null_threshold (float, optional): _description_. Defaults to 0.9.
        repeated_threshold (float, optional): _description_. Defaults to 0.9.

    Raises:
        ValueError: _description_

    Returns:
        PreparedFraudData: _description_
    """
    if "isFraud" not in train.columns:
        raise ValueError("train must include the isFraud target column.")

    y = train["isFraud"].copy()
    x_train = train.drop(columns=["isFraud"]).copy()
    x_test = test.copy()

    x_train, x_test = add_uid_features(x_train, x_test)
    x_train, x_test = add_transaction_amount_aggregates(x_train, x_test)
    x_train, x_test = add_email_features(x_train, x_test)
    x_train, x_test = add_time_features(x_train, x_test)
    x_train, x_test = add_browser_features(x_train, x_test)
    x_train, x_test = add_device_features(x_train, x_test)
    x_train, x_test = add_frequency_features(x_train, x_test)

    dropped_columns: list[str] = []
    if drop_useless:
        dropped_columns = get_useless_columns(
            x_train,
            null_threshold=null_threshold,
            repeated_threshold=repeated_threshold,
        )
        x_train = x_train.drop(columns=dropped_columns, errors="ignore")
        x_test = x_test.drop(columns=dropped_columns, errors="ignore")

    x_train, x_test = align_columns(x_train, x_test)
    x_train, x_test, encoders = label_encode_object_columns(x_train, x_test)
    x_train = x_train.drop(columns=["TransactionDT", "DT"], errors="ignore")
    x_test = x_test.drop(columns=["TransactionDT", "DT"], errors="ignore")
    x_train = x_train.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
    x_test = x_test.replace([np.inf, -np.inf], fill_value).fillna(fill_value)

    return PreparedFraudData(
        x_train=x_train,
        y=y,
        x_test=x_test,
        metadata={
            "dropped_columns": dropped_columns,
            "label_encoders": encoders,
            "feature_names": list(x_train.columns),
        },
    )


def add_uid_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-resolution entity ID. 
    
    由于用户uuid，所以新增三种不同的表示用户uuid的方式，同时拥有 粗/中/细 三种实体分辨率。
    
    https://towardsdatascience.com/an-introduction-to-entity-resolution-needs-and-challenges-97fba052dde5/

    Args:
        train (pd.DataFrame): 训练数据
        test (pd.DataFrame): 测试数据

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 经过实体处理后的数据
    """
    for frame in (train, test):
        frame["uid"] = _join_as_string(frame, ["card1", "card2"])
        frame["uid2"] = _join_as_string(frame, ["uid", "card3", "card5"])
        frame["uid3"] = _join_as_string(frame, ["uid2", "addr1", "addr2"])
        if "D9" in frame.columns:
            frame["D9"] = np.where(frame["D9"].isna(), 0, 1)
    return train, test


def add_transaction_amount_aggregates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: tuple[str, ...] = (
        "card1",
        "card2",
        "card3",
        "card5",
        "uid",
        "uid2",
        "uid3",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    
    按照 `group_columns` 里的列分别分组，计算每组中 `TransactionAmt` 的 `mean` 以及 `std`.
    
    这里一个小问题是，把 train 和 test 拼在一起计算了，所以训练集会预先得知未来的信息，存在一定的信息泄漏，kaggle 常见技巧。

    Args:
        train (pd.DataFrame): 训练数据
        test (pd.DataFrame): 测试数据
        group_columns (tuple[str, ...], optional): 分组维度. Defaults to ( "card1", "card2", "card3", "card5", "uid", "uid2", "uid3", ).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 处理后的 训练集、测试集
    """
    if "TransactionAmt" not in train.columns or "TransactionAmt" not in test.columns:
        return train, test
    for col in group_columns:
        if col not in train.columns or col not in test.columns:
            continue
        both = pd.concat(
            [train[[col, "TransactionAmt"]], test[[col, "TransactionAmt"]]],
            axis=0,
        )
        for agg_type in ("mean", "std"):
            new_col = f"{col}_TransactionAmt_{agg_type}"
            mapping = both.groupby(col, dropna=False)["TransactionAmt"].agg(agg_type)
            train[new_col] = train[col].map(mapping)
            test[new_col] = test[col].map(mapping)
    train["TransactionAmt"] = np.log1p(train["TransactionAmt"])
    test["TransactionAmt"] = np.log1p(test["TransactionAmt"])
    return train, test


def add_transaction_amount_aggregates_by_time(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: tuple[str, ...] = (
        "card1",
        "card2",
        "card3",
        "card5",
        "uid",
        "uid2",
        "uid3",
    ),
    time_col: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered amount aggregates that only use past rows.

    Unlike ``add_transaction_amount_aggregates`` (train+test global stats),
    each row's mean/std is computed from earlier transactions with the same
    group key (``TransactionDT`` strictly smaller). The current row is
    excluded. New columns are named ``*_hist_mean`` / ``*_hist_std`` so they
    do not clash with the leaky global version.

    Do not call this together with ``add_transaction_amount_aggregates`` on
    the same frames if both apply ``log1p`` to ``TransactionAmt``.
    """
    if (
        "TransactionAmt" not in train.columns
        or "TransactionAmt" not in test.columns
        or time_col not in train.columns
        or time_col not in test.columns
    ):
        return train, test

    train = train.copy()
    test = test.copy()
    train_index = train.index
    test_index = test.index

    train_part = train.reset_index(drop=True)
    test_part = test.reset_index(drop=True)
    train_part["_split"] = "train"
    test_part["_split"] = "test"
    train_part["_row_id"] = np.arange(len(train_part))
    test_part["_row_id"] = np.arange(len(test_part))

    both = pd.concat([train_part, test_part], axis=0, ignore_index=True)
    # Stable order when timestamps tie: train before test, then original row order.
    both = both.sort_values(
        [time_col, "_split", "_row_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    for col in group_columns:
        if col not in both.columns:
            continue
        mean_col = f"{col}_TransactionAmt_hist_mean"
        std_col = f"{col}_TransactionAmt_hist_std"
        hist_mean, hist_std = _past_group_mean_std(
            both[col],
            both["TransactionAmt"].astype(float),
        )
        both[mean_col] = hist_mean
        both[std_col] = hist_std

    train_out = (
        both.loc[both["_split"] == "train"]
        .sort_values("_row_id", kind="mergesort")
        .drop(columns=["_split", "_row_id"])
    )
    test_out = (
        both.loc[both["_split"] == "test"]
        .sort_values("_row_id", kind="mergesort")
        .drop(columns=["_split", "_row_id"])
    )
    train_out.index = train_index
    test_out.index = test_index

    train_out["TransactionAmt"] = np.log1p(train_out["TransactionAmt"])
    test_out["TransactionAmt"] = np.log1p(test_out["TransactionAmt"])
    return train_out, test_out


def _past_group_mean_std(
    group_keys: pd.Series,
    values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    对于每一个维度,都只用其时间之前的数据计算 avg 和 std.

    Args:
        group_keys (pd.Series): _description_
        values (pd.Series): _description_

    Returns:
        tuple[pd.Series, pd.Series]: _description_
    """
    values = values.astype(float)
    keys = group_keys
    # Replace NaN keys with a sentinel so they form one group; restore via index.
    key_for_group = keys.where(keys.notna(), "__missing__")

    cumsum = values.groupby(key_for_group, sort=False).cumsum()
    cumsum_sq = (values * values).groupby(key_for_group, sort=False).cumsum()
    prev_count = values.groupby(key_for_group, sort=False).cumcount()
    prev_sum = cumsum - values
    prev_sum_sq = cumsum_sq - values * values

    with np.errstate(invalid="ignore", divide="ignore"):
        hist_mean = prev_sum / prev_count.replace(0, np.nan)
        # Match pandas groupby.std (ddof=1); NaN when fewer than 2 past rows.
        hist_var = (prev_sum_sq - prev_count * hist_mean**2) / (prev_count - 1)
        hist_var = hist_var.where(prev_count >= 2)
        hist_std = np.sqrt(hist_var.clip(lower=0))

    return hist_mean, hist_std


def add_email_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """处理邮箱信息:
    
    1. 填补缺失值为 "email_not_provided"
    2. 保留特定公司的邮箱tag为特定的类进行公司分桶, 小众的全部标记为 "other"
    3. 按照前缀后缀分别筛选出品牌和地区
    4. 检查是否P/R是相同的域名, 0-1, tag

    Args:
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 处理后的 train, test
    """
    unknown = "email_not_provided"
    for frame in (train, test):
        for col in ("P_emaildomain", "R_emaildomain"):
            if col not in frame.columns:
                continue
            filled = frame[col].fillna(unknown).astype(str)
            frame[col] = filled
            frame[f"{col}_bin"] = filled.map(EMAIL_PROVIDER_MAP).fillna("other")
            suffix = filled.str.split(".").str[-1]
            frame[f"{col}_suffix"] = suffix.where(~suffix.isin(US_EMAIL_SUFFIXES), "us")
            frame[f"{col}_prefix"] = filled.str.split(".").str[0]
        if {"P_emaildomain", "R_emaildomain"}.issubset(frame.columns):
            frame["email_check"] = np.where(
                (frame["P_emaildomain"] == frame["R_emaildomain"])
                & (frame["P_emaildomain"] != unknown),
                1,
                0,
            )
    return train, test


def add_time_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    start_date: str = START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    经典的时间特征的处理: 
    
    | 列 | 例（约） | 含义 |
    |---|---|---|
    | `TransactionDT` | `86400` | 暂时保留,后面会删除 |
    | `DT` | `2017-12-01 00:00:00` | 还原成绝对时间戳 |
    | `DT_M` | `12` | 从 2017 起算的“月第几” |
    | `DT_W` | `48` 之类 | 从 2017 起算的“周第几” |
    | `DT_D` | `335` 之类 | 从 2017 起算的“日第几” |
    | `DT_hour` | `0`–`23` | 小时 |
    | `DT_day_week` | `0`–`6` | 星期几 |
    | `DT_day` | `1`–`31` | 几号 |

    Args:
        train (pd.DataFrame): _description_
        test (pd.DataFrame): _description_
        start_date (str, optional): _description_. Defaults to START_DATE.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    for frame in (train, test):
        if "TransactionDT" not in frame.columns:
            continue
        seconds = frame["TransactionDT"].fillna(frame["TransactionDT"].median())
        frame["DT"] = pd.to_datetime(seconds.apply(lambda value: start + dt.timedelta(seconds=float(value))))
        iso_week = frame["DT"].dt.isocalendar().week.astype(int)
        frame["DT_M"] = (frame["DT"].dt.year - 2017) * 12 + frame["DT"].dt.month
        frame["DT_W"] = (frame["DT"].dt.year - 2017) * 52 + iso_week
        frame["DT_D"] = (frame["DT"].dt.year - 2017) * 365 + frame["DT"].dt.dayofyear
        frame["DT_hour"] = frame["DT"].dt.hour
        frame["DT_day_week"] = frame["DT"].dt.dayofweek
        frame["DT_day"] = frame["DT"].dt.day
    return train, test


def add_browser_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (train, test):
        if "id_31" not in frame.columns:
            continue
        frame["lastest_browser"] = frame["id_31"].isin(LATEST_BROWSERS).astype(int)
    return train, test


def add_browser_features_enriched(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enriched browser features with the same ``(train, test)`` signature.

    Adds:
    - ``lastest_browser``: original hand-curated 0/1 latest flag
    - ``browser_family``: coarse browser family (chrome / firefox / ...)
    - ``browser_version``: parsed major.minor version as float
    - ``browser_fq_enc``: how often this exact ``id_31`` appeared in the *past*
      (0 = first time seen; no future leakage when ``TransactionDT`` exists)
    """
    if "id_31" not in train.columns or "id_31" not in test.columns:
        return train, test

    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        frame["lastest_browser"] = frame["id_31"].isin(LATEST_BROWSERS).astype(int)
        family, version = _parse_browser_family_version(frame["id_31"])
        frame["browser_family"] = family
        frame["browser_version"] = version

    time_col = "TransactionDT"
    if time_col in train.columns and time_col in test.columns:
        train["browser_fq_enc"], test["browser_fq_enc"] = _past_frequency_by_key(
            train_keys=train["id_31"],
            test_keys=test["id_31"],
            train_time=train[time_col],
            test_time=test[time_col],
        )
    else:
        # Fallback without timestamps: global counts (Kaggle-style, not time-safe).
        both = pd.concat([train["id_31"], test["id_31"]], axis=0)
        mapping = both.value_counts(dropna=False).to_dict()
        train["browser_fq_enc"] = train["id_31"].map(mapping).fillna(0).astype(np.int32)
        test["browser_fq_enc"] = test["id_31"].map(mapping).fillna(0).astype(np.int32)
    return train, test


def _parse_browser_family_version(
    browser: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    text = browser.fillna("unknown").astype(str).str.lower().str.strip()
    family = pd.Series(np.full(len(text), "other", dtype=object), index=text.index)
    patterns = [
        ("chrome", r"chrome"),
        ("firefox", r"firefox"),
        ("safari", r"safari"),
        ("edge", r"edge"),
        ("opera", r"opera"),
        ("samsung", r"samsung"),
        ("ie", r"ie |internet explorer"),
        ("google_search", r"google search"),
    ]
    for name, pattern in patterns:
        family = family.where(~text.str.contains(pattern, regex=True, na=False), name)

    version = text.str.extract(r"(\d+\.\d+|\d+)", expand=False)
    version = pd.to_numeric(version, errors="coerce")
    return family, version


def _past_frequency_by_key(
    *,
    train_keys: pd.Series,
    test_keys: pd.Series,
    train_time: pd.Series,
    test_time: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Count prior occurrences of each key (current row excluded). Missing -> 0."""
    train_part = pd.DataFrame(
        {
            "key": train_keys.astype("object"),
            "time": train_time,
            "_split": "train",
            "_row_id": np.arange(len(train_keys)),
        }
    )
    test_part = pd.DataFrame(
        {
            "key": test_keys.astype("object"),
            "time": test_time,
            "_split": "test",
            "_row_id": np.arange(len(test_keys)),
        }
    )
    both = pd.concat([train_part, test_part], axis=0, ignore_index=True)
    both["key"] = both["key"].where(both["key"].notna(), "__missing__")
    both = both.sort_values(
        ["time", "_split", "_row_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    # Past-only count: 0 for the first time a UA appears.
    both["browser_fq_enc"] = both.groupby("key", sort=False).cumcount().astype(np.int32)
    train_out = (
        both.loc[both["_split"] == "train"]
        .sort_values("_row_id", kind="mergesort")["browser_fq_enc"]
        .to_numpy()
    )
    test_out = (
        both.loc[both["_split"] == "test"]
        .sort_values("_row_id", kind="mergesort")["browser_fq_enc"]
        .to_numpy()
    )
    return (
        pd.Series(train_out, index=train_keys.index),
        pd.Series(test_out, index=test_keys.index),
    )


def add_device_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (train, test):
        if "DeviceInfo" not in frame.columns:
            continue
        device = frame["DeviceInfo"].fillna("unknown_device").astype(str)
        frame["DeviceInfo"] = device.str.lower()
        frame["device_name"] = frame["DeviceInfo"].str.split("/", n=1).str[0]
        _normalize_device_name(frame)
        frame["had_id"] = 1
    return train, test


def add_frequency_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequency_columns = [
        "card1",
        "card2",
        "card3",
        "card5",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "device_name",
        "id_30",
        "id_33",
        "uid",
        "uid2",
        "uid3",
    ]
    for col in frequency_columns:
        if col not in train.columns or col not in test.columns:
            continue
        both = pd.concat([train[col], test[col]], axis=0)
        mapping = both.value_counts(dropna=False).to_dict()
        train[f"{col}_fq_enc"] = train[col].map(mapping)
        test[f"{col}_fq_enc"] = test[col].map(mapping)

    for period in ("DT_M", "DT_W", "DT_D"):
        if period not in train.columns or period not in test.columns:
            continue
        both = pd.concat([train[period], test[period]], axis=0)
        mapping = both.value_counts(dropna=False).to_dict()
        train[f"{period}_total"] = train[period].map(mapping)
        test[f"{period}_total"] = test[period].map(mapping)
        if "uid" in train.columns and "uid" in test.columns:
            new_col = f"uid_{period}"
            both_uid_period = pd.concat(
                [
                    train[["uid", period]].astype(str),
                    test[["uid", period]].astype(str),
                ],
                axis=0,
            )
            joint_key = both_uid_period["uid"] + "_" + both_uid_period[period]
            joint_counts = joint_key.value_counts(dropna=False).to_dict()
            train_key = train["uid"].astype(str) + "_" + train[period].astype(str)
            test_key = test["uid"].astype(str) + "_" + test[period].astype(str)
            train[new_col] = train_key.map(joint_counts) / train[f"{period}_total"]
            test[new_col] = test_key.map(joint_counts) / test[f"{period}_total"]
    return train, test


def get_useless_columns(
    data: pd.DataFrame,
    *,
    null_threshold: float = 0.9,
    repeated_threshold: float = 0.9,
) -> list[str]:
    many_null = [
        col for col in data.columns if data[col].isna().mean() > null_threshold
    ]
    repeated = []
    for col in data.columns:
        normalized_counts = data[col].value_counts(dropna=False, normalize=True)
        if not normalized_counts.empty and normalized_counts.iloc[0] > repeated_threshold:
            repeated.append(col)
    return sorted(set(many_null + repeated))


def align_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_columns = [col for col in train.columns if col in test.columns]
    return train[common_columns].copy(), test[common_columns].copy()


def label_encode_object_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    encoders: dict[str, dict[str, int]] = {}
    for col in train.columns:
        if _needs_label_encoding(train[col]) or _needs_label_encoding(test[col]):
            combined = pd.concat(
                [train[col].astype(str), test[col].astype(str)],
                axis=0,
                ignore_index=True,
            )
            codes, uniques = pd.factorize(combined, sort=True)
            train[col] = codes[: len(train)].astype("int32")
            test[col] = codes[len(train) :].astype("int32")
            encoders[col] = {str(value): int(idx) for idx, value in enumerate(uniques)}
    return train, test, encoders


def _needs_label_encoding(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _join_as_string(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """把若干列转字符串后合并成一个新的列. 
    
       e.g., col_1 : 201, col_2: "ABC", then new col_3 = "201_ABC"

    Args:
        frame (pd.DataFrame): _description_
        columns (list[str]): _description_

    Returns:
        pd.Series: _description_
    """
    values = []
    for col in columns:
        if col in frame.columns:
            values.append(frame[col].astype(str))
        else:
            values.append(pd.Series("missing", index=frame.index))
    result = values[0]
    for value in values[1:]:
        result = result + "_" + value
    return result


def _normalize_device_name(frame: pd.DataFrame) -> None:
    replacements = [
        ("SM", "Samsung"),
        ("SAMSUNG", "Samsung"),
        ("GT-", "Samsung"),
        ("Moto G", "Motorola"),
        ("Moto", "Motorola"),
        ("moto", "Motorola"),
        ("LG-", "LG"),
        ("rv:", "RV"),
        ("HUAWEI", "Huawei"),
        ("ALE-", "Huawei"),
        ("-L", "Huawei"),
        ("Blade", "ZTE"),
        ("BLADE", "ZTE"),
        ("Linux", "Linux"),
        ("XT", "Sony"),
        ("HTC", "HTC"),
        ("ASUS", "Asus"),
    ]
    for pattern, value in replacements:
        mask = frame["device_name"].str.contains(pattern, na=False, regex=False)
        frame.loc[mask, "device_name"] = value

