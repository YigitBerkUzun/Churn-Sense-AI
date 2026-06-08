"""Business-oriented feature engineering for the BigML Telecom Churn dataset.

Every feature here is designed to be interpretable by a business stakeholder
rather than to squeeze out marginal accuracy. The same function is used during
training and at inference time (API / dashboard) so there is never any skew
between how features are built offline and online.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# Names of the engineered columns, exported so other modules stay in sync.
ENGINEERED_FEATURES: list[str] = [
    "tenure_group",
    "service_count",
    "total_charge",
    "avg_charge_per_minute",
    "customer_value_score",
    "service_call_risk",
]


def _tenure_group(account_length: pd.Series) -> pd.Series:
    """Bucket account length (the tenure analog) into lifecycle stages."""
    bins = [-1, 50, 100, 150, 200, np.inf]
    labels = ["New", "Growing", "Established", "Loyal", "Veteran"]
    return pd.cut(account_length, bins=bins, labels=labels).astype(str)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append all engineered features to a cleaned DataFrame."""
    df = df.copy()

    # tenure_group — customer lifecycle stage based on account length.
    df["tenure_group"] = _tenure_group(df["Account length"])

    # service_count — number of value-added plans the customer subscribes to.
    plans = [c for c in config.PLAN_COLUMNS if c in df.columns]
    df["service_count"] = (df[plans] == "Yes").sum(axis=1).astype(int)

    # total_charge — total spend across day/eve/night/international usage.
    charge_cols = [c for c in config.CHARGE_COLUMNS if c in df.columns]
    df["total_charge"] = df[charge_cols].sum(axis=1).round(2)

    # total minutes used (intermediate, also feeds avg_charge_per_minute).
    minute_cols = [c for c in config.MINUTE_COLUMNS if c in df.columns]
    total_minutes = df[minute_cols].sum(axis=1)

    # avg_charge_per_minute — pricing efficiency proxy (guard against zero use).
    df["avg_charge_per_minute"] = np.where(
        total_minutes > 0,
        df["total_charge"] / total_minutes,
        0.0,
    ).round(4)

    # customer_value_score — approximate lifetime value: spend weighted by how
    # long the account has been active.
    df["customer_value_score"] = (
        df["total_charge"] * (df["Account length"] / 100.0)
    ).round(2)

    # service_call_risk — heavy support contact is the single strongest churn
    # signal in this dataset; scaled to roughly [0, 1+].
    df["service_call_risk"] = (
        df["Customer service calls"] / config.SERVICE_CALL_RISK_THRESHOLD
    ).round(3)

    logger.info("Added %d engineered features.", len(ENGINEERED_FEATURES))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Public entry point — alias kept for readability in pipelines."""
    return add_features(df)


if __name__ == "__main__":
    from src.data_preprocessing import load_clean_split

    train, _ = load_clean_split()
    frame = add_features(train)
    print(frame[ENGINEERED_FEATURES].head())
