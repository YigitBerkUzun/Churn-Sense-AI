"""Data loading and cleaning for the BigML Telecom Churn dataset.

The dataset is delivered pre-split into an 80% training file and a 20% test
file, so this module loads each one and turns it into a clean, typed DataFrame.
Feature creation lives in :mod:`src.feature_engineering` and encoding/scaling
lives inside the model pipeline built in :mod:`src.train_model`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load a raw BigML churn CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Place 'churn-bigml-80.csv' and 'churn-bigml-20.csv' in data/raw/."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, %d columns", path.name, *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw BigML data.

    Steps:
      * drop exact duplicate rows
      * strip whitespace from string columns
      * cast "Area code" to a categorical string (only a few discrete codes)
      * normalise the boolean target to integers (True=1, False=0)
    """
    df = df.copy()

    # Drop exact duplicate rows (none in the shipped data, but guards against
    # dirtier inputs and keeps the pipeline robust).
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("Dropped %d duplicate rows.", removed)

    # Strip stray whitespace from object columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Area code is a code, not a magnitude -> treat as categorical text.
    if "Area code" in df.columns:
        df["Area code"] = df["Area code"].astype(str)

    # Encode the target. pandas may read it as bool or as the strings
    # "True"/"False"; handle both robustly.
    if config.TARGET_COLUMN in df.columns:
        df[config.TARGET_COLUMN] = (
            df[config.TARGET_COLUMN]
            .map({True: 1, False: 0, "True": 1, "False": 0})
            .astype(int)
        )

    df = df.replace({"": np.nan})
    logger.info("Cleaned data: %d rows, %d columns", *df.shape)
    return df


def load_clean_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both the train (80%) and test (20%) splits."""
    train = clean_data(load_raw_data(config.RAW_TRAIN_PATH))
    test = clean_data(load_raw_data(config.RAW_TEST_PATH))
    return train, test


def preprocess(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run load -> clean for both splits and optionally persist the results."""
    config.ensure_directories()
    train, test = load_clean_split()
    if save:
        train.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
        test.to_csv(config.PROCESSED_TEST_PATH, index=False)
        logger.info(
            "Saved cleaned splits to %s and %s",
            config.PROCESSED_TRAIN_PATH,
            config.PROCESSED_TEST_PATH,
        )
    return train, test


if __name__ == "__main__":
    preprocess()
