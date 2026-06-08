"""Central configuration for the ChurnGuard AI platform.

All paths, column definitions, thresholds and model settings live here so that
no business value is ever hardcoded across the codebase. Importing from a single
config module keeps preprocessing, training, the API and the dashboard in sync.

Dataset: BigML Telecom Churn (pre-split into an 80% train and 20% test file).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# The dataset arrives already split 80/20.
RAW_TRAIN_PATH: Path = RAW_DATA_DIR / "churn-bigml-80.csv"
RAW_TEST_PATH: Path = RAW_DATA_DIR / "churn-bigml-20.csv"

PROCESSED_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "cleaned_train.csv"
PROCESSED_TEST_PATH: Path = PROCESSED_DATA_DIR / "cleaned_test.csv"

MODEL_PATH: Path = MODELS_DIR / "xgboost_model.pkl"
PREPROCESSOR_PATH: Path = MODELS_DIR / "scaler.pkl"  # fitted ColumnTransformer
SHAP_EXPLAINER_PATH: Path = MODELS_DIR / "shap_explainer.pkl"
FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.pkl"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"


def ensure_directories() -> None:
    """Create all output directories if they do not yet exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset schema (BigML Telecom Churn)
# --------------------------------------------------------------------------- #
TARGET_COLUMN: str = "Churn"  # True = churned, False = retained

# Plan columns carry Yes/No semantics; used for the engineered service_count.
PLAN_COLUMNS: list[str] = ["International plan", "Voice mail plan"]

# Per-period charge columns; summed into total_charge.
CHARGE_COLUMNS: list[str] = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

# Per-period minute columns; summed into total_minutes.
MINUTE_COLUMNS: list[str] = [
    "Total day minutes",
    "Total eve minutes",
    "Total night minutes",
    "Total intl minutes",
]

NUMERIC_COLUMNS: list[str] = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

# "Area code" is numeric-looking but has only a few discrete values -> treat as
# categorical. "State" is high-cardinality categorical.
CATEGORICAL_COLUMNS: list[str] = [
    "State",
    "Area code",
    "International plan",
    "Voice mail plan",
]

# --------------------------------------------------------------------------- #
# Model & training settings
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Risk segmentation thresholds (configurable, business-facing)
# A churn probability >= "high" => "High Risk", etc.
# --------------------------------------------------------------------------- #
RISK_THRESHOLDS: dict[str, float] = {
    "medium": 0.30,  # Low < 30% | Medium 30-70% | High >= 70%
    "high": 0.70,
}

RISK_LABELS: dict[str, str] = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk",
}

# Number of customer service calls at/above which churn risk spikes sharply.
SERVICE_CALL_RISK_THRESHOLD: int = 4
