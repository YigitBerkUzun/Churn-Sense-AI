"""Model training, comparison and artifact persistence.

Trains and compares Logistic Regression, Random Forest and XGBoost on the
BigML Telecom Churn dataset. The data arrives already split 80/20, so we train
on the 80% file and evaluate on the 20% file (no internal split).

Everything except dataset cleaning is wrapped in a **single sklearn Pipeline**:

    raw features
      -> FunctionTransformer(add_features)      # feature engineering
      -> ColumnTransformer(OneHotEncoder +      # encoding + scaling
                           StandardScaler)
      -> classifier                             # XGBoost (production)

So one fitted object handles feature engineering, encoding, scaling and
inference, guaranteeing zero train/serve skew. (Dataset-level cleaning —
duplicate removal and target encoding — stays in ``clean_data`` since those are
training-data hygiene steps, not per-row inference transforms.)

Persists:
  * ``models/xgboost_model.pkl``   — the full fitted Pipeline (production)
  * ``models/scaler.pkl``          — the fitted ColumnTransformer step (for SHAP)
  * ``models/shap_explainer.pkl``  — SHAP explainer for the model step
  * ``models/feature_names.pkl``   — transformed feature names (for SHAP/UI)
  * ``reports/metrics.json``       — comparison metrics for all three models

Run with:  ``python -m src.train_model``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import preprocess
from src.evaluate import evaluate_model, format_metrics
from src.feature_engineering import add_features
from src.utils import get_logger, save_artifact, save_json

logger = get_logger(__name__)

# Categorical columns after feature engineering (original + tenure_group).
CATEGORICAL_FEATURES: list[str] = config.CATEGORICAL_COLUMNS + ["tenure_group"]

# Numeric columns after feature engineering (original + engineered numerics).
NUMERIC_FEATURES: list[str] = config.NUMERIC_COLUMNS + [
    "service_count",
    "total_charge",
    "avg_charge_per_minute",
    "customer_value_score",
    "service_call_risk",
]


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals and standard-scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(model: object) -> Pipeline:
    """Assemble the end-to-end pipeline: FE -> encode/scale -> model.

    The pipeline takes a raw (cleaned) feature DataFrame and produces a
    prediction, so the same object is reused unchanged at inference time.
    """
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def get_clean_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both splits (feature engineering happens in the pipeline)."""
    return preprocess(save=True)


def _candidate_models(scale_pos_weight: float) -> dict[str, object]:
    """Return the three models to compare, configured for class imbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train() -> dict:
    """Full training routine. Returns the metrics dictionary."""
    config.ensure_directories()

    train_df, test_df = get_clean_splits()
    y_train = train_df[config.TARGET_COLUMN]
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])

    # Handle class imbalance for XGBoost via positive class weighting.
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    logger.info(
        "Class balance (train): neg=%d pos=%d weight=%.2f", neg, pos, scale_pos_weight
    )

    all_metrics: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}

    # Each candidate is wrapped in its own full pipeline (FE + encode/scale +
    # model) and fit on the raw training features.
    for name, model in _candidate_models(scale_pos_weight).items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test.to_numpy(), y_pred, y_proba)
        all_metrics[name] = metrics
        pipelines[name] = pipe
        logger.info(format_metrics(name, metrics))

    # XGBoost is the designated production model.
    production_name = "XGBoost"
    production_pipe = pipelines[production_name]

    # Pull out the fitted sub-components for SHAP and inspection.
    model_step = production_pipe.named_steps["model"]
    preprocess_step = production_pipe.named_steps["preprocess"]
    feature_names = preprocess_step.get_feature_names_out().tolist()
    # Matrix as seen by the model (FE + encode/scale applied) for SHAP plots.
    X_test_t = production_pipe[:-1].transform(X_test)

    # Persist artifacts: full pipeline is the production model.
    save_artifact(production_pipe, config.MODEL_PATH)
    save_artifact(preprocess_step, config.PREPROCESSOR_PATH)
    save_artifact(feature_names, config.FEATURE_NAMES_PATH)

    # Build and persist a SHAP explainer + summary plot (on the model step).
    from src.explainability import build_explainer, save_summary_plot

    explainer = build_explainer(model_step, save=True)
    try:
        save_summary_plot(explainer, X_test_t, feature_names)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Could not save SHAP summary plot: %s", exc)
    logger.info("SHAP explainer ready: %s", type(explainer).__name__)

    report = {
        "dataset": "BigML Telecom Churn",
        "production_model": production_name,
        "pipeline_steps": [name for name, _ in production_pipe.steps],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_names),
        "models": all_metrics,
    }
    save_json(report, config.METRICS_PATH)
    logger.info("Saved metrics report to %s", config.METRICS_PATH)
    logger.info("Training complete. Production model: %s (single Pipeline)", production_name)
    return report


if __name__ == "__main__":
    train()
