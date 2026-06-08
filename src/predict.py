"""Inference layer: load artifacts, run predictions and segment risk.

This module is the single source of truth for turning raw customer attributes
into a churn probability, a risk segment and (optionally) a SHAP explanation.
It is consumed by both the FastAPI backend and the Streamlit dashboard.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, load_artifact

logger = get_logger(__name__)


class ChurnPredictor:
    """Encapsulates the production Pipeline and the SHAP explainer.

    ``xgboost_model.pkl`` is a full sklearn Pipeline (feature engineering ->
    encoding/scaling -> model), so inference is a single ``predict_proba`` call
    on the raw feature DataFrame — no manual transforms, no train/serve skew.
    """

    def __init__(self) -> None:
        self.pipeline = load_artifact(config.MODEL_PATH)
        self.feature_names: list[str] = load_artifact(config.FEATURE_NAMES_PATH)
        self._explainer = None  # lazily loaded

    # ----------------------------------------------------------------- #
    # Core transforms
    # ----------------------------------------------------------------- #
    def _transform(self, data: pd.DataFrame) -> np.ndarray:
        """Run the pipeline up to (but excluding) the model step, for SHAP."""
        return self.pipeline[:-1].transform(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return churn probabilities for one or more raw customer rows."""
        return self.pipeline.predict_proba(data)[:, 1]

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def predict(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Predict churn for each row, returning probability + risk segment."""
        probabilities = self.predict_proba(data)
        return [
            {
                "churn_probability": round(float(p), 4),
                "risk_segment": segment_risk(float(p)),
                "will_churn": bool(p >= config.RISK_THRESHOLDS["medium"]),
            }
            for p in probabilities
        ]

    def explain(self, data: pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
        """Return prediction + SHAP local explanation for a single row."""
        from src.explainability import local_explanation

        if self._explainer is None:
            self._explainer = load_artifact(config.SHAP_EXPLAINER_PATH)

        X = self._transform(data.head(1))
        prediction = self.predict(data.head(1))[0]

        explanation = local_explanation(
            self._explainer, X, self.feature_names, top_n=top_n
        )
        return {**prediction, "explanation": explanation}

    def waterfall(self, data: pd.DataFrame, max_display: int = 12):
        """Return a SHAP waterfall matplotlib Figure for a single customer."""
        from src.explainability import waterfall_figure

        if self._explainer is None:
            self._explainer = load_artifact(config.SHAP_EXPLAINER_PATH)
        X = self._transform(data.head(1))
        return waterfall_figure(self._explainer, X, self.feature_names, max_display)


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Return a cached predictor instance (artifacts loaded once)."""
    return ChurnPredictor()


def segment_risk(probability: float) -> str:
    """Map a churn probability to a configurable risk segment label."""
    if probability >= config.RISK_THRESHOLDS["high"]:
        return config.RISK_LABELS["high"]
    if probability >= config.RISK_THRESHOLDS["medium"]:
        return config.RISK_LABELS["medium"]
    return config.RISK_LABELS["low"]


def predict_single(customer: dict[str, Any]) -> dict[str, Any]:
    """Convenience helper: predict churn for one customer given as a dict."""
    df = pd.DataFrame([customer])
    return get_predictor().predict(df)[0]
