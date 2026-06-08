"""Unit tests for feature engineering, evaluation and the recommendation engine.

These tests do not require a trained model artifact, so they run in CI without
the dataset. Tests that need the model are skipped automatically when the
artifacts are absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.evaluate import evaluate_model
from src.feature_engineering import ENGINEERED_FEATURES, add_features
from src.predict import segment_risk
from src.recommendation_engine import generate_recommendations


def test_feature_engineering_adds_expected_columns(sample_frame):
    out = add_features(sample_frame)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    # service_count: International plan = Yes, Voice mail plan = No -> 1.
    assert out["service_count"].iloc[0] == 1
    # total_charge = day + eve + night + intl charge.
    assert out["total_charge"].iloc[0] == pytest.approx(50.9 + 5.26 + 8.86 + 1.78)
    # service_call_risk = 5 / 4.
    assert out["service_call_risk"].iloc[0] == pytest.approx(5 / 4)


def test_avg_charge_per_minute_handles_zero_minutes(sample_frame):
    frame = sample_frame.copy()
    for col in config.MINUTE_COLUMNS:
        frame.loc[0, col] = 0.0
    out = add_features(frame)
    # No division by zero when there is no usage.
    assert out["avg_charge_per_minute"].iloc[0] == 0.0


def test_segment_risk_thresholds():
    assert segment_risk(0.10) == config.RISK_LABELS["low"]
    assert segment_risk(config.RISK_THRESHOLDS["medium"]) == config.RISK_LABELS["medium"]
    assert segment_risk(config.RISK_THRESHOLDS["high"]) == config.RISK_LABELS["high"]
    assert segment_risk(0.99) == config.RISK_LABELS["high"]


def test_evaluate_model_returns_all_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.6])
    metrics = evaluate_model(y_true, y_pred, y_proba)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
        assert key in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_recommendations_for_high_risk_customer(sample_customer):
    recs = generate_recommendations(
        customer=sample_customer,
        churn_probability=0.85,
        risk_segment=config.RISK_LABELS["high"],
    )
    assert recs, "Expected at least one recommendation"
    titles = " ".join(r["title"].lower() for r in recs)
    # 5 service calls + international plan should trigger relevant rules.
    assert "retention specialist" in titles or "international" in titles


def test_recommendations_low_risk_fallback():
    customer = {
        "State": "VA",
        "Account length": 80,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Customer service calls": 0,
        "Total day minutes": 120.0,
        "Total day charge": 20.0,
        "Total eve charge": 10.0,
        "Total night charge": 8.0,
        "Total intl charge": 1.0,
    }
    recs = generate_recommendations(customer, 0.05, config.RISK_LABELS["low"])
    assert len(recs) == 1
    assert recs[0]["priority"] == "Low"


@pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="Trained model artifact not available"
)
def test_predict_with_trained_model(sample_frame):
    from src.predict import get_predictor

    result = get_predictor().predict(sample_frame)[0]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_segment"] in config.RISK_LABELS.values()
