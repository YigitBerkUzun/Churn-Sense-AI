"""API tests using FastAPI's TestClient.

The /health endpoint is always testable. Inference endpoints are exercised only
when a trained model artifact exists, otherwise they should return 503.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src import config

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "ChurnGuard AI API"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_predict_endpoint(sample_customer):
    resp = client.post("/predict", json=sample_customer)
    if not config.MODEL_PATH.exists():
        assert resp.status_code == 503
        return
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_segment"] in config.RISK_LABELS.values()


def test_recommend_endpoint(sample_customer):
    resp = client.post("/recommend", json=sample_customer)
    if not config.MODEL_PATH.exists():
        assert resp.status_code == 503
        return
    assert resp.status_code == 200
    assert isinstance(resp.json()["recommendations"], list)


def test_predict_validation_rejects_bad_payload():
    resp = client.post("/predict", json={"Account length": -5})
    assert resp.status_code == 422
