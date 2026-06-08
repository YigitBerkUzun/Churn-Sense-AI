"""API route handlers for prediction, explanation and recommendations."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    CustomerFeatures,
    ExplainResponse,
    HealthResponse,
    PredictionResponse,
    RecommendResponse,
)
from src import __version__
from src.predict import get_predictor
from src.recommendation_engine import generate_recommendations
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _to_frame(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.to_raw()])


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe that also reports whether the model is loadable."""
    model_loaded = True
    try:
        get_predictor()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Model not loaded: %s", exc)
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=__version__,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Return churn probability and risk segment for one customer."""
    try:
        result = get_predictor().predict(_to_frame(customer))[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(customer: CustomerFeatures) -> ExplainResponse:
    """Return the prediction plus a SHAP-based local explanation."""
    try:
        result = get_predictor().explain(_to_frame(customer))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplainResponse(**result)


@router.post("/recommend", response_model=RecommendResponse, tags=["inference"])
def recommend(customer: CustomerFeatures) -> RecommendResponse:
    """Return the prediction plus rule-based retention recommendations."""
    try:
        prediction = get_predictor().predict(_to_frame(customer))[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    recommendations = generate_recommendations(
        customer=customer.to_raw(),
        churn_probability=prediction["churn_probability"],
        risk_segment=prediction["risk_segment"],
    )
    return RecommendResponse(**prediction, recommendations=recommendations)
