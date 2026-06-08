"""FastAPI application entry point for ChurnGuard AI.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Interactive docs are then available at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src import __version__

app = FastAPI(
    title="ChurnGuard AI API",
    description=(
        "Customer churn prediction, SHAP explainability and rule-based "
        "retention recommendations."
    ),
    version=__version__,
)

# Allow the Streamlit dashboard (and other clients) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Friendly root endpoint pointing users to the docs."""
    return {
        "service": "ChurnGuard AI API",
        "version": __version__,
        "docs": "/docs",
    }
