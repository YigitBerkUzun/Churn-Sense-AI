"""Model evaluation utilities.

Provides a single ``evaluate_model`` function returning the full set of metrics
required by the project (precision, recall, F1, ROC-AUC, confusion matrix) so
that accuracy is never reported in isolation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Compute the standard churn classification metrics.

    Args:
        y_true: ground-truth labels (0/1).
        y_pred: predicted labels (0/1).
        y_proba: predicted probability of the positive (churn) class.

    Returns:
        Dictionary of metric name -> value. The confusion matrix is returned as
        a nested list so it is JSON serialisable.
    """
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def format_metrics(name: str, metrics: dict[str, Any]) -> str:
    """Return a human-readable one-line summary of a model's metrics."""
    return (
        f"{name:<20} | acc={metrics['accuracy']:.3f} "
        f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
        f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f}"
    )
