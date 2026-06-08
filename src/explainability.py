"""SHAP-based explainability for the production model.

Supports both global explainability (feature importance / summary plots) and
local explainability (per-customer waterfall plots and signed contributions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src import config
from src.utils import get_logger, load_artifact, save_artifact

logger = get_logger(__name__)


def build_explainer(model: Any, save: bool = True) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for a tree-based model (XGBoost/RF)."""
    explainer = shap.TreeExplainer(model)
    if save:
        save_artifact(explainer, config.SHAP_EXPLAINER_PATH)
        logger.info("Saved SHAP explainer to %s", config.SHAP_EXPLAINER_PATH)
    return explainer


def load_explainer() -> shap.TreeExplainer:
    """Load the persisted SHAP explainer."""
    return load_artifact(config.SHAP_EXPLAINER_PATH)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
) -> np.ndarray:
    """Return SHAP values for the positive (churn) class.

    Works across SHAP versions that return either a single array or a list of
    arrays (one per class).
    """
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        # Binary classifier returning [class0, class1]; take churn class.
        shap_values = shap_values[1]
    return np.asarray(shap_values)


def local_explanation(
    explainer: shap.TreeExplainer,
    x_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 8,
) -> dict[str, Any]:
    """Explain a single prediction.

    Returns the top contributing features split into those that push churn risk
    up versus down, ready for display in the dashboard or API.
    """
    x_row = np.asarray(x_row).reshape(1, -1)
    shap_values = compute_shap_values(explainer, x_row)[0]

    contributions = sorted(
        zip(feature_names, shap_values.tolist()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]

    increases = [
        {"feature": f, "impact": round(v, 4)} for f, v in contributions if v > 0
    ]
    decreases = [
        {"feature": f, "impact": round(v, 4)} for f, v in contributions if v < 0
    ]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[-1])

    return {
        "base_value": round(float(base_value), 4),
        "increases_risk": increases,
        "decreases_risk": decreases,
    }


def global_importance(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Return mean absolute SHAP value per feature (global importance)."""
    shap_values = compute_shap_values(explainer, X_transformed)
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def save_summary_plot(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: list[str],
    out_path: str | Path | None = None,
) -> Path:
    """Generate and save a SHAP summary (beeswarm) plot to reports/figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path) if out_path else config.FIGURES_DIR / "shap_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = compute_shap_values(explainer, X_transformed)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", out_path)
    return out_path


def waterfall_figure(
    explainer: shap.TreeExplainer,
    x_row: np.ndarray,
    feature_names: list[str],
    max_display: int = 12,
):
    """Build a SHAP **waterfall** plot for a single prediction.

    Returns a matplotlib Figure ready for ``st.pyplot`` in the dashboard.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_row = np.asarray(x_row).reshape(1, -1)
    shap_values = compute_shap_values(explainer, x_row)[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[-1])

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=x_row[0],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
