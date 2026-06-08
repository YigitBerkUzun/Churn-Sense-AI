"""Prediction page: customer input form and churn scoring."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "src").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from app.components.charts import gauge
from app.components.risk_card import kpi_card, risk_card
from app.components.sidebar import customer_input_form, render_sidebar

st.set_page_config(page_title="Prediction · ChurnGuard AI", page_icon="🔮", layout="wide")
render_sidebar()

st.title("🔮 Churn Prediction")
st.caption("Enter a customer's details to estimate their churn risk.")

customer = customer_input_form(key_prefix="predict")

if st.button("Predict churn", type="primary", use_container_width=True):
    try:
        from src.predict import get_predictor

        result = get_predictor().predict(pd.DataFrame([customer]))[0]
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Train the model first with "
            "`python -m src.train_model`."
        )
        st.stop()

    # Derive the extra KPIs: customer value (engineered score) and the top
    # retention priority from the rule-based recommendation engine.
    from src.feature_engineering import add_features
    from src.recommendation_engine import generate_recommendations

    customer_value = float(
        add_features(pd.DataFrame([customer]))["customer_value_score"].iloc[0]
    )
    recommendations = generate_recommendations(
        customer, result["churn_probability"], result["risk_segment"]
    )
    retention_priority = recommendations[0]["priority"]  # sorted High-first

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        risk_card(result["churn_probability"], result["risk_segment"])
    with right:
        st.plotly_chart(
            gauge(result["churn_probability"], result["risk_segment"]),
            use_container_width=True,
        )

    # KPI cards: churn probability, customer value, retention priority.
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Churn Probability", f"{result['churn_probability'] * 100:.1f}%")
    with k2:
        kpi_card("Risk Segment", result["risk_segment"])
    with k3:
        kpi_card(
            "Customer Value",
            f"{customer_value:,.0f}",
            help_text="Engineered customer_value_score (spend × account length).",
        )
    with k4:
        _icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(retention_priority, "")
        kpi_card(
            "Retention Priority",
            f"{_icon} {retention_priority}",
            help_text="Highest-priority recommended retention action.",
        )

    st.info(
        "Head to the **SHAP Analysis** page to understand *why*, or the "
        "**Recommendations** page for retention actions."
    )
