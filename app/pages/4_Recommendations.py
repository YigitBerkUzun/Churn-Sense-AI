"""Recommendations page: rule-based retention strategies."""
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

from app.components.risk_card import risk_card
from app.components.sidebar import customer_input_form, render_sidebar

st.set_page_config(
    page_title="Recommendations · ChurnGuard AI", page_icon="💡", layout="wide"
)
render_sidebar()

st.title("💡 Retention Recommendations")
st.caption("Turn churn risk into concrete, business-ready retention actions.")

_PRIORITY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

customer = customer_input_form(key_prefix="rec")

if st.button("Generate recommendations", type="primary", use_container_width=True):
    try:
        from src.predict import get_predictor

        prediction = get_predictor().predict(pd.DataFrame([customer]))[0]
    except FileNotFoundError:
        st.error("Model artifacts not found. Train first with `python -m src.train_model`.")
        st.stop()

    from src.recommendation_engine import generate_recommendations

    recommendations = generate_recommendations(
        customer=customer,
        churn_probability=prediction["churn_probability"],
        risk_segment=prediction["risk_segment"],
    )

    st.divider()
    risk_card(prediction["churn_probability"], prediction["risk_segment"])

    st.markdown("### Recommended actions")
    for rec in recommendations:
        icon = _PRIORITY_ICON.get(rec["priority"], "•")
        with st.container(border=True):
            st.markdown(f"#### {icon} {rec['title']}  ·  _{rec['priority']} priority_")
            st.write(rec["detail"])
