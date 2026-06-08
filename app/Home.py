"""ChurnGuard AI — Streamlit dashboard home page.

Launch with:  ``streamlit run app/Home.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (the folder containing ``src``) is importable, since
# Streamlit puts the script's own directory on sys.path, not the project root.
_root = Path(__file__).resolve()
while not (_root / "src").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("🛡️ ChurnGuard AI")
st.subheader("Customer Churn Prediction & Retention Intelligence Platform")

st.markdown(
    """
    ChurnGuard AI is a production-oriented analytics platform that predicts which
    customers are likely to **churn**, explains **why**, and recommends concrete
    **retention actions** — turning a machine learning model into business value.
    """
)

st.divider()

# ---- Business problem -------------------------------------------------------
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📉 The Business Problem")
    st.markdown(
        """
        Acquiring a new customer can cost **5–7× more** than retaining an existing
        one. For a subscription telco business, even a small reduction in churn
        translates into significant recurring revenue.

        ChurnGuard AI helps retention teams:

        - **Predict** churn probability for each customer
        - **Understand** the drivers behind each prediction (SHAP)
        - **Segment** customers into Low / Medium / High risk
        - **Act** with tailored, rule-based retention recommendations
        """
    )

with col2:
    st.markdown("### 🧰 How It Works")
    st.markdown(
        """
        1. **Data** — IBM Telco Customer Churn dataset
        2. **Features** — business-oriented engineering
        3. **Model** — XGBoost vs Logistic Regression vs Random Forest
        4. **Explain** — global & local SHAP
        5. **Recommend** — transparent rule engine
        6. **Serve** — FastAPI + this Streamlit dashboard
        """
    )

st.divider()

st.markdown("### 🚀 Get Started")
nav1, nav2, nav3, nav4 = st.columns(4)
nav1.info("**Prediction**\n\nScore a customer in real time.")
nav2.info("**SHAP Analysis**\n\nSee what drives churn.")
nav3.info("**Customer Insights**\n\nExplore the data.")
nav4.info("**Recommendations**\n\nGet retention actions.")

st.caption("Use the sidebar to navigate between pages.")
