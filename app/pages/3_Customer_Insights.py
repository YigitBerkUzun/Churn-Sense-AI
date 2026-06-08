"""Customer Insights page: exploratory data analytics."""
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

from app.components.charts import (
    churn_by_category,
    churn_distribution,
    numeric_distribution,
)
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Customer Insights · ChurnGuard AI", page_icon="📊", layout="wide"
)
render_sidebar()

st.title("📊 Customer Insights")
st.caption("Exploratory analytics on the customer base (train + test combined).")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """Load and combine the cleaned train/test splits if available."""
    from src import config
    from src.data_preprocessing import load_clean_split

    try:
        train, test = load_clean_split()
    except FileNotFoundError:
        return None
    return pd.concat([train, test], ignore_index=True)


df = load_data()

if df is None:
    st.error(
        "Dataset not found. Place `churn-bigml-80.csv` and `churn-bigml-20.csv` "
        "in `data/raw/`."
    )
    st.stop()

# ---- KPI row ---------------------------------------------------------------
total = len(df)
churned = int(df["Churn"].sum()) if "Churn" in df else 0
churn_rate = (churned / total * 100) if total else 0
avg_day_charge = df["Total day charge"].mean() if "Total day charge" in df else 0
avg_account = df["Account length"].mean() if "Account length" in df else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{total:,}")
k2.metric("Churn Rate", f"{churn_rate:.1f}%")
k3.metric("Avg Day Charge", f"${avg_day_charge:.2f}")
k4.metric("Avg Account Length", f"{avg_account:.0f}")

st.divider()

# ---- Charts ----------------------------------------------------------------
if "Churn" in df.columns:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(churn_distribution(df), use_container_width=True)
    with c2:
        cat_candidates = [
            c
            for c in ["International plan", "Voice mail plan", "Area code",
                      "Customer service calls", "State"]
            if c in df.columns
        ]
        cat_col = st.selectbox("Churn rate by category", cat_candidates)
        st.plotly_chart(churn_by_category(df, cat_col), use_container_width=True)

    num_candidates = [
        c
        for c in [
            "Account length",
            "Total day minutes",
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
            "Customer service calls",
        ]
        if c in df.columns
    ]
    num_col = st.selectbox("Numeric feature distribution", num_candidates)
    st.plotly_chart(numeric_distribution(df, num_col), use_container_width=True)

st.divider()
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)
