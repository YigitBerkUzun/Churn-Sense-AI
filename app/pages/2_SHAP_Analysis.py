"""SHAP Analysis page: global and local explainability."""
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

from app.components.charts import contribution_bar
from app.components.sidebar import customer_input_form, render_sidebar

st.set_page_config(page_title="SHAP Analysis · ChurnGuard AI", page_icon="🧠", layout="wide")
render_sidebar()

st.title("🧠 SHAP Explainability")
st.caption("Understand the model globally and explain individual predictions.")

tab_global, tab_local = st.tabs(["🌍 Global importance", "👤 Single customer"])

# --------------------------------------------------------------------------- #
# Global explainability
# --------------------------------------------------------------------------- #
with tab_global:
    st.markdown(
        "Global importance shows which features drive churn predictions across "
        "the whole customer base (mean absolute SHAP value)."
    )
    from src import config

    summary_path = config.FIGURES_DIR / "shap_summary.png"
    if summary_path.exists():
        st.image(str(summary_path), caption="SHAP summary plot", use_column_width=True)
    else:
        st.warning(
            "SHAP summary plot not found. Generate it during/after training, e.g. "
            "via `src.explainability.save_summary_plot`."
        )

# --------------------------------------------------------------------------- #
# Local explainability
# --------------------------------------------------------------------------- #
with tab_local:
    st.markdown("Explain a single prediction with local SHAP contributions.")
    customer = customer_input_form(key_prefix="shap")

    if st.button("Explain prediction", type="primary", use_container_width=True):
        df = pd.DataFrame([customer])
        try:
            from src.predict import get_predictor

            predictor = get_predictor()
            result = predictor.explain(df)
        except FileNotFoundError:
            st.error(
                "Model / explainer artifacts not found. Train first with "
                "`python -m src.train_model`."
            )
            st.stop()

        st.metric(
            "Churn probability",
            f"{result['churn_probability'] * 100:.1f}%",
            help=result["risk_segment"],
        )

        # SHAP waterfall plot — how each feature pushes the prediction from the
        # base value to this customer's churn score.
        st.markdown("#### 💧 SHAP waterfall")
        fig = predictor.waterfall(df, max_display=12)
        st.pyplot(fig, use_container_width=True)

        explanation = result["explanation"]
        st.markdown("#### Contribution breakdown")
        st.plotly_chart(
            contribution_bar(
                explanation["increases_risk"], explanation["decreases_risk"]
            ),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔺 Increases churn risk")
            if explanation["increases_risk"]:
                st.table(pd.DataFrame(explanation["increases_risk"]))
            else:
                st.write("None.")
        with c2:
            st.markdown("#### 🔻 Reduces churn risk")
            if explanation["decreases_risk"]:
                st.table(pd.DataFrame(explanation["decreases_risk"]))
            else:
                st.write("None.")
