"""Risk indicator and KPI card components."""
from __future__ import annotations

import streamlit as st

# Colour palette mapped to each risk segment.
_RISK_COLORS: dict[str, str] = {
    "Low Risk": "#16a34a",     # green
    "Medium Risk": "#f59e0b",  # amber
    "High Risk": "#dc2626",    # red
}


def risk_color(segment: str) -> str:
    """Return the brand colour for a risk segment."""
    return _RISK_COLORS.get(segment, "#6b7280")


def kpi_card(label: str, value: str, help_text: str | None = None) -> None:
    """Render a simple KPI metric card."""
    st.metric(label=label, value=value, help=help_text)


def risk_card(probability: float, segment: str) -> None:
    """Render a prominent churn-risk card with a coloured banner and gauge."""
    color = risk_color(segment)
    pct = probability * 100

    st.markdown(
        f"""
        <div style="
            border-radius: 14px;
            padding: 20px 24px;
            background: linear-gradient(135deg, {color}22, {color}05);
            border-left: 6px solid {color};
            margin-bottom: 12px;">
            <div style="font-size: 0.9rem; color: #6b7280; text-transform: uppercase;
                        letter-spacing: 0.05em;">Churn Risk</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: {color};">
                {pct:.1f}%
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {color};">
                {segment}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
