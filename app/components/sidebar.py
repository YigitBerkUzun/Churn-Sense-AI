"""Shared sidebar and customer input form for the dashboard."""
from __future__ import annotations

from typing import Any

import streamlit as st

# US state codes present in the BigML dataset.
_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
]
_AREA_CODES = ["415", "408", "510"]


def render_sidebar() -> None:
    """Render the common sidebar branding and navigation hint."""
    with st.sidebar:
        st.markdown("## 🛡️ ChurnGuard AI")
        st.caption("Customer Churn Prediction & Retention Intelligence")
        st.divider()
        st.markdown(
            "Use the pages above to predict churn, explore SHAP explanations, "
            "review customer insights and generate retention strategies."
        )
        st.divider()
        st.caption("Built with Streamlit · XGBoost · SHAP · FastAPI")


def customer_input_form(key_prefix: str = "form") -> dict[str, Any]:
    """Render a customer attribute form and return the collected values.

    Keys exactly match the raw BigML dataset column names expected by the model.
    """
    st.markdown("##### 👤 Account")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state = st.selectbox("State", _STATES, index=_STATES.index("KS"),
                             key=f"{key_prefix}_state")
    with c2:
        area_code = st.selectbox("Area code", _AREA_CODES, key=f"{key_prefix}_area")
    with c3:
        account_length = st.number_input(
            "Account length", 0, 300, 100, key=f"{key_prefix}_acct"
        )
    with c4:
        service_calls = st.number_input(
            "Customer service calls", 0, 20, 1, key=f"{key_prefix}_svc"
        )

    st.markdown("##### 📦 Plans")
    p1, p2, p3 = st.columns(3)
    with p1:
        intl_plan = st.selectbox("International plan", ["No", "Yes"],
                                key=f"{key_prefix}_intl")
    with p2:
        vmail_plan = st.selectbox("Voice mail plan", ["No", "Yes"],
                                 key=f"{key_prefix}_vmplan")
    with p3:
        vmail_msgs = st.number_input("Number vmail messages", 0, 60, 0,
                                    key=f"{key_prefix}_vmmsg")

    st.markdown("##### ☎️ Usage")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.caption("Day")
        day_min = st.number_input("Day minutes", 0.0, 400.0, 180.0,
                                 key=f"{key_prefix}_daymin")
        day_calls = st.number_input("Day calls", 0, 200, 100,
                                   key=f"{key_prefix}_daycalls")
        day_charge = st.number_input("Day charge", 0.0, 100.0, 30.6,
                                    key=f"{key_prefix}_daychg")
    with d2:
        st.caption("Evening")
        eve_min = st.number_input("Eve minutes", 0.0, 400.0, 200.0,
                                 key=f"{key_prefix}_evemin")
        eve_calls = st.number_input("Eve calls", 0, 200, 100,
                                   key=f"{key_prefix}_evecalls")
        eve_charge = st.number_input("Eve charge", 0.0, 100.0, 17.0,
                                    key=f"{key_prefix}_evechg")
    with d3:
        st.caption("Night")
        night_min = st.number_input("Night minutes", 0.0, 400.0, 200.0,
                                   key=f"{key_prefix}_nightmin")
        night_calls = st.number_input("Night calls", 0, 200, 100,
                                     key=f"{key_prefix}_nightcalls")
        night_charge = st.number_input("Night charge", 0.0, 100.0, 9.0,
                                      key=f"{key_prefix}_nightchg")

    st.markdown("##### 🌍 International usage")
    i1, i2, i3 = st.columns(3)
    with i1:
        intl_min = st.number_input("Intl minutes", 0.0, 60.0, 10.0,
                                  key=f"{key_prefix}_intlmin")
    with i2:
        intl_calls = st.number_input("Intl calls", 0, 30, 4,
                                    key=f"{key_prefix}_intlcalls")
    with i3:
        intl_charge = st.number_input("Intl charge", 0.0, 20.0, 2.7,
                                     key=f"{key_prefix}_intlchg")

    return {
        "State": state,
        "Account length": account_length,
        "Area code": area_code,
        "International plan": intl_plan,
        "Voice mail plan": vmail_plan,
        "Number vmail messages": vmail_msgs,
        "Total day minutes": day_min,
        "Total day calls": day_calls,
        "Total day charge": day_charge,
        "Total eve minutes": eve_min,
        "Total eve calls": eve_calls,
        "Total eve charge": eve_charge,
        "Total night minutes": night_min,
        "Total night calls": night_calls,
        "Total night charge": night_charge,
        "Total intl minutes": intl_min,
        "Total intl calls": intl_calls,
        "Total intl charge": intl_charge,
        "Customer service calls": service_calls,
    }
