"""Plotly chart helpers for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.risk_card import risk_color


def churn_distribution(df: pd.DataFrame, target: str = "Churn") -> go.Figure:
    """Pie chart of churned vs retained customers."""
    counts = df[target].map({1: "Churned", 0: "Retained"}).value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.5,
        color=counts.index,
        color_discrete_map={"Churned": "#dc2626", "Retained": "#16a34a"},
        title="Churn Distribution",
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def churn_by_category(df: pd.DataFrame, column: str, target: str = "Churn") -> go.Figure:
    """Grouped bar chart of churn rate by a categorical column."""
    grouped = (
        df.groupby(column)[target].mean().mul(100).round(1).reset_index(name="churn_rate")
    )
    fig = px.bar(
        grouped,
        x=column,
        y="churn_rate",
        text="churn_rate",
        title=f"Churn Rate by {column} (%)",
        color="churn_rate",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
    return fig


def numeric_distribution(df: pd.DataFrame, column: str, target: str = "Churn") -> go.Figure:
    """Overlaid histogram of a numeric feature split by churn."""
    plot_df = df.copy()
    plot_df["Status"] = plot_df[target].map({1: "Churned", 0: "Retained"})
    fig = px.histogram(
        plot_df,
        x=column,
        color="Status",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Churned": "#dc2626", "Retained": "#16a34a"},
        title=f"{column} Distribution by Churn",
    )
    return fig


def feature_importance_bar(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of SHAP global feature importance."""
    df = importance_df.sort_values("importance")
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Global Feature Importance (mean |SHAP|)",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Impact")
    return fig


def contribution_bar(increases: list[dict], decreases: list[dict]) -> go.Figure:
    """Diverging bar chart of local SHAP contributions for one customer."""
    rows = [
        {"feature": d["feature"], "impact": d["impact"], "direction": "Increases risk"}
        for d in increases
    ] + [
        {"feature": d["feature"], "impact": d["impact"], "direction": "Decreases risk"}
        for d in decreases
    ]
    df = pd.DataFrame(rows).sort_values("impact")
    fig = px.bar(
        df,
        x="impact",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "Increases risk": "#dc2626",
            "Decreases risk": "#16a34a",
        },
        title="Why this customer? (local SHAP contributions)",
    )
    fig.update_layout(yaxis_title="", xaxis_title="SHAP value")
    return fig


def gauge(probability: float, segment: str) -> go.Figure:
    """Speedometer-style gauge for churn probability.

    The coloured bands are derived from ``config.RISK_THRESHOLDS`` so the gauge
    always matches the configured Low/Medium/High segmentation.
    """
    from src import config

    med = config.RISK_THRESHOLDS["medium"] * 100
    high = config.RISK_THRESHOLDS["high"] * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": f"Churn Risk · {segment}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color(segment)},
                "steps": [
                    {"range": [0, med], "color": "#dcfce7"},
                    {"range": [med, high], "color": "#fef9c3"},
                    {"range": [high, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=50, b=10))
    return fig
