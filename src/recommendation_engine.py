"""Rule-based retention recommendation engine (BigML Telecom Churn).

Translates a customer's attributes and churn risk into concrete, business-ready
retention actions. Rules are intentionally transparent so that a retention team
can understand and trust every suggestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src import config


@dataclass
class Recommendation:
    """A single actionable retention suggestion."""

    title: str
    detail: str
    priority: str  # "High" | "Medium" | "Low"

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "detail": self.detail, "priority": self.priority}


@dataclass
class RecommendationContext:
    """Inputs the rules reason over."""

    churn_probability: float
    risk_segment: str
    customer: dict[str, Any]
    is_high_value: bool = False
    recommendations: list[Recommendation] = field(default_factory=list)


def _total_charge(customer: dict[str, Any]) -> float:
    """Sum the per-period charges available on the customer record."""
    return float(
        sum(float(customer.get(col, 0) or 0) for col in config.CHARGE_COLUMNS)
    )


def _high_value(customer: dict[str, Any]) -> bool:
    """Heuristic: long-tenured, high-spending customers are high value."""
    account_length = float(customer.get("Account length", 0) or 0)
    return _total_charge(customer) >= 60 or account_length >= 150


def generate_recommendations(
    customer: dict[str, Any],
    churn_probability: float,
    risk_segment: str,
) -> list[dict[str, str]]:
    """Apply retention rules and return ordered recommendations."""
    ctx = RecommendationContext(
        churn_probability=churn_probability,
        risk_segment=risk_segment,
        customer=customer,
        is_high_value=_high_value(customer),
    )
    recs = ctx.recommendations

    high_risk = churn_probability >= config.RISK_THRESHOLDS["high"]
    medium_risk = churn_probability >= config.RISK_THRESHOLDS["medium"]

    service_calls = int(float(customer.get("Customer service calls", 0) or 0))
    intl_plan = customer.get("International plan") == "Yes"
    has_voicemail = customer.get("Voice mail plan") == "Yes"
    day_minutes = float(customer.get("Total day minutes", 0) or 0)

    # Rule 1: High churn risk + high value -> loyalty discount.
    if high_risk and ctx.is_high_value:
        recs.append(
            Recommendation(
                title="Offer a loyalty discount",
                detail=(
                    "This is a high-value customer at high risk of churning. "
                    "Proactively offer a 15-20% loyalty discount or bill credit "
                    "to protect recurring revenue."
                ),
                priority="High",
            )
        )

    # Rule 2: Many customer service calls -> the strongest churn driver here.
    if service_calls >= config.SERVICE_CALL_RISK_THRESHOLD:
        recs.append(
            Recommendation(
                title="Escalate to retention specialist",
                detail=(
                    f"Customer has contacted support {service_calls} times — a "
                    "very strong churn signal. Route to a senior retention "
                    "agent to resolve the underlying issue and apologise with a "
                    "goodwill gesture."
                ),
                priority="High",
            )
        )
    elif service_calls >= 3:
        recs.append(
            Recommendation(
                title="Proactive support follow-up",
                detail=(
                    "Repeated support contacts detected. Trigger a proactive "
                    "follow-up call to confirm the problem is fully resolved."
                ),
                priority="Medium",
            )
        )

    # Rule 3: International plan -> often expensive and churn-prone.
    if intl_plan:
        recs.append(
            Recommendation(
                title="Review international plan pricing",
                detail=(
                    "International-plan subscribers churn more often. Offer a "
                    "better-value international bundle or usage cap to reduce "
                    "bill shock."
                ),
                priority="High" if high_risk else "Medium",
            )
        )

    # Rule 4: Heavy daytime usage but no voicemail -> upsell engagement add-ons.
    if not has_voicemail and day_minutes >= 250:
        recs.append(
            Recommendation(
                title="Upsell value-added plans",
                detail=(
                    "Heavy daytime caller without a voicemail plan. Offer a free "
                    "trial of voicemail / productivity add-ons to deepen "
                    "engagement and stickiness."
                ),
                priority="Low",
            )
        )

    # Rule 5: Medium/high risk new-ish customer -> onboarding outreach.
    if medium_risk and float(customer.get("Account length", 0) or 0) <= 50:
        recs.append(
            Recommendation(
                title="Strengthen onboarding",
                detail=(
                    "Newer account showing early churn signals. Trigger a "
                    "personalised onboarding / check-in call to build loyalty."
                ),
                priority="Medium",
            )
        )

    # Fallback: low-risk, retained customer.
    if not recs:
        recs.append(
            Recommendation(
                title="Maintain & nurture",
                detail=(
                    "Customer is currently low risk. Keep engagement high with "
                    "periodic value check-ins and loyalty rewards."
                ),
                priority="Low",
            )
        )

    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: priority_rank[r.priority])
    return [r.as_dict() for r in recs]
