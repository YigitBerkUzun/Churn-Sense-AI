"""Pydantic request/response schemas for the ChurnGuard AI API.

Mirrors the BigML Telecom Churn feature set.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Raw customer attributes as found in the BigML Telecom Churn dataset."""

    State: str = Field("KS", description="US state code, e.g. 'KS'")
    Account_length: int = Field(128, ge=0, alias="Account length")
    Area_code: str = Field("415", alias="Area code")
    International_plan: Literal["Yes", "No"] = Field("No", alias="International plan")
    Voice_mail_plan: Literal["Yes", "No"] = Field("No", alias="Voice mail plan")
    Number_vmail_messages: int = Field(0, ge=0, alias="Number vmail messages")
    Total_day_minutes: float = Field(180.0, ge=0, alias="Total day minutes")
    Total_day_calls: int = Field(100, ge=0, alias="Total day calls")
    Total_day_charge: float = Field(30.6, ge=0, alias="Total day charge")
    Total_eve_minutes: float = Field(200.0, ge=0, alias="Total eve minutes")
    Total_eve_calls: int = Field(100, ge=0, alias="Total eve calls")
    Total_eve_charge: float = Field(17.0, ge=0, alias="Total eve charge")
    Total_night_minutes: float = Field(200.0, ge=0, alias="Total night minutes")
    Total_night_calls: int = Field(100, ge=0, alias="Total night calls")
    Total_night_charge: float = Field(9.0, ge=0, alias="Total night charge")
    Total_intl_minutes: float = Field(10.0, ge=0, alias="Total intl minutes")
    Total_intl_calls: int = Field(4, ge=0, alias="Total intl calls")
    Total_intl_charge: float = Field(2.7, ge=0, alias="Total intl charge")
    Customer_service_calls: int = Field(1, ge=0, alias="Customer service calls")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "State": "OH",
                "Account length": 84,
                "Area code": "408",
                "International plan": "Yes",
                "Voice mail plan": "No",
                "Number vmail messages": 0,
                "Total day minutes": 299.4,
                "Total day calls": 71,
                "Total day charge": 50.9,
                "Total eve minutes": 61.9,
                "Total eve calls": 88,
                "Total eve charge": 5.26,
                "Total night minutes": 196.9,
                "Total night calls": 89,
                "Total night charge": 8.86,
                "Total intl minutes": 6.6,
                "Total intl calls": 7,
                "Total intl charge": 1.78,
                "Customer service calls": 5,
            }
        },
    }

    def to_raw(self) -> dict:
        """Return the record keyed by the original dataset column names."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_segment: str
    will_churn: bool


class FeatureImpact(BaseModel):
    feature: str
    impact: float


class Explanation(BaseModel):
    base_value: float
    increases_risk: list[FeatureImpact]
    decreases_risk: list[FeatureImpact]


class ExplainResponse(PredictionResponse):
    explanation: Explanation


class RecommendationItem(BaseModel):
    title: str
    detail: str
    priority: str


class RecommendResponse(PredictionResponse):
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    # `model_loaded` would otherwise clash with Pydantic's protected "model_" namespace.
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    version: str
