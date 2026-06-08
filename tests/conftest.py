"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_customer() -> dict:
    """A representative raw customer record (high-risk profile)."""
    return {
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


@pytest.fixture
def sample_frame(sample_customer: dict) -> pd.DataFrame:
    return pd.DataFrame([sample_customer])
