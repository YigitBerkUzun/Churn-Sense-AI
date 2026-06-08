"""Shared utilities: logging, artifact persistence and small helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_artifact(obj: Any, path: str | Path) -> None:
    """Persist any Python object (model, preprocessor, explainer) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path) -> Any:
    """Load a previously persisted artifact, raising a clear error if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. Run `python -m src.train_model` first."
        )
    return joblib.load(path)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to disk as nicely formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
