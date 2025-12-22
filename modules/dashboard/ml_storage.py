from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import pickle

from django.conf import settings


@dataclass
class ModelBundle:
    model: Any
    features: list[str]
    target: str
    scaler: Any = None
    encoders: Optional[Dict[str, Any]] = None
    target_encoder: Any = None
    problem_type: Optional[str] = None
    impute_values: Optional[Dict[str, float]] = None
    missing_token: str = '__MISSING__'


def _models_dir() -> Path:
    p = Path(settings.EXPORT_DIR) / 'models'
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_bundle(session_key: str, bundle: ModelBundle) -> str:
    path = _models_dir() / f"{session_key}__model_bundle.pkl"
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return str(path)


def load_bundle(path: str) -> ModelBundle:
    with open(path, 'rb') as f:
        return pickle.load(f)
