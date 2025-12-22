from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import pandas as pd

from django.conf import settings

from modules.data_loader import load_data
from modules.data_profiler import profile_dataframe
from utils.validation import detect_problem_type


SESSION_KEY_FILE_PATH = 'data.file_path'
SESSION_KEY_SEPARATOR = 'data.separator'
SESSION_KEY_TARGET = 'ml.target'
SESSION_KEY_FEATURES = 'ml.features'
SESSION_KEY_MODEL_BUNDLE_PATH = 'ml.model_bundle_path'
SESSION_KEY_MANUAL_TYPES = 'data.manual_types'
SESSION_KEY_SAMPLING = 'data.sampling'
SESSION_KEY_LAST_RESULTS = 'ui.last_results'


@dataclass(frozen=True)
class DataContext:
    df: pd.DataFrame
    profile: Dict[str, Any]
    problem_type: Optional[str]
    problem_description: Optional[str]


def ensure_session_key(request) -> str:
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def get_loaded_file_path(request) -> Optional[str]:
    return request.session.get(SESSION_KEY_FILE_PATH)


def set_loaded_file_path(request, file_path: str, separator: str) -> None:
    request.session[SESSION_KEY_FILE_PATH] = file_path
    request.session[SESSION_KEY_SEPARATOR] = separator


def clear_session_state(request) -> None:
    for key in [
        SESSION_KEY_FILE_PATH,
        SESSION_KEY_SEPARATOR,
        SESSION_KEY_TARGET,
        SESSION_KEY_FEATURES,
        SESSION_KEY_MODEL_BUNDLE_PATH,
        SESSION_KEY_MANUAL_TYPES,
        SESSION_KEY_SAMPLING,
        SESSION_KEY_LAST_RESULTS,
    ]:
        if key in request.session:
            del request.session[key]


def save_uploaded_bytes(request, filename: str, content: bytes) -> str:
    session_key = ensure_session_key(request)
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = filename.replace('..', '.').replace('/', '_').replace('\\', '_')
    path = upload_dir / f"{session_key}__{safe_name}"
    path.write_bytes(content)
    return str(path)


def apply_manual_types(df: pd.DataFrame, manual_types: Dict[str, str]) -> pd.DataFrame:
    df2 = df.copy()
    for col, declared in (manual_types or {}).items():
        if col not in df2.columns:
            continue
        if declared == 'numeric':
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
        elif declared == 'date':
            df2[col] = pd.to_datetime(df2[col], errors='coerce')
        elif declared == 'boolean':
            # Conversion prudente
            df2[col] = df2[col].astype('object').map(lambda x: True if str(x).lower() in {'true','1','yes','y'} else (False if str(x).lower() in {'false','0','no','n'} else pd.NA))
        elif declared in {'categorical', 'text'}:
            df2[col] = df2[col].astype('string')
    return df2


def sample_dataframe(
    df: pd.DataFrame,
    enabled: bool,
    method: str,
    n_rows: int,
    random_state: int,
    target: Optional[str],
) -> pd.DataFrame:
    if not enabled:
        return df
    if n_rows is None or n_rows <= 0 or n_rows >= len(df):
        return df
    rs = 42 if random_state is None else int(random_state)

    if method == 'stratified' and target and target in df.columns:
        # Stratifié uniquement si cible pas trop manquante
        s = df[target]
        not_na = s.notna()
        df2 = df[not_na]
        if len(df2) == 0:
            return df.sample(n=n_rows, random_state=rs)
        frac = n_rows / len(df2)
        if frac >= 1.0:
            return df2
        return df2.groupby(target, group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=rs))

    return df.sample(n=n_rows, random_state=rs)


def get_data_context(request) -> Tuple[Optional[DataContext], Optional[str]]:
    file_path = get_loaded_file_path(request)
    if not file_path:
        return None, None

    sep = request.session.get(SESSION_KEY_SEPARATOR, ',')
    df, err = load_data(file_path, separator=sep)
    if df is None:
        return None, err

    manual_types = request.session.get(SESSION_KEY_MANUAL_TYPES) or {}
    df = apply_manual_types(df, manual_types)

    # Sampling (gros datasets)
    sampling = request.session.get(SESSION_KEY_SAMPLING) or {}
    df = sample_dataframe(
        df,
        enabled=bool(sampling.get('enabled')),
        method=str(sampling.get('method') or 'random'),
        n_rows=sampling.get('n_rows') or 0,
        random_state=sampling.get('random_state') or 42,
        target=request.session.get(SESSION_KEY_TARGET),
    )

    prof = profile_dataframe(df)

    problem_type = None
    problem_desc = None
    target = request.session.get(SESSION_KEY_TARGET)
    if target and target in df.columns:
        problem_type, problem_desc = detect_problem_type(df[target])

    return DataContext(df=df, profile=prof, problem_type=problem_type, problem_description=problem_desc), None


def set_last_results(request, payload: Dict[str, Any]) -> None:
    request.session[SESSION_KEY_LAST_RESULTS] = payload


def get_last_results(request) -> Optional[Dict[str, Any]]:
    return request.session.get(SESSION_KEY_LAST_RESULTS)


def export_session_payload(request, ctx: Optional[DataContext]) -> Dict[str, Any]:
    return {
        'meta': {
            'app': 'DataAnalyzer V2 (Django)',
        },
        'data': {
            'file_path': request.session.get(SESSION_KEY_FILE_PATH),
            'separator': request.session.get(SESSION_KEY_SEPARATOR),
            'n_rows': ctx.df.shape[0] if ctx else None,
            'n_cols': ctx.df.shape[1] if ctx else None,
        },
        'selection': {
            'target': request.session.get(SESSION_KEY_TARGET),
            'features': request.session.get(SESSION_KEY_FEATURES) or [],
            'manual_types': request.session.get(SESSION_KEY_MANUAL_TYPES) or {},
            'sampling': request.session.get(SESSION_KEY_SAMPLING) or {},
        },
        'last_results': request.session.get(SESSION_KEY_LAST_RESULTS) or {},
    }
