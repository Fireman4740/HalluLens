from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_bool(value: Any) -> float:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return np.nan
    txt = str(value).strip().lower()
    if txt in {"true", "1", "yes"}:
        return True
    if txt in {"false", "0", "no"}:
        return False
    return np.nan


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_int(value: Any) -> float:
    try:
        return int(value)
    except Exception:
        return np.nan


def sorted_unique(series: pd.Series) -> list[Any]:
    values = [v for v in series.dropna().unique().tolist()]
    if not values:
        return []
    try:
        return sorted(values)
    except Exception:
        return sorted(values, key=lambda item: str(item))


def apply_multiselect_filter(df: pd.DataFrame, column: str, selected: list[Any]) -> pd.DataFrame:
    if not selected:
        return df
    return df[df[column].isin(selected)]


def option_index(options: list[Any], preferred: Any, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    if not options:
        return 0
    return min(max(fallback, 0), len(options) - 1)

