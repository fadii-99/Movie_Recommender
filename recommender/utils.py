from __future__ import annotations
import numpy as np
import pandas as pd

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def nanmean(series: pd.Series, default: float = 0.0) -> float:
    v = series.dropna()
    return float(v.mean()) if len(v) else default

def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]
