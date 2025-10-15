"""
Utilities for loading and feature-engineering the Kaggle "Hospital Triage & Patient History" dataset so that the rest
of the codebase can treat it just like the in-house `triaj_data.csv`.

The implementation is a trimmed-down, reusable version of the logic that lived in
`experimental/kaggle_enhanced_final_fix_v2.py`.

Usage
-----
from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data

kaggle_df_raw = load_kaggle_triage_data()
kaggle_df_fe  = feature_engineer_kaggle_data(kaggle_df_raw.copy())
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional heavy deps – we import lazily so that the rest of the repo still works
# even if they are missing.
try:
    import kagglehub  # type: ignore
    import pyreadr    # type: ignore
except ImportError:  # pragma: no cover – only needed if user wants auto-download.
    kagglehub = None  # type: ignore
    pyreadr = None   # type: ignore


###############################################################################
# Public helpers
###############################################################################

_KAGGLE_CSV = Path("src/kaggle_triage_data.csv")
_KAGGLE_DATASET = "maalona/hospital-triage-and-patient-history-data"


def load_kaggle_triage_data(force_download: bool = False) -> pd.DataFrame:
    """Return the raw Kaggle triage dataframe (caches a CSV locally)."""
    if _KAGGLE_CSV.exists() and not force_download:
        return pd.read_csv(_KAGGLE_CSV)

    if kagglehub is None or pyreadr is None:
        raise ImportError(
            "kagglehub / pyreadr not installed. Install with `pip install kagglehub pyreadr` "
            "or provide `src/kaggle_triage_data.csv` manually."
        )

    # Download (large – will hit network once and cache under ~/.cache/kagglehub)
    print("📥  Downloading Kaggle hospital triage dataset … (may take a minute)")
    dataset_path: Path = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
    rdata_path = dataset_path / "5v_cleandf.rdata"
    result = pyreadr.read_r(rdata_path)
    df = result[list(result.keys())[0]]  # first object in the RData file

    # Persist for future runs
    _KAGGLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_KAGGLE_CSV, index=False)
    print(f"✅  Saved CSV cache → {_KAGGLE_CSV.relative_to(Path.cwd())}")
    return df


###############################################################################
# Feature engineering (condensed, accuracy-first like earlier baseline)
###############################################################################


@dataclass
class KaggleFeatureSpec:
    features: List[str]
    groups: Dict[str, List[str]]


def feature_engineer_kaggle_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, KaggleFeatureSpec]:
    """Condensed feature engineering used in the earlier 88% baseline.

    Keeps a small, stable set of numerical features:
      - vitals: HR, SBP, DBP, RR, SpO2, Temp (renamed)
      - age (as `yaş` for compatibility with analysis code)
      - pain score if present
      - core labs: *_last values (keeps original names with `_last` suffix)

    Fills missing values with medians and casts to float32.
    Returns a KaggleFeatureSpec with simple pathway grouping.
    """

    if "esi" not in df.columns:
        raise ValueError("Expected `esi` column in Kaggle dataframe")

    df = df.copy()
    df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
    df.dropna(subset=["esi"], inplace=True)
    esi_vals = df["esi"].astype(int).clip(lower=1, upper=5)
    df["esi_5class_encoded"] = (5 - esi_vals).astype("Int64")

    # Carry raw gender for fairness if available
    if "gender" in df.columns and "gender_original" not in df.columns:
        df["gender_original"] = df["gender"].fillna("Unknown").astype(str)

    # Vital mapping (rename to concise names)
    VITAL_MAP = {
        "triage_vital_hr": "hr",
        "triage_vital_sbp": "sbp",
        "triage_vital_dbp": "dbp",
        "triage_vital_rr": "rr",
        "triage_vital_o2": "spo2",
        "triage_vital_temp": "temp",
    }
    # Core labs (use the *_last columns directly to cooperate with downstream filters)
    LAB_KEEP = [
        "glucose_last",
        "creatinine_last",
        "bun_last",
        "hemoglobin_last",
        "wbc_last",
        "sodium_last",
        "potassium_last",
    ]

    features: List[str] = []
    # Vitals
    for raw, new in VITAL_MAP.items():
        if raw in df.columns:
            df[new] = pd.to_numeric(df[raw], errors="coerce")
            features.append(new)

    # Age → `yaş` for compatibility with fairness utils
    df["yaş"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    features.append("yaş")

    # Pain score
    if "triage_pain" in df.columns:
        df["pain_score"] = pd.to_numeric(df["triage_pain"], errors="coerce")
        features.append("pain_score")

    # Labs
    for col in LAB_KEEP:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            features.append(col)

    # Fill missing per-column median and cast
    for col in features:
        series = df[col]
        med = float(np.nanmedian(series.to_numpy())) if series.notna().any() else 0.0
        df[col] = series.fillna(med).astype(np.float32)

    # Build simple groups compatible with the model
    groups: Dict[str, List[str]] = {k: [] for k in ("vital", "symptom", "risk", "context", "lab", "interaction")}
    for col in features:
        if col in {"hr", "sbp", "dbp", "rr", "spo2", "temp"}:
            groups["vital"].append(col)
        elif col == "pain_score":
            groups["symptom"].append(col)
        elif col == "yaş":
            groups["context"].append(col)
        elif col.endswith("_last"):
            groups["lab"].append(col)
        else:
            groups["risk"].append(col)

    # Final dataframe order: features + target + fairness aux (if present)
    extra = ["esi_5class_encoded"]
    if "gender_original" in df.columns:
        extra.append("gender_original")
    out_cols = features + extra
    df = df[out_cols]

    spec = KaggleFeatureSpec(features=features, groups=groups)
    return df, spec
