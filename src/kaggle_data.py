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
from pathlib import Path
from typing import List, Tuple

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
# Feature engineering (condensed)
###############################################################################

# Column groups – hard-coded names taken from the original notebook. If a column
# is missing we silently ignore it.
_VITAL_MAP = {
    "triage_vital_hr": "hr",
    "triage_vital_sbp": "sbp",
    "triage_vital_dbp": "dbp",
    "triage_vital_rr": "rr",
    "triage_vital_o2": "spo2",
    "triage_vital_temp": "temp",
}

_LAB_MAP = {
    "glucose_last": "glucose",
    "creatinine_last": "creatinine",
    "bun_last": "bun",
    "hemoglobin_last": "hemoglobin",
    "wbc_last": "wbc",
    "sodium_last": "sodium",
    "potassium_last": "potassium",
}


def feature_engineer_kaggle_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return a *numerical-only* dataframe ready for modelling + list of feature names.

    This is **much** simpler than the original experimental script – the goal is
    to integrate with the current FL pipeline quickly.  We keep:
      • numeric vitals (6)
      • age
      • simple pain score (numeric) if present

    Everything is converted to `float32` and NaNs filled with median.
    """

    # ----- target mapping to 3-class ------------------------------
    if "esi" not in df.columns:
        raise ValueError("Expected `esi` column in Kaggle dataframe")

    df = df.copy()
    df["doğru triyaj_encoded"] = (
        df["esi"].map({1: 2, 2: 2, 3: 1, 4: 0, 5: 0}).astype("Int64")
    )
    df.dropna(subset=["doğru triyaj_encoded"], inplace=True)

    feature_cols: List[str] = []

    # ----- vitals -------------------------------------------------
    for raw, new in _VITAL_MAP.items():
        if raw in df.columns:
            df[new] = pd.to_numeric(df[raw], errors="coerce")
            feature_cols.append(new)

    # age ----------------------------------------------------------
    df["yaş"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    feature_cols.append("yaş")

    # pain ---------------------------------------------------------
    if "triage_pain" in df.columns:
        df["pain_score"] = pd.to_numeric(df["triage_pain"], errors="coerce")
        feature_cols.append("pain_score")

    # labs (basic numeric, no abnormal flags for now) --------------
    for raw, new in _LAB_MAP.items():
        if raw in df.columns:
            df[new] = pd.to_numeric(df[raw], errors="coerce")
            feature_cols.append(new)

    # fill missing with median
    df[feature_cols] = df[feature_cols].apply(lambda s: s.fillna(s.median()))

    # cast for torch (memory)
    df[feature_cols] = df[feature_cols].astype("float32")

    return df, feature_cols 