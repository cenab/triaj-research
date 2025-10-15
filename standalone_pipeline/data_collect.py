#!/usr/bin/env python3
"""
Data Collection & Readiness Pipeline for Triage (centralized + federated)

What it does
------------
1) Loads data from:
   - Kaggle "Hospital Triage & Patient History" (default), or
   - one or more local CSVs (glob), each optionally tagged with site_id.
2) Harmonizes columns, engineers features (vitals/labs/age/pain), encodes ESI (5->0..4).
3) Runs QC gates (missingness, plausibility, label sanity) and writes per-site
   Data Readiness Reports (JSON).
4) Produces:
   - data/processed/triage_processed.csv (unified, numeric features + target + site_id)
   - data/splits/centralized_splits.json  (train/val/test indices)
   - data/splits/fl_splits.json           (Dirichlet non-IID client splits)

Usage
-----
# Default: Kaggle source (requires `kagglehub` + `pyreadr` once, or cached CSV)
python -m standalone_pipeline.data_collect

# Local CSVs (comma-separated globs), assign site_id by filename stem if missing
python -m standalone_pipeline.data_collect --local-csv "data/siteA*.csv,data/siteB*.csv"

# Control splits and QC thresholds
python -m standalone_pipeline.data_collect --clients 5 --dirichlet-alpha 1.0 --qc-miss 0.4

Notes
-----
- If you already have src/kaggle_triage_data.csv, Kaggle deps are not required.
- Outputs are deterministic (seed=42).
"""

from __future__ import annotations
import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

# ---------- Optional imports from your repo if available ----------
try:
    # Your trimmed Kaggle loader + FE (fast path)
    from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
except Exception:
    load_kaggle_triage_data = None   # type: ignore
    feature_engineer_kaggle_data = None  # type: ignore

# Conservative physiologic plausibility bounds (edit if your sites differ)
PHYSIO_BOUNDS = {
    "hr":      (20, 260),     # bpm
    "sbp":     (50, 260),     # mmHg
    "dbp":     (25, 150),     # mmHg
    "rr":      (6,  60),      # breaths/min
    "spo2":    (50, 100),     # %
    "temp":    (30, 43),      # C
    "glucose": (20, 800),     # mg/dL or mmol/L converted—KEEP UNIT CONSISTENT PER SITE
    "creatinine": (0.1, 15),  # mg/dL
    "bun":     (1, 200),      # mg/dL
    "hemoglobin": (3, 22),    # g/dL
    "wbc":     (0.5, 200),    # 10^9/L (or K/µL, ensure consistent)
    "sodium":  (110, 170),    # mmol/L
    "potassium": (2, 8),      # mmol/L
    "age":     (0, 110)       # years
}

# Column maps aligning with your repo’s feature groups
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

# Fairness buckets
AGE_BINS   = [0, 18, 35, 50, 65, np.inf]
AGE_LABELS = ["<18", "18-34", "35-49", "50-64", "65+"]

OUT_BASE = Path("data")
OUT_PROC = OUT_BASE / "processed"
OUT_SPLT = OUT_BASE / "splits"
OUT_REP  = OUT_BASE / "reports"

RNG = default_rng(42)


def _ensure_dirs() -> None:
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    OUT_SPLT.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)


def _read_local_csvs(globs: List[str]) -> pd.DataFrame:
    """Resolve local CSV inputs from comma-separated patterns or paths.

    - Supports absolute and relative paths.
    - Supports wildcard patterns (e.g., data/*.csv) including absolute ones.
    - De-duplicates matches and filters to files only.
    """
    files: List[Path] = []
    seen: set = set()
    for g in globs:
        if not g:
            continue
        # Expand user (~) and env vars
        g_exp = os.path.expandvars(os.path.expanduser(g))

        # Use glob so absolute patterns are supported; if it's an exact path,
        # glob.glob will return [path] if it exists.
        matches = glob.glob(g_exp, recursive=True)

        # If no matches but it's an existing file path, include directly
        if not matches and os.path.isfile(g_exp):
            matches = [g_exp]

        for m in matches:
            p = Path(m)
            try:
                # Resolve to avoid duplicate paths differing by relative/absolute
                rp = p.resolve()
            except Exception:
                rp = p
            if rp.is_file() and str(rp) not in seen:
                seen.add(str(rp))
                files.append(rp)

    if not files:
        raise FileNotFoundError("No local CSV files matched the provided patterns.")

    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        if "site_id" not in df.columns:
            # Infer site_id from filename stem prefix before first non-alpha char
            site = f.stem.split("_")[0]
            df["site_id"] = site
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def _harmonize_minimal(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Minimal FE to match your model groups when kaggle_data module is unavailable."""
    if "esi" not in df.columns:
        raise ValueError("Expected 'esi' column in data for target mapping.")

    out = df.copy()

    # Map vitals
    for raw, new in _VITAL_MAP.items():
        if raw in out.columns:
            out[new] = pd.to_numeric(out[raw], errors="coerce")

    # Age
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    elif "Age" in out.columns:
        out["age"] = pd.to_numeric(out["Age"], errors="coerce")
    else:
        out["age"] = np.nan
    out["yaş"] = out["age"]  # keep Turkish name used in your code

    # Pain
    if "triage_pain" in out.columns:
        out["pain_score"] = pd.to_numeric(out["triage_pain"], errors="coerce")

    # Labs
    for raw, new in _LAB_MAP.items():
        if raw in out.columns:
            out[new] = pd.to_numeric(out[raw], errors="coerce")

    # Target: 0..4 => ESI5..ESI1
    out["esi"] = pd.to_numeric(out["esi"], errors="coerce")
    out = out.dropna(subset=["esi"])
    esi_vals = out["esi"].astype(int).clip(1, 5)
    out["esi_5class_encoded"] = (5 - esi_vals).astype("Int64")

    # Feature set (numeric only)
    feature_cols: List[str] = []
    feature_cols += [c for c in ["hr", "sbp", "dbp", "rr", "spo2", "temp"] if c in out.columns]
    feature_cols += ["yaş"] if "yaş" in out.columns else []
    feature_cols += ["pain_score"] if "pain_score" in out.columns else []
    feature_cols += [c for c in ["glucose", "creatinine", "bun", "hemoglobin", "wbc", "sodium", "potassium"] if c in out.columns]

    # Impute with median (train-set medians will be enforced later; this is pre-QC)
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[feature_cols] = out[feature_cols].apply(lambda s: s.fillna(s.median()))

    # Cast float32 for memory parity
    out[feature_cols] = out[feature_cols].astype("float32")

    # site_id fallback
    if "site_id" not in out.columns:
        out["site_id"] = "site_0"

    return out, feature_cols


def _feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prefer your repo's feature_engineer_kaggle_data; otherwise use minimal path."""
    if feature_engineer_kaggle_data is not None:
        fe, spec = feature_engineer_kaggle_data(df.copy())
        if "site_id" not in fe.columns:
            fe["site_id"] = df.get("site_id", "site_0")
        return fe, spec.features
    return _harmonize_minimal(df)


def _qc_site(df: pd.DataFrame,
             features: List[str],
             qc_missing_thresh: float = 0.40) -> Dict:
    """Run QC gates and return a site readiness report dict."""
    n = len(df)
    miss = df[features].isna().mean().to_dict()

    # Physiologic plausibility
    out_of_range = {}
    for col, (lo, hi) in PHYSIO_BOUNDS.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            frac = float(((s < lo) | (s > hi)).mean())
            out_of_range[col] = frac

    # Label sanity
    if "esi_5class_encoded" not in df.columns:
        raise ValueError("esi_5class_encoded missing after FE.")
    y = df["esi_5class_encoded"].astype(int).to_numpy()
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    # Fairness buckets availability
    age = df.get("yaş", np.nan)
    age_groups = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    gender = df.get("gender_original", df.get("gender", df.get("Gender", pd.Series(["Unknown"] * n))))
    gender = gender.fillna("Unknown").astype(str)
    if "gender_male" in df.columns:
        gender = np.where(df["gender_male"] > 0.5, "Male", "Female")

    # Compute simple fairness sizes
    grp_counts_age = age_groups.value_counts(dropna=False).to_dict()
    grp_counts_gender = pd.Series(gender).value_counts(dropna=False).to_dict()

    readiness = {
        "rows": n,
        "features": features,
        "missingness": miss,
        "plausibility_oob_fraction": out_of_range,
        "label_distribution_esi5_to_esi1": label_dist,
        "fairness_group_sizes": {
            "age_group": {str(k): int(v) for k, v in grp_counts_age.items()},
            "gender": {str(k): int(v) for k, v in grp_counts_gender.items()},
        },
        "thresholds": {
            "missingness_max": qc_missing_thresh,
            "oob_warn_if_gt": 0.01
        }
    }

    # Simple pass/fail flags
    readiness["gates"] = {
        "missingness_ok": all((v or 0.0) <= qc_missing_thresh for v in miss.values()),
        "plausibility_ok": all((v or 0.0) <= 0.01 for v in out_of_range.values()),
        "labels_ok": n > 0 and sum(label_dist.values()) == n
    }
    readiness["ready"] = all(readiness["gates"].values())
    return readiness


def _centralized_splits(df: pd.DataFrame,
                        target_col: str = "esi_5class_encoded",
                        test_size: float = 0.2,
                        val_size: float = 0.2,
                        seed: int = 42) -> Dict[str, List[int]]:
    """Return 60/20/20 stratified indices for train/val/test."""
    idx = np.arange(len(df))
    y = df[target_col].astype(int).to_numpy()
    # First: holdout test (20%)
    idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    # Then split train/val (20% of total -> 25% of trainval)
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size/(1 - test_size),
                                          random_state=seed, stratify=y_trainval)
    return {
        "train": sorted(map(int, idx_train.tolist())),
        "val":   sorted(map(int, idx_val.tolist())),
        "test":  sorted(map(int, idx_test.tolist()))
    }


def _federated_splits(y: np.ndarray,
                      clients: int = 5,
                      alpha: float = 1.0,
                      seed: int = 42) -> Dict[str, List[int]]:
    """Dirichlet over labels → non-IID indices per client (like your fl_splits)."""
    rng = default_rng(seed)
    indices = np.arange(len(y))
    classes = np.unique(y)
    class_indices = {c: indices[y == c] for c in classes}
    buckets: List[List[int]] = [[] for _ in range(clients)]
    for c in classes:
        idx = rng.permutation(class_indices[c])
        props = rng.dirichlet([alpha] * clients)
        counts = (props * len(idx)).astype(int)
        while counts.sum() < len(idx):
            counts[rng.integers(0, clients)] += 1
        while counts.sum() > len(idx):
            j = rng.integers(0, clients)
            if counts[j] > 0:
                counts[j] -= 1
        start = 0
        for k in range(clients):
            end = start + counts[k]
            buckets[k].extend(idx[start:end].tolist())
            start = end
    return {str(i): sorted(map(int, b)) for i, b in enumerate(buckets)}


def main():
    ap = argparse.ArgumentParser(description="Collect, harmonize, QC, and split triage data.")
    ap.add_argument("--local-csv", type=str, default="", help="Comma-separated globs for local CSVs (overrides Kaggle).")
    ap.add_argument("--clients", type=int, default=5, help="Number of FL clients.")
    ap.add_argument("--dirichlet-alpha", type=float, default=1.0, help="Dirichlet alpha for non-IID FL splits.")
    ap.add_argument("--qc-miss", type=float, default=0.40, help="Max allowed missingness per feature.")
    args = ap.parse_args()

    _ensure_dirs()

    # --------- Load raw ----------
    if args.local_csv.strip():
        globs = [g.strip() for g in args.local_csv.split(",") if g.strip()]
        raw = _read_local_csvs(globs)
        source = "local_csv"
    else:
        if load_kaggle_triage_data is None:
            raise RuntimeError("Kaggle path requested but kaggle_data module unavailable and no --local-csv specified.")
        raw = load_kaggle_triage_data()  # uses cache if exists
        source = "kaggle"

    # Normalize site_id presence
    if "site_id" not in raw.columns:
        raw["site_id"] = "site_0"

    # --------- Per-site QC + FE ----------
    processed: List[pd.DataFrame] = []
    site_reports: Dict[str, Dict] = {}

    for site, df_site in raw.groupby("site_id", dropna=False):
        # Raw missingness snapshot before any FE/imputation
        try:
            raw_miss = df_site.isna().mean().round(4).to_dict()
        except Exception:
            raw_miss = {}
        # Feature engineering (uses your fast path if available)
        fe_site, feats = _feature_engineer(df_site)
        # Keep site_id
        fe_site["site_id"] = site

        # QC report
        rep = _qc_site(fe_site, feats, qc_missing_thresh=args.qc_miss)
        rep["raw_missingness_before_fe"] = raw_miss
        site_reports[str(site)] = rep

        # Save per-site report
        with (OUT_REP / f"data_readiness_{site}.json").open("w", encoding="utf-8") as f:
            json.dump({
                "source": source,
                "site_id": site,
                **rep
            }, f, indent=2)

        if not rep["ready"]:
            print(f"[WARN] Site '{site}' did NOT pass QC gates; keeping in unified set but flagging in report.")
        processed.append(fe_site)

    # --------- Concatenate & finalize ----------
    df_all = pd.concat(processed, axis=0, ignore_index=True)

    # Ensure required target exists and is int
    if "esi_5class_encoded" not in df_all.columns:
        raise RuntimeError("esi_5class_encoded missing after FE; cannot proceed.")
    df_all["esi_5class_encoded"] = df_all["esi_5class_encoded"].astype(int)

    # Persist processed CSV
    out_csv = OUT_PROC / "triage_processed.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"✅ Saved processed dataset → {out_csv}")

    # --------- Centralized splits (60/20/20 stratified) ----------
    splits = _centralized_splits(df_all)
    with (OUT_SPLT / "centralized_splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"✅ Saved centralized splits → {OUT_SPLT / 'centralized_splits.json'}")

    # --------- Federated splits (Dirichlet non-IID) ----------
    y = df_all["esi_5class_encoded"].to_numpy()
    fl_splits = _federated_splits(y, clients=args.clients, alpha=args.dirichlet_alpha)
    with (OUT_SPLT / "fl_splits.json").open("w", encoding="utf-8") as f:
        json.dump(fl_splits, f, indent=2)
    print(f"✅ Saved FL splits → {OUT_SPLT / 'fl_splits.json'}")

    # --------- Global readiness summary ----------
    global_summary = {
        "source": source,
        "sites": list(site_reports.keys()),
        "rows_total": int(len(df_all)),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "reports": {k: {"ready": v.get("ready", False), "rows": v.get("rows", 0)} for k, v in site_reports.items()},
    }
    with (OUT_REP / "data_readiness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)
    print(f"✅ Saved readiness summary → {OUT_REP / 'data_readiness_summary.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
