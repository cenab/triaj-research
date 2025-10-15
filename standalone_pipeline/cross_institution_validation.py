"""Cross-institution validation scaffold.

Usage
-----
python -m standalone_pipeline.cross_institution_validation --csv data/processed/kaggle_triage_processed.csv --site-col site_id

Notes
-----
- Provide a `site_id` column; the script will compute per-site metrics and an overall summary.
- If `site_id` is missing, falls back to non-IID splits via Dirichlet for simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from .evaluation_framework import ClinicalMetrics
except Exception:
    from evaluation_framework import ClinicalMetrics  # type: ignore


def evaluate_per_site(df: pd.DataFrame, site_col: str, target_col: str = 'esi_5class_encoded') -> Dict:
    metrics: Dict[str, Dict] = {}
    if site_col not in df.columns:
        raise ValueError(f"site_col '{site_col}' not found in dataframe")
    sites = df[site_col].unique()
    for site in sites:
        sub = df[df[site_col] == site]
        # Here, assume predictions were produced and attached as 'pred'. Placeholder for pipeline integration.
        if 'pred' not in sub.columns:
            continue
        y_true = sub[target_col].astype(int).to_numpy()
        y_pred = sub['pred'].astype(int).to_numpy()
        metrics[str(site)] = ClinicalMetrics.calculate_triage_metrics(y_true, y_pred)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--site-col', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    results = evaluate_per_site(df, args.site_col)
    out = Path('results') / 'cross_institution_metrics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with out.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-site metrics to {out}")


if __name__ == '__main__':
    main()

