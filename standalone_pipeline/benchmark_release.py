"""Generate a processed triage benchmark and federated splits.

Outputs
-------
- data/processed/kaggle_triage_processed.csv
- data/splits/fl_splits.json (list of indices per client)

Notes
-----
- Ensure you have `src/kaggle_triage_data.csv` or Kaggle credentials.
- Review licensing before redistribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

try:
    from .kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
except Exception:
    from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data  # type: ignore


def main(clients: int = 5, dirichlet_alpha: float = 1.0) -> None:
    df_raw = load_kaggle_triage_data()
    df, feature_spec = feature_engineer_kaggle_data(df_raw.copy())
    features = feature_spec.features
    out_dir = Path('data') / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = out_dir / 'kaggle_triage_processed.csv'
    df.to_csv(processed_path, index=False)

    # Federated splits (Dirichlet over labels) on processed rows with valid target
    y = df['esi_5class_encoded'].astype(int).to_numpy()
    indices = np.arange(len(df))
    rng = np.random.default_rng(42)
    classes = np.unique(y)
    class_indices = {c: indices[y == c] for c in classes}
    client_buckets: List[List[int]] = [[] for _ in range(clients)]
    for c in classes:
        idx = class_indices[c]
        rng.shuffle(idx)
        props = rng.dirichlet([dirichlet_alpha] * clients)
        counts = (props * len(idx)).astype(int)
        # fix rounding
        while counts.sum() < len(idx):
            counts[rng.integers(0, clients)] += 1
        while counts.sum() > len(idx):
            j = rng.integers(0, clients)
            if counts[j] > 0:
                counts[j] -= 1
        start = 0
        for k in range(clients):
            end = start + counts[k]
            client_buckets[k].extend(idx[start:end].tolist())
            start = end

    splits = [sorted(bucket) for bucket in client_buckets]
    Path('data/splits').mkdir(parents=True, exist_ok=True)
    with open('data/splits/fl_splits.json', 'w', encoding='utf-8') as f:
        json.dump({str(i): splits[i] for i in range(clients)}, f, indent=2)

    print(f"Saved processed dataset to {processed_path}")
    print("Saved federated splits to data/splits/fl_splits.json")


if __name__ == '__main__':
    main()
