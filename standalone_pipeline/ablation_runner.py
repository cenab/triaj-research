"""Run a small set of ablations and record metrics.

Usage
-----
python -m standalone_pipeline.ablation_runner --runs 3

Note: This is a lightweight scaffold; adjust epochs/runs for full studies.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    from .advanced_training import train_advanced_model
except Exception:
    from advanced_training import train_advanced_model  # type: ignore


def main(runs: int = 3) -> None:
    configs = [
        {"name": "baseline", "loss_kwargs": {}},
        {"name": "no_focal", "loss_kwargs": {"alpha": 0.0}},
        {"name": "no_critical", "loss_kwargs": {"critical_miss_penalty": 0.0}},
        {"name": "gamma_low", "loss_kwargs": {"gamma": 1.0}},
    ]
    out = {"timestamp": datetime.utcnow().isoformat(), "runs": []}
    for cfg in configs[:runs]:
        report, _ = train_advanced_model(
            epochs=3,
            batch_size=128,
            learning_rate=5e-3,
            threshold_sweep=False,
            loss_kwargs=cfg.get("loss_kwargs", {}),
        )
        out["runs"].append({
            "name": cfg["name"],
            "overall_accuracy": report.get("clinical_metrics", {}).get("overall_accuracy"),
            "critical_sensitivity": report.get("clinical_metrics", {}).get("clinical_safety", {}).get("critical_sensitivity"),
            "under_triage_rate": report.get("clinical_metrics", {}).get("clinical_safety", {}).get("under_triage_rate"),
        })
    Path('results').mkdir(parents=True, exist_ok=True)
    path = Path('results') / f"ablation_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with path.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Ablation summary written to {path}")


if __name__ == '__main__':
    main()
