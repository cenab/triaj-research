#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact curated grid sweep (robust to trainer variants).

- Detects trainer signature and only passes supported kwargs
  (e.g., 'prioritize_critical' may not exist in the baseline trainer).
- Runs the curated grid over model choice, LR, batch, epochs, loss presets.
- Writes a CSV + JSONL with metrics & artifact paths.

Usage:
  python run_grid_exact.py --outdir results/sweeps/grid_v1
  python run_grid_exact.py --outdir results/sweeps/quick --max-runs 10
"""

from __future__ import annotations
import argparse, csv, json, re, sys, inspect
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import shutil
import numpy as np
import pandas as pd

# ---- Try to import centralized + federated trainers ---------------------------
_import_errors: List[str] = []

# Centralized trainers (baseline + red-focus variant)
central_trainers: List[Dict[str, Any]] = []
for mod, label in [
    ("standalone_pipeline.advanced_training", "centralized_baseline"),
    ("standalone_pipeline.advanced_training_red_focus", "centralized_red_focus"),
    ("src.advanced_training", "centralized_baseline_alt"),
    ("advanced_training", "centralized_baseline_legacy"),
]:
    try:
        fn = __import__(mod, fromlist=["train_advanced_model"]).train_advanced_model  # type: ignore
        central_trainers.append({
            "module": mod,
            "label": label,
            "func": fn,
            "sig": inspect.signature(fn),
        })
    except Exception as e:
        _import_errors.append(f"{mod}: {e}")

if not central_trainers:
    raise ImportError("Could not import any centralized train_advanced_model.\n" + "\n".join(_import_errors))

# Federated training (optional)
federated_trainer: Dict[str, Any] | None = None
try:
    mod = "standalone_pipeline.fl_training"
    ft = __import__(mod, fromlist=["federated_training"]).federated_training  # type: ignore
    federated_trainer = {
        "module": mod,
        "label": "federated",
        "func": ft,
        "sig": inspect.signature(ft),
    }
except Exception as e:
    _import_errors.append(f"standalone_pipeline.fl_training: {e}")

def _supports(sig: inspect.Signature, name: str) -> bool:
    return name in set(sig.parameters.keys())

# ---- Fixed settings (not in grid) --------------------------------------------
FIXED = {
    "calibrate_temperature_grid": [0.75, 0.85, 0.90, 0.95, 1.00],
    "threshold_sweep": True,
    "threshold_sample": 20000,
}

# ---- Loss presets -------------------------------------------------------------
LOSS_PRESETS = {
    # Core loss configurations for publishable comparisons
    "balanced": {
        "alpha": 0.25, "gamma": 2.0,
        "critical_miss_penalty": 100.0,
    },
    "safety_tilted": {
        "alpha": 0.35, "gamma": 2.5,
        "critical_miss_penalty": 150.0,
    },
    "safety_aggressive": {
        "alpha": 0.45, "gamma": 3.0,
        "critical_miss_penalty": 200.0,
    },
}

# ---- Grid ---------------------------------------------------------------------
# Note: Some trainers don't use hierarchical internally (they'll just ignore it).
GRID = {
    "model_kwargs": [
        {},
    ],
    "learning_rate": [5e-4, 1e-3],
    "batch_size": [256],
    "epochs": [15],
    # This dimension is auto-collapsed if 'prioritize_critical' isn't supported:
    "prioritize_critical": [True, False],
    "loss_kwargs": [
        LOSS_PRESETS["balanced"],
        LOSS_PRESETS["safety_tilted"],
        LOSS_PRESETS["safety_aggressive"],
    ],
}

def _collapse_grid_if_needed(grid: Dict[str, Iterable], sig: inspect.Signature) -> Dict[str, Iterable]:
    g = dict(grid)
    if not _supports(sig, "prioritize_critical"):
        # Remove this dimension entirely to avoid passing/printing it
        g.pop("prioritize_critical", None)
    return g

def product_dict(grid: Dict[str, Iterable]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for vals in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))

def extract_model_path_from_report_path(report_path: str | None) -> str | None:
    if not report_path:
        return None
    m = re.search(r"advanced_evaluation_report_(\d{8}_\d{6})\.json$", report_path)
    return str(Path(report_path).parent / f"advanced_model_{m.group(1)}.pth") if m else None

def _extract_timestamp(path: str | None) -> str:
    if not path:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    m = re.search(r"(\d{8}_\d{6})", Path(path).name)
    return m.group(1) if m else datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _safe_label_slug(label: str) -> str:
    slug = label.strip().replace(" | ", "__").replace(" ", "_").replace("=", "-")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "", slug)
    return slug[:180] or "run"

def ensure_named_artifacts(label: str, report_path: str | None, model_path: str | None) -> tuple[str | None, str | None]:
    if not report_path:
        return report_path, model_path
    label_slug = _safe_label_slug(label)
    timestamp = _extract_timestamp(report_path)
    report_p = Path(report_path)
    named_report = report_p.with_name(f"{label_slug}_{timestamp}{report_p.suffix}")
    if not named_report.exists() and report_p.exists():
        try:
            shutil.copy2(report_p, named_report)
        except Exception:
            named_report = report_p
    if model_path:
        model_p = Path(model_path)
        timestamp_model = _extract_timestamp(model_path) or timestamp
        named_model = model_p.with_name(f"{label_slug}_{timestamp_model}{model_p.suffix}")
        if not named_model.exists() and model_p.exists():
            try:
                shutil.copy2(model_p, named_model)
            except Exception:
                named_model = model_p
    else:
        named_model = None
    return str(named_report), str(named_model) if named_model is not None else None

def mean_macro_f1(class_metrics: Dict[str, Dict[str, float]]) -> float:
    if not class_metrics:
        return float("nan")
    f1s = [float(m.get("f1_score", 0.0)) for m in class_metrics.values()]
    return float(np.mean(f1s)) if f1s else float("nan")

def _latest_glob(path: Path, pattern: str) -> str | None:
    files = sorted(path.glob(pattern))
    return str(files[-1]) if files else None


def main():
    ap = argparse.ArgumentParser(description="Exact curated grid sweep (centralized + federated)")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for consolidated results")
    ap.add_argument("--max-runs", type=int, default=None, help="Only run first N combos across all trainers")
    ap.add_argument("--mode", choices=["both", "centralized", "federated"], default="both",
                    help="Select which training paradigms to run")
    ap.add_argument("--ab-test", action="store_true",
                    help="Also generate A/B tasks varying one variable at a time (per paradigm)")
    ap.add_argument("--ab-only", action="store_true",
                    help="Only run A/B tasks (skip the full grid)")
    ap.add_argument("--allow-all-fl-combos", action="store_true",
                    help="Disable pruning for FL grid; run full cartesian product of FL options")
    args = ap.parse_args()

    # Prepare output files
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    jsonl_path = outdir / "results.jsonl"
    fails_path = outdir / "failures.jsonl"

    # Build tasks: centralized (all available trainers)
    tasks: List[Dict[str, Any]] = []
    ab_tasks: List[Dict[str, Any]] = []
    if args.mode in ("both", "centralized"):
        for tr in central_trainers:
            grid_eff = _collapse_grid_if_needed(GRID, tr["sig"])
            grid_eff = dict(grid_eff)
            if tr["module"] == "standalone_pipeline.advanced_training_red_focus":
                grid_eff["model_kwargs"] = [
                    {"hierarchical": False},
                    {"hierarchical": True},
                ]
            else:
                grid_eff["model_kwargs"] = [{}]
            # Full grid
            if not args.ab_only:
                for cfg in product_dict(grid_eff):
                    tasks.append({
                        "type": "centralized",
                        "trainer": tr,
                        "cfg": cfg,
                    })
            # A/B tasks: vary one dimension at a time around a base
            if args.ab_test or args.ab_only:
                # Choose base as first element per key
                base_cfg = {k: (list(v)[0] if isinstance(v, (list, tuple)) else next(iter(v))) for k, v in grid_eff.items()}
                seen: set[str] = set()
                for key, values in grid_eff.items():
                    for val in values:
                        cfg_ab = dict(base_cfg)
                        cfg_ab[key] = val
                        # freeze for dedup
                        try:
                            frozen = json.dumps(cfg_ab, sort_keys=True)
                        except Exception:
                            frozen = str(cfg_ab)
                        if frozen in seen:
                            continue
                        seen.add(frozen)
                        ab_tasks.append({
                            "type": "centralized",
                            "trainer": tr,
                            "cfg": cfg_ab,
                        })

    # Build tasks: federated variants (if available)
    FL_GRID = {
        "fl_variant": ["baseline", "red_focus"],
        "rounds": [10],
        "clients": [10],
        "local_epochs": [1, 2],
        "batch_size": [256],
        "learning_rate": [5e-4, 1e-3],
        "aggregator": ["fedavg", "median"],
        "momentum": [0.9],  # only used for fedavgm (not in grid)
        "trim_fraction": [0.1],  # not used but kept for signature compatibility
        "non_iid": [False, True],
        "dirichlet_alpha": [1.0, 0.5],
        "dp": [False, True],
        "dp_noise_multiplier": [1.0],
        "dp_max_grad_norm": [1.0],
        "dp_delta": [1e-5],
        "personalize_epochs": [0],
        "personalize_lr": [5e-4],
    }
    if federated_trainer is not None and args.mode in ("both", "federated"):
        # Full grid with pruning
        if not args.ab_only:
            for cfg in product_dict(FL_GRID):
                # Prune invalid/redundant combinations unless exhaustive requested
                if not args.allow_all_fl_combos:
                    if not cfg["non_iid"] and cfg["dirichlet_alpha"] != 1.0:
                        continue
                tasks.append({
                    "type": "federated",
                    "trainer": federated_trainer,
                    "cfg": cfg,
                })
        # A/B tasks: vary clients and local_epochs independently (others fixed to base)
        if args.ab_test or args.ab_only:
            base_f = {k: (v[0] if isinstance(v, list) else v) for k, v in FL_GRID.items()}
            # Ensure aggregator-specific defaults are valid
            if base_f.get("aggregator") != "trim":
                base_f["trim_fraction"] = 0.1
            if base_f.get("aggregator") != "fedavgm":
                base_f["momentum"] = 0.9
            # IID base
            base_f["non_iid"] = False
            base_f["dirichlet_alpha"] = 1.0
            # DP base off
            base_f["dp"] = False
            base_f["dp_noise_multiplier"] = 1.0
            base_f["dp_max_grad_norm"] = 1.0
            seen_f: set[str] = set()
            # Vary clients
            for c in FL_GRID["clients"]:
                cfg_ab = dict(base_f); cfg_ab["clients"] = c
                frozen = json.dumps(cfg_ab, sort_keys=True)
                if frozen not in seen_f:
                    seen_f.add(frozen)
                    ab_tasks.append({"type": "federated", "trainer": federated_trainer, "cfg": cfg_ab})
            # Vary local epochs
            for le in FL_GRID["local_epochs"]:
                cfg_ab = dict(base_f); cfg_ab["local_epochs"] = le
                frozen = json.dumps(cfg_ab, sort_keys=True)
                if frozen not in seen_f:
                    seen_f.add(frozen)
                    ab_tasks.append({"type": "federated", "trainer": federated_trainer, "cfg": cfg_ab})
            # Optionally vary DP on/off and a couple noise values
            for dp_flag in [False, True]:
                cfg_ab = dict(base_f); cfg_ab["dp"] = dp_flag
                if dp_flag:
                    cfg_ab["dp_noise_multiplier"] = 1.0
                    cfg_ab["dp_max_grad_norm"] = 1.0
                frozen = json.dumps(cfg_ab, sort_keys=True)
                if frozen not in seen_f:
                    seen_f.add(frozen)
                    ab_tasks.append({"type": "federated", "trainer": federated_trainer, "cfg": cfg_ab})
            # Vary non_iid with alpha
            for non_iid in [False, True]:
                for alpha in ([1.0] if not non_iid else [0.5, 0.3]):
                    cfg_ab = dict(base_f)
                    cfg_ab["non_iid"] = non_iid
                    cfg_ab["dirichlet_alpha"] = alpha
                    frozen = json.dumps(cfg_ab, sort_keys=True)
                    if frozen not in seen_f:
                        seen_f.add(frozen)
                        ab_tasks.append({"type": "federated", "trainer": federated_trainer, "cfg": cfg_ab})

    # Merge tasks + A/B tasks per flags
    if args.ab_only:
        tasks = ab_tasks
    elif args.ab_test:
        tasks = tasks + ab_tasks

    total = len(tasks)
    if args.max_runs is not None:
        tasks = tasks[: int(args.max_runs)]
    print(f"Planned runs: {len(tasks)} (total combos available: {total})")

    # CSV columns (unified schema)
    cols = [
        "run_id", "run_type", "trainer_label", "trainer_module",
        "hierarchical", "learning_rate", "batch_size", "epochs",
        "prioritize_critical", "loss_name",
        # Federated-only knobs
        "fl_rounds", "fl_clients", "fl_local_epochs", "fl_aggregator", "fl_non_iid",
        "fl_dirichlet_alpha", "fl_dp", "fl_dp_noise_multiplier", "fl_dp_max_grad_norm", "fl_dp_delta",
        "fl_momentum", "fl_trim_fraction", "fl_personalize_epochs", "fl_personalize_lr",
        # Metrics
        "overall_accuracy", "macro_f1", "under_triage_rate", "critical_sensitivity",
        "avg_inference_time_ms", "throughput_samples_per_sec", "model_size_mb", "total_parameters",
        # Optional reliability (FL)
        "val_ece", "val_nll", "val_macro_f1",
        # Artifacts
        "report_path", "model_path",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf, \
         open(jsonl_path, "a", encoding="utf-8") as jf, \
         open(fails_path, "a", encoding="utf-8") as ff:
        cw = csv.DictWriter(cf, fieldnames=cols)
        if write_header: cw.writeheader()

        for i, task in enumerate(tasks, start=1):
            ttype = task["type"]
            tr = task["trainer"]
            cfg = task["cfg"]

            if ttype == "centralized":
                hier = bool(cfg.get("model_kwargs", {}).get("hierarchical", False))
                loss_kwargs = cfg["loss_kwargs"]
                loss_name = next((k for k, v in LOSS_PRESETS.items() if v == loss_kwargs), "custom")

                banner = (
                    f"[{i}/{len(tasks)}] CENTR | {tr['label']} | "
                    f"{'Hier' if hier else 'Flat'} | lr={cfg['learning_rate']} | "
                    f"bs={cfg['batch_size']} | ep={cfg['epochs']} | "
                )
                if "prioritize_critical" in cfg:
                    banner += f"crit_policy={cfg['prioritize_critical']} | "
                banner += f"loss={loss_name}"
                print("\n" + banner)

                # Build kwargs dynamically based on this trainer's signature
                sig: inspect.Signature = tr["sig"]
                kwargs: Dict[str, Any] = {}
                base_kwargs = dict(
                    epochs=int(cfg["epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    learning_rate=float(cfg["learning_rate"]),
                    model_kwargs=dict(cfg["model_kwargs"]),
                    loss_kwargs=dict(loss_kwargs),
                    threshold_sweep=bool(FIXED["threshold_sweep"]),
                    calibrate_temperature_grid=list(FIXED["calibrate_temperature_grid"]),
                    threshold_sample=int(FIXED["threshold_sample"]),
                    output_dir="results",
                )
                for k, v in base_kwargs.items():
                    if _supports(sig, k):
                        kwargs[k] = v
                if "prioritize_critical" in cfg and _supports(sig, "prioritize_critical"):
                    kwargs["prioritize_critical"] = bool(cfg["prioritize_critical"])

                try:
                    report, _ = tr["func"](**kwargs)
                except Exception as e:
                    err = {
                        "run_id": i, "type": ttype, "trainer": tr["module"],
                        "config": cfg, "kwargs_sent": {k: v for k, v in kwargs.items() if k != 'loss_kwargs'},
                        "error": f"{type(e).__name__}: {e}",
                    }
                    print(f"[FAIL] {err['error']}")
                    ff.write(json.dumps(err) + "\n"); ff.flush()
                    continue

                summary_name = banner.strip()
                # Extract metrics/artifacts
                clinical = report.get("clinical_metrics", {}) or {}
                class_metrics = clinical.get("class_metrics", {}) or {}
                safety = clinical.get("clinical_safety", {}) or {}
                perf = report.get("performance_metrics", {}) or {}
                report_path_original = report.get("report_path", None)
                model_path_original = extract_model_path_from_report_path(report_path_original)
                report_path_named, model_path_named = ensure_named_artifacts(summary_name, report_path_original, model_path_original)
                report_path = report_path_named or report_path_original
                model_path = model_path_named or model_path_original

                row = {
                    "run_id": i,
                    "run_type": "centralized",
                    "trainer_label": tr["label"],
                    "trainer_module": tr["module"],
                    "hierarchical": hier,
                    "learning_rate": cfg["learning_rate"],
                    "batch_size": cfg["batch_size"],
                    "epochs": cfg["epochs"],
                    "prioritize_critical": cfg.get("prioritize_critical", None) if "prioritize_critical" in GRID else None,
                    "loss_name": loss_name,
                    # Federated placeholders
                    "fl_rounds": None, "fl_clients": None, "fl_local_epochs": None, "fl_aggregator": None,
                    "fl_non_iid": None, "fl_dirichlet_alpha": None, "fl_dp": None, "fl_dp_noise_multiplier": None,
                    "fl_dp_max_grad_norm": None, "fl_dp_delta": None, "fl_momentum": None, "fl_trim_fraction": None,
                    "fl_personalize_epochs": None, "fl_personalize_lr": None,
                    # Metrics
                    "overall_accuracy": float(clinical.get("overall_accuracy", np.nan)),
                    "macro_f1": mean_macro_f1(class_metrics),
                    "under_triage_rate": float(safety.get("under_triage_rate", np.nan)),
                    "critical_sensitivity": float(safety.get("critical_sensitivity", np.nan)),
                    "avg_inference_time_ms": perf.get("avg_inference_time_ms", None),
                    "throughput_samples_per_sec": perf.get("throughput_samples_per_sec", None),
                    "model_size_mb": perf.get("model_size_mb", None),
                    "total_parameters": perf.get("total_parameters", None),
                    # Reliability (centralized N/A)
                    "val_ece": None, "val_nll": None, "val_macro_f1": None,
                    # Artifacts
                    "report_path": report_path,
                    "model_path": model_path,
                }
                cw.writerow(row); cf.flush()

                trainer_meta = {
                    "module": tr["module"],
                    "label": tr["label"],
                }
                jf.write(json.dumps({
                    "run_id": i,
                    "type": "centralized",
                    "trainer": trainer_meta,
                    "config": cfg,
                    "fixed": FIXED,
                    "summary": summary_name,
                    "kwargs_sent": {k: v for k, v in kwargs.items() if k != 'loss_kwargs'},
                    "metrics": {
                        "clinical_metrics": clinical,
                        "performance_metrics": perf,
                        "calibration": report.get("calibration", {}),
                        "thresholds": report.get("thresholds", {}),
                    },
                    "artifacts": {
                        "report_path": row["report_path"],
                        "model_path": row["model_path"],
                        "original_report_path": report_path_original,
                        "original_model_path": model_path_original,
                    },
                }) + "\n"); jf.flush()

                print(
                    f"[OK] acc={row['overall_accuracy']:.4f} | macroF1={row['macro_f1']:.4f} "
                    f"| crit_sens={row['critical_sensitivity']:.4f} | under={row['under_triage_rate']:.4f}"
                )

            else:  # Federated
                if federated_trainer is None:
                    continue
                variant = cfg["fl_variant"]
                hier = True if variant == "red_focus" else False
                loss_name = "red_focus" if variant == "red_focus" else "baseline"
                # Presets mirroring fl_training.main
                if variant == "baseline":
                    loss_kwargs = {}
                    threshold_kwargs = {}
                    calibrate = [0.8, 0.85, 0.9, 0.95, 1.0]
                    output_suffix = "advanced"
                else:
                    loss_kwargs = {
                        "alpha": 0.45,
                        "gamma": 3.0,
                        "critical_miss_penalty": 150.0,
                    }
                    threshold_kwargs = {
                        # 5-class tuning knobs
                        "min_critical_recall": 0.98,
                        "min_critical_precision": 0.90,
                        "tune_esi5": True,
                        "bottom_range": (-0.20, -0.16, -0.12, -0.08, -0.04, 0.0),
                        "min_esi5_recall": 0.10,
                    }
                    calibrate = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                    output_suffix = "advanced_red_focus"

                banner = (
                    f"[{i}/{len(tasks)}] FEDER | {variant} | "
                    f"{'Hier' if hier else 'Flat'} | rounds={cfg['rounds']} | clients={cfg['clients']} | "
                    f"lr={cfg['learning_rate']} | bs={cfg['batch_size']} | le={cfg['local_epochs']} | agg={cfg['aggregator']}"
                )
                print("\n" + banner)

                # Build kwargs based on federated trainer signature
                sig_f: inspect.Signature = federated_trainer["sig"]
                base_kwargs_f = dict(
                    rounds=int(cfg["rounds"]),
                    clients=int(cfg["clients"]),
                    local_epochs=int(cfg["local_epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    learning_rate=float(cfg["learning_rate"]),
                    hierarchical=bool(hier),
                    loss_kwargs=loss_kwargs,
                    output_suffix=output_suffix,
                    threshold_kwargs=threshold_kwargs,
                    calibrate_temperatures=calibrate,
                    non_iid=bool(cfg["non_iid"]),
                    dirichlet_alpha=float(cfg["dirichlet_alpha"]),
                    aggregator=str(cfg["aggregator"]),
                    momentum=float(cfg.get("momentum", 0.9)),
                    trim_fraction=float(cfg.get("trim_fraction", 0.1)),
                    personalize_epochs=int(cfg.get("personalize_epochs", 0)),
                    personalize_lr=float(cfg.get("personalize_lr", 5e-4)),
                    dp=bool(cfg["dp"]),
                    dp_noise_multiplier=float(cfg.get("dp_noise_multiplier", 1.0)),
                    dp_max_grad_norm=float(cfg.get("dp_max_grad_norm", 1.0)),
                    dp_delta=float(cfg.get("dp_delta", 1e-5)),
                )
                kwargs_f: Dict[str, Any] = {}
                for k, v in base_kwargs_f.items():
                    if _supports(sig_f, k):
                        kwargs_f[k] = v

                try:
                    report = federated_trainer["func"](**kwargs_f)  # type: ignore[call-arg]
                except Exception as e:
                    err = {
                        "run_id": i, "type": ttype, "trainer": federated_trainer["module"],
                        "config": cfg, "kwargs_sent": {k: v for k, v in kwargs_f.items() if k != 'loss_kwargs'},
                        "error": f"{type(e).__name__}: {e}",
                    }
                    print(f"[FAIL] {err['error']}")
                    ff.write(json.dumps(err) + "\n"); ff.flush()
                    continue

                summary_name = banner.strip()
                # Extract metrics + attempt to locate artifacts by convention
                clinical = report.get("clinical_metrics", {}) or {}
                class_metrics = clinical.get("class_metrics", {}) or {}
                safety = clinical.get("clinical_safety", {}) or {}
                perf = report.get("performance_metrics", {}) or {}
                rel = report.get("reliability_history", {}) or {}
                rounds_axis = report.get("training_history", {}).get("rounds", []) if isinstance(report.get("training_history"), dict) else []

                # Find artifacts (best-effort)
                results_dir = Path("results")
                report_path_original = _latest_glob(results_dir, f"fl_{output_suffix}_evaluation_report_*.json")
                model_path_original = _latest_glob(results_dir, f"fl_{output_suffix}_model_*.pth")
                report_path_named, model_path_named = ensure_named_artifacts(summary_name, report_path_original, model_path_original)
                report_path = report_path_named or report_path_original
                model_path = model_path_named or model_path_original

                row = {
                    "run_id": i,
                    "run_type": "federated",
                    "trainer_label": f"federated_{variant}",
                    "trainer_module": federated_trainer["module"],
                    "hierarchical": hier,
                    "learning_rate": cfg["learning_rate"],
                    "batch_size": cfg["batch_size"],
                    "epochs": None,
                    "prioritize_critical": None,
                    "loss_name": loss_name,
                    "fl_rounds": cfg["rounds"],
                    "fl_clients": cfg["clients"],
                    "fl_local_epochs": cfg["local_epochs"],
                    "fl_aggregator": cfg["aggregator"],
                    "fl_non_iid": cfg["non_iid"],
                    "fl_dirichlet_alpha": cfg["dirichlet_alpha"],
                    "fl_dp": cfg["dp"],
                    "fl_dp_noise_multiplier": base_kwargs_f.get("dp_noise_multiplier"),
                    "fl_dp_max_grad_norm": base_kwargs_f.get("dp_max_grad_norm"),
                    "fl_dp_delta": base_kwargs_f.get("dp_delta"),
                    "fl_momentum": base_kwargs_f.get("momentum"),
                    "fl_trim_fraction": base_kwargs_f.get("trim_fraction"),
                    "fl_personalize_epochs": base_kwargs_f.get("personalize_epochs"),
                    "fl_personalize_lr": base_kwargs_f.get("personalize_lr"),
                    "overall_accuracy": float(clinical.get("overall_accuracy", np.nan)),
                    "macro_f1": mean_macro_f1(class_metrics),
                    "under_triage_rate": float(safety.get("under_triage_rate", np.nan)),
                    "critical_sensitivity": float(safety.get("critical_sensitivity", np.nan)),
                    "avg_inference_time_ms": perf.get("avg_inference_time_ms", None),
                    "throughput_samples_per_sec": perf.get("throughput_samples_per_sec", None),
                    "model_size_mb": perf.get("model_size_mb", None),
                    "total_parameters": perf.get("total_parameters", None),
                    "val_ece": (rel.get("val_ece", []) or [None])[-1] if isinstance(rel.get("val_ece", []), list) else None,
                    "val_nll": (rel.get("val_nll", []) or [None])[-1] if isinstance(rel.get("val_nll", []), list) else None,
                    "val_macro_f1": (report.get("training_history", {}).get("val_macro_f1", []) or [None])[-1]
                        if isinstance(report.get("training_history"), dict) else None,
                    "report_path": report_path,
                    "model_path": model_path,
                }
                cw.writerow(row); cf.flush()

                trainer_meta = {
                    "module": federated_trainer["module"],
                    "label": f"federated_{variant}",
                }
                jf.write(json.dumps({
                    "run_id": i,
                    "type": "federated",
                    "trainer": trainer_meta,
                    "config": cfg,
                    "summary": summary_name,
                    "kwargs_sent": {k: v for k, v in kwargs_f.items() if k != 'loss_kwargs'},
                    "metrics": {
                        "clinical_metrics": clinical,
                        "performance_metrics": perf,
                        "reliability_history": rel,
                    },
                    "artifacts": {
                        "report_path": row["report_path"],
                        "model_path": row["model_path"],
                        "original_report_path": report_path_original,
                        "original_model_path": model_path_original,
                    },
                }) + "\n"); jf.flush()

                print(
                    f"[OK] acc={row['overall_accuracy']:.4f} | macroF1={row['macro_f1']:.4f} "
                    f"| crit_sens={row['critical_sensitivity']:.4f} | under={row['under_triage_rate']:.4f}"
                )

    # Quick preview (centralized + federated together)
    try:
        df = pd.read_csv(csv_path)
        top = df.sort_values(["overall_accuracy", "macro_f1"], ascending=[False, False]).head(10)
        print("\nTop 10 by (accuracy, macro_f1):")
        print(top[["run_id","run_type","trainer_label","hierarchical","learning_rate","batch_size","epochs",
                   "prioritize_critical","loss_name","overall_accuracy","macro_f1",
                   "critical_sensitivity","under_triage_rate","report_path"]].to_string(index=False))
    except Exception as e:
        print(f"Preview skipped: {e}")

if __name__ == "__main__":
    main()
