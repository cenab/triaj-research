"""End-to-end training script for the advanced hierarchical triage model.

This module mirrors the behaviour of the legacy fixed-training pipeline but
uses :class:`AdvancedHierarchicalTriageModel` and
:class:`AdvancedClinicalSafetyLoss`.  It loads the Kaggle triage dataset,
applies the enhanced feature-engineering pipeline when available, trains the
model with stratified splits, and emits a JSON report compatible with the
existing evaluation artefacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import time

try:
    from .advanced_model_architecture import (
        AdvancedClinicalSafetyLoss,
        AdvancedHierarchicalTriageEnsemble,
        AdvancedHierarchicalTriageModel,
    )
    from .evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics
    from .kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
    from .conformal import fit_conformal_thresholds, apply_conformal_thresholds
    from .utils import construct_fairness_data, get_critical_class_indices
except ImportError:  # pragma: no cover - script mode
    from advanced_model_architecture import (  # type: ignore
        AdvancedClinicalSafetyLoss,
        AdvancedHierarchicalTriageEnsemble,
        AdvancedHierarchicalTriageModel,
    )
    from evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics  # type: ignore
    from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data  # type: ignore
    from conformal import fit_conformal_thresholds, apply_conformal_thresholds  # type: ignore
    from utils import construct_fairness_data, get_critical_class_indices  # type: ignore

try:  # optional dependency for enhanced features
    from .experimental.kaggle_enhanced_final_fix_v2 import (
        advanced_kaggle_feature_engineering,
    )
except Exception:  # pragma: no cover - optional module
    try:
        from experimental.kaggle_enhanced_final_fix_v2 import (
            advanced_kaggle_feature_engineering,
        )
    except Exception:
        advanced_kaggle_feature_engineering = None  # type: ignore


@dataclass
class FeatureGroups:
    vital: List[str]
    symptom: List[str]
    risk: List[str]
    context: List[str]
    lab: List[str]
    interaction: List[str]

    def all(self) -> List[str]:
        return (
            self.vital
            + self.symptom
            + self.risk
            + self.context
            + self.lab
            + self.interaction
        )


def _ensure_group(df: pd.DataFrame, cols: List[str], prefix: str) -> List[str]:
    if cols:
        return cols
    placeholder = f"{prefix}_placeholder"
    if placeholder not in df.columns:  # type: ignore[attr-defined]
        df[placeholder] = 0.0  # type: ignore[index]
    return [placeholder]


def load_feature_engineered_dataframe(
    *,
    csv_path: Path = Path("src/kaggle_triage_data.csv"),
) -> Tuple[pd.DataFrame, FeatureGroups, str]:
    """Load cached Kaggle dataset and return a feature-engineered dataframe."""

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = load_kaggle_triage_data()

    df_engineered, spec = feature_engineer_kaggle_data(df.copy())
    target_col = "esi_5class_encoded"

    all_features = spec.features
    miss = df_engineered[all_features].isna().mean()
    keep = [col for col in all_features if miss.get(col, 0.0) <= 0.40]

    lab_feats_all = spec.groups.get("lab", [])
    lab_last = [col for col in lab_feats_all if col.endswith("_last")]
    keep = [col for col in keep if (col not in lab_feats_all) or (col in lab_last)]
    keep = sorted(set(keep))

    df_engineered = df_engineered[keep + [target_col]]

    vital_feats = [c for c in spec.groups.get("vital", []) if c in keep]
    symptom_feats = [c for c in spec.groups.get("symptom", []) if c in keep]
    risk_feats = [c for c in spec.groups.get("risk", []) if c in keep]
    context_feats = [c for c in spec.groups.get("context", []) if c in keep]
    lab_feats = [c for c in lab_last if c in keep]
    interaction_feats = [c for c in spec.groups.get("interaction", []) if c in keep]

    df_engineered = df_engineered.reset_index(drop=True)

    feature_groups = FeatureGroups(
        vital=_ensure_group(df_engineered, vital_feats, "vital"),
        symptom=_ensure_group(df_engineered, symptom_feats, "symptom"),
        risk=_ensure_group(df_engineered, risk_feats, "risk"),
        context=_ensure_group(df_engineered, context_feats, "context"),
        lab=_ensure_group(df_engineered, lab_feats, "lab"),
        interaction=_ensure_group(df_engineered, interaction_feats, "interaction"),
    )

    return df_engineered, feature_groups, target_col


def _to_tensor(df, columns: Sequence[str]) -> torch.Tensor:
    data = df[columns].to_numpy(dtype=np.float32)
    return torch.from_numpy(data)


def _make_dataset(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_groups: FeatureGroups,
) -> TensorDataset:
    tensors = [
        _to_tensor(df, feature_groups.vital),
        _to_tensor(df, feature_groups.symptom),
        _to_tensor(df, feature_groups.risk),
        _to_tensor(df, feature_groups.context),
        _to_tensor(df, feature_groups.lab),
        _to_tensor(df, feature_groups.interaction),
    ]
    targets = torch.from_numpy(y.astype(np.int64))
    return TensorDataset(*tensors, targets)


def _get_temperature(model: nn.Module) -> float:
    temp = getattr(model, "calibration_temperature", None)
    if temp is None:
        return 1.0
    if isinstance(temp, torch.Tensor):
        return float(temp.detach().cpu().item())
    return float(temp)


def _compose_probs(
    outputs: torch.Tensor | Tuple[torch.Tensor, ...], temperature: float | None = None
) -> torch.Tensor:
    def _apply_temp(t: torch.Tensor) -> torch.Tensor:
        if temperature is None or abs(temperature - 1.0) < 1e-9:
            return t
        scale = torch.tensor(float(temperature), device=t.device, dtype=t.dtype).clamp_min(1e-6)
        return t / scale

    if isinstance(outputs, tuple):
        heads = tuple(_apply_temp(t) for t in outputs)
        if len(heads) == 2:
            gate_logits, nc_logits = heads
            gate = torch.softmax(gate_logits, dim=1)
            nc_probs = torch.softmax(nc_logits, dim=1)
            noncrit = gate[:, 0:1]
            crit = gate[:, 1:2]
            nc0 = noncrit * nc_probs[:, 0:1]
            nc1 = noncrit * nc_probs[:, 1:2]
            return torch.cat([nc0, nc1, crit], dim=1).clamp_min(1e-8)
        if len(heads) == 3:
            gate_logits, nc_logits, crit_logits = heads
            gate = torch.softmax(gate_logits, dim=1)
            noncrit = gate[:, 0:1]
            crit = gate[:, 1:2]
            nc_probs = torch.softmax(nc_logits, dim=1)
            crit_probs = torch.softmax(crit_logits, dim=1)
            return torch.cat([noncrit * nc_probs, crit * crit_probs], dim=1).clamp_min(1e-8)
        raise ValueError("Unsupported hierarchical outputs tuple length")

    logits = outputs
    temp_val = float(temperature) if temperature is not None else 1.0
    logits = logits / max(temp_val, 1e-6)
    return torch.softmax(logits, dim=1)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdvancedClinicalSafetyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        *features, targets = batch
        features = [t.to(device) for t in features]
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(*features)
        if isinstance(outputs, tuple):
            loss = criterion.compute_hierarchical_loss_general(outputs, targets)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        temp = _get_temperature(model)
        probs = _compose_probs(outputs, temperature=temp)
        preds = probs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdvancedClinicalSafetyLoss,
    device: torch.device,
    collect_outputs: bool = False,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            *features, targets = batch
            features = [t.to(device) for t in features]
            targets = targets.to(device)

            outputs = model(*features)
            if isinstance(outputs, tuple):
                loss = criterion.compute_hierarchical_loss_general(outputs, targets)
            else:
                loss = criterion(outputs, targets)

            temperature = _get_temperature(model)
            probs = _compose_probs(outputs, temperature=temperature)
            preds = probs.argmax(dim=1)

            running_loss += loss.item() * targets.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if collect_outputs:
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                y_prob.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    if collect_outputs:
        return avg_loss, accuracy, np.array(y_true), np.array(y_pred), np.array(y_prob)
    return avg_loss, accuracy


# Removed legacy 3-class threshold sweep (ESI-only)

def _sweep_thresholds_top2(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    critical_range: Sequence[float] = (-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02),
    noncritical_default: float = 0.0,
    class_weights: Sequence[float] | None = None,
    min_critical_recall: float | None = 0.98,
    min_critical_precision: float | None = 0.9,
    metric: str = "f1_weighted_custom",
) -> Tuple[List[float], Dict[str, float]]:
    """Top-2 critical (ESI1–2) threshold sweep for K>=5 classes."""
    k = probs.shape[1]
    assert k >= 5, "red-focus top2 sweep expects K>=5"
    crit_start = max(1, k - 2)
    c1, c2 = crit_start, crit_start + 1
    best_thr = [noncritical_default] * k
    best_score = -1.0
    base_argmax = probs.argmax(axis=1)
    w = np.array(class_weights if class_weights is not None else (0.05, 0.15, 0.20, 0.25, 0.35), dtype=np.float32)
    if w.shape[0] != k:
        w = np.ones(k, dtype=np.float32)

    for t1 in critical_range:
        for t2 in critical_range:
            thr = np.full(k, noncritical_default, dtype=np.float32)
            thr[c1] = float(t1)
            thr[c2] = float(t2)
            adjusted = probs - thr
            pred = adjusted.argmax(axis=1)
            invalid = adjusted.max(axis=1) < 0.0
            if invalid.any():
                pred[invalid] = base_argmax[invalid]

            true_crit = y_true >= crit_start
            pred_crit = pred >= crit_start
            if true_crit.any():
                recall_c = np.sum(pred_crit & true_crit) / np.sum(true_crit)
            else:
                recall_c = 1.0
            if pred_crit.any():
                precision_c = np.sum(pred_crit & true_crit) / np.sum(pred_crit)
            else:
                precision_c = 1.0
            if min_critical_recall is not None and recall_c < float(min_critical_recall):
                continue
            if min_critical_precision is not None and precision_c < float(min_critical_precision):
                continue

            if metric == "accuracy":
                score = accuracy_score(y_true, pred)
            elif metric == "f1_weighted_custom":
                f1_per = f1_score(y_true, pred, average=None, zero_division=0)
                if f1_per.shape[0] != k:
                    f1_per = np.pad(f1_per, (0, k - f1_per.shape[0]))
                score = float(np.dot(f1_per, w / (w.sum() + 1e-12)))
            else:
                score = f1_score(y_true, pred, average="macro")
            if score > best_score:
                best_score = score
                best_thr = thr.tolist()

    return best_thr, {"score": best_score, "metric": metric}


def train_advanced_model(
    *,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 5e-3,
    output_dir: str = "results",
    model_kwargs: Dict | None = None,
    loss_kwargs: Dict | None = None,
    temperature: float | None = None,
    calibrate_temperature_grid: Sequence[float] | None = None,
    threshold_sweep: bool = True,
    threshold_sample: int = 20000,
    prioritize_critical: bool = True,
    use_conformal: bool = True,
    conformal_alpha: float = 0.05,
) -> Tuple[Dict, AdvancedHierarchicalTriageModel]:
    """Train the advanced model and return (report_dict, trained_model)."""

    np.random.seed(42)
    torch.manual_seed(42)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    df_engineered, feature_groups, target_col = load_feature_engineered_dataframe()
    X = df_engineered[feature_groups.all()]
    y = df_engineered[target_col].astype(int).to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    train_indices = X_train.index.to_numpy(dtype=int)
    val_indices = X_val.index.to_numpy(dtype=int)
    test_indices = X_test.index.to_numpy(dtype=int)

    from sklearn.preprocessing import StandardScaler

    all_cols = X_train.columns.tolist()
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.copy().fillna(train_medians)
    X_val = X_val.copy().fillna(train_medians)
    X_test = X_test.copy().fillna(train_medians)

    scaler = StandardScaler()
    X_train[all_cols] = scaler.fit_transform(X_train[all_cols])
    X_val[all_cols] = scaler.transform(X_val[all_cols])
    X_test[all_cols] = scaler.transform(X_test[all_cols])
    preprocess = {
        "columns": all_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "train_medians": {str(k): float(v) for k, v in train_medians.items()},
    }

    _, val_group_labels = construct_fairness_data(df_engineered, X_val.index)
    fairness_attributes_test, test_group_labels = construct_fairness_data(df_engineered, X_test.index)

    train_dataset = _make_dataset(X_train, y_train, feature_groups)
    val_dataset = _make_dataset(X_val, y_val, feature_groups)
    test_dataset = _make_dataset(X_test, y_test, feature_groups)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(model_kwargs or {})
    use_hierarchical = model_kwargs.pop("hierarchical", False)
    if use_hierarchical:
        k = int(len(np.unique(y)))
        noncritical_classes = 2 if k == 3 else 3
        critical_classes = 1 if k == 3 else 2
        model = AdvancedHierarchicalTriageEnsemble(
            num_vital_features=len(feature_groups.vital),
            num_symptom_features=len(feature_groups.symptom),
            num_risk_features=len(feature_groups.risk),
            num_context_features=len(feature_groups.context),
            num_lab_features=len(feature_groups.lab),
            num_interaction_features=len(feature_groups.interaction),
            noncritical_classes=noncritical_classes,
            critical_classes=critical_classes,
        ).to(device)
    else:
        model = AdvancedHierarchicalTriageModel(
            num_vital_features=len(feature_groups.vital),
            num_symptom_features=len(feature_groups.symptom),
            num_risk_features=len(feature_groups.risk),
            num_context_features=len(feature_groups.context),
            num_lab_features=len(feature_groups.lab),
            num_interaction_features=len(feature_groups.interaction),
            num_classes=len(np.unique(y)),
            **model_kwargs,
        ).to(device)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Default loss shaping for critical-first behaviour if not provided (ESI-only)
    if loss_kwargs is None and prioritize_critical:
        loss_kwargs = {
            "alpha": 0.45,
            "gamma": 3.0,
            "critical_miss_penalty": 150.0,
        }
    loss_kwargs = loss_kwargs or {}
    criterion = AdvancedClinicalSafetyLoss(class_weights=class_weights_tensor, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epochs": []}
    best_state = deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)
        history["epochs"].append(epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
        )

    model.load_state_dict(best_state)

    if temperature is not None:
        model.set_temperature(temperature)
        chosen_temperature = temperature
    elif calibrate_temperature_grid:
        base_temp = _get_temperature(model)
        chosen_temperature = base_temp
        best_nll = float("inf")
        for temp in calibrate_temperature_grid:
            model.set_temperature(temp)
            _, _, y_val_tmp, _, prob_tmp = _evaluate(model, val_loader, criterion, device, collect_outputs=True)
            if prob_tmp is None:
                continue
            nll = ReliabilityMetrics.negative_log_likelihood(y_val_tmp, prob_tmp)
            if nll < best_nll:
                best_nll = nll
                chosen_temperature = temp
        model.set_temperature(chosen_temperature)
    else:
        chosen_temperature = _get_temperature(model)

    # Collect validation probs for threshold sweep
    val_loss_collect, val_acc_collect, y_val, y_val_pred, val_prob = _evaluate(
        model, val_loader, criterion, device, collect_outputs=True
    )

    # Initialize thresholds with correct dimensionality
    num_classes = int(len(np.unique(y)))
    critical_classes = get_critical_class_indices(num_classes)
    critical_threshold = min(critical_classes)

    tau_global: float | None = None
    tau_by_group: Dict[str, float] = {}
    class_thresholds = [0.0] * num_classes
    threshold_info = None

    if use_conformal:
        tau_global, tau_by_group = fit_conformal_thresholds(
            y_val,
            val_prob,
            critical_classes=critical_classes,
            alpha=conformal_alpha,
            group_labels=val_group_labels,
        )
        val_preds_conf = apply_conformal_thresholds(
            val_prob,
            tau_global,
            critical_classes,
            tau_by_group=tau_by_group,
            group_labels=val_group_labels,
        )
        val_metrics_conf = ClinicalMetrics.calculate_triage_metrics(y_val, val_preds_conf)
        val_group_recall: Dict[str, float] = {}
        if val_group_labels.size:
            unique_groups = np.unique(val_group_labels.astype(str))
            val_crit_mask = y_val >= critical_threshold
            for group in unique_groups:
                mask = (val_group_labels == group) & val_crit_mask
                if np.any(mask):
                    val_group_recall[str(group)] = float(np.mean(val_preds_conf[mask] >= critical_threshold))
        threshold_info = {
            "method": "C3",
            "alpha": conformal_alpha,
            "tau": float(tau_global),
            "tau_by_group": {str(k): float(v) for k, v in tau_by_group.items()},
            "val_critical_sensitivity": float(val_metrics_conf["clinical_safety"]["critical_sensitivity"]),
            "val_under_triage_rate": float(val_metrics_conf["clinical_safety"]["under_triage_rate"]),
            "val_group_critical_sensitivity": val_group_recall,
        }
    elif threshold_sweep:
        if len(val_prob) > threshold_sample:
            idx = np.random.RandomState(42).choice(len(val_prob), size=threshold_sample, replace=False)
            probs_sweep = val_prob[idx]
            y_true_sweep = y_val[idx]
        else:
            probs_sweep = val_prob
            y_true_sweep = y_val
        if num_classes >= 5:
            class_thresholds, threshold_info = _sweep_thresholds_top2(
                probs_sweep,
                y_true_sweep,
                critical_range=tuple(x / 100 for x in (-10, -8, -6, -4, -2, 0, 2)),
                noncritical_default=0.0,
                class_weights=(0.05, 0.15, 0.20, 0.25, 0.35),
                min_critical_recall=0.98 if prioritize_critical else 0.95,
                min_critical_precision=0.9 if prioritize_critical else 0.85,
                metric="f1_weighted_custom",
            )
        else:
            raise ValueError(f"ESI-only configuration expects >=5 classes; got {num_classes}.")
        if hasattr(model, "set_class_thresholds"):
            model.set_class_thresholds(class_thresholds, enable=True)

    # Evaluate on test with final thresholds
    test_loss, test_acc, y_true, y_pred, y_prob = _evaluate(
        model, test_loader, criterion, device, collect_outputs=True
    )
    # Apply conformal thresholds or classical offsets
    if use_conformal and tau_global is not None:
        y_pred = apply_conformal_thresholds(
            y_prob,
            tau_global,
            critical_classes,
            tau_by_group=tau_by_group,
            group_labels=test_group_labels,
        )
    elif threshold_sweep and (num_classes == 3 or num_classes >= 5):
        adjusted = y_prob - np.array(class_thresholds, dtype=np.float32)
        pred = adjusted.argmax(axis=1)
        invalid = (adjusted.max(axis=1) < 0.0)
        if invalid.any():
            base_argmax = y_prob.argmax(axis=1)
            pred[invalid] = base_argmax[invalid]
        y_pred = pred

    # Performance profiling
    total_samples = len(test_dataset)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            *features, _ = batch
            features = [t.to(device) for t in features]
            model(*features)
    end = time.perf_counter()
    inference_duration = max(end - start, 1e-6)
    avg_inference_time_ms = (inference_duration / total_samples) * 1000
    throughput = total_samples / inference_duration

    # Metrics
    clinical_metrics = ClinicalMetrics.calculate_triage_metrics(y_true, y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    nll = float(ReliabilityMetrics.negative_log_likelihood(y_true, y_prob))
    ece = float(ReliabilityMetrics.expected_calibration_error(y_true, y_prob))
    brier = float(ReliabilityMetrics.brier_score(y_true, y_prob))

    reliability_metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "nll": nll,
        "ece": ece,
        "brier": brier,
    }

    bootstrap_ci = {
        "accuracy": ReliabilityMetrics.bootstrap_ci(y_true, y_pred=y_pred, metric="accuracy"),
        "macro_f1": ReliabilityMetrics.bootstrap_ci(y_true, y_pred=y_pred, metric="macro_f1"),
        "nll": ReliabilityMetrics.bootstrap_ci(y_true, y_prob=y_prob, metric="nll"),
        "ece": ReliabilityMetrics.bootstrap_ci(y_true, y_prob=y_prob, metric="ece"),
        "critical_sensitivity": ReliabilityMetrics.bootstrap_ci(y_true, y_pred=y_pred, metric="critical_sensitivity"),
        "under_triage_rate": ReliabilityMetrics.bootstrap_ci(y_true, y_pred=y_pred, metric="under_triage_rate"),
    }

    if use_conformal and tau_global is not None and threshold_info is not None:
        test_group_recall: Dict[str, float] = {}
        if test_group_labels.size:
            unique_test_groups = np.unique(test_group_labels.astype(str))
            test_crit_mask = y_true >= critical_threshold
            for group in unique_test_groups:
                mask = (test_group_labels == group) & test_crit_mask
                if np.any(mask):
                    test_group_recall[str(group)] = float(np.mean(y_pred[mask] >= critical_threshold))
        threshold_info["test_critical_sensitivity"] = float(clinical_metrics["clinical_safety"]["critical_sensitivity"])
        threshold_info["test_under_triage_rate"] = float(clinical_metrics["clinical_safety"]["under_triage_rate"])
        threshold_info["test_group_critical_sensitivity"] = test_group_recall

    fairness_metrics = None
    if fairness_attributes_test:
        evaluator = FairnessEvaluator()
        fairness_metrics = evaluator.evaluate_fairness(
            np.array(y_true),
            np.array(y_pred),
            fairness_attributes_test,
        )

    def _serialise(obj):
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        return obj

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 ** 2  # float32

    performance_metrics = {
        "avg_inference_time_ms": avg_inference_time_ms,
        "throughput_samples_per_sec": throughput,
        "model_size_mb": model_size_mb,
        "total_parameters": total_params,
        "total_samples_tested": total_samples,
    }

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "clinical_metrics": clinical_metrics,
        "performance_metrics": performance_metrics,
        "training_history": history,
        "model_info": {
            "architecture": "AdvancedHierarchicalTriageEnsemble" if use_hierarchical else "AdvancedHierarchicalTriageModel",
            "total_parameters": total_params,
            "model_size_mb": model_size_mb,
        },
        "calibration": {"temperature": chosen_temperature},
        "thresholds": {
            "class_thresholds": [float(x) for x in class_thresholds],
            "selection": _serialise(threshold_info),
        },
        "conformal": _serialise(threshold_info) if use_conformal else None,
        "reliability_metrics": reliability_metrics,
        "bootstrap_ci": {k: tuple(map(float, v)) for k, v in bootstrap_ci.items()},
        "fairness_metrics": _serialise(fairness_metrics),
        "data_info": {
            "total_samples": len(df_engineered),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "class_distribution": np.bincount(y).tolist(),
            "feature_dimensions": {
                "vital": len(feature_groups.vital),
                "symptom": len(feature_groups.symptom),
                "risk": len(feature_groups.risk),
                "context": len(feature_groups.context),
                "lab": len(feature_groups.lab),
                "interaction": len(feature_groups.interaction),
            },
            "indices": {
                "train": [int(i) for i in train_indices],
                "val": [int(i) for i in val_indices],
                "test": [int(i) for i in test_indices],
            },
        },
    }

    # Reproducibility: include git SHA and pip freeze
    metadata = {}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        metadata["git_sha"] = sha
    except Exception:
        metadata["git_sha"] = None
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode().splitlines()
        metadata["pip_freeze"] = pip_freeze
    except Exception:
        metadata["pip_freeze"] = None
    report["metadata"] = metadata

    overall_accuracy = clinical_metrics.get("overall_accuracy", 0.0)
    critical_sensitivity = clinical_metrics.get("clinical_safety", {}).get("critical_sensitivity", 0.0)
    under_triage_rate = clinical_metrics.get("clinical_safety", {}).get("under_triage_rate", 0.0)

    summary = {
        "overall_performance": "Good" if overall_accuracy >= 0.7 else "Needs Improvement",
        "key_findings": [
            f"Overall accuracy: {overall_accuracy:.3f}",
            f"Critical case sensitivity: {critical_sensitivity:.3f}",
            f"Under-triage rate: {under_triage_rate:.3f}",
            f"Macro-F1: {macro_f1:.3f}",
            f"Average inference time: {avg_inference_time_ms:.2f}ms",
        ],
        "recommendations": [],
        "risk_assessment": "Low" if critical_sensitivity >= 0.8 else "High",
        "calibration_temperature": chosen_temperature,
    }

    if use_conformal and threshold_info is not None and "val_critical_sensitivity" in threshold_info:
        summary["key_findings"].append(
            f"C³ guarantee (alpha={conformal_alpha:.2f}) with val critical sensitivity "
            f"{threshold_info['val_critical_sensitivity']:.3f}"
        )

    if overall_accuracy < 0.7:
        summary["recommendations"].append("Improve overall accuracy via hyperparameter tuning or data augmentation")
    if critical_sensitivity < 0.9:
        summary["recommendations"].append("Critical: Boost detection of top-2 severity (ESI1–2) cases")
    if under_triage_rate > 0.2:
        summary["recommendations"].append("Reduce under-triage rate to enhance patient safety")
    if not summary["recommendations"]:
        summary["recommendations"].append("Model performance meets clinical safety targets")

    report["summary"] = summary

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"advanced_evaluation_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["report_path"] = report_path

    model_path = os.path.join(output_dir, f"advanced_model_{timestamp}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_groups": feature_groups.__dict__,
        "class_weights": class_weights.tolist(),
        "training_history": history,
        "preprocess": preprocess,
        "split_indices": {
            "train": [int(i) for i in train_indices],
            "val": [int(i) for i in val_indices],
            "test": [int(i) for i in test_indices],
        },
    }, model_path)

    print(f"Report saved to: {report_path}")
    print(f"Model checkpoint saved to: {model_path}")

    return report, model


if __name__ == "__main__":  # pragma: no cover
    train_advanced_model()
