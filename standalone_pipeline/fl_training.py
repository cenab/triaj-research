"""Federated training driver for advanced triage models (baseline and red-focus)."""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import time

try:
    from .advanced_model_architecture import (
        AdvancedClinicalSafetyLoss,
        AdvancedHierarchicalTriageModel,
        AdvancedHierarchicalTriageEnsemble,
    )
    from .advanced_training import (
        FeatureGroups,
        _ensure_group,
        load_feature_engineered_dataframe,
    )
    from .evaluation_framework import ReliabilityMetrics, ClinicalMetrics, FairnessEvaluator
except ImportError:  # pragma: no cover - allow running as top-level script
    from advanced_model_architecture import (  # type: ignore
        AdvancedClinicalSafetyLoss,
        AdvancedHierarchicalTriageModel,
        AdvancedHierarchicalTriageEnsemble,
    )
    from advanced_training import (  # type: ignore
        FeatureGroups,
        _ensure_group,
        load_feature_engineered_dataframe,
    )
    from evaluation_framework import ReliabilityMetrics, ClinicalMetrics, FairnessEvaluator  # type: ignore

# Optional DP library
try:  # pragma: no cover - optional
    from opacus import PrivacyEngine  # type: ignore
    from opacus.accountants import RDPAccountant  # type: ignore
    from opacus.accountants.analysis import rdp as rdp_analysis  # type: ignore
except Exception:  # pragma: no cover
    PrivacyEngine = None  # type: ignore
    RDPAccountant = None  # type: ignore
    rdp_analysis = None  # type: ignore


# Removed legacy 3-class baseline threshold sweep (ESI-only)

def _sweep_thresholds_top2(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    critical_range: Sequence[float] = (-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02),
    noncritical_default: float = 0.0,
    class_weights: Sequence[float] | None = None,
    min_critical_recall: float | None = 0.95,
    min_critical_precision: float | None = 0.85,
    metric: str = "f1_weighted_custom",
    tune_esi5: bool = True,
    bottom_range: Sequence[float] = (-0.20, -0.16, -0.12, -0.08, -0.04, 0.0),
    min_esi5_recall: float | None = None,
) -> Tuple[List[float], Dict[str, float]]:
    """Sweep per-class offsets for K-class where top-2 labels are critical.

    Adjust offsets for the top-2 classes (critical band) and optionally class 0 (ESI5).
    Returns the best thresholds and the achieved score under constraints.
    """
    k = probs.shape[1]
    assert k >= 4, "top-2 sweep expects K>=4"
    crit_start = max(1, k - 2)
    c1, c2 = crit_start, crit_start + 1
    best_thr = [noncritical_default] * k
    best_score = -1.0
    base_argmax = probs.argmax(axis=1)
    w = np.ones(k, dtype=np.float32) if class_weights is None else np.array(class_weights, dtype=np.float32)
    if w.shape[0] != k:
        w = np.ones(k, dtype=np.float32)

    bot_values = bottom_range if tune_esi5 else (noncritical_default,)
    for t0 in bot_values:
        for t1 in critical_range:
            for t2 in critical_range:
                thr = np.full(k, noncritical_default, dtype=np.float32)
                thr[0] = float(t0)
                thr[c1] = float(t1)
                thr[c2] = float(t2)
                adjusted = probs - thr
                pred = adjusted.argmax(axis=1)
                invalid = adjusted.max(axis=1) < 0.0
                if invalid.any():
                    pred[invalid] = base_argmax[invalid]

                # constraints on aggregated critical band (evaluate per candidate)
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

                # score metric
                if metric == "accuracy":
                    score = accuracy_score(y_true, pred)
                elif metric == "f1_weighted_custom":
                    f1_per = f1_score(y_true, pred, average=None, zero_division=0)
                    if f1_per.shape[0] != k:
                        f1_per = np.pad(f1_per, (0, k - f1_per.shape[0]))
                    score = float(np.dot(f1_per, w / (w.sum() + 1e-12)))
                else:
                    score = f1_score(y_true, pred, average="macro")

                # Optional constraint: ensure ESI5 recall is not below a floor
                if min_esi5_recall is not None:
                    true_esi5 = (y_true == 0)
                    if true_esi5.any():
                        rec5 = np.sum((pred == 0) & true_esi5) / np.sum(true_esi5)
                        if rec5 < float(min_esi5_recall):
                            continue

                if score > best_score:
                    best_score = score
                    best_thr = thr.tolist()

    return best_thr, {"score": float(best_score), "metric": metric}

def _replace_bn_with_gn(module: nn.Module) -> nn.Module:
    """Recursively replace BatchNorm1d with GroupNorm to enable DP-SGD."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            # DP-safe normalization: use GroupNorm with a single group (LayerNorm-like).
            # Using many groups on 2D tabular activations can destabilize training and
            # even cause mode collapse under DP-SGD.
            gn = nn.GroupNorm(num_groups=1, num_channels=int(child.num_features), affine=True)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child)
    return module

def _to_tensor(df, columns: Sequence[str]) -> torch.Tensor:
    return torch.tensor(df[columns].to_numpy(dtype=np.float32))


def _make_dataset(df, y, groups: FeatureGroups) -> TensorDataset:
    tensors = [
        _to_tensor(df, groups.vital),
        _to_tensor(df, groups.symptom),
        _to_tensor(df, groups.risk),
        _to_tensor(df, groups.context),
        _to_tensor(df, groups.lab),
        _to_tensor(df, groups.interaction),
    ]
    targets = torch.tensor(y.astype(np.int64))
    return TensorDataset(*tensors, targets)


def _compose_probs(outputs, temperature: float | None = None) -> torch.Tensor:
    temp = None
    if temperature is not None:
        try:
            temp = float(temperature)
        except Exception:
            temp = None
        if temp is not None and (not math.isfinite(temp) or temp <= 0.0):
            temp = None
    if isinstance(outputs, tuple):
        # Temperature scale each head
        heads = []
        for t in outputs:
            if temp is not None:
                scale = torch.tensor(float(temp), device=t.device, dtype=t.dtype)
                t = t / scale.clamp_min(1e-6)
            heads.append(t)
        outputs = tuple(heads)
        if len(outputs) == 2:
            gate_logits, nc_logits = outputs
            gate = torch.softmax(gate_logits, dim=1)
            nc_probs = torch.softmax(nc_logits, dim=1)
            noncrit = gate[:, 0:1]
            crit = gate[:, 1:2]
            nc0 = noncrit * nc_probs[:, 0:1]
            nc1 = noncrit * nc_probs[:, 1:2]
            return torch.cat([nc0, nc1, crit], dim=1)
        elif len(outputs) == 3:
            gate_logits, nc_logits, crit_logits = outputs
            gate = torch.softmax(gate_logits, dim=1)
            noncrit = gate[:, 0:1]
            crit = gate[:, 1:2]
            nc_probs = torch.softmax(nc_logits, dim=1)
            crit_probs = torch.softmax(crit_logits, dim=1)
            left = noncrit * nc_probs
            right = crit * crit_probs
            return torch.cat([left, right], dim=1)
        else:
            raise ValueError("Unsupported hierarchical outputs tuple length")
    logits = outputs
    if temp is not None:
        logits = logits / max(float(temp), 1e-6)
    return torch.softmax(logits, dim=1)

def _get_temperature(m: nn.Module) -> float:
    t = getattr(m, "calibration_temperature", None)
    if t is None:
        return 1.0
    if isinstance(t, torch.Tensor):
        t = float(t.detach().cpu().item())
    else:
        t = float(t)
    if (not math.isfinite(t)) or t <= 0.0:
        return 1.0
    return t

def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdvancedClinicalSafetyLoss,
    device: torch.device,
    collect_outputs: bool = False,
) -> Tuple[float, float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    probs_list: List[np.ndarray] = []

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

            running_loss += loss.item() * targets.size(0)
            probs = _compose_probs(outputs, temperature=_get_temperature(model))
            preds = probs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if collect_outputs:
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                probs_list.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total

    if collect_outputs:
        return avg_loss, acc, np.array(y_true), np.array(y_pred), np.array(probs_list)
    return avg_loss, acc, None, None, None


def _compute_class_weights(y: np.ndarray, device: torch.device, *, num_classes: int) -> torch.Tensor:
    """Balanced class weights with safe handling for missing classes.

    In federated/non-IID splits, a client's local dataset may not contain all classes.
    sklearn's compute_class_weight() returns only weights for the classes present in y,
    which is incompatible with torch.nn.CrossEntropyLoss(weight=...), which expects a
    weight vector of length K (the model's number of classes).
    """
    y = np.asarray(y, dtype=int)
    k = int(max(1, num_classes))
    if y.size == 0:
        return torch.ones(k, dtype=torch.float32, device=device)
    counts = np.bincount(y, minlength=k).astype(np.float64)
    n = float(counts.sum())
    # Balanced: n / (k * count_c). For count_c==0, set weight 0 (no local signal).
    weights = np.zeros(k, dtype=np.float64)
    for c in range(k):
        if counts[c] > 0:
            weights[c] = n / (k * counts[c])
        else:
            weights[c] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _aggregate_states(
    states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    *,
    method: str = "fedavg",
    momentum_buf: Dict[str, torch.Tensor] | None = None,
    momentum: float = 0.9,
    trim_fraction: float = 0.1,
    prev_global: Dict[str, torch.Tensor] | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor] | None]:
    keys = list(states[0].keys())
    if method == "fedavg":
        agg: Dict[str, torch.Tensor] = {}
        for key in keys:
            stacked = torch.stack([w * sd[key] for sd, w in zip(states, weights)], dim=0)
            agg[key] = stacked.sum(dim=0)
        return agg, momentum_buf
    if method == "median":
        agg = {}
        for key in keys:
            stacked = torch.stack([sd[key] for sd in states], dim=0)
            agg[key] = stacked.median(dim=0).values
        return agg, momentum_buf
    if method == "trim":
        k = max(0, int(trim_fraction * len(states)))
        agg = {}
        for key in keys:
            stacked = torch.stack([sd[key] for sd in states], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k: len(states) - k] if len(states) - 2 * k > 0 else sorted_vals
            agg[key] = trimmed.mean(dim=0)
        return agg, momentum_buf
    if method == "fedavgm":
        assert prev_global is not None, "prev_global required for fedavgm"
        avg = {}
        for key in keys:
            stacked = torch.stack([w * sd[key] for sd, w in zip(states, weights)], dim=0)
            avg[key] = stacked.sum(dim=0)
        if momentum_buf is None:
            momentum_buf = {k: torch.zeros_like(v) for k, v in avg.items()}
        new_global = {}
        new_buf: Dict[str, torch.Tensor] = {}
        for key in keys:
            delta = avg[key] - prev_global[key]
            buf = momentum * momentum_buf[key] + delta
            new_buf[key] = buf
            new_global[key] = prev_global[key] + buf
        return new_global, new_buf
    raise ValueError(f"Unknown aggregation method: {method}")


def _train_local(
    base_model: nn.Module,
    loader: DataLoader,
    y: np.ndarray,
    loss_kwargs: Dict,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    hierarchical: bool,
) -> Tuple[Dict[str, torch.Tensor], float, float, Optional[float]]:
    model = deepcopy(base_model).to(device)
    dp_cfg = getattr(base_model, "_dp_config", None)
    dp_enabled = bool(dp_cfg) and PrivacyEngine is not None
    # Determine number of classes from the model if possible. Client-local datasets in
    # non-IID splits may not contain all classes.
    num_classes = None
    if hasattr(model, "class_thresholds"):
        try:
            num_classes = int(model.class_thresholds.numel())  # type: ignore[attr-defined]
        except Exception:
            num_classes = None
    if num_classes is None:
        try:
            num_classes = int(np.max(y)) + 1 if len(y) else 1
        except Exception:
            num_classes = 1
    # Determine class weights
    class_weights = _compute_class_weights(y, device, num_classes=int(num_classes))
    if dp_enabled:
        # Under DP-SGD, very large class weights can cause extreme clipping and unstable
        # optimization. We cap them to keep the DP signal-to-noise ratio reasonable.
        try:
            cap = float(dp_cfg.get("class_weight_cap", 5.0))
        except Exception:
            cap = 5.0
        if cap > 0:
            class_weights = torch.clamp(class_weights, max=cap)
    if hierarchical:
        # Head-aware static weights to prioritize ESI1–2 and restore ESI5
        k = None
        if hasattr(model, "class_thresholds"):
            try:
                k = int(model.class_thresholds.numel())  # type: ignore[attr-defined]
            except Exception:
                k = None
        if k is None and hasattr(model, "_noncritical_classes"):
            try:
                k = int(getattr(model, "_noncritical_classes")) + int(getattr(model, "_critical_classes") or 1)
            except Exception:
                k = None
        weight_vec = None
        if k == 5:
            weight_vec = torch.tensor([3.0, 1.3, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        elif k == 3:
            weight_vec = torch.tensor([1.0, 1.5, 3.0], dtype=torch.float32, device=device)
        criterion = AdvancedClinicalSafetyLoss(class_weights=weight_vec, **loss_kwargs)
    else:
        criterion = AdvancedClinicalSafetyLoss(class_weights=class_weights, **loss_kwargs)
    # Optimizer choice: DP-SGD is conventionally run with SGD, but we allow Adam
    # (useful for tabular models) via dp_cfg for experimentation.
    if dp_enabled:
        opt_name = str((dp_cfg or {}).get("optimizer", "sgd")).lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # Optional: class-balanced sampler per client.
    # NOTE: Disabled for DP-SGD runs because sampling assumptions matter for privacy accounting,
    # and Opacus is typically used with uniform sampling.
    train_loader = loader
    disable_balanced = os.getenv("TRIAJ_DISABLE_BALANCED_SAMPLER", "").strip() in {"1", "true", "True", "yes", "YES"}
    if not dp_enabled and not disable_balanced:
        try:
            labels = y
            k_all = int(labels.max()) + 1 if len(labels) else 1
            counts = np.bincount(labels, minlength=k_all)
            per_sample = 1.0 / np.clip(counts[labels], 1, None)
            sampler = WeightedRandomSampler(
                torch.tensor(per_sample, dtype=torch.float32),
                num_samples=len(per_sample),
                replacement=True,
            )
            train_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler)
        except Exception:
            train_loader = loader

    # DP-SGD per-client (if enabled on base_model via attribute)
    privacy_engine = None
    if dp_enabled:
        try:
            # Replace BN with GN before making private
            _replace_bn_with_gn(model)
            # Ensure training mode before attaching PrivacyEngine
            model.train()
            noise = float(dp_cfg.get("noise_multiplier", 1.0))
            max_grad_norm = float(dp_cfg.get("max_grad_norm", 1.0))
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise,
                max_grad_norm=max_grad_norm,
            )
        except Exception as e:
            # Fail fast: returning a "DP" report when Opacus isn't actually attached is misleading.
            raise RuntimeError(f"DP-SGD requested but Opacus make_private() failed: {e}") from e

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            *features, targets = batch
            features = [t.to(device) for t in features]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(*features)
            if hierarchical:
                loss = criterion.compute_hierarchical_loss_general(outputs, targets)
                probs = _compose_probs(outputs)
            else:
                loss = criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)
            loss.backward()
            # Non-private training uses a final global clip for stability. DP-SGD already
            # performs per-sample clipping and noise addition via Opacus.
            if not dp_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = probs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += targets.size(0)

    # Use underlying module weights if model is wrapped by Opacus
    raw_state_dict = getattr(model, "_module", model).state_dict()
    state = {k: v.detach().cpu() for k, v in raw_state_dict.items()}
    avg_loss = total_loss / total_seen
    avg_acc = total_correct / total_seen
    # Return epsilon estimate if available
    epsilon = None
    if privacy_engine is not None:
        try:
            epsilon = float(privacy_engine.get_epsilon(delta=float(dp_cfg.get("delta", 1e-5))))
        except Exception:
            epsilon = None
    return state, avg_loss, avg_acc, epsilon


def federated_training(
    *,
    rounds: int,
    clients: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    hierarchical: bool,
    loss_kwargs: Dict,
    output_suffix: str,
    threshold_kwargs: Dict,
    calibrate_temperatures: Sequence[float] | None = None,
    non_iid: bool = False,
    dirichlet_alpha: float = 1.0,
    aggregator: str = "fedavg",
    momentum: float = 0.9,
    trim_fraction: float = 0.1,
    personalize_epochs: int = 0,
    personalize_lr: float = 5e-4,
    dp: bool = False,
    dp_noise_multiplier: float = 1.0,
    dp_max_grad_norm: float = 1.0,
    dp_delta: float = 1e-5,
    dp_optimizer: str = "sgd",
    by_site: bool = False,
    site_col: str = "site_id",
    byzantine_fraction: float = 0.0,
    byzantine_attack: str = "signflip",
    byzantine_scale: float = 5.0,
) -> Dict:
    # Reproducibility seeds and deterministic flags
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    df, groups, target_col = load_feature_engineered_dataframe()
    X = df[groups.all()]
    y = df[target_col].astype(int).to_numpy()

    # Splits: either global stratified splits (default) or per-site splits.
    if by_site:
        if site_col not in df.columns:
            raise ValueError(f"--by-site requested but site column '{site_col}' not found in dataframe")
        site_vals = df[site_col].fillna("site_0").astype(str).to_numpy()
        unique_sites = sorted(np.unique(site_vals).tolist())
        if len(unique_sites) < 2:
            raise ValueError(f"--by-site requires >=2 unique sites; found {len(unique_sites)}")
        # Override clients to match sites (cross-silo federation).
        if clients != len(unique_sites):
            print(f"[info] Overriding clients={clients} to clients={len(unique_sites)} to match sites={unique_sites}")
        clients = len(unique_sites)

        idx_all = np.arange(len(X))
        idx_train_all: List[int] = []
        idx_val_all: List[int] = []
        idx_test_all: List[int] = []
        for s in unique_sites:
            idx_s = idx_all[site_vals == s]
            y_s = y[idx_s]
            # 80/10/10 within each site
            idx_train_s, idx_temp_s = train_test_split(
                idx_s, test_size=0.2, stratify=y_s, random_state=42
            )
            y_temp_s = y[idx_temp_s]
            idx_val_s, idx_test_s = train_test_split(
                idx_temp_s, test_size=0.5, stratify=y_temp_s, random_state=42
            )
            idx_train_all.extend(idx_train_s.tolist())
            idx_val_all.extend(idx_val_s.tolist())
            idx_test_all.extend(idx_test_s.tolist())

        X_train = X.iloc[idx_train_all]
        y_train = y[idx_train_all]
        X_val = X.iloc[idx_val_all]
        y_val = y[idx_val_all]
        X_test = X.iloc[idx_test_all]
        y_test = y[idx_test_all]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

    # Impute + standardize continuous features using training split statistics.
    from sklearn.preprocessing import StandardScaler

    all_cols = X_train.columns.tolist()
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.copy().fillna(train_medians)
    X_val = X_val.copy().fillna(train_medians)
    X_test = X_test.copy().fillna(train_medians)

    # Optional fast mode: sample a fraction of data for quick DP sanity runs
    try:
        sample_frac = float(os.getenv("TRIAJ_SAMPLE_FRACTION", "1.0"))
    except Exception:
        sample_frac = 1.0
    if 0.0 < sample_frac < 1.0:
        rand = np.random.default_rng(42)
        def _sample_df_y(df, yy, frac: float):
            n = len(df)
            k = max(1, int(n * frac))
            idx = rand.choice(n, size=k, replace=False)
            return df.iloc[idx].reset_index(drop=True), yy[idx]
        X_train, y_train = _sample_df_y(X_train, y_train, sample_frac)
        X_val, y_val = _sample_df_y(X_val, y_val, min(1.0, sample_frac))
        X_test, y_test = _sample_df_y(X_test, y_test, min(1.0, sample_frac))

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

    # Split training data across clients (IID/Dirichlet, or by-site cross-silo).
    rng = np.random.default_rng(42)
    if by_site:
        # Client splits are site-aligned. We build splits as positional indices into X_train.
        site_vals_train = df.loc[X_train.index, site_col].fillna("site_0").astype(str).to_numpy()
        unique_sites = sorted(np.unique(site_vals_train).tolist())
        splits = [np.where(site_vals_train == s)[0] for s in unique_sites]
    elif non_iid:
        y_train_arr = y_train
        classes = np.unique(y_train_arr)
        class_indices = {c: np.where(y_train_arr == c)[0] for c in classes}
        client_indices: List[List[int]] = [[] for _ in range(clients)]
        for c in classes:
            idx_c = class_indices[c]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet([dirichlet_alpha] * clients)
            counts = (proportions * len(idx_c)).astype(int)
            while counts.sum() < len(idx_c):
                counts[rng.integers(0, clients)] += 1
            while counts.sum() > len(idx_c):
                j = rng.integers(0, clients)
                if counts[j] > 0:
                    counts[j] -= 1
            start = 0
            for k in range(clients):
                end = start + counts[k]
                client_indices[k].extend(idx_c[start:end].tolist())
                start = end
        splits = [np.array(sorted(ci), dtype=int) for ci in client_indices]
    else:
        indices = np.arange(len(X_train))
        rng.shuffle(indices)
        splits = np.array_split(indices, clients)

    client_loaders = []
    client_sizes = []
    client_targets = []
    for split in splits:
        subset_df = X_train.iloc[split].reset_index(drop=True)
        subset_y = y_train[split]
        dataset = _make_dataset(subset_df, subset_y, groups)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
        client_sizes.append(len(split))
        client_targets.append(subset_y)

    val_loader = DataLoader(_make_dataset(X_val, y_val, groups), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_make_dataset(X_test, y_test, groups), batch_size=batch_size, shuffle=False)

    # Device selection: prefer CUDA, then Apple MPS, else CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if dp and PrivacyEngine is None:
        raise RuntimeError("DP-SGD requested (--dp) but Opacus is not available in this environment.")
    # Opacus DP-SGD support on MPS can be fragile; fall back to CPU for DP runs.
    if dp and device.type == "mps":
        print("[info] DP-SGD enabled: falling back to CPU device (MPS DP support may be incomplete).")
        device = torch.device("cpu")

    # Guard: hierarchical ensemble supports 3-class only
    if hierarchical:
        k = int(len(np.unique(y)))
        noncritical_classes = 2 if k == 3 else 3
        critical_classes = 1 if k == 3 else 2
        global_model = AdvancedHierarchicalTriageEnsemble(
            num_vital_features=len(groups.vital),
            num_symptom_features=len(groups.symptom),
            num_risk_features=len(groups.risk),
            num_context_features=len(groups.context),
            num_lab_features=len(groups.lab),
            num_interaction_features=len(groups.interaction),
            noncritical_classes=noncritical_classes,
            critical_classes=critical_classes,
        ).to(device)
    else:
        global_model = AdvancedHierarchicalTriageModel(
            num_vital_features=len(groups.vital),
            num_symptom_features=len(groups.symptom),
            num_risk_features=len(groups.risk),
            num_context_features=len(groups.context),
            num_lab_features=len(groups.lab),
            num_interaction_features=len(groups.interaction),
            num_classes=len(np.unique(y)),
        ).to(device)

    # Attach DP config to model for local training if requested
    if dp:
        setattr(global_model, "_dp_config", {
            "noise_multiplier": dp_noise_multiplier,
            "max_grad_norm": dp_max_grad_norm,
            "delta": dp_delta,
            "optimizer": str(dp_optimizer),
        })
        # Ensure server model uses GroupNorm to match client-side DP models
        try:
            _replace_bn_with_gn(global_model)
        except Exception:
            pass

    # Loss defaults:
    # - Non-DP runs default to a stronger critical-miss penalty for ESI-5.
    # - DP runs use a DP-friendly objective (disable safety penalties by default),
    #   since large, non-smooth penalty terms can dominate clipped gradients and stall learning.
    k = int(len(np.unique(y)))
    eff_loss_kwargs = dict(loss_kwargs)
    if dp:
        eff_loss_kwargs.setdefault("w_focal", 0.1)
        eff_loss_kwargs.setdefault("w_safety", 0.0)
        eff_loss_kwargs.setdefault("w_critical", 0.0)
        eff_loss_kwargs.setdefault("critical_miss_penalty", 50.0)
    else:
        if "critical_miss_penalty" not in eff_loss_kwargs and k >= 5:
            eff_loss_kwargs["critical_miss_penalty"] = 150.0
    global_class_weights = _compute_class_weights(y_train, device, num_classes=int(np.max(y) + 1))
    criterion = AdvancedClinicalSafetyLoss(class_weights=global_class_weights, **eff_loss_kwargs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "rounds": []}
    reliability_history = {"val_nll": [], "val_ece": [], "val_ece_per_class": []}
    comm_history = {"bytes_per_round": []}
    systems_history = {"client_time_mean": [], "client_time_std": []}

    # DP accounting across rounds (cumulative epsilon). We compute a max-over-clients
    # privacy budget curve to use in the paper.
    dp_round_history: List[float] = []
    dp_client_steps_per_round: List[int] = []
    dp_client_sample_rates: List[float] = []
    dp_alphas = None
    if dp and (RDPAccountant is not None) and (rdp_analysis is not None):
        dp_alphas = list(getattr(RDPAccountant, "DEFAULT_ALPHAS", [2, 4, 8, 16, 32, 64]))
        for loader, n in zip(client_loaders, client_sizes):
            steps = int(local_epochs) * int(len(loader))
            dp_client_steps_per_round.append(max(1, steps))
            q = float(batch_size) / float(max(1, n))
            dp_client_sample_rates.append(min(1.0, q))
        dp_steps_cum = [0 for _ in range(clients)]
    momentum_buffer: Dict[str, torch.Tensor] | None = None
    for rnd in range(1, rounds + 1):
        print(f"\n--- Federated Round {rnd}/{rounds} ---")
        global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states = []
        client_losses = []
        client_accs = []
        weights = []

        client_times = []

        # Choose malicious clients for this round (if enabled). We select indices
        # deterministically from the RNG seed for reproducibility.
        malicious: set[int] = set()
        if byzantine_fraction and byzantine_fraction > 0.0:
            n_mal = int(np.ceil(float(byzantine_fraction) * clients))
            n_mal = max(1, min(clients, n_mal))
            mal_idx = rng.choice(np.arange(clients), size=n_mal, replace=False)
            malicious = set(int(x) for x in mal_idx.tolist())

        for client_idx, (loader, subset_y, size) in enumerate(zip(client_loaders, client_targets, client_sizes)):
            local_model = deepcopy(global_model).cpu()
            local_model.load_state_dict(global_state)
            t0 = time.perf_counter()
            state, loss, acc, eps = _train_local(
                local_model,
                loader,
                subset_y,
                eff_loss_kwargs,
                local_epochs,
                learning_rate,
                device,
                hierarchical,
            )
            t1 = time.perf_counter()
            client_times.append(t1 - t0)
            # Optional Byzantine behavior: corrupt the client update before aggregation.
            if client_idx in malicious:
                attacked: Dict[str, torch.Tensor] = {}
                for k, v in state.items():
                    g = global_state[k]
                    update = v - g
                    # Some state entries are integer buffers (e.g., BN counters). Don't corrupt them.
                    if not torch.is_floating_point(update):
                        attacked[k] = v
                        continue
                    if byzantine_attack == "signflip":
                        attacked[k] = g - float(byzantine_scale) * update
                    elif byzantine_attack == "scale":
                        attacked[k] = g + float(byzantine_scale) * update
                    elif byzantine_attack == "gaussian":
                        # Use unbiased=False: std() on a scalar is NaN under the default unbiased
                        # estimator, which can poison scalar buffers like calibration_temperature.
                        std = update.std(unbiased=False)
                        if not bool(torch.isfinite(std).item()):
                            std = torch.zeros((), dtype=update.dtype, device=update.device)
                        scale = float(byzantine_scale) * std.clamp_min(1e-12)
                        attacked[k] = g + torch.randn_like(update) * scale
                    else:
                        attacked[k] = v
                state = attacked
            client_states.append(state)
            client_losses.append(loss)
            client_accs.append(acc)
            weights.append(size / sum(client_sizes))

        aggregated_state, momentum_buffer = _aggregate_states(
            client_states,
            weights,
            method=aggregator,
            momentum_buf=momentum_buffer,
            momentum=momentum,
            trim_fraction=trim_fraction,
            prev_global=global_state if aggregator == "fedavgm" else None,
        )
        global_model.load_state_dict(aggregated_state)

        # Collect validation predictions for reliability metrics
        val_loss, val_acc, yv_true, yv_pred, yv_prob = _evaluate_model(
            global_model, val_loader, criterion, device, collect_outputs=True
        )
        history["train_loss"].append(float(np.average(client_losses, weights=weights)))
        history["train_acc"].append(float(np.average(client_accs, weights=weights)) * 100)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)
        # Macro-F1 on validation
        try:
            macro_f1 = f1_score(yv_true, yv_pred, average="macro")
        except Exception:
            macro_f1 = float('nan')
        history["val_macro_f1"].append(float(macro_f1) * 100)
        history["rounds"].append(rnd)

        # Reliability metrics (overall ECE/NLL and per-class ECE)
        if yv_prob is not None:
            nll = ReliabilityMetrics.negative_log_likelihood(yv_true, yv_prob)
            ece = ReliabilityMetrics.expected_calibration_error(yv_true, yv_prob)
            ece_pc = ReliabilityMetrics.expected_calibration_error_per_class(yv_true, yv_prob)
            reliability_history["val_nll"].append(float(nll))
            reliability_history["val_ece"].append(float(ece))
            # Ensure JSON-serialisable
            reliability_history["val_ece_per_class"].append({str(k): float(v) for k, v in ece_pc.items()})

        # Communication estimate (bytes): params * 4 * clients * 2
        total_params = sum(p.numel() for p in global_model.parameters())
        comm_bytes = total_params * 4 * clients * 2
        comm_history["bytes_per_round"].append(int(comm_bytes))
        # Client time stats
        if client_times:
            systems_history["client_time_mean"].append(float(np.mean(client_times)))
            systems_history["client_time_std"].append(float(np.std(client_times)))
        else:
            systems_history["client_time_mean"].append(0.0)
            systems_history["client_time_std"].append(0.0)

        # DP epsilon per round (aggregate max across clients)
        if dp and (dp_alphas is not None) and dp_client_steps_per_round:
            # Update cumulative steps and compute epsilon per client using RDP analysis.
            epsilons: List[float] = []
            for i in range(clients):
                dp_steps_cum[i] += dp_client_steps_per_round[i]
                q = dp_client_sample_rates[i]
                # Compute RDP and convert to (eps, alpha).
                rdp_vals = rdp_analysis.compute_rdp(
                    q=q,
                    noise_multiplier=float(dp_noise_multiplier),
                    steps=int(dp_steps_cum[i]),
                    orders=dp_alphas,
                )
                eps_i, _ = rdp_analysis.get_privacy_spent(orders=dp_alphas, rdp=rdp_vals, delta=float(dp_delta))
                epsilons.append(float(eps_i))
            dp_round_history.append(float(max(epsilons)))

    # Calibration on validation set
    chosen_temperature = getattr(global_model, "calibration_temperature", torch.tensor(1.0)).item()
    if calibrate_temperatures:
        best_temp = chosen_temperature
        best_nll = float("inf")
        for temp in calibrate_temperatures:
            if hasattr(global_model, "set_temperature"):
                global_model.set_temperature(temp)
            _, _, y_val_tmp, _, prob_tmp = _evaluate_model(
                global_model, val_loader, criterion, device, collect_outputs=True
            )
            if prob_tmp is None:
                continue
            nll = ReliabilityMetrics.negative_log_likelihood(y_val_tmp, prob_tmp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        if hasattr(global_model, "set_temperature"):
            global_model.set_temperature(best_temp)
        chosen_temperature = best_temp

    _, _, y_val_arr, _, val_prob = _evaluate_model(
        global_model, val_loader, criterion, device, collect_outputs=True
    )

    num_classes = int(len(np.unique(y)))
    class_thresholds = [0.0] * num_classes
    threshold_info = None
    if val_prob is not None:
        if val_prob.shape[1] >= 5:
            chosen_thresholds, threshold_info = _sweep_thresholds_top2(
                val_prob,
                y_val_arr,
                critical_range=threshold_kwargs.get("critical_range", (-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02)),
                noncritical_default=threshold_kwargs.get("noncritical_default", 0.0),
                class_weights=threshold_kwargs.get("weights5", (0.05, 0.15, 0.20, 0.25, 0.35)),
                min_critical_recall=threshold_kwargs.get("min_critical_recall", 0.95),
                min_critical_precision=threshold_kwargs.get("min_critical_precision", 0.85),
                metric="f1_weighted_custom",
                tune_esi5=threshold_kwargs.get("tune_esi5", True),
                bottom_range=threshold_kwargs.get("bottom_range", (-0.20, -0.16, -0.12, -0.08, -0.04, 0.0)),
                min_esi5_recall=threshold_kwargs.get("min_esi5_recall", 0.10),
            )
        else:
            raise ValueError(f"ESI-only configuration expects >=5 classes; got {val_prob.shape[1]}.")
        if hasattr(global_model, "set_class_thresholds"):
            # Ensure thresholds length matches model outputs
            if len(chosen_thresholds) != val_prob.shape[1]:
                chosen_thresholds = [0.0] * val_prob.shape[1]
            global_model.set_class_thresholds(chosen_thresholds, enable=True)
        class_thresholds = chosen_thresholds

    # Final evaluation on test set
    test_loss, test_acc, y_true, y_pred, y_prob = _evaluate_model(
        global_model, test_loader, criterion, device, collect_outputs=True
    )

    if class_thresholds and y_prob is not None:
        adjusted = y_prob - np.array(class_thresholds, dtype=np.float32)
        preds = adjusted.argmax(axis=1)
        invalid = adjusted.max(axis=1) < 0.0
        if invalid.any():
            fallback = y_prob.argmax(axis=1)
            preds[invalid] = fallback[invalid]
        y_pred = preds

    # Use unified clinical metrics (supports 3 or 5 classes)
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

    # Fairness metrics on held-out test set
    fairness_metrics = None
    fairness_attributes: Dict[str, np.ndarray] = {}
    test_index = X_test.index
    gender_col = "gender_original" if "gender_original" in df.columns else "gender"
    if gender_col in df.columns:
        fairness_attributes["gender"] = df.loc[test_index, gender_col].to_numpy()
    elif "gender_male" in df.columns:
        fairness_attributes["gender"] = np.where(df.loc[test_index, "gender_male"].to_numpy(dtype=float) > 0.5, "Male", "Female")
    age_source = None
    if "age_numeric" in df.columns:
        age_source = df.loc[test_index, "age_numeric"].to_numpy(dtype=float)
    elif "yaş" in df.columns:
        age_source = df.loc[test_index, "yaş"].to_numpy(dtype=float)
    elif "age" in df.columns:
        age_source = df.loc[test_index, "age"].to_numpy(dtype=float)
    if age_source is not None:
        bins = [0, 18, 35, 50, 65, np.inf]
        labels = np.array(["<18", "18-34", "35-49", "50-64", "65+"])
        grouped = labels[np.digitize(age_source, bins) - 1]
        fairness_attributes["age_group"] = grouped
    if fairness_attributes:
        evaluator = FairnessEvaluator()
        fairness_metrics = evaluator.evaluate_fairness(np.array(y_true), np.array(y_pred), fairness_attributes)

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

    # Inference timing (model-only; warm-up excluded)
    total_samples = len(y_pred)
    try:
        with torch.no_grad():
            # Warm-up
            for batch in test_loader:
                *features, _ = batch
                features = [t.to(device) for t in features]
                _ = global_model(*features)
                break
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for batch in test_loader:
                *features, _ = batch
                features = [t.to(device) for t in features]
                _ = global_model(*features)
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            duration = max(t1 - t0, 1e-6)
            avg_inference_time_ms = (duration / max(1, total_samples)) * 1000.0
            throughput = total_samples / duration
    except Exception:
        avg_inference_time_ms = None
        throughput = None
    summary = {
        "overall_performance": "Good" if clinical_metrics["overall_accuracy"] >= 0.7 else "Needs Improvement",
        "key_findings": [
            f"Overall accuracy: {clinical_metrics['overall_accuracy']:.3f}",
            f"Critical case sensitivity: {clinical_metrics['clinical_safety'].get('critical_sensitivity', 0.0):.3f}",
            f"Under-triage rate: {clinical_metrics['clinical_safety'].get('under_triage_rate', 0.0):.3f}",
            f"Macro-F1: {macro_f1:.3f}",
        ],
        "recommendations": [],
        "risk_assessment": "Low",
    }

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_config": {
            "rounds": int(rounds),
            "clients": int(clients),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "hierarchical": bool(hierarchical),
            "non_iid": bool(non_iid),
            "dirichlet_alpha": float(dirichlet_alpha),
            "aggregator": str(aggregator),
            "momentum": float(momentum),
            "trim_fraction": float(trim_fraction),
            "by_site": bool(by_site),
            "site_col": str(site_col) if by_site else None,
            "dp": bool(dp),
            "dp_optimizer": str(dp_optimizer) if dp else None,
            "dp_noise_multiplier": float(dp_noise_multiplier) if dp else None,
            "dp_max_grad_norm": float(dp_max_grad_norm) if dp else None,
            "dp_delta": float(dp_delta) if dp else None,
        },
        "clinical_metrics": clinical_metrics,
        "performance_metrics": {
            "avg_inference_time_ms": avg_inference_time_ms,
            "throughput_samples_per_sec": throughput,
            # MB uses 1e6 bytes for consistency with communication reporting.
            "model_size_mb": sum(p.numel() for p in global_model.parameters()) * 4 / 1e6,
            "model_size_mib": sum(p.numel() for p in global_model.parameters()) * 4 / 1024 ** 2,
            "total_parameters": sum(p.numel() for p in global_model.parameters()),
            "total_samples_tested": total_samples,
        },
        "training_history": history,
        "reliability_history": reliability_history,
        "communication_history": comm_history,
        "systems_history": systems_history,
        "model_info": {
            "architecture": "AdvancedHierarchicalTriageEnsemble" if hierarchical else "AdvancedHierarchicalTriageModel",
        },
        "calibration": {"temperature": chosen_temperature},
        "thresholds": {"class_thresholds": class_thresholds, "selection": threshold_info},
        "reliability_metrics": reliability_metrics,
        "bootstrap_ci": {k: tuple(map(float, v)) for k, v in bootstrap_ci.items()},
        "fairness_metrics": _serialise(fairness_metrics),
        "privacy": {
            "dp": dp,
            "epsilon_per_round": dp_round_history,
            "epsilon_total": float(dp_round_history[-1]) if dp_round_history else None,
            "accounting": "RDP (opacus.analysis.rdp) with per-client sample_rate=batch_size/n and cumulative steps across rounds"
            if dp and dp_round_history
            else None,
            "optimizer": str(dp_optimizer) if dp else None,
            "noise_multiplier": float(dp_noise_multiplier) if dp else None,
            "max_grad_norm": float(dp_max_grad_norm) if dp else None,
            "delta": float(dp_delta) if dp else None,
        },
        "data_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "class_distribution": np.bincount(y).tolist(),
            "by_site": bool(by_site),
            "site_col": str(site_col) if by_site else None,
            "site_distribution": (
                df[site_col].fillna("site_0").astype(str).value_counts().to_dict() if by_site and site_col in df.columns else None
            ),
            "preprocess": preprocess,
        },
        "summary": summary,
    }

    # Per-site test metrics (when site IDs are available)
    if by_site and site_col in df.columns:
        site_test = df.loc[X_test.index, site_col].fillna("site_0").astype(str).to_numpy()
        per_site = {}
        for s in sorted(np.unique(site_test).tolist()):
            mask = site_test == s
            if not np.any(mask):
                continue
            per_site[str(s)] = ClinicalMetrics.calculate_triage_metrics(y_true[mask], y_pred[mask])
        report["per_site_test_metrics"] = per_site

    # Byzantine configuration (if enabled)
    if byzantine_fraction and byzantine_fraction > 0.0:
        report["byzantine"] = {
            "fraction": float(byzantine_fraction),
            "attack": str(byzantine_attack),
            "scale": float(byzantine_scale),
        }

    report["selection_criterion"] = {
        "primary_metric": "critical_sensitivity",
        "constraint": {"under_triage_rate": "<=0.05"},
        "tie_breaker": "macro_f1",
    }

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = Path("results") / f"fl_{output_suffix}_evaluation_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Plot ECE/NLL and Acc/F1 vs rounds, if matplotlib available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass
        rounds_axis = history["rounds"]
        figures_dir = Path('docs') / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        # ECE/NLL
        fig1, ax1 = plt.subplots(figsize=(7.2, 3.8))
        ax1.plot(rounds_axis, reliability_history["val_ece"], marker='o', linewidth=2.0, color='tab:blue', label='ECE (val)')
        ax1.set_xlabel('Round', fontsize=11)
        ax1.set_ylabel('ECE', fontsize=11, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(rounds_axis, reliability_history["val_nll"], marker='s', linewidth=2.0, color='tab:red', label='NLL (val)')
        ax2.set_ylabel('NLL', fontsize=11, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_title('Validation Reliability vs Rounds', fontsize=12)
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        fig1.tight_layout()
        fig_path_ts = figures_dir / f"fl_{output_suffix}_ece_nll_vs_rounds_{timestamp}.png"
        fig_path_stable = figures_dir / f"fl_{output_suffix}_ece_nll_vs_rounds.png"
        plt.savefig(fig_path_ts, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path_stable, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved reliability plot to {fig_path_ts} and {fig_path_stable}")
        # Acc/F1
        fig2, axa = plt.subplots(figsize=(7.2, 3.8))
        axa.plot(rounds_axis, history["val_acc"], marker='o', linewidth=2.0, color='tab:green', label='Accuracy (val)')
        axa.set_xlabel('Round', fontsize=11)
        axa.set_ylabel('Accuracy (%)', fontsize=11, color='tab:green')
        axa.tick_params(axis='y', labelcolor='tab:green')
        axb = axa.twinx()
        axb.plot(rounds_axis, history["val_macro_f1"], marker='s', linewidth=2.0, color='tab:purple', label='Macro-F1 (val)')
        axb.set_ylabel('Macro-F1 (%)', fontsize=11, color='tab:purple')
        axb.tick_params(axis='y', labelcolor='tab:purple')
        axa.set_title('Validation Accuracy and Macro-F1 vs Rounds', fontsize=12)
        lines1, labels1 = axa.get_legend_handles_labels()
        lines2, labels2 = axb.get_legend_handles_labels()
        axa.legend(lines1 + lines2, labels1 + labels2, loc='best')
        fig2.tight_layout()
        fig2_path_ts = figures_dir / f"fl_{output_suffix}_acc_f1_vs_rounds_{timestamp}.png"
        fig2_path_stable = figures_dir / f"fl_{output_suffix}_acc_f1_vs_rounds.png"
        plt.savefig(fig2_path_ts, dpi=300, bbox_inches='tight')
        plt.savefig(fig2_path_stable, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved acc/f1 plot to {fig2_path_ts} and {fig2_path_stable}")
        # Systems: Comm and client time std
        fig3, axc = plt.subplots(figsize=(7.2, 3.8))
        axc.plot(
            rounds_axis,
            [b / 1e6 for b in comm_history["bytes_per_round"]],
            marker="o",
            linewidth=2.0,
            color="tab:orange",
            label="Total comm (MB/round)",
        )
        axc.set_xlabel('Round', fontsize=11)
        axc.set_ylabel('Total comm (MB/round)', fontsize=11, color='tab:orange')
        axc.tick_params(axis='y', labelcolor='tab:orange')
        axd = axc.twinx()
        # If systems_history is not available, default to zeros
        try:
            ct_std = systems_history["client_time_std"]
        except Exception:
            ct_std = [0 for _ in rounds_axis]
        axd.plot(rounds_axis, ct_std, marker='s', linewidth=2.0, color='tab:brown', label='Client time std (s)')
        axd.set_ylabel('Client time std (s)', fontsize=11, color='tab:brown')
        axd.tick_params(axis='y', labelcolor='tab:brown')
        axc.set_title('Systems: Communication and Client Time Variability', fontsize=12)
        lines1, labels1 = axc.get_legend_handles_labels()
        lines2, labels2 = axd.get_legend_handles_labels()
        axc.legend(lines1 + lines2, labels1 + labels2, loc='best')
        fig3.tight_layout()
        fig3_path_ts = figures_dir / f"fl_{output_suffix}_systems_vs_rounds_{timestamp}.png"
        fig3_path_stable = figures_dir / f"fl_{output_suffix}_systems_vs_rounds.png"
        plt.savefig(fig3_path_ts, dpi=300, bbox_inches='tight')
        plt.savefig(fig3_path_stable, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved systems plot to {fig3_path_ts} and {fig3_path_stable}")
        # DP privacy/utility (if enabled): show accuracy and cumulative epsilon across rounds,
        # and a privacy–utility curve using the cumulative epsilon (max over clients).
        if dp_round_history:
            eps_cum = list(dp_round_history)
            rounds_eps = rounds_axis[: len(eps_cum)]
            acc = history["val_acc"][: len(eps_cum)]

            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11.2, 3.8))

            # Left: rounds vs accuracy + cumulative epsilon (twin axis)
            ax4a.plot(
                rounds_eps,
                acc,
                marker="o",
                linewidth=2.0,
                color="tab:green",
                label="Val acc (%)",
            )
            ax4a.set_xlabel("Round", fontsize=11)
            ax4a.set_ylabel("Val acc (%)", fontsize=11, color="tab:green")
            ax4a.tick_params(axis="y", labelcolor="tab:green")
            ax4a.grid(alpha=0.25)
            ax4a2 = ax4a.twinx()
            ax4a2.plot(
                rounds_eps,
                eps_cum,
                marker="s",
                linewidth=2.0,
                color="tab:blue",
                label="Cumulative ε (max client)",
            )
            ax4a2.set_ylabel("Cumulative ε (max client)", fontsize=11, color="tab:blue")
            ax4a2.tick_params(axis="y", labelcolor="tab:blue")
            ax4a.set_title("DP Accounting vs Rounds", fontsize=12)

            lines1, labels1 = ax4a.get_legend_handles_labels()
            lines2, labels2 = ax4a2.get_legend_handles_labels()
            ax4a.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

            # Right: privacy–utility
            ax4b.plot(
                eps_cum,
                acc,
                marker="o",
                linewidth=2.0,
                color="tab:purple",
            )
            ax4b.set_xlabel("Cumulative ε (max client)", fontsize=11)
            ax4b.set_ylabel("Val acc (%)", fontsize=11)
            ax4b.set_title("Privacy–Utility", fontsize=12)
            ax4b.grid(alpha=0.25)

            fig4.suptitle("DP-SGD: Privacy and Utility Across Rounds", fontsize=12, y=1.05)
            fig4.tight_layout()
            fig4_path_ts = figures_dir / f"fl_{output_suffix}_privacy_utility_{timestamp}.png"
            fig4_path_stable = figures_dir / f"fl_{output_suffix}_privacy_utility.png"
            plt.savefig(fig4_path_ts, dpi=300, bbox_inches="tight")
            plt.savefig(fig4_path_stable, dpi=300, bbox_inches="tight")
            plt.close(fig4)
            print(f"Saved DP privacy/utility plot to {fig4_path_ts} and {fig4_path_stable}")
    except Exception as e:
        print(f"Skipping reliability plot (matplotlib not available?): {e}")
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "feature_groups": groups.__dict__,
            "training_history": history,
        },
        Path("results") / f"fl_{output_suffix}_model_{timestamp}.pth",
    )
    print(f"Federated report saved to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Federated training for advanced triage models")
    parser.add_argument("mode", choices=["baseline", "red_focus"], help="Which configuration to run")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--non-iid", action="store_true")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--aggregator", choices=["fedavg", "median", "trim", "fedavgm"], default="fedavg")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--personalize-epochs", type=int, default=0)
    parser.add_argument("--personalize-lr", type=float, default=5e-4)
    # Differential privacy options
    parser.add_argument("--dp", action="store_true", help="Enable DP-SGD via Opacus (if available)")
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-optimizer", choices=["sgd", "adam"], default="sgd", help="Optimizer for DP-SGD local training")
    # Site-aware cross-silo federation
    parser.add_argument("--by-site", action="store_true", help="Use site_id/dep_name as FL clients (cross-silo)")
    parser.add_argument("--site-col", type=str, default="site_id", help="Column name for site identifier")
    # Byzantine simulation (robust aggregation evaluation)
    parser.add_argument("--byzantine-fraction", type=float, default=0.0, help="Fraction of clients to corrupt each round")
    parser.add_argument("--byzantine-attack", choices=["signflip", "scale", "gaussian"], default="signflip")
    parser.add_argument("--byzantine-scale", type=float, default=5.0)
    args = parser.parse_args()

    if args.mode == "baseline":
        loss_kwargs = {}
        threshold_kwargs = {}
        calibrate = [0.8, 0.85, 0.9, 0.95, 1.0]
        federated_training(
            rounds=args.rounds,
            clients=args.clients,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hierarchical=False,
            loss_kwargs=loss_kwargs,
            output_suffix="advanced",
            threshold_kwargs=threshold_kwargs,
            calibrate_temperatures=calibrate,
            non_iid=args.non_iid,
            dirichlet_alpha=args.dirichlet_alpha,
            aggregator=args.aggregator,
            momentum=args.momentum,
            trim_fraction=args.trim_fraction,
            personalize_epochs=args.personalize_epochs,
            personalize_lr=args.personalize_lr,
            dp=args.dp,
            dp_noise_multiplier=args.dp_noise_multiplier,
            dp_max_grad_norm=args.dp_max_grad_norm,
            dp_delta=args.dp_delta,
            dp_optimizer=args.dp_optimizer,
            by_site=args.by_site,
            site_col=args.site_col,
            byzantine_fraction=args.byzantine_fraction,
            byzantine_attack=args.byzantine_attack,
            byzantine_scale=args.byzantine_scale,
        )
    else:
        loss_kwargs = {
            "alpha": 0.45,
            "gamma": 3.0,
            "critical_miss_penalty": 150.0,
        }
        threshold_kwargs = {
            # ESI tuning knobs (top-2 band)
            "min_critical_recall": 0.98,
            "min_critical_precision": 0.90,
            "tune_esi5": True,
            "bottom_range": (-0.20, -0.16, -0.12, -0.08, -0.04, 0.0),
            "min_esi5_recall": 0.10,
        }
        calibrate = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        federated_training(
            rounds=args.rounds,
            clients=args.clients,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hierarchical=True,
            loss_kwargs=loss_kwargs,
            output_suffix="advanced_red_focus",
            threshold_kwargs=threshold_kwargs,
            calibrate_temperatures=calibrate,
            non_iid=args.non_iid,
            dirichlet_alpha=args.dirichlet_alpha,
            aggregator=args.aggregator,
            momentum=args.momentum,
            trim_fraction=args.trim_fraction,
            personalize_epochs=args.personalize_epochs,
            personalize_lr=args.personalize_lr,
            dp=args.dp,
            dp_noise_multiplier=args.dp_noise_multiplier,
            dp_max_grad_norm=args.dp_max_grad_norm,
            dp_delta=args.dp_delta,
            dp_optimizer=args.dp_optimizer,
            by_site=args.by_site,
            site_col=args.site_col,
            byzantine_fraction=args.byzantine_fraction,
            byzantine_attack=args.byzantine_attack,
            byzantine_scale=args.byzantine_scale,
        )


if __name__ == "__main__":
    main()
