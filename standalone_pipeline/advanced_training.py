"""Baseline advanced training pipeline (no critical-focused adjustments)."""

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
import pandas as pd
import time
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

try:
    from .advanced_model_architecture import AdvancedClinicalSafetyLoss, AdvancedHierarchicalTriageModel
    from .evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics
    from .kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
    from .conformal import fit_conformal_thresholds, apply_conformal_thresholds
    from .utils import construct_fairness_data, get_critical_class_indices
except ImportError:  # pragma: no cover - script mode
    from advanced_model_architecture import AdvancedClinicalSafetyLoss, AdvancedHierarchicalTriageModel  # type: ignore
    from evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics  # type: ignore
    from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data  # type: ignore
    from conformal import fit_conformal_thresholds, apply_conformal_thresholds  # type: ignore
    from utils import construct_fairness_data, get_critical_class_indices  # type: ignore

# Optional DP library (Opacus)
try:  # pragma: no cover - optional
    from opacus import PrivacyEngine  # type: ignore
except Exception:  # pragma: no cover
    PrivacyEngine = None  # type: ignore

try:
    from .experimental.kaggle_enhanced_final_fix_v2 import advanced_kaggle_feature_engineering  # type: ignore
except Exception:  # pragma: no cover
    try:
        from experimental.kaggle_enhanced_final_fix_v2 import advanced_kaggle_feature_engineering  # type: ignore
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


def _get_temperature(model: nn.Module) -> float:
    temp = getattr(model, "calibration_temperature", None)
    if temp is None:
        return 1.0
    if isinstance(temp, torch.Tensor):
        return float(temp.detach().cpu().item())
    return float(temp)


def _compose_probs(
    outputs: torch.Tensor | Tuple[torch.Tensor, ...],
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    """Compose probabilities from model outputs, supporting flat, 2-head, or 3-head structures."""

    if isinstance(outputs, tuple):
        def _apply_temp(t: torch.Tensor) -> torch.Tensor:
            if temperature is None or abs(temperature - 1.0) < 1e-9:
                return t
            scale = torch.tensor(float(temperature), device=t.device, dtype=t.dtype).clamp_min(1e-6)
            return t / scale

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


def _ensure_group(df: pd.DataFrame, cols: List[str], prefix: str) -> List[str]:
    if cols:
        return cols
    placeholder = f"{prefix}_placeholder"
    if placeholder not in df.columns:
        df[placeholder] = 0.0
    return [placeholder]


def load_feature_engineered_dataframe(
    *, csv_path: Path = Path("src/kaggle_triage_data.csv"), use_advanced_fe: bool = True
) -> Tuple[pd.DataFrame, FeatureGroups, str]:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = load_kaggle_triage_data()

    # Optional: use advanced FE from experimental module (reproduces prior high-accuracy regime)
    if use_advanced_fe and advanced_kaggle_feature_engineering is not None:
        dff, vital_feats, symptom_feats, risk_feats, context_feats, lab_feats, interaction_feats = (
            advanced_kaggle_feature_engineering(df.copy())
        )
        if "esi" not in dff.columns:
            raise ValueError("Expected 'esi' column after advanced FE")
        esi_vals = pd.to_numeric(dff["esi"], errors="coerce").astype(int).clip(1, 5)
        dff["esi_5class_encoded"] = (5 - esi_vals).astype(int)
        target_col = "esi_5class_encoded"

        # Preserve raw gender for fairness if present
        if "gender" in dff.columns and "gender_original" not in dff.columns:
            dff["gender_original"] = dff["gender"].fillna("Unknown").astype(str)

        keep = list({*vital_feats, *symptom_feats, *risk_feats, *context_feats, *lab_feats, *interaction_feats})
        extra = [target_col] + (["gender_original"] if "gender_original" in dff.columns else [])
        df_engineered = dff[keep + extra].reset_index(drop=True)

        feature_groups = FeatureGroups(
            vital=_ensure_group(df_engineered, vital_feats, "vital"),
            symptom=_ensure_group(df_engineered, symptom_feats, "symptom"),
            risk=_ensure_group(df_engineered, risk_feats, "risk"),
            context=_ensure_group(df_engineered, context_feats, "context"),
            lab=_ensure_group(df_engineered, lab_feats, "lab"),
            interaction=_ensure_group(df_engineered, interaction_feats, "interaction"),
        )
        return df_engineered, feature_groups, target_col

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


def _to_tensor(df: pd.DataFrame, columns: Sequence[str]) -> torch.Tensor:
    return torch.tensor(df[columns].to_numpy(dtype=np.float32))


def _make_dataset(df: pd.DataFrame, y: np.ndarray, groups: FeatureGroups) -> TensorDataset:
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


def _train_one_epoch(
    model: AdvancedHierarchicalTriageModel,
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
    model: AdvancedHierarchicalTriageModel,
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

            running_loss += loss.item() * targets.size(0)
            temp = _get_temperature(model)
            probs = _compose_probs(outputs, temperature=temp)
            preds = probs.argmax(dim=1)
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
    min_critical_recall: float | None = 0.95,
    min_critical_precision: float | None = 0.85,
    metric: str = "f1_weighted_custom",
    tune_esi5: bool = True,
    bottom_range: Sequence[float] = (-0.06, -0.04, -0.02, 0.0),
) -> Tuple[List[float], Dict[str, float]]:
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

                true_crit = y_true >= crit_start
                pred_crit = pred >= crit_start
                recall_c = (np.sum(pred_crit & true_crit) / np.sum(true_crit)) if true_crit.any() else 1.0
                precision_c = (np.sum(pred_crit & true_crit) / np.sum(pred_crit)) if pred_crit.any() else 1.0
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
    dp_config: Optional[Dict] = None,
    use_conformal: bool = True,
    conformal_alpha: float = 0.05,
    use_advanced_fe: bool = True,
    calibrate_by: str = "nll",
) -> Tuple[Dict, AdvancedHierarchicalTriageModel]:
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    df_engineered, groups, target_col = load_feature_engineered_dataframe(use_advanced_fe=use_advanced_fe)
    X = df_engineered[groups.all()]
    y = df_engineered[target_col].astype(int).to_numpy()

    # Optional fast mode for DP sanity checks: sample a fraction before split
    try:
        sample_frac = float(os.getenv("TRIAJ_SAMPLE_FRACTION", "1.0"))
    except Exception:
        sample_frac = 1.0
    if 0.0 < sample_frac < 1.0:
        n_all = len(X)
        k_all = max(1, int(n_all * sample_frac))
        rng = np.random.default_rng(42)
        sel = rng.choice(n_all, size=k_all, replace=False)
        X = X.iloc[sel].reset_index(drop=True)
        y = y[sel]

    # Stratified splits (capture original indices for reproducibility)
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

    # Fairness metadata for validation/test splits (used for guarantees + reporting)
    _, val_group_labels = construct_fairness_data(df_engineered, X_val.index)
    fairness_attributes_test, test_group_labels = construct_fairness_data(df_engineered, X_test.index)

    train_loader = DataLoader(_make_dataset(X_train, y_train, groups), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_make_dataset(X_val, y_val, groups), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_make_dataset(X_test, y_test, groups), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = model_kwargs or {}
    model = AdvancedHierarchicalTriageModel(
        num_vital_features=len(groups.vital),
        num_symptom_features=len(groups.symptom),
        num_risk_features=len(groups.risk),
        num_context_features=len(groups.context),
        num_lab_features=len(groups.lab),
        num_interaction_features=len(groups.interaction),
        num_classes=len(np.unique(y)),
        **model_kwargs,
    ).to(device)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    loss_kwargs = dict(loss_kwargs or {})
    # Baseline: keep loss safety-neutral for accuracy sanity checks
    # (heavy penalties are reserved for the red-focus pipeline)
    if "critical_miss_penalty" in loss_kwargs and loss_kwargs["critical_miss_penalty"] is None:
        loss_kwargs.pop("critical_miss_penalty", None)
    criterion = AdvancedClinicalSafetyLoss(
        class_weights=torch.tensor(class_weights, dtype=torch.float32, device=device),
        # Accuracy-first baseline: down-weight safety penalties unless caller overrides
        w_focal=float(loss_kwargs.pop("w_focal", 0.1)),
        w_safety=float(loss_kwargs.pop("w_safety", 0.0)),
        w_critical=float(loss_kwargs.pop("w_critical", 0.0)),
        **loss_kwargs,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optional DP-SGD via Opacus
    dp_epsilon_per_epoch: list[float] = []
    if dp_config and PrivacyEngine is not None:
        try:
            noise = float(dp_config.get("noise_multiplier", 1.0))
            max_grad_norm = float(dp_config.get("max_grad_norm", 1.0))
            delta = float(dp_config.get("delta", 1e-5))
            sample_rate = min(1.0, batch_size / max(1, len(X_train)))
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise,
                max_grad_norm=max_grad_norm,
            )
        except Exception as e:
            print(f"Warning: DP-SGD setup failed ({e}); continuing without DP.")
            privacy_engine = None
    else:
        privacy_engine = None

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
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
        )

        if privacy_engine is not None:
            try:
                eps = privacy_engine.get_epsilon(delta=float(dp_config.get("delta", 1e-5)))  # type: ignore[arg-type]
                dp_epsilon_per_epoch.append(float(eps))
            except Exception:
                pass

    model.load_state_dict(best_state)

    if temperature is not None:
        model.set_temperature(temperature)
        chosen_temperature = temperature
    elif calibrate_temperature_grid:
        base_temp = model.get_temperature()
        best_temp = base_temp
        if calibrate_by == "accuracy":
            best_score = float("-inf")
        else:
            best_score = float("inf")
        for temp in calibrate_temperature_grid:
            model.set_temperature(temp)
            if calibrate_by == "accuracy":
                _, val_acc_tmp = _evaluate(model, val_loader, criterion, device)
                if val_acc_tmp > best_score:
                    best_score = val_acc_tmp
                    best_temp = temp
            else:
                _, _, y_val_tmp, _, prob_tmp = _evaluate(model, val_loader, criterion, device, collect_outputs=True)
                if prob_tmp is None or y_val_tmp is None:
                    continue
                nll = ReliabilityMetrics.negative_log_likelihood(y_val_tmp, prob_tmp)
                if nll < best_score:
                    best_score = nll
                    best_temp = temp
        model.set_temperature(best_temp)
        chosen_temperature = best_temp
    else:
        chosen_temperature = model.get_temperature()

    val_loss_collect, val_acc_collect, y_val_arr, _, val_prob = _evaluate(
        model, val_loader, criterion, device, collect_outputs=True
    )

    num_classes = int(len(np.unique(y)))
    critical_classes = get_critical_class_indices(num_classes)
    critical_threshold = min(critical_classes)

    tau_global: float | None = None
    tau_by_group: Dict[str, float] = {}
    class_thresholds = [0.0] * num_classes
    threshold_info = None

    if use_conformal:
        tau_global, tau_by_group = fit_conformal_thresholds(
            y_val_arr,
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
        val_metrics_conf = ClinicalMetrics.calculate_triage_metrics(y_val_arr, val_preds_conf)
        val_group_recall: Dict[str, float] = {}
        if val_group_labels.size:
            unique_groups = np.unique(val_group_labels.astype(str))
            val_crit_mask = y_val_arr >= critical_threshold
            for group in unique_groups:
                mask = (val_group_labels == group) & val_crit_mask
                if np.any(mask):
                    val_group_recall[str(group)] = float(
                        np.mean(val_preds_conf[mask] >= critical_threshold)
                    )
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
        # Accuracy-oriented threshold sweep for baseline (no safety constraints)
        if num_classes >= 5:
            if len(val_prob) > threshold_sample:
                idx = np.random.RandomState(42).choice(len(val_prob), size=threshold_sample, replace=False)
                probs_sweep = val_prob[idx]
                y_true_sweep = y_val_arr[idx]
            else:
                probs_sweep = val_prob
                y_true_sweep = y_val_arr
            class_thresholds, threshold_info = _sweep_thresholds_top2(
                probs_sweep,
                y_true_sweep,
                # Keep thresholds near-zero and optimise for accuracy
                critical_range=(0.0,),
                noncritical_default=0.0,
                class_weights=None,
                min_critical_recall=None,
                min_critical_precision=None,
                metric="accuracy",
                tune_esi5=False,
                bottom_range=(0.0,),
            )
            model.set_class_thresholds(class_thresholds, enable=True)
        else:
            raise ValueError(f"ESI-only configuration expects >=5 classes; got {num_classes}.")

    test_loss, test_acc, y_true, y_pred, y_prob = _evaluate(
        model, test_loader, criterion, device, collect_outputs=True
    )

    if use_conformal and tau_global is not None:
        y_pred = apply_conformal_thresholds(
            y_prob,
            tau_global,
            critical_classes,
            tau_by_group=tau_by_group,
            group_labels=test_group_labels,
        )
    elif threshold_sweep and (num_classes >= 5):
        adjusted = y_prob - np.array(class_thresholds, dtype=np.float32)
        preds = adjusted.argmax(axis=1)
        invalid = adjusted.max(axis=1) < 0.0
        if invalid.any():
            fallback = y_prob.argmax(axis=1)
            preds[invalid] = fallback[invalid]
        y_pred = preds

    total_samples = len(y_true)
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            *features, _ = batch
            features = [t.to(device) for t in features]
            model(*features)
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
    duration = max(time.perf_counter() - start, 1e-6)
    avg_inference_time_ms = (duration / total_samples) * 1000
    throughput = total_samples / duration

    clinical_metrics = ClinicalMetrics.calculate_triage_metrics(y_true, y_pred)

    # Reliability + calibration metrics
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
    model_size_mb = total_params * 4 / 1024 ** 2

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
        "privacy": {"dp": bool(privacy_engine is not None), "epsilon_per_epoch": dp_epsilon_per_epoch},
        "model_info": {
            "architecture": "AdvancedHierarchicalTriageModel",
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
                "vital": len(groups.vital),
                "symptom": len(groups.symptom),
                "risk": len(groups.risk),
                "context": len(groups.context),
                "lab": len(groups.lab),
                "interaction": len(groups.interaction),
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
    }

    if use_conformal and threshold_info is not None and "val_critical_sensitivity" in threshold_info:
        summary["key_findings"].append(
            f"C³ guarantee (alpha={conformal_alpha:.2f}) with val critical sensitivity "
            f"{threshold_info['val_critical_sensitivity']:.3f}"
        )

    if overall_accuracy < 0.7:
        summary["recommendations"].append("Improve overall accuracy via hyperparameter tuning or feature engineering")
    if critical_sensitivity < 0.9:
        summary["recommendations"].append("Increase sensitivity to top-2 severity (ESI1–2) cases (adjust loss weights or thresholds)")
    if under_triage_rate > 0.2:
        summary["recommendations"].append("Reduce under-triage rate to enhance patient safety")
    if not summary["recommendations"]:
        summary["recommendations"].append("Model performance meets clinical targets")

    report["summary"] = summary
    report["selection_criterion"] = {
        "primary_metric": "critical_sensitivity",
        "constraint": {"under_triage_rate": "<=0.05"},
        "tie_breaker": "macro_f1",
    }

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"advanced_evaluation_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = report_path

    model_path = os.path.join(output_dir, f"advanced_model_{timestamp}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_groups": groups.__dict__,
            "class_weights": class_weights.tolist(),
            "training_history": history,
            "preprocess": preprocess,
            "split_indices": {
                "train": [int(i) for i in train_indices],
                "val": [int(i) for i in val_indices],
                "test": [int(i) for i in test_indices],
            },
        },
        model_path,
    )

    print(f"Report saved to: {report_path}")
    print(f"Model checkpoint saved to: {model_path}")

    return report, model


if __name__ == "__main__":  # pragma: no cover
    train_advanced_model()
