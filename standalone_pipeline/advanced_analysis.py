"""Post-training analysis (fairness + SHAP) for the advanced triage model.

Includes publication-quality plotting with clear titles, labels, and styling.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from typing import List as _ListAlias
import pandas as pd

try:
    import shap
except ImportError:
    shap = None  # SHAP is optional; analysis will skip SHAP if unavailable

try:
    from .advanced_training import (
        FeatureGroups,
        _ensure_group,
        load_feature_engineered_dataframe,
        _make_dataset,
    )
    from .advanced_model_architecture import AdvancedHierarchicalTriageModel, AdvancedHierarchicalTriageEnsemble
    from .evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics
except ImportError:  # pragma: no cover - script mode
    from advanced_training import FeatureGroups, _ensure_group, load_feature_engineered_dataframe, _make_dataset  # type: ignore
    from advanced_model_architecture import AdvancedHierarchicalTriageModel, AdvancedHierarchicalTriageEnsemble  # type: ignore
    from evaluation_framework import ClinicalMetrics, FairnessEvaluator, ReliabilityMetrics  # type: ignore


def _load_checkpoint(path: Path) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def _class_names_from_k(k: int) -> _ListAlias[str]:
    """Return human-friendly class names for plots given K classes.

    - For ESI 5-class encoding (0..4), return ESI5..ESI1
    - For 3-class hierarchical variants, provide generic names
    - Otherwise, fallback to Class 0..Class K-1
    """
    if k == 5:
        return ["ESI5", "ESI4", "ESI3", "ESI2", "ESI1"]
    if k == 3:
        return ["NonCrit-Low", "NonCrit-High", "Critical"]
    return [f"Class {i}" for i in range(k)]


def _prepare_splits(
    feature_groups: FeatureGroups, target: np.ndarray, *, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(target))
    _, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=target
    )
    return indices, test_idx, np.delete(indices, test_idx)


def analyse_checkpoint(checkpoint_path: str, output_dir: str = "results") -> Dict:
    df, feature_groups, target_col = load_feature_engineered_dataframe()
    y = df[target_col].astype(int).to_numpy()
    feature_matrix = df[feature_groups.all()]

    _, test_idx, _ = _prepare_splits(feature_groups, y)
    test_df = feature_matrix.iloc[test_idx].reset_index(drop=True)
    test_y = y[test_idx]

    checkpoint = _load_checkpoint(Path(checkpoint_path))
    state_dict = checkpoint["model_state_dict"]

    # If training saved preprocess stats, reapply here for numerical consistency
    pre = checkpoint.get("preprocess")
    if pre:
        cols = pre.get("columns", feature_groups.all())
        # Create a view on the full matrix to avoid column-order issues
        full = feature_matrix.copy()
        med = pd.Series(pre.get("train_medians", {}))
        mu = pd.Series(pre.get("mean", []), index=cols)
        sig = pd.Series(pre.get("scale", []), index=cols).replace(0, 1.0)
        # Impute + standardize the entire matrix on saved stats, but keep original ordering
        full[cols] = full[cols].fillna(med).astype(np.float32)
        full[cols] = (full[cols] - mu) / sig
        feature_matrix = full
        test_df = feature_matrix.iloc[test_idx].reset_index(drop=True)

    dataset = _make_dataset(test_df, test_y, feature_groups)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    feature_groups_ckpt = checkpoint.get("feature_groups")
    if feature_groups_ckpt:
        feature_groups = FeatureGroups(**feature_groups_ckpt)
        feature_matrix = df[feature_groups.all()]
        test_df = feature_matrix.iloc[test_idx].reset_index(drop=True)
        dataset = _make_dataset(test_df, test_y, feature_groups)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    model_kwargs = {
        "num_vital_features": len(feature_groups.vital),
        "num_symptom_features": len(feature_groups.symptom),
        "num_risk_features": len(feature_groups.risk),
        "num_context_features": len(feature_groups.context),
        "num_lab_features": len(feature_groups.lab),
        "num_interaction_features": len(feature_groups.interaction),
    }

    if any(k.startswith("backbone.") for k in state_dict.keys()):
        model = AdvancedHierarchicalTriageEnsemble(**model_kwargs)
    else:
        model = AdvancedHierarchicalTriageModel(num_classes=len(np.unique(y)), **model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred, y_prob = [], [], []
    def _compose(outputs):
        if isinstance(outputs, tuple):
            temp = getattr(model, "calibration_temperature", None)
            heads = []
            for t in outputs:
                if temp is not None:
                    temp_val = float(temp.detach().cpu().item()) if isinstance(temp, torch.Tensor) else float(temp)
                    t = t / max(temp_val, 1e-6)
                heads.append(t)
            outputs = tuple(heads)
            if len(outputs) == 2:
                # Legacy 2-head: gate (noncritical vs critical) + noncritical detail head
                red_logits, yg_logits = outputs
                red_probs = torch.softmax(red_logits, dim=1)
                yg_probs = torch.softmax(yg_logits, dim=1)
                noncrit_gate = red_probs[:, 0:1]
                crit_gate = red_probs[:, 1:2]
                nc0 = noncrit_gate * yg_probs[:, 0:1]
                nc1 = noncrit_gate * yg_probs[:, 1:2]
                probs = torch.cat([nc0, nc1, crit_gate], dim=1)
            elif len(outputs) == 3:
                gate_logits, nc_logits, crit_logits = outputs
                gate = torch.softmax(gate_logits, dim=1)
                noncrit = gate[:, 0:1]
                crit = gate[:, 1:2]
                nc_probs = torch.softmax(nc_logits, dim=1)
                crit_probs = torch.softmax(crit_logits, dim=1)
                left = noncrit * nc_probs
                right = crit * crit_probs
                probs = torch.cat([left, right], dim=1)
            else:
                raise ValueError("Unsupported hierarchical outputs tuple length")
        else:
            temp = getattr(model, "calibration_temperature", None)
            logits = outputs
            if temp is not None:
                temp_val = float(temp.detach().cpu().item()) if isinstance(temp, torch.Tensor) else float(temp)
                logits = logits / max(temp_val, 1e-6)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    with torch.no_grad():
        for batch in loader:
            *features, targets = batch
            features = [t.to(device) for t in features]
            outputs = model(*features)
            probs = _compose(outputs)
            preds = probs.argmax(axis=1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)
    clinical_metrics = ClinicalMetrics.calculate_triage_metrics(y_true_arr, y_pred_arr)

    fairness_attributes = {}
    gender_col = "gender_original" if "gender_original" in df.columns else "gender"
    if gender_col in df.columns:
        fairness_attributes["gender"] = df.loc[test_idx, gender_col].to_numpy()
    elif "gender_male" in df.columns:
        fairness_attributes["gender"] = np.where(df.loc[test_idx, "gender_male"] > 0.5, "Male", "Female")
    age_values = None
    if "age_numeric" in df.columns:
        age_values = df.loc[test_idx, "age_numeric"].to_numpy(dtype=float)
    elif "yaş" in df.columns:
        age_values = df.loc[test_idx, "yaş"].to_numpy(dtype=float)
    elif "age" in df.columns:
        age_values = df.loc[test_idx, "age"].to_numpy(dtype=float)
    if age_values is not None:
        bins = [0, 18, 35, 50, 65, np.inf]
        labels = ["<18", "18-34", "35-49", "50-64", "65+"]
        age_groups = np.array(labels)[np.digitize(age_values, bins) - 1]
        fairness_attributes["age_group"] = age_groups

    fairness_results = None
    if fairness_attributes:
        evaluator = FairnessEvaluator()
        fairness_results = evaluator.evaluate_fairness(
            np.array(y_true), np.array(y_pred), fairness_attributes
        )

    # Reliability metrics
    reliability = {
        "nll": ReliabilityMetrics.negative_log_likelihood(y_true_arr, y_prob_arr),
        "ece": ReliabilityMetrics.expected_calibration_error(y_true_arr, y_prob_arr),
        "ece_per_class": ReliabilityMetrics.expected_calibration_error_per_class(y_true_arr, y_prob_arr),
        "brier": ReliabilityMetrics.brier_score(y_true_arr, y_prob_arr),
        "bootstrap": {
            "accuracy": ReliabilityMetrics.bootstrap_ci(y_true_arr, y_pred=y_pred_arr, metric="accuracy"),
            "macro_f1": ReliabilityMetrics.bootstrap_ci(y_true_arr, y_pred=y_pred_arr, metric="macro_f1"),
            "nll": ReliabilityMetrics.bootstrap_ci(y_true_arr, y_prob=y_prob_arr, metric="nll"),
            "ece": ReliabilityMetrics.bootstrap_ci(y_true_arr, y_prob=y_prob_arr, metric="ece"),
        },
    }

    # SHAP (optional)
    shap_payload = None
    if shap is not None:
        combined_matrix = feature_matrix.to_numpy(dtype=np.float32)
        background = combined_matrix[np.random.choice(len(combined_matrix), size=100, replace=False)]
        test_matrix = test_df.to_numpy(dtype=np.float32)

        def model_fn(x: np.ndarray) -> np.ndarray:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            splits = []
            start = 0
            for cols in [
                feature_groups.vital,
                feature_groups.symptom,
                feature_groups.risk,
                feature_groups.context,
                feature_groups.lab,
                feature_groups.interaction,
            ]:
                width = len(cols)
                splits.append(x_tensor[:, start : start + width])
                start += width
            with torch.no_grad():
                outputs = model(*splits)
                probs = _compose(outputs)
                return probs

        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(test_matrix[:50])  # limit for speed
        shap_payload = {
            "feature_names": feature_groups.all(),
            "values_per_class": [sv.tolist() for sv in shap_values],
        }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    def _to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serialisable(v) for v in obj]
        return obj

    analysis = {
        "checkpoint": checkpoint_path,
        "timestamp": timestamp,
        "clinical_metrics": _to_serialisable(clinical_metrics),
        "fairness": _to_serialisable(fairness_results) if fairness_results else None,
        "reliability": _to_serialisable(reliability),
        "shap": shap_payload,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"advanced_analysis_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Advanced analysis written to {path}")
    return analysis


def _plot_reliability_diagrams(y_true: np.ndarray, y_prob: np.ndarray, *, stable_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Use a clean, readable style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    figs = Path('docs') / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    # Overall reliability with histogram
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, 16)
    mids = (bins[:-1] + bins[1:]) / 2
    accs = []
    counts = []
    for i in range(15):
        mask = (conf >= bins[i]) & (conf < bins[i+1]) if i < 14 else (conf >= bins[i]) & (conf <= bins[i+1])
        if np.any(mask):
            accs.append(correct[mask].mean())
            counts.append(mask.sum())
        else:
            accs.append(np.nan)
            counts.append(0)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1.2, label='Perfect calibration')
    ax1.plot(mids, accs, marker='o', linewidth=2.0, label='Observed accuracy')
    ax1.set_xlabel('Predicted confidence', fontsize=11)
    ax1.set_ylabel('Observed accuracy', fontsize=11)
    ax1.set_title('Reliability Diagram (Overall)', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right', fontsize=9)
    ax2 = ax1.twinx()
    ax2.bar(mids, counts / max(1, np.sum(counts)), width=0.06, alpha=0.25, color='tab:gray', label='Frequency')
    ax2.set_ylabel('Frequency', fontsize=11)
    fig.tight_layout()
    plt.savefig(figs / f'rel_diagrams_{stable_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Classwise reliability
    k = y_prob.shape[1]
    class_names = _class_names_from_k(k)
    fig, axes = plt.subplots(1, k, figsize=(6.0 * k / 2.2, 3.4), sharey=True)
    if k == 1:
        axes = [axes]
    for c in range(k):
        p_c = y_prob[:, c]
        is_c = (y_true == c).astype(float)
        mids_c = mids
        accs_c = []
        for i in range(15):
            mask = (p_c >= bins[i]) & (p_c < bins[i+1]) if i < 14 else (p_c >= bins[i]) & (p_c <= bins[i+1])
            if np.any(mask):
                accs_c.append(is_c[mask].mean())
            else:
                accs_c.append(np.nan)
        axes[c].plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1.0)
        axes[c].plot(mids_c, accs_c, marker='o', linewidth=1.6)
        axes[c].set_xlabel(f'P({class_names[c]})', fontsize=11)
        axes[c].set_title(class_names[c], fontsize=12)
        axes[c].set_xlim(0, 1)
        axes[c].set_ylim(0, 1)
    axes[0].set_ylabel('Empirical frequency', fontsize=11)
    fig.suptitle('Reliability Diagram (Per Class)', fontsize=13)
    fig.tight_layout()
    plt.savefig(figs / f'rel_diagrams_classwise_{stable_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_decision_and_cost_curves(y_true: np.ndarray, y_prob: np.ndarray, *, stable_prefix: str, class_index: int = 2) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    figs = Path('docs') / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    n = len(y_true)
    p = y_prob[:, class_index]
    y_pos = (y_true == class_index).astype(int)
    thresholds = np.linspace(0.0, 1.0, 101)
    net_benefit = []
    under = []
    over = []
    for t in thresholds:
        pred_pos = (p >= t).astype(int)
        tp = np.sum((pred_pos == 1) & (y_pos == 1))
        fp = np.sum((pred_pos == 1) & (y_pos == 0))
        fn = np.sum((pred_pos == 0) & (y_pos == 1))
        # Decision curve net benefit (binary one-vs-rest)
        odds = t / (1 - t + 1e-12)
        nb = (tp / n) - (fp / n) * odds
        net_benefit.append(nb)
        under.append(fn / max(1, np.sum(y_pos)))
        over.append(fp / max(1, np.sum(pred_pos)))
    # Decision curve
    import matplotlib.pyplot as plt
    k = y_prob.shape[1]
    class_names = _class_names_from_k(k)
    label_name = class_names[class_index]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(thresholds, net_benefit, color='tab:blue', linewidth=2.0, label=f'Model ({label_name} vs rest)')
    ax.set_xlabel(f'Threshold (P({label_name}))', fontsize=11)
    ax.set_ylabel('Net benefit', fontsize=11)
    ax.set_title(f'Decision Curve – {label_name} vs Rest', fontsize=12)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(figs / f'decision_curves_{stable_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Cost trade-offs
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(thresholds, under, color='tab:red', linewidth=2.0, label=f'Under-triage (missed {label_name})')
    ax.plot(thresholds, over, color='tab:orange', linewidth=2.0, label='Over-triage (false positives)')
    ax.set_xlabel(f'Threshold (P({label_name}))', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title(f'Triage Trade-offs – {label_name}', fontsize=12)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(figs / f'cost_tradeoffs_{stable_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_fairness_calibration_bars(y_true: np.ndarray, y_prob: np.ndarray, fairness_attributes: Dict[str, np.ndarray]) -> None:
    if not fairness_attributes:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    figs = Path('docs') / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    # For each attribute, compute ECE per group
    fig, axes = plt.subplots(1, len(fairness_attributes), figsize=(6.0 * len(fairness_attributes), 3.6))
    if len(fairness_attributes) == 1:
        axes = [axes]
    for ax, (attr, values) in zip(axes, fairness_attributes.items()):
        groups = np.unique(values)
        eces = []
        for g in groups:
            mask = values == g
            ece_g = ReliabilityMetrics.expected_calibration_error(y_true[mask], y_prob[mask]) if np.any(mask) else 0.0
            eces.append(ece_g)
        ax.bar([str(g) for g in groups], eces, color='tab:blue', alpha=0.85)
        ax.set_title(f'Calibration Error by {attr}', fontsize=12)
        ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=11)
        ax.set_ylim(0, max(eces + [0.02]) * 1.3)
        ax.set_xlabel('Group', fontsize=11)
    fig.tight_layout()
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    plt.savefig(figs / f'fairness_calibration_bars_{ts}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _split_train_val_test(df, y, feature_groups, *, seed: int = 42):
    from sklearn.model_selection import train_test_split
    X = df[feature_groups.all()]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _run_baseline_histgb(output_dir: str = "results") -> Dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    df, feature_groups, target_col = load_feature_engineered_dataframe()
    y = df[target_col].astype(int).to_numpy()
    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_val_test(df, y, feature_groups)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Use validation for nothing special (could calibrate), then test
    prob = model.predict_proba(X_test)
    pred = prob.argmax(axis=1)

    clinical = ClinicalMetrics.calculate_triage_metrics(y_test, pred)
    reliability = {
        "nll": ReliabilityMetrics.negative_log_likelihood(y_test, prob),
        "ece": ReliabilityMetrics.expected_calibration_error(y_test, prob),
        "ece_per_class": ReliabilityMetrics.expected_calibration_error_per_class(y_test, prob),
        "brier": ReliabilityMetrics.brier_score(y_test, prob),
        "bootstrap": {
            "accuracy": ReliabilityMetrics.bootstrap_ci(y_test, y_pred=pred, metric="accuracy"),
            "macro_f1": ReliabilityMetrics.bootstrap_ci(y_test, y_pred=pred, metric="macro_f1"),
            "nll": ReliabilityMetrics.bootstrap_ci(y_test, y_prob=prob, metric="nll"),
            "ece": ReliabilityMetrics.bootstrap_ci(y_test, y_prob=prob, metric="ece"),
        },
    }
    out = {
        "model": "histgb",
        "clinical_metrics": clinical,
        "reliability": reliability,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with open(Path(output_dir) / f"baseline_histgb_analysis_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Plots (optional)
    try:
        import matplotlib.pyplot as plt
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass
        # Reliability diagram
        conf = prob.max(axis=1)
        correct = (pred == y_test).astype(float)
        bins = np.linspace(0.0, 1.0, 16)
        indices = np.digitize(conf, bins) - 1
        accs = []
        mids = []
        for i in range(15):
            mask = indices == i
            if not np.any(mask):
                continue
            accs.append(correct[mask].mean())
            mids.append((bins[i] + bins[i+1]) / 2)
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1.2, label='Perfect calibration')
        ax.plot(mids, accs, marker='o', linewidth=2.0, label='Observed accuracy')
        ax.set_xlabel('Predicted confidence', fontsize=11)
        ax.set_ylabel('Observed accuracy', fontsize=11)
        ax.set_title('Reliability Diagram (HistGB Baseline)', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        figs = Path('docs') / 'figures'
        figs.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(figs / 'rel_diagrams_histgb.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass
    return out


def _run_baseline_logreg(calibrate: str | None = None, output_dir: str = "results") -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    df, feature_groups, target_col = load_feature_engineered_dataframe()
    y = df[target_col].astype(int).to_numpy()
    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_val_test(df, y, feature_groups)

    base = LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto', n_jobs=None)
    if calibrate == 'platt':
        model = CalibratedClassifierCV(base, method='sigmoid', cv=5)
    else:
        model = base
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    pred = prob.argmax(axis=1)

    clinical = ClinicalMetrics.calculate_triage_metrics(y_test, pred)
    reliability = {
        "nll": ReliabilityMetrics.negative_log_likelihood(y_test, prob),
        "ece": ReliabilityMetrics.expected_calibration_error(y_test, prob),
        "ece_per_class": ReliabilityMetrics.expected_calibration_error_per_class(y_test, prob),
        "brier": ReliabilityMetrics.brier_score(y_test, prob),
        "bootstrap": {
            "accuracy": ReliabilityMetrics.bootstrap_ci(y_test, y_pred=pred, metric="accuracy"),
            "macro_f1": ReliabilityMetrics.bootstrap_ci(y_test, y_pred=pred, metric="macro_f1"),
            "nll": ReliabilityMetrics.bootstrap_ci(y_test, y_prob=prob, metric="nll"),
            "ece": ReliabilityMetrics.bootstrap_ci(y_test, y_prob=prob, metric="ece"),
        },
    }
    out = {
        "model": "logreg_platt" if calibrate == 'platt' else "logreg",
        "clinical_metrics": clinical,
        "reliability": reliability,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with open(Path(output_dir) / f"baseline_logreg_analysis_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Advanced analysis: checkpoints or baselines")
    parser.add_argument("checkpoint", nargs='?', help="Path to advanced model checkpoint")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--baseline", choices=["histgb", "logreg"], default=None)
    parser.add_argument("--calibrate", choices=[None, "platt"], default=None)
    parser.add_argument("--plots", nargs='*', default=[])
    parser.add_argument("--stable-prefix", default="centralized", help="Prefix for stable figure filenames")
    args = parser.parse_args()

    if args.baseline:
        if args.baseline == 'histgb':
            _run_baseline_histgb(args.output_dir)
        elif args.baseline == 'logreg':
            _run_baseline_logreg(args.calibrate, args.output_dir)
    else:
        assert args.checkpoint, "checkpoint path is required when --baseline is not set"
        res = analyse_checkpoint(args.checkpoint, args.output_dir)
        # Optional plotting to stable filenames
        try:
            df, feature_groups, target_col = load_feature_engineered_dataframe()
            y = df[target_col].astype(int).to_numpy()
            _, test_idx, _ = _prepare_splits(feature_groups, y)
            # We need y_true/y_prob; recompute quickly using the same code path
            # Reuse checkpoint to get probs
            checkpoint = _load_checkpoint(Path(args.checkpoint))
            state_dict = checkpoint["model_state_dict"]
            model_kwargs = {
                "num_vital_features": len(feature_groups.vital),
                "num_symptom_features": len(feature_groups.symptom),
                "num_risk_features": len(feature_groups.risk),
                "num_context_features": len(feature_groups.context),
                "num_lab_features": len(feature_groups.lab),
                "num_interaction_features": len(feature_groups.interaction),
            }
            if any(k.startswith("backbone.") for k in state_dict.keys()):
                model = AdvancedHierarchicalTriageEnsemble(**model_kwargs)
            else:
                model = AdvancedHierarchicalTriageModel(num_classes=len(np.unique(y)), **model_kwargs)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            feature_matrix = df[feature_groups.all()]
            test_df = feature_matrix.iloc[test_idx].reset_index(drop=True)
            test_y = y[test_idx]
            dataset = _make_dataset(test_df, test_y, feature_groups)
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
            y_true_list: List[int] = []
            y_prob_list: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    *features, targets = batch
                    features = [t.to(device) for t in features]
                    outputs = model(*features)
                    # compose probs (same as above)
                    if isinstance(outputs, tuple):
                        # Legacy 2-head composition (see above) or 3-head (ESI)
                        if len(outputs) == 2:
                            red_logits, yg_logits = outputs
                            temp = getattr(model, "calibration_temperature", None)
                            if temp is not None:
                                temp_val = float(temp.detach().cpu().item()) if isinstance(temp, torch.Tensor) else float(temp)
                                scale = torch.tensor(temp_val, device=red_logits.device, dtype=red_logits.dtype)
                                red_logits = red_logits / scale.clamp_min(1e-6)
                                yg_logits = yg_logits / scale.clamp_min(1e-6)
                            red_probs = torch.softmax(red_logits, dim=1)
                            yg_probs = torch.softmax(yg_logits, dim=1)
                            noncrit_gate = red_probs[:, 0:1]
                            crit_gate = red_probs[:, 1:2]
                            nc0 = noncrit_gate * yg_probs[:, 0:1]
                            nc1 = noncrit_gate * yg_probs[:, 1:2]
                            probs = torch.cat([nc0, nc1, crit_gate], dim=1)
                        else:
                            gate_logits, nc_logits, crit_logits = outputs
                            temp = getattr(model, "calibration_temperature", None)
                            if temp is not None:
                                temp_val = float(temp.detach().cpu().item()) if isinstance(temp, torch.Tensor) else float(temp)
                                scale = torch.tensor(temp_val, device=gate_logits.device, dtype=gate_logits.dtype)
                                gate_logits = gate_logits / scale.clamp_min(1e-6)
                                nc_logits = nc_logits / scale.clamp_min(1e-6)
                                crit_logits = crit_logits / scale.clamp_min(1e-6)
                            gate = torch.softmax(gate_logits, dim=1)
                            nc_probs = torch.softmax(nc_logits, dim=1)
                            crit_probs = torch.softmax(crit_logits, dim=1)
                            left = gate[:, 0:1] * nc_probs
                            right = gate[:, 1:2] * crit_probs
                            probs = torch.cat([left, right], dim=1)
                    else:
                        temp = getattr(model, "calibration_temperature", None)
                        logits = outputs
                        if temp is not None:
                            temp_val = float(temp.detach().cpu().item()) if isinstance(temp, torch.Tensor) else float(temp)
                            logits = logits / max(temp_val, 1e-6)
                        probs = torch.softmax(logits, dim=1)
                    y_true_list.extend(targets.numpy())
                    y_prob_list.extend(probs.cpu().numpy())
            y_true_arr = np.array(y_true_list)
            y_prob_arr = np.array(y_prob_list)
            if 'reliability' in args.plots or 'reliability' in (args.plots or []):
                _plot_reliability_diagrams(y_true_arr, y_prob_arr, stable_prefix=args.stable_prefix)
            if 'decision_curves' in args.plots:
                # Use the highest-severity class by default
                k = y_prob_arr.shape[1]
                _plot_decision_and_cost_curves(y_true_arr, y_prob_arr, stable_prefix=args.stable_prefix, class_index=k-1)
            if 'fairness' in args.plots:
                # rebuild fairness attributes on test split
                fairness_attributes = {}
                gender_col = "gender_original" if "gender_original" in df.columns else "gender"
                if gender_col in df.columns:
                    fairness_attributes["gender"] = df.loc[test_idx, gender_col].to_numpy()
                elif "gender_male" in df.columns:
                    fairness_attributes["gender"] = np.where(df.loc[test_idx, "gender_male"] > 0.5, "Male", "Female")
                age_values = None
                if "age_numeric" in df.columns:
                    age_values = df.loc[test_idx, "age_numeric"].to_numpy(dtype=float)
                elif "yaş" in df.columns:
                    age_values = df.loc[test_idx, "yaş"].to_numpy(dtype=float)
                elif "age" in df.columns:
                    age_values = df.loc[test_idx, "age"].to_numpy(dtype=float)
                if age_values is not None:
                    bins = [0, 18, 35, 50, 65, np.inf]
                    labels = ["<18", "18-34", "35-49", "50-64", "65+"]
                    age_groups = np.array(labels)[np.digitize(age_values, bins) - 1]
                    fairness_attributes["age_group"] = age_groups
                _plot_fairness_calibration_bars(y_true_arr, y_prob_arr, fairness_attributes)
        except Exception as e:
            print(f"Plotting skipped: {e}")
