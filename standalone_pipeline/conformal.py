"""Conformal Critical Control (C³) utilities.

These helpers compute deployment-time thresholds that guarantee a minimum
critical-case recall (overall and optionally per subgroup) by applying
conformal prediction to the summed probability mass over the critical classes.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "fit_conformal_thresholds",
    "apply_conformal_thresholds",
]


def _quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the conformal quantile for the provided scores."""
    if scores.size == 0:
        return 1.0  # no critical samples observed; fall back to safest option
    scores = np.sort(scores)
    n = scores.size
    # Finite-sample upper (1 - alpha) quantile with conformal guarantee
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def fit_conformal_thresholds(
    y_true: Sequence[int],
    probs: np.ndarray,
    *,
    critical_classes: Sequence[int],
    alpha: float = 0.05,
    group_labels: Sequence[str] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Fit global and optional group-wise conformal thresholds.

    Parameters
    ----------
    y_true:
        Sequence of integer labels (ascending severity encoding).
    probs:
        Array of shape (n_samples, n_classes) with predicted probabilities.
    critical_classes:
        Indices corresponding to the critical severity band (e.g., top-2 ESI).
    alpha:
        Allowed miss-rate (default 5%), yielding recall guarantee ≥ 1 - alpha.
    group_labels:
        Optional sequence of length n_samples specifying subgroup membership.

    Returns
    -------
    tau_global:
        Global conformal threshold τ such that predicting critical whenever
        1 - p_crit(x) ≤ τ yields recall ≥ 1 - alpha.
    tau_by_group:
        Mapping subgroup → τ_g. Groups with no critical samples fall back to
        the global threshold.
    """

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    critical_classes = np.asarray(list(critical_classes), dtype=int)

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (n_samples, n_classes)")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have the same number of samples")

    crit_prob = probs[:, critical_classes].sum(axis=1)
    crit_prob = np.clip(crit_prob, 0.0, 1.0)

    critical_mask = np.isin(y_true, critical_classes)
    scores = 1.0 - crit_prob[critical_mask]
    tau_global = _quantile(scores, alpha)

    tau_by_group: Dict[str, float] = {}
    if group_labels is not None:
        group_labels = np.asarray(group_labels)
        if group_labels.shape[0] != probs.shape[0]:
            raise ValueError("group_labels must align with probs/y_true length")
        unique_groups = np.unique(group_labels.astype(str))
        for group in unique_groups:
            group_mask = (group_labels == group) & critical_mask
            group_scores = 1.0 - crit_prob[group_mask]
            if group_scores.size == 0:
                continue  # insufficient data; fall back to global tau
            tau_by_group[str(group)] = _quantile(group_scores, alpha)

    return float(tau_global), {g: float(t) for g, t in tau_by_group.items()}


def apply_conformal_thresholds(
    probs: np.ndarray,
    tau_global: float,
    critical_classes: Sequence[int],
    *,
    tau_by_group: Dict[str, float] | None = None,
    group_labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Apply conformal thresholds to convert probabilities into class labels.

    Parameters
    ----------
    probs:
        Array of shape (n_samples, n_classes) with predicted probabilities.
    tau_global:
        Global conformal threshold τ from `fit_conformal_thresholds`.
    critical_classes:
        Indices corresponding to the critical severity band.
    tau_by_group:
        Optional subgroup-specific thresholds τ_g.
    group_labels:
        Optional subgroup labels aligning with `probs`.

    Returns
    -------
    preds:
        Array of predicted class indices after enforcing the conformal rule.
    """

    probs = np.asarray(probs, dtype=np.float64)
    critical_classes = np.asarray(list(critical_classes), dtype=int)
    preds = probs.argmax(axis=1).astype(int)

    crit_prob = probs[:, critical_classes].sum(axis=1)
    crit_prob = np.clip(crit_prob, 0.0, 1.0)

    thresholds = np.full(probs.shape[0], float(tau_global), dtype=np.float64)
    if tau_by_group and group_labels is not None:
        group_labels = np.asarray(group_labels)
        if group_labels.shape[0] != probs.shape[0]:
            raise ValueError("group_labels must align with probs length")
        for group, tau in tau_by_group.items():
            thresholds[group_labels == group] = tau

    critical_decision = crit_prob >= (1.0 - thresholds)
    if np.any(critical_decision):
        crit_sub_probs = probs[critical_decision][:, critical_classes]
        best_crit_idx = crit_sub_probs.argmax(axis=1)
        preds[critical_decision] = critical_classes[best_crit_idx]

    return preds

