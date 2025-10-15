"""Shared utilities for fairness metadata and severity helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "construct_fairness_data",
    "get_critical_class_indices",
]


def construct_fairness_data(
    df: pd.DataFrame,
    indices: Sequence[int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build fairness attribute arrays and composite group labels for indices."""

    if isinstance(indices, np.ndarray):
        idx = indices
    else:
        idx = np.array(list(indices))

    subset = df.loc[idx].reindex(idx)
    n = len(subset)

    # Gender handling --------------------------------------------------------
    if "gender_original" in subset.columns:
        gender = subset["gender_original"].fillna("Unknown").astype(str).to_numpy()
    elif "gender" in subset.columns:
        gender = subset["gender"].fillna("Unknown").astype(str).to_numpy()
    elif "gender_male" in subset.columns:
        gender = np.where(subset["gender_male"].to_numpy(dtype=float) > 0.5, "Male", "Female")
    else:
        gender = np.array(["Unknown"] * n, dtype=object)

    # Age buckets ------------------------------------------------------------
    age_values = None
    for col in ("age_numeric", "yaş", "age"):
        if col in subset.columns:
            age_values = pd.to_numeric(subset[col], errors="coerce").to_numpy()
            break

    age_labels = np.array(["Unknown"] * n, dtype=object)
    if age_values is not None:
        bins = np.array([0, 18, 35, 50, 65, np.inf])
        labels = np.array(["<18", "18-34", "35-49", "50-64", "65+"])
        with np.errstate(invalid="ignore"):
            digitized = np.digitize(age_values, bins) - 1
        digitized = np.clip(digitized, 0, len(labels) - 1)
        age_labels = labels[digitized]
        age_labels[~np.isfinite(age_values)] = "Unknown"

    fairness_attributes: Dict[str, np.ndarray] = {
        "gender": gender,
        "age_group": age_labels,
    }

    group_labels = np.array(
        [f"{g}|{a}" for g, a in zip(gender, age_labels)],
        dtype=object,
    )

    return fairness_attributes, group_labels


def get_critical_class_indices(num_classes: int) -> List[int]:
    """Return class indices corresponding to the critical severity band."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if num_classes <= 2:
        return [num_classes - 1]
    if num_classes == 3:
        return [2]
    # For 5-class ESI encoding 0..4, top-2 classes are critical
    start = max(1, num_classes - 2)
    return list(range(start, num_classes))

