#!/usr/bin/env python3
"""Generate paper-ready figures from saved experiment reports.

This script is intentionally lightweight and depends only on matplotlib/numpy
from the project's virtualenv.

Outputs (written to ./figures):
- pipeline_overview.png
- dataset_class_distribution.png
- confusion_matrices_esi5.png
- confusion_matrices_3class.png
- safety_tradeoff.png
 - fl_advanced_acc_f1_vs_rounds.png
 - fl_advanced_ece_nll_vs_rounds.png
 - fl_advanced_systems_vs_rounds.png
 - fl_advanced_privacy_utility.png
 - baseline_comparison.png
 - byzantine_robustness.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
DOCS_FIG_DIR = REPO_ROOT / "docs" / "figures"


def _extract_ts(path: Path) -> str:
    m = re.search(r"_(\d{8}_\d{6})$", path.stem)
    return m.group(1) if m else ""


def _pick_latest_report(candidates: List[Path], predicate) -> Path | None:
    chosen: List[tuple[str, Path]] = []
    for p in candidates:
        try:
            with p.open("r", encoding="utf-8") as f:
                rep = json.load(f)
        except Exception:
            continue
        if predicate(rep, p):
            chosen.append((_extract_ts(p), p))
    if not chosen:
        return None
    chosen.sort(key=lambda x: x[0] or x[1].name)
    return chosen[-1][1]


def _resolve_reports_core() -> Dict[str, Path]:
    res_dir = REPO_ROOT / "results"
    all_reports = sorted(res_dir.glob("*evaluation_report_*.json"))

    out: Dict[str, Path] = {}

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: path.name.startswith("advanced_evaluation_report_")
        and not bool((rep.get("privacy") or {}).get("dp", False)),
    )
    if p:
        out["Centralized (non-DP)"] = p

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: path.name.startswith("fl_advanced_evaluation_report_")
        and bool((rep.get("run_config") or {}).get("by_site", False))
        and int((rep.get("run_config") or {}).get("rounds", -1)) == 20
        and int((rep.get("run_config") or {}).get("clients", -1)) == 3
        and not bool((rep.get("run_config") or {}).get("dp", False))
        and str((rep.get("run_config") or {}).get("aggregator", "")) == "fedavg",
    )
    if p:
        out["FL Flat (non-DP; 3 sites, 20 rounds)"] = p

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: path.name.startswith("fl_advanced_red_focus_evaluation_report_")
        and bool((rep.get("run_config") or {}).get("by_site", False))
        and int((rep.get("run_config") or {}).get("rounds", -1)) == 20
        and int((rep.get("run_config") or {}).get("clients", -1)) == 3
        and not bool((rep.get("run_config") or {}).get("dp", False))
        and str((rep.get("run_config") or {}).get("aggregator", "")) == "fedavg",
    )
    if p:
        out["FL Red-Focus (non-DP; 3 sites, 20 rounds)"] = p

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: path.name.startswith("fl_advanced_evaluation_report_")
        and bool((rep.get("run_config") or {}).get("by_site", False))
        and bool((rep.get("run_config") or {}).get("dp", False))
        and int((rep.get("run_config") or {}).get("rounds", -1)) == 5
        and int((rep.get("run_config") or {}).get("clients", -1)) == 3
        and abs(float((rep.get("run_config") or {}).get("dp_noise_multiplier", -1.0)) - 1.0) < 1e-9
        and str((rep.get("run_config") or {}).get("dp_optimizer", "")) == "adam",
    )
    if p:
        out["DP-SGD (3 sites, 5 rounds; $\\sigma{=}1.0$; Adam)"] = p

    return out


def _resolve_reports_byz() -> Dict[str, Path]:
    res_dir = REPO_ROOT / "results"
    all_reports = sorted(res_dir.glob("fl_advanced_evaluation_report_*.json"))

    def _is_target_run(rep: Dict) -> bool:
        rc = rep.get("run_config") or {}
        return (
            int(rc.get("rounds", -1)) == 10
            and int(rc.get("clients", -1)) == 7
            and bool(rc.get("non_iid", False))
            and abs(float(rc.get("dirichlet_alpha", -1.0)) - 0.3) < 1e-9
            and not bool(rc.get("by_site", False))
            and not bool(rc.get("dp", False))
        )

    out: Dict[str, Path] = {}
    p = _pick_latest_report(
        all_reports,
        lambda rep, path: _is_target_run(rep)
        and str((rep.get("run_config") or {}).get("aggregator", "")) == "fedavg"
        and rep.get("byzantine") is None,
    )
    if p:
        out["Control (FedAvg)"] = p

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: _is_target_run(rep)
        and str((rep.get("run_config") or {}).get("aggregator", "")) == "fedavg"
        and abs(float((rep.get("byzantine") or {}).get("fraction", -1.0)) - 0.29) < 1e-9
        and str((rep.get("byzantine") or {}).get("attack", "")) == "gaussian"
        and abs(float((rep.get("byzantine") or {}).get("scale", -1.0)) - 2.0) < 1e-9,
    )
    if p:
        out["Attack (FedAvg)"] = p

    p = _pick_latest_report(
        all_reports,
        lambda rep, path: _is_target_run(rep)
        and str((rep.get("run_config") or {}).get("aggregator", "")) == "median"
        and abs(float((rep.get("byzantine") or {}).get("fraction", -1.0)) - 0.29) < 1e-9
        and str((rep.get("byzantine") or {}).get("attack", "")) == "gaussian"
        and abs(float((rep.get("byzantine") or {}).get("scale", -1.0)) - 2.0) < 1e-9,
    )
    if p:
        out["Attack (Median)"] = p
    return out


def _resolve_reports_baselines() -> Dict[str, Path]:
    res_dir = REPO_ROOT / "results"
    histgb = sorted(res_dir.glob("baseline_histgb_analysis_*.json"))
    out: Dict[str, Path] = {}
    if histgb:
        out["HistGradientBoosting"] = histgb[-1]
    return out


# Core (ESI-5) reports used for most paper figures.
REPORTS_CORE = _resolve_reports_core()

# Robust aggregation / adversarial evaluation (non-IID, 7 clients).
REPORTS_BYZ = _resolve_reports_byz()

# Strong traditional ML baseline (tabular GBDT).
REPORTS_BASELINES = _resolve_reports_baselines()

# 3-class (Green/Yellow/Red) reports used in Table 3.
REPORTS_3CLASS = {
    "Centralized": REPO_ROOT / "results" / "advanced_evaluation_report_20251005_071734.json",
    "FL Flat": REPO_ROOT / "results" / "fl_advanced_evaluation_report_20251005_074646.json",
    "FL Red-Focus": REPO_ROOT / "results" / "fl_advanced_red_focus_evaluation_report_20251007_032147.json",
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_class_names(report: Dict) -> List[str]:
    keys = list(report["clinical_metrics"]["class_metrics"].keys())
    esi5 = ["ESI5", "ESI4", "ESI3", "ESI2", "ESI1"]
    if set(keys) == set(esi5):
        return esi5
    tri = ["Green", "Yellow", "Red"]
    if set(keys) == set(tri):
        return tri
    # Fallback: preserve insertion order from JSON
    return keys


def _confusion_matrix(report: Dict) -> np.ndarray:
    return np.asarray(report["clinical_metrics"]["confusion_matrix"], dtype=float)


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return cm / denom


def plot_pipeline_overview(out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # Clean schematic: boxes + arrows (no external deps)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 3.0))
    ax.set_axis_off()

    steps = [
        ("Dataset", "Kaggle ED triage\n(558k encounters)"),
        ("Preprocess", "Impute + standardize\nGroup features (6 groups)"),
        ("Client Train", "Safety-aware loss\n(Optional DP-SGD)"),
        ("Server", "Aggregate updates\n(FedAvg / robust)"),
        ("Post-hoc", "Temp scaling\nThreshold selection"),
        ("Evaluate", "Accuracy + safety\nECE/NLL + comm/fairness"),
    ]

    n = len(steps)
    x0, x1 = 0.04, 0.96
    y = 0.52
    width = (x1 - x0) / n * 0.92
    gap = (x1 - x0) / n * 0.08
    box_h = 0.46

    xs: List[float] = []
    for i in range(n):
        xs.append(x0 + i * (width + gap))

    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        rect = plt.Rectangle(
            (x, y - box_h / 2),
            width,
            box_h,
            linewidth=1.2,
            edgecolor="#222222",
            facecolor="#f6f6f6" if i % 2 == 0 else "#ffffff",
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            y + 0.12,
            title,
            ha="center",
            va="center",
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            x + width / 2,
            y - 0.05,
            body,
            ha="center",
            va="center",
            color="#222222",
            linespacing=1.2,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.2, y),
                xytext=(x + width, y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#444444"),
            )

    ax.set_title("End-to-End Federated Triage Pipeline", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety_tradeoff(reports: Dict[str, Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    titles = list(reports.keys())
    acc = [reports[t]["clinical_metrics"]["overall_accuracy"] for t in titles]
    crit = [reports[t]["clinical_metrics"]["clinical_safety"]["critical_sensitivity"] for t in titles]
    under = [reports[t]["clinical_metrics"]["clinical_safety"]["under_triage_rate"] for t in titles]

    # Convert to percent for display
    acc_p = [a * 100 for a in acc]
    crit_p = [c * 100 for c in crit]
    under_p = [u * 100 for u in under]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    colors = ["#2b6cb0", "#2f855a", "#c05621", "#805ad5"]

    def _bar(ax, values, title, ylabel):
        x = np.arange(len(titles))
        ax.bar(x, values, color=colors[: len(titles)], alpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(values):
            ax.text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    _bar(axes[0], acc_p, "Accuracy", "Test Accuracy (%)")
    _bar(axes[1], crit_p, "Critical Sensitivity", "Recall (ESI-2/ESI-1) (%)")
    _bar(axes[2], under_p, "Under-Triage", "Under-triage Rate (%)")

    fig.suptitle("Safety and Utility Trade-Offs (ESI-5)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_class_distribution(report: Dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    counts = report["data_info"]["class_distribution"]
    names = _ordered_class_names(report)

    # Ensure ESI ordering in counts
    # class_distribution is over encoded labels 0..K-1 (0=ESI5 ...), so map:
    if set(names) == {"ESI5", "ESI4", "ESI3", "ESI2", "ESI1"} and len(counts) == 5:
        names = ["ESI5", "ESI4", "ESI3", "ESI2", "ESI1"]
        counts = [counts[0], counts[1], counts[2], counts[3], counts[4]]

    total = float(sum(counts))
    frac = [c / total for c in counts]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(names, counts, color="#2b6cb0", alpha=0.9)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (ESI-5 to ESI-1)")
    ax.grid(axis="y", alpha=0.25)

    for i, (c, p) in enumerate(zip(counts, frac)):
        ax.text(i, c + total * 0.004, f"{p*100:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices_esi5(reports: Dict[str, Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    titles = list(reports.keys())
    mats = [_row_normalize(_confusion_matrix(reports[t])) for t in titles]
    classes = ["ESI5", "ESI4", "ESI3", "ESI2", "ESI1"]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
    axes = axes.flatten()
    vmin, vmax = 0.0, 1.0
    im = None
    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate percentages (row-normalized)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j] * 100.0
                if 0.0 < val < 1.0:
                    label = "<1"
                else:
                    label = f"{val:.0f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if mat[i, j] > 0.5 else "#111111",
                )

    # Leave room for a shared colorbar outside the grids.
    fig.subplots_adjust(right=0.86, wspace=0.35, hspace=0.35, top=0.90)

    # Shared colorbar (outside)
    assert im is not None
    cax = fig.add_axes([0.89, 0.18, 0.02, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized (%)", rotation=90)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])

    fig.suptitle("Row-Normalized Confusion Matrices (ESI-5)", fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices_3class(reports: Dict[str, Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    titles = list(reports.keys())
    mats = [_row_normalize(_confusion_matrix(reports[t])) for t in titles]
    classes = ["Green", "Yellow", "Red"]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    vmin, vmax = 0.0, 1.0
    im = None
    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j] * 100.0
                if 0.0 < val < 1.0:
                    label = "<1"
                else:
                    label = f"{val:.0f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if mat[i, j] > 0.5 else "#111111",
                )

    fig.subplots_adjust(right=0.88, wspace=0.35, top=0.86)

    assert im is not None
    cax = fig.add_axes([0.91, 0.22, 0.02, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized (%)", rotation=90)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])

    fig.suptitle("Row-Normalized Confusion Matrices (Green/Yellow/Red)", fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_privacy_utility_dp(dp_report: Dict, out_path: Path) -> None:
    """Plot DP accounting and utility from a federated DP-SGD report.

    Our FL driver stores a *cumulative* privacy budget curve (max over clients)
    computed via RDP accounting across rounds/steps. This plot shows:
    - Val accuracy vs rounds (left axis)
    - Cumulative epsilon vs rounds (right axis)
    - Privacy–utility view using cumulative epsilon on the x-axis
    """
    import matplotlib.pyplot as plt  # type: ignore

    eps_cum = list(dp_report.get("privacy", {}).get("epsilon_per_round", []) or [])
    hist = dp_report.get("training_history", {}) or {}
    rounds = list(hist.get("rounds", []) or list(range(1, len(eps_cum) + 1)))
    val_acc = list(hist.get("val_acc", []) or [])

    n = min(len(eps_cum), len(rounds), len(val_acc))
    if n <= 0:
        return

    eps_cum = eps_cum[:n]
    rounds = rounds[:n]
    val_acc = val_acc[:n]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))

    # Panel A: round-by-round view
    ax = axes[0]
    ax.plot(rounds, val_acc, marker="o", linewidth=2.0, color="tab:green", label="Val Acc (%)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Accuracy (%)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(alpha=0.25)
    ax.set_title("Utility and DP Accounting vs Rounds", fontsize=11)

    axr = ax.twinx()
    axr.plot(rounds, eps_cum, marker="s", linewidth=2.0, color="tab:blue", label="Cumulative ε (max client)")
    axr.set_ylabel("Cumulative ε (max client)", color="tab:blue")
    axr.tick_params(axis="y", labelcolor="tab:blue")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    # Panel B: privacy–utility view (using cumulative epsilon)
    axb = axes[1]
    axb.plot(
        eps_cum,
        val_acc,
        marker="o",
        linewidth=2.0,
        color="#2b6cb0",
    )
    for r, x, y in zip(rounds, eps_cum, val_acc):
        axb.text(x, y + 0.12, str(r), ha="center", va="bottom", fontsize=8, color="#111111")
    axb.set_xlabel("Cumulative ε (max client)")
    axb.set_ylabel("Validation Accuracy (%)")
    axb.grid(alpha=0.25)
    axb.set_title("Privacy–Utility (DP-SGD)", fontsize=11)

    fig.suptitle("DP-SGD: Privacy Accounting and Utility", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_hist(rep: Dict) -> Dict:
    return rep.get("training_history", {}) or {}


def plot_fl_acc_f1_vs_rounds(fl_reports: Dict[str, Dict], out_path: Path) -> None:
    """Compare validation accuracy and macro-F1 across rounds for FL runs."""
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6))
    ax_acc, ax_f1 = axes
    for title, rep in fl_reports.items():
        hist = _get_hist(rep)
        r = hist.get("rounds", [])
        acc = hist.get("val_acc", [])
        f1 = hist.get("val_macro_f1", [])
        if not r:
            continue
        ax_acc.plot(r, acc, marker="o", linewidth=2.0, label=title)
        ax_f1.plot(r, f1, marker="s", linewidth=2.0, label=title)

    ax_acc.set_title("Validation Accuracy", fontsize=12)
    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Val Acc (%)")
    ax_acc.grid(alpha=0.25)

    ax_f1.set_title("Validation Macro-F1", fontsize=12)
    ax_f1.set_xlabel("Round")
    ax_f1.set_ylabel("Val Macro-F1 (%)")
    ax_f1.grid(alpha=0.25)

    # Shared legend
    handles, labels = ax_acc.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fl_ece_nll_vs_rounds(fl_reports: Dict[str, Dict], out_path: Path) -> None:
    """Compare validation calibration metrics across rounds for FL runs."""
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6))
    ax_ece, ax_nll = axes
    for title, rep in fl_reports.items():
        hist = _get_hist(rep)
        r = hist.get("rounds", [])
        rel = rep.get("reliability_history", {}) or {}
        ece = rel.get("val_ece", [])
        nll = rel.get("val_nll", [])
        if not r:
            continue
        ax_ece.plot(r[: len(ece)], ece, marker="o", linewidth=2.0, label=title)
        ax_nll.plot(r[: len(nll)], nll, marker="s", linewidth=2.0, label=title)

    ax_ece.set_title("Validation ECE", fontsize=12)
    ax_ece.set_xlabel("Round")
    ax_ece.set_ylabel("ECE")
    ax_ece.grid(alpha=0.25)

    ax_nll.set_title("Validation NLL", fontsize=12)
    ax_nll.set_xlabel("Round")
    ax_nll.set_ylabel("NLL")
    ax_nll.grid(alpha=0.25)

    handles, labels = ax_ece.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fl_systems_vs_rounds(fl_reports: Dict[str, Dict], out_path: Path) -> None:
    """Plot communication plus simple systems variability across rounds.

    Communication is often constant across rounds when it's approximated as
    `params * 4 bytes * 2 directions * num_clients`. To make the figure useful,
    we also plot the per-round client wall-clock time standard deviation when
    available in the report.
    """
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax2 = ax.twinx()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (title, rep) in enumerate(fl_reports.items()):
        hist = _get_hist(rep)
        r = hist.get("rounds", [])
        comm = (rep.get("communication_history", {}) or {}).get("bytes_per_round", [])
        if not r or not comm:
            continue
        mb = [b / 1e6 for b in comm[: len(r)]]
        color = colors[idx % len(colors)]
        ax.plot(r[: len(mb)], mb, marker="o", linewidth=2.0, color=color, label=f"{title} (comm)")

        sys_hist = rep.get("systems_history", {}) or {}
        ct_std = sys_hist.get("client_time_std", []) or []
        if ct_std:
            n = min(len(r), len(ct_std))
            ax2.plot(
                r[:n],
                ct_std[:n],
                marker="s",
                linewidth=2.0,
                linestyle="--",
                color=color,
                alpha=0.85,
                label=f"{title} (client time std)",
            )

    ax.set_title("Systems: Communication and Client Time Variability", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Total comm (MB/round)", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylabel("Client time std (s)", color="tab:brown")
    ax2.tick_params(axis="y", labelcolor="tab:brown")
    ax.grid(alpha=0.25)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        # Put legend outside to avoid covering either y-axis series.
        ax.legend(
            h1 + h2,
            l1 + l2,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metric(rep: Dict, key: str) -> float:
    cm = rep.get("clinical_metrics", {}) or {}
    if key == "accuracy":
        return float(cm.get("overall_accuracy", 0.0))
    if key == "macro_f1":
        v = cm.get("macro_f1")
        if v is not None:
            return float(v)
        # Backward-compatibility: older reports only stored per-class F1 and confusion matrix.
        class_metrics = cm.get("class_metrics") or {}
        if not class_metrics:
            return 0.0
        # Prefer stable ordering for ESI-5.
        names = list(class_metrics.keys())
        if set(names) == {"ESI5", "ESI4", "ESI3", "ESI2", "ESI1"}:
            names = ["ESI5", "ESI4", "ESI3", "ESI2", "ESI1"]
        if set(names) == {"Green", "Yellow", "Red"}:
            names = ["Green", "Yellow", "Red"]
        f1 = [float((class_metrics.get(n) or {}).get("f1_score", 0.0)) for n in names]
        if not f1:
            return 0.0
        return float(np.mean(np.asarray(f1, dtype=float)))
    if key == "critical_sensitivity":
        return float((cm.get("clinical_safety", {}) or {}).get("critical_sensitivity", 0.0))
    if key == "under_triage":
        return float((cm.get("clinical_safety", {}) or {}).get("under_triage_rate", 0.0))
    if key == "over_triage":
        return float((cm.get("clinical_safety", {}) or {}).get("over_triage_rate", 0.0))
    raise KeyError(key)


def plot_baseline_comparison(reports: Dict[str, Dict], out_path: Path) -> None:
    """Compare key metrics across strong centralized and FL systems."""
    import matplotlib.pyplot as plt  # type: ignore

    systems = list(reports.keys())
    vals = {
        "Accuracy (%)": [_metric(reports[s], "accuracy") * 100 for s in systems],
        "Macro-F1 (%)": [_metric(reports[s], "macro_f1") * 100 for s in systems],
        "Crit. Sens. (%)": [_metric(reports[s], "critical_sensitivity") * 100 for s in systems],
        "Under-Triage (%)": [_metric(reports[s], "under_triage") * 100 for s in systems],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.2))
    axes = axes.flatten()
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for ax, (title, v) in zip(axes, vals.items()):
        x = np.arange(len(systems))
        ax.bar(x, v, color=colors[: len(systems)], alpha=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=18, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        for i, val in enumerate(v):
            ax.text(i, val + (max(v) * 0.015 if max(v) else 0.2), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Baseline Comparison (ESI-5)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_byzantine_robustness(reports: Dict[str, Dict], out_path: Path) -> None:
    """Plot the effect of a Byzantine attack and robust aggregation."""
    import matplotlib.pyplot as plt  # type: ignore

    systems = list(reports.keys())
    acc = [_metric(reports[s], "accuracy") * 100 for s in systems]
    crit = [_metric(reports[s], "critical_sensitivity") * 100 for s in systems]
    over = [_metric(reports[s], "over_triage") * 100 for s in systems]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    def _bar(ax, values, title, ylabel):
        x = np.arange(len(systems))
        ax.bar(x, values, color=colors[: len(systems)], alpha=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=18, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(values):
            ax.text(i, v + (max(values) * 0.02 if max(values) else 0.2), f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    _bar(axes[0], acc, "Accuracy", "Test Acc (%)")
    _bar(axes[1], crit, "Critical Sens.", "Recall (ESI1–2) (%)")
    _bar(axes[2], over, "Over-Triage", "Over-triage (%)")

    fig.suptitle("Byzantine Robustness (Non-IID; 7 Clients; 10 Rounds)", fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load reports (skip missing)
    loaded_core: Dict[str, Dict] = {}
    for title, path in REPORTS_CORE.items():
        if not path.exists():
            print(f"[warn] Missing report: {path}")
            continue
        loaded_core[title] = _load_json(path)

    if not loaded_core:
        raise SystemExit("No core reports found; cannot generate figures.")

    loaded_byz: Dict[str, Dict] = {}
    for title, path in REPORTS_BYZ.items():
        if path.exists():
            loaded_byz[title] = _load_json(path)
        else:
            print(f"[warn] Missing robustness report: {path}")

    loaded_baselines: Dict[str, Dict] = {}
    for title, path in REPORTS_BASELINES.items():
        if path.exists():
            loaded_baselines[title] = _load_json(path)
        else:
            print(f"[warn] Missing baseline report: {path}")

    # Use any loaded report for dataset distribution (prefer DP report as it carries full data_info)
    dist_report = next(iter(loaded_core.values()))
    for title, rep in loaded_core.items():
        if rep.get("data_info", {}).get("total_samples") == 558029:
            dist_report = rep
            break

    plot_pipeline_overview(FIG_DIR / "pipeline_overview.png")
    plot_dataset_class_distribution(dist_report, FIG_DIR / "dataset_class_distribution.png")

    # Confusion matrices require ESI-5 (5-class) reports
    needed = [
        "Centralized (non-DP)",
        "FL Flat (non-DP; 3 sites, 20 rounds)",
        "FL Red-Focus (non-DP; 3 sites, 20 rounds)",
        "DP-SGD (3 sites, 5 rounds; $\\sigma{=}1.0$; Adam)",
    ]
    cm_reports = {k: loaded_core[k] for k in needed if k in loaded_core}
    if len(cm_reports) == 4:
        plot_confusion_matrices_esi5(cm_reports, FIG_DIR / "confusion_matrices_esi5.png")
    else:
        print("[warn] Missing one or more ESI-5 reports; skipping confusion matrices.")

    # 3-class confusion matrices (Green/Yellow/Red)
    loaded3: Dict[str, Dict] = {}
    for title, path in REPORTS_3CLASS.items():
        if path.exists():
            loaded3[title] = _load_json(path)
        else:
            print(f"[warn] Missing 3-class report: {path}")
    if len(loaded3) == 3:
        plot_confusion_matrices_3class(loaded3, FIG_DIR / "confusion_matrices_3class.png")
    else:
        print("[warn] Missing one or more 3-class reports; skipping 3-class confusion matrices.")

    # Trade-off summary (needs safety metrics present in all reports)
    short_names = {
        "Centralized (non-DP)": "Centralized",
        "FL Flat (non-DP; 3 sites, 20 rounds)": "FL Flat",
        "FL Red-Focus (non-DP; 3 sites, 20 rounds)": "FL Red-Focus",
        "DP-SGD (3 sites, 5 rounds; $\\sigma{=}1.0$; Adam)": "DP-SGD",
    }
    trade = {}
    for k in needed:
        if k in loaded_core:
            trade[short_names[k]] = loaded_core[k]
    if len(trade) == 4:
        plot_safety_tradeoff(trade, FIG_DIR / "safety_tradeoff.png")
    else:
        print("[warn] Missing one or more reports; skipping safety tradeoff plot.")

    # DP privacy/utility
    dp_key = "DP-SGD (3 sites, 5 rounds; $\\sigma{=}1.0$; Adam)"
    dp_rep = loaded_core.get(dp_key)
    if dp_rep and dp_rep.get("privacy", {}).get("epsilon_per_round"):
        plot_privacy_utility_dp(dp_rep, FIG_DIR / "fl_advanced_privacy_utility.png")

    # FL training dynamics (cross-silo runs)
    fl_dyn_keys = {
        "FL Flat (by-site)": "FL Flat (non-DP; 3 sites, 20 rounds)",
        "FL Red-Focus (by-site)": "FL Red-Focus (non-DP; 3 sites, 20 rounds)",
    }
    fl_dyn = {pretty: loaded_core[key] for pretty, key in fl_dyn_keys.items() if key in loaded_core}
    if len(fl_dyn) >= 1:
        plot_fl_acc_f1_vs_rounds(fl_dyn, FIG_DIR / "fl_advanced_acc_f1_vs_rounds.png")
        plot_fl_ece_nll_vs_rounds(fl_dyn, FIG_DIR / "fl_advanced_ece_nll_vs_rounds.png")
        plot_fl_systems_vs_rounds(fl_dyn, FIG_DIR / "fl_advanced_systems_vs_rounds.png")

    # Baseline comparison figure (tree baseline + NN + FL variants)
    baseline_bundle: Dict[str, Dict] = {}
    if "HistGradientBoosting" in loaded_baselines:
        baseline_bundle["HistGB"] = loaded_baselines["HistGradientBoosting"]
    if "Centralized (non-DP)" in loaded_core:
        baseline_bundle["Centralized NN"] = loaded_core["Centralized (non-DP)"]
    if "FL Red-Focus (non-DP; 3 sites, 20 rounds)" in loaded_core:
        baseline_bundle["FL Red-Focus"] = loaded_core["FL Red-Focus (non-DP; 3 sites, 20 rounds)"]
    if "FL Flat (non-DP; 3 sites, 20 rounds)" in loaded_core:
        baseline_bundle["FL Flat"] = loaded_core["FL Flat (non-DP; 3 sites, 20 rounds)"]
    if len(baseline_bundle) >= 2:
        plot_baseline_comparison(baseline_bundle, FIG_DIR / "baseline_comparison.png")

    # Byzantine robustness figure
    if len(loaded_byz) == 3:
        plot_byzantine_robustness(loaded_byz, FIG_DIR / "byzantine_robustness.png")

    print(f"✅ Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
