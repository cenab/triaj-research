#!/usr/bin/env python3
"""Sanity-check paper artifacts (reports + figures) for missing/degenerate outputs.

This script is meant to catch issues like:
- Missing figure files referenced by `paper.tex`
- NaNs / constant histories in report JSONs
- Prediction collapse in confusion matrices (e.g., always predicting one class)
- Blank / low-variance PNG outputs

Run:
  .venv/bin/python scripts/audit_paper_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _isfinite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _uniq_count(seq: Iterable[object]) -> int:
    vals = []
    for x in seq:
        if _isfinite(x):
            vals.append(round(float(x), 8))
    return len(set(vals))


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class AuditFinding:
    kind: str
    subject: str
    message: str


def _audit_report(path: Path) -> List[AuditFinding]:
    rep = _load_json(path)
    findings: List[AuditFinding] = []

    # NaN scan in key histories
    for hist_name in ["training_history", "reliability_history", "communication_history", "systems_history"]:
        hist = rep.get(hist_name, {}) or {}
        if not isinstance(hist, dict):
            continue
        for k, v in hist.items():
            if isinstance(v, list) and any(isinstance(x, (int, float)) and not math.isfinite(float(x)) for x in v):
                findings.append(
                    AuditFinding(
                        "report_nonfinite",
                        f"{path.name}:{hist_name}.{k}",
                        "Contains non-finite (NaN/inf) values.",
                    )
                )

    # Flatness checks (comm is allowed to be constant)
    th = rep.get("training_history", {}) or {}
    if isinstance(th, dict):
        for k in ["val_acc", "val_macro_f1", "val_loss", "train_loss"]:
            v = th.get(k)
            if isinstance(v, list) and len(v) >= 3 and _uniq_count(v) <= 1:
                findings.append(
                    AuditFinding(
                        "report_constant_history",
                        f"{path.name}:training_history.{k}",
                        f"Constant series (n={len(v)}).",
                    )
                )

    # DP epsilon monotonic
    priv = rep.get("privacy", {}) or {}
    if isinstance(priv, dict) and priv.get("dp"):
        eps = priv.get("epsilon_per_round")
        if isinstance(eps, list) and len(eps) >= 2:
            mono = all(float(eps[i + 1]) >= float(eps[i]) - 1e-9 for i in range(len(eps) - 1))
            if not mono:
                findings.append(
                    AuditFinding(
                        "report_privacy_nonmono",
                        f"{path.name}:privacy.epsilon_per_round",
                        "Not nondecreasing.",
                    )
                )

    # Confusion-matrix collapse check (predicted distribution)
    cm = (rep.get("clinical_metrics", {}) or {}).get("confusion_matrix")
    if cm is not None:
        cm = np.asarray(cm, dtype=float)
        col = cm.sum(axis=0)
        tot = float(col.sum())
        if tot > 0:
            p = col / tot
            m = float(p.max())
            if m > 0.98:
                findings.append(
                    AuditFinding(
                        "report_prediction_collapse",
                        f"{path.name}:clinical_metrics.confusion_matrix",
                        f"Predictions collapse to one class (max column share={m:.3f}).",
                    )
                )

    # Headline metric sanity
    clin = rep.get("clinical_metrics", {}) or {}
    if isinstance(clin, dict):
        for k in ["overall_accuracy", "macro_f1"]:
            v = clin.get(k)
            if isinstance(v, (int, float)) and float(v) <= 0.0:
                findings.append(
                    AuditFinding(
                        "report_zero_metric",
                        f"{path.name}:clinical_metrics.{k}",
                        f"Non-positive value: {v}.",
                    )
                )

    return findings


def _audit_png(path: Path) -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    size = path.stat().st_size
    if size < 10_000:
        findings.append(AuditFinding("png_too_small", path.name, f"File size is very small: {size} bytes."))

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
    std = float(arr.std())
    if std < 0.03:
        findings.append(AuditFinding("png_low_variance", path.name, f"Low pixel variance (std={std:.4f})."))
    return findings


def _parse_paper_figures(tex_path: Path) -> List[str]:
    tex = tex_path.read_text(encoding="utf-8")
    # Capture \includegraphics[...]{path}
    paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return [p.strip() for p in paths if p.strip()]


def main() -> int:
    findings: List[AuditFinding] = []

    # 1) Paper -> figure paths exist
    paper_tex = REPO_ROOT / "paper.tex"
    if paper_tex.exists():
        for rel in _parse_paper_figures(paper_tex):
            p = (REPO_ROOT / rel).resolve()
            if not p.exists():
                findings.append(AuditFinding("paper_missing_figure", rel, "Referenced by paper.tex but missing."))
            elif p.suffix.lower() == ".png":
                findings.extend(_audit_png(p))
    else:
        findings.append(AuditFinding("paper_missing", "paper.tex", "paper.tex not found."))

    # 2) Reports referenced by the figure script exist + are sane
    try:
        sys.path.insert(0, str((REPO_ROOT / "scripts").resolve()))
        import generate_paper_figures as gpf  # type: ignore

        report_dicts: List[Tuple[str, Dict[str, Path]]] = [
            ("REPORTS_CORE", getattr(gpf, "REPORTS_CORE", {})),
            ("REPORTS_BYZ", getattr(gpf, "REPORTS_BYZ", {})),
            ("REPORTS_BASELINES", getattr(gpf, "REPORTS_BASELINES", {})),
            ("REPORTS_3CLASS", getattr(gpf, "REPORTS_3CLASS", {})),
        ]
        for dict_name, d in report_dicts:
            if not isinstance(d, dict):
                continue
            for title, path in d.items():
                path = Path(path)
                if not path.exists():
                    findings.append(AuditFinding("missing_report", f"{dict_name}:{title}", f"Missing: {path}"))
                    continue
                findings.extend(_audit_report(path))
    except Exception as e:
        findings.append(AuditFinding("audit_error", "import generate_paper_figures", f"Failed: {e}"))

    # 3) Figures directory: basic checks (size, hashes, variance)
    fig_dir = REPO_ROOT / "figures"
    if fig_dir.exists():
        pngs = sorted(fig_dir.glob("*.png"))
        hashes: Dict[str, List[str]] = {}
        for p in pngs:
            h = _sha256_12(p)
            hashes.setdefault(h, []).append(p.name)
            findings.extend(_audit_png(p))
        for h, names in hashes.items():
            if len(names) > 1:
                findings.append(AuditFinding("png_duplicate", ",".join(names), f"Duplicate sha256 prefix: {h}"))
    else:
        findings.append(AuditFinding("missing_figures_dir", "figures/", "Directory not found."))

    if not findings:
        print("AUDIT OK: no findings.")
        return 0

    by_kind: Dict[str, List[AuditFinding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)

    print("AUDIT FAIL: findings detected.")
    for kind in sorted(by_kind.keys()):
        print(f"\n[{kind}] ({len(by_kind[kind])})")
        for f in by_kind[kind]:
            print(f"- {f.subject}: {f.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
