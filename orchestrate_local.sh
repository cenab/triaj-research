#!/usr/bin/env bash
# ===========================================================
#  Full Experiment Orchestration Script (LOCAL CSV VERSION)
#  Updated for the new centralized + red-focus + FL pipeline
# ===========================================================

set -Eeuo pipefail
set -o pipefail

echo "🔧 Setting up environment..."
# Make PYTHONPATH robust under `set -u` (avoid unbound var when empty)
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

# Resolve Python interpreter (prefer local venv)
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "❌ No Python interpreter found. Install Python 3 or create .venv." >&2
  exit 1
fi
echo "Using Python: $($PY -V 2>/dev/null || echo unknown)"
mkdir -p results/sweeps data/processed data/splits data/reports docs/figures

# Path to your cached Kaggle CSV (override by: DATA_CSV=/path/to/file bash orchestrate_local.sh)
DATA_CSV="${DATA_CSV:-/Users/batu/Documents/DEVELOPMENT/triaj-research/src/kaggle_triage_data.csv}"

# -------- 0) DATA COLLECTION (skipped: resuming from Step 5) ------------------
# echo "📦 Collecting and processing triage data from local CSV..."
# $PY -m standalone_pipeline.data_collect \
#   --local-csv "$DATA_CSV" \
#   --clients 5 \
#   --dirichlet-alpha 1.0 \
#   --qc-miss 0.40

# -------- 1) CENTRALIZED TRAINING (skipped) -----------------------------------
# echo "🏥 Centralized training (baseline flat model)…"
# $PY -m standalone_pipeline.advanced_training
# echo "🏥 Centralized training (hierarchical red-focus model)…"
# $PY -m standalone_pipeline.advanced_training_red_focus

# -------- 2) FEDERATED TRAINING (skipped) -------------------------------------
# echo "🌐 Federated training (baseline IID)…"
# $PY -m standalone_pipeline.fl_training baseline \
#   --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
#   --learning_rate 5e-3
# echo "🌐 Federated training (baseline non-IID)…"
# $PY -m standalone_pipeline.fl_training baseline \
#   --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
#   --learning_rate 5e-3 --non-iid --dirichlet-alpha 0.5
# echo "🌐 Federated training (hierarchical red-focus, non-IID, median agg)…"
# $PY -m standalone_pipeline.fl_training red_focus \
#   --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
#   --learning_rate 5e-3 --non-iid --dirichlet-alpha 0.5 --aggregator median

# -------- 3) ABLATIONS (skipped) ----------------------------------------------
# echo "⚗️ Running ablation study…"
# $PY -m standalone_pipeline.ablation_runner --runs 3

# -------- 4) POST-TRAINING ANALYSIS (skipped) ---------------------------------
# echo "🔍 Post-training analysis (centralized latest)…"
LATEST_CKPT=$(ls -t results/advanced_model_*.pth 2>/dev/null | head -n 1 || true)
# if [[ -n "$LATEST_CKPT" ]]; then
#   echo "Analyzing checkpoint: $LATEST_CKPT"
#   $PY -m standalone_pipeline.advanced_analysis "$LATEST_CKPT" \
#     --plots reliability fairness decision_curves \
#     --stable-prefix centralized
#   # Optional baselines for context
#   $PY -m standalone_pipeline.advanced_analysis --baseline histgb
#   $PY -m standalone_pipeline.advanced_analysis --baseline logreg --calibrate platt
# else
#   echo "⚠️ No centralized checkpoint found; skipping advanced_analysis."
# fi

# -------- 5) SITE-LEVEL PREDICTIONS (for cross-site validation) ----------------
# Creates data/processed/triage_processed_with_pred.csv (adds 'pred' column)
if [[ -n "$LATEST_CKPT" ]]; then
  echo "🧮 Creating per-row predictions for cross-site validation…"
  $PY - <<PYCODE
import numpy as np, pandas as pd, torch
from pathlib import Path
from standalone_pipeline.advanced_model_architecture import (
    AdvancedHierarchicalTriageModel, AdvancedHierarchicalTriageEnsemble
)
from standalone_pipeline.advanced_training import FeatureGroups, _to_tensor

CSV = Path("data/processed/triage_processed.csv")
CKPT = Path("${LATEST_CKPT}")
if not CKPT.exists() or not CSV.exists():
    raise SystemExit("Missing checkpoint or processed CSV")

df = pd.read_csv(CSV)
ckpt = torch.load(CKPT, map_location="cpu")
fg = FeatureGroups(**ckpt["feature_groups"])
num_classes = int(df["esi_5class_encoded"].astype(int).nunique())

sd = ckpt["model_state_dict"]
is_ens = any(k.startswith("backbone.") for k in sd.keys())
# Instantiate model consistent with checkpoint (match advanced_analysis logic)
common_kwargs = dict(
    num_vital_features=len(fg.vital),
    num_symptom_features=len(fg.symptom),
    num_risk_features=len(fg.risk),
    num_context_features=len(fg.context),
    num_lab_features=len(fg.lab),
    num_interaction_features=len(fg.interaction),
)
if is_ens:
    model = AdvancedHierarchicalTriageEnsemble(**common_kwargs)
else:
    model = AdvancedHierarchicalTriageModel(num_classes=num_classes, **common_kwargs)
model.load_state_dict(sd, strict=False)
model.eval()

# Ensure all expected feature columns exist; create any missing as zeros
cols = fg.vital + fg.symptom + fg.risk + fg.context + fg.lab + fg.interaction
missing = [c for c in cols if c not in df.columns]
if missing:
    for c in missing:
        df[c] = 0.0

# Try to reapply saved preprocessing (mean/scale + train medians). Fallback to best-effort standardization.
X = df[cols].copy()
pre = ckpt.get("preprocess")
if pre:
    pre_cols = pre.get("columns", cols)
    # Create any preprocess columns that are missing
    for c in pre_cols:
        if c not in X.columns:
            X[c] = 0.0
    med = pd.Series(pre.get("train_medians", {}))
    mean = pd.Series(pre.get("mean", [0.0]*len(pre_cols)), index=pre_cols)
    scale = pd.Series(pre.get("scale", [1.0]*len(pre_cols)), index=pre_cols).replace(0, 1.0)
    X = X.reindex(columns=pre_cols)  # enforce training order
    X = X.fillna(med).astype(np.float32)
    X = (X - mean) / scale
else:
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X = (X - X.mean()) / (X.std(ddof=0).replace(0, 1.0))

tensors = [
    _to_tensor(X, fg.vital),
    _to_tensor(X, fg.symptom),
    _to_tensor(X, fg.risk),
    _to_tensor(X, fg.context),
    _to_tensor(X, fg.lab),
    _to_tensor(X, fg.interaction),
]
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=1024, shuffle=False)

def compose(outputs, temp=1.0):
    if isinstance(outputs, tuple):
        heads = tuple(o / max(temp, 1e-6) for o in outputs)
        if len(heads) == 2:
            g, nc = heads
            g = torch.softmax(g, 1); nc = torch.softmax(nc, 1)
            noncrit, crit = g[:, 0:1], g[:, 1:2]
            return torch.cat([noncrit*nc[:, 0:1], noncrit*nc[:, 1:2], crit], 1)
        elif len(heads) == 3:
            gate, nc, crit = heads
            gate = torch.softmax(gate, 1)
            return torch.cat([gate[:, 0:1]*torch.softmax(nc,1), gate[:, 1:2]*torch.softmax(crit,1)], 1)
        else:
            raise RuntimeError("Unexpected head tuple length")
    return torch.softmax(outputs / max(temp, 1e-6), 1)

temp = float(getattr(model, "calibration_temperature", torch.tensor(1.0)).detach().cpu().item())
preds = []
with torch.no_grad():
    for batch in loader:
        outputs = model(*batch)
        probs = compose(outputs, temp=temp)
        preds.extend(probs.argmax(1).cpu().numpy().tolist())

out = df.copy()
out["pred"] = np.array(preds, dtype=int)
out.to_csv("data/processed/triage_processed_with_pred.csv", index=False)
print("✅ Wrote predictions → data/processed/triage_processed_with_pred.csv")
PYCODE
fi

# -------- 6) CROSS-INSTITUTION VALIDATION -------------------------------------
echo "🏥 Cross-institution validation…"
CSV_FOR_XVAL="data/processed/triage_processed_with_pred.csv"
if [[ -f "$CSV_FOR_XVAL" ]]; then
  $PY -m standalone_pipeline.cross_institution_validation \
    --csv "$CSV_FOR_XVAL" --site-col site_id
else
  echo "⚠️ No per-row predictions file found; running validator on processed CSV (will skip metrics)."
  $PY -m standalone_pipeline.cross_institution_validation \
    --csv data/processed/triage_processed.csv --site-col site_id
fi

# -------- 7) CURATED GRID SWEEP -----------------------------------------------
echo "🧮 Curated grid sweep (centralized + federated)…"
$PY run_grid_exact.py --outdir results/sweeps/grid_v1

# Optional quick sanity:
# python run_grid_exact.py --outdir results/sweeps/quick --max-runs 10

# -------- 8) SUMMARY PREVIEW ---------------------------------------------------
echo "📊 Top 10 by (accuracy, macro-F1)…"
if [[ -f results/sweeps/grid_v1/results.csv ]]; then
$PY - <<'PY'
import pandas as pd
df = pd.read_csv("results/sweeps/grid_v1/results.csv")
top = df.sort_values(["overall_accuracy","macro_f1"], ascending=[False,False]).head(10)
cols = ["run_id","run_type","trainer_label","hierarchical","learning_rate","batch_size",
        "epochs","overall_accuracy","macro_f1","critical_sensitivity","under_triage_rate"]
print(top[cols].to_string(index=False))
PY
else
  echo "⚠️ No sweep results found yet."
fi

echo "✅ All experiments completed."
