#!/usr/bin/env bash
# ===========================================================
#  Full Experiment Orchestration Script (LOCAL CSV VERSION)
#  Uses your cached Kaggle triage dataset instead of KaggleHub
# ===========================================================

set -e  # Exit on any error
set -o pipefail

# -------- ENVIRONMENT SETUP --------------------------------
echo "🔧 Setting up environment..."
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p results/sweeps
mkdir -p data/processed data/splits data/reports

# -------- 0. DATA COLLECTION -------------------------------
echo "📦 Collecting and processing triage data from local CSV..."
python -m standalone_pipeline.data_collect \
  --local-csv "/Users/batu/Documents/DEVELOPMENT/triaj-research/src/kaggle_triage_data.csv" \
  --clients 5 \
  --dirichlet-alpha 1.0

# -------- 1. CENTRALIZED TRAINING ---------------------------
echo "🏥 Running centralized training (baseline flat model)..."
python -m standalone_pipeline.advanced_training

echo "🏥 Running centralized training (hierarchical red-focus model)..."
python -m standalone_pipeline.advanced_training_red_focus

# -------- 2. FEDERATED TRAINING -----------------------------
echo "🌐 Running federated training (baseline IID)..."
python -m standalone_pipeline.fl_training baseline \
  --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
  --learning_rate 5e-3

echo "🌐 Running federated training (baseline non-IID)..."
python -m standalone_pipeline.fl_training baseline \
  --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
  --learning_rate 5e-3 --non-iid --dirichlet-alpha 0.5

echo "🌐 Running federated training (hierarchical red-focus non-IID)..."
python -m standalone_pipeline.fl_training red_focus \
  --rounds 10 --clients 5 --local_epochs 1 --batch_size 256 \
  --learning_rate 5e-3 --non-iid --dirichlet-alpha 0.5 --aggregator median

# -------- 3. ABLATIONS -------------------------------------
echo "⚗️ Running ablation study..."
python -m standalone_pipeline.ablation_runner --runs 3

# -------- 4. POST-TRAINING ANALYSIS -------------------------
echo "🔍 Running post-training analysis..."
LATEST_CKPT=$(ls -t results/advanced_model_*.pth 2>/dev/null | head -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
  echo "Analyzing latest checkpoint: $LATEST_CKPT"
  python -m standalone_pipeline.advanced_analysis "$LATEST_CKPT" --plots reliability fairness decision_curves
else
  echo "⚠️ No checkpoint found for analysis; skipping."
fi

# -------- 5. CROSS-INSTITUTION VALIDATION -------------------
echo "🏥 Running cross-institution validation..."
python -m standalone_pipeline.cross_institution_validation \
  --csv data/processed/triage_processed.csv --site-col site_id

# -------- 6. CURATED GRID SWEEP -----------------------------
echo "🧮 Running curated grid sweep (centralized + federated)..."
python run_grid_exact.py --outdir results/sweeps/grid_v1

# Optional quick sanity run:
# python run_grid_exact.py --outdir results/sweeps/quick --max-runs 10

# -------- 7. SUMMARY PREVIEW -------------------------------
echo "📊 Showing top 10 results by accuracy and F1..."
if [[ -f results/sweeps/grid_v1/results.csv ]]; then
  python - <<'PYCODE'
import pandas as pd
df = pd.read_csv("results/sweeps/grid_v1/results.csv")
top = df.sort_values(["overall_accuracy","macro_f1"], ascending=[False,False]).head(10)
print(top[["run_id","run_type","trainer_label","hierarchical","learning_rate","batch_size",
           "epochs","overall_accuracy","macro_f1","critical_sensitivity","under_triage_rate"]])
PYCODE
else
  echo "⚠️ No sweep results found yet."
fi

echo "✅ All experiments completed successfully."