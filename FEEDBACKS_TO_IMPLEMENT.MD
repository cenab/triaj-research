# Feedback Implementation Status

This file tracks which reviewer feedback items are already implemented in the codebase, with concrete code references, plus partials/gaps to address next.

## Implemented

- Clinical safety objective (penalty matrix + focal + critical-miss penalty)
  - AdvancedClinicalSafetyLoss implements weighted CE, focal component and a clinical penalty matrix with extra penalty for critical misses.
    - standalone_pipeline/advanced_model_architecture.py:357
    - standalone_pipeline/advanced_model_architecture.py:376
    - standalone_pipeline/advanced_model_architecture.py:405
    - standalone_pipeline/advanced_model_architecture.py:417

- Decision-time optimization (temperature scaling + per-class offsets/thresholds for top-2 critical band)
  - Model supports global temperature scaling and per-class thresholds; inference applies thresholds via probability offsets.
    - Temperature: standalone_pipeline/advanced_model_architecture.py:152, standalone_pipeline/advanced_model_architecture.py:234
    - Thresholds API and application: standalone_pipeline/advanced_model_architecture.py:244, standalone_pipeline/advanced_model_architecture.py:223
  - Threshold sweep utilities target the top-2 critical classes (ESI1–2) and set thresholds on the model.
    - Centralized: standalone_pipeline/advanced_training.py:227
    - Federated: standalone_pipeline/fl_training.py:57, standalone_pipeline/fl_training.py:640

- Federated learning with robust aggregation and optional DP
  - Aggregators implemented: FedAvg, coordinate-wise median, trimmed mean, momentum (FedAvgM).
    - standalone_pipeline/fl_training.py:254
  - Non-IID Dirichlet client splits with per-client weighting by sample count.
    - standalone_pipeline/fl_training.py:446
  - Optional DP-SGD via Opacus; BatchNorm→GroupNorm swap; per-round epsilon tracking and privacy–utility plot.
    - BN→GN: standalone_pipeline/fl_training.py:132
    - DP wiring: standalone_pipeline/fl_training.py:520, standalone_pipeline/fl_training.py:540
    - Privacy–utility plot: standalone_pipeline/fl_training.py:810

- Reliability and clinical safety evaluation
  - Reliability metrics: NLL, ECE, per-class ECE, Brier; bootstrap utility.
    - standalone_pipeline/evaluation_framework.py:216
  - Clinical safety: under-/over-triage and critical sensitivity.
    - standalone_pipeline/evaluation_framework.py:52
  - FL reliability over rounds (ECE/NLL, Acc/Macro-F1) with plots.
    - standalone_pipeline/fl_training.py:740

- Reproducibility: deterministic, artifacted grid runner
  - Curated grid that detects trainer signatures, passes only supported kwargs, names artifacts, and writes CSV + JSONL.
    - run_grid_exact.py:1, run_grid_exact.py:52, run_grid_exact.py:111, run_grid_exact.py:160, run_grid_exact.py:274, run_grid_exact.py:520

- Data pipeline clarity (Kaggle loader + feature engineering → 5-class ESI)
  - Self-contained Kaggle loader with cached CSV; simple, numeric-only feature engineering; consistent 5-class ESI encoding.
    - standalone_pipeline/kaggle_data.py:39, standalone_pipeline/kaggle_data.py:94
  - Centralized/FL pipelines consume engineered features with consistent groupings.
    - standalone_pipeline/advanced_training.py:130
    - standalone_pipeline/fl_training.py:435

- Fairness evaluation (available in analysis path)
  - FairnessEvaluator provides group metrics, including positive_rate and critical_rate.
    - standalone_pipeline/evaluation_framework.py:119, standalone_pipeline/evaluation_framework.py:186
  - Advanced analysis computes fairness by gender and age bands and serializes results.
    - standalone_pipeline/advanced_analysis.py:132, standalone_pipeline/advanced_analysis.py:245

- Baselines for comparison
  - Logistic regression (+optional Platt calibration) and HistGradientBoosting baselines with reliability metrics and bootstrap CIs.
    - standalone_pipeline/advanced_analysis.py:292, standalone_pipeline/advanced_analysis.py:324

- Systems metrics for FL
  - Communication bytes/round and client time dispersion tracked and plotted.
    - standalone_pipeline/fl_training.py:603, standalone_pipeline/fl_training.py:784

- Class-balanced sampling per client
  - WeightedRandomSampler is used to rebalance per-client minibatches; import added so it activates (instead of silently falling back).
    - standalone_pipeline/fl_training.py:20, standalone_pipeline/fl_training.py:349

- Bootstrap CIs + reliability metrics in reports
  - Centralized and FL reports now log accuracy/macro-F1/NLL/ECE and safety CIs alongside point estimates.
    - standalone_pipeline/advanced_training.py:507
    - standalone_pipeline/fl_training.py:700

- Fairness metrics surfaced in core artifacts
  - Test-set fairness evaluation (gender + age bands) stored in centralized & federated reports and bubbled through run_grid_exact outputs.
    - standalone_pipeline/advanced_training.py:489
    - standalone_pipeline/fl_training.py:672
    - run_grid_exact.py:350

- Full Kaggle feature ingestion
  - `feature_engineer_kaggle_data` now numerically encodes every Kaggle column (demographics, comorbidities, labs, meds, chief complaints) and groups them for the hierarchical model.
    - standalone_pipeline/kaggle_data.py
    - standalone_pipeline/advanced_training.py:96
    - standalone_pipeline/advanced_training_red_focus.py:93
    - standalone_pipeline/fl_training.py:435

- Conformal Critical Control (C³) thresholds
  - Centralized and red-focus training calibrate conformal thresholds (overall + subgroup) to guarantee critical recall, replacing heuristic sweeps.
    - standalone_pipeline/conformal.py
    - standalone_pipeline/advanced_training.py:400
    - standalone_pipeline/advanced_training_red_focus.py:492

- Multi-token pathway attention
  - Attention now attends over pathway tokens (vital/symptom/risk/context/lab/interaction) with shared projections, fixing the degenerate single-token self-attention bug.
    - standalone_pipeline/advanced_model_architecture.py:24

- Train-split imputation only
  - Median imputation is now fitted on the train split and applied to val/test/FL clients, eliminating leakage from global preprocessing.
    - standalone_pipeline/advanced_training.py:338
    - standalone_pipeline/advanced_training_red_focus.py:371
    - standalone_pipeline/fl_training.py:429

## Partially Implemented

- Inference-time measurement improvements
  - Model-only latency captured everywhere; still need optional end-to-end timings (with data transfer) and documentation in PAPER.md.
    - standalone_pipeline/advanced_training.py:471
    - standalone_pipeline/evaluation_framework.py
    - standalone_pipeline/fl_training.py:689

- Threshold sweep for top-2 critical band
  - Implemented across centralized/red_focus/FL; next step is exposing per-class offsets in appendix tables.
    - standalone_pipeline/advanced_training.py:227
    - standalone_pipeline/advanced_training_red_focus.py:309
    - standalone_pipeline/fl_training.py:57

- Bootstrap CIs usage in paper tables
  - Reports now contain CIs; PAPER.md and publication tables still need to incorporate them explicitly.
    - standalone_pipeline/advanced_training.py:507
    - standalone_pipeline/fl_training.py:700

## Gaps To Address (from feedback)

- (Fixed) Threshold-sweep indentation/flow
  - Federated _sweep_thresholds_top2 scoring block dedented and constraints evaluated per candidate; returns non-trivial thresholds and score.
    - standalone_pipeline/fl_training.py:57

- Credible latency reporting
  - Model-only timings added with perf_counter and CUDA sync in centralized (advanced_training), evaluation benchmark, and FL report; end-to-end vs model-only split can be expanded later if needed.
    - standalone_pipeline/advanced_training.py (CUDA sync added)
    - standalone_pipeline/evaluation_framework.py (perf_counter + warm-up + CUDA sync)
    - standalone_pipeline/fl_training.py (perf_counter + warm-up + CUDA sync)

- (Fixed) Ensure class-balanced sampling actually activates
  - WeightedRandomSampler import added in fl_training.
    - standalone_pipeline/fl_training.py:20

- Use bootstrap CIs and explicit selection criteria in main results
  - Reports now include CIs + selection criterion; propagate into PAPER.md tables and narrative.
    - standalone_pipeline/advanced_training.py:507
    - standalone_pipeline/fl_training.py:700

- Fairness results in paper-level artifacts
  - Fairness is saved in reports/grid JSON; still need to render in PAPER.md tables and figures.
    - standalone_pipeline/advanced_training.py:489
    - standalone_pipeline/fl_training.py:672
    - run_grid_exact.py:350

- Engineering notes from feedback
  - _evaluate_model could optionally return logits for richer post-hoc calibration experiments (classwise temperatures).
    - standalone_pipeline/fl_training.py:200
  - Communication bytes/round assume float32; reflect mixed precision/compression if enabled.
    - standalone_pipeline/fl_training.py:603
  - Document feature-group counts in appendix; analysis already logs counts.
    - standalone_pipeline/advanced_analysis.py:582
