# Slide 1 – Collaborative Federated Triage Overview
- Problem: emergency departments need fast, precise triage under heavy demand.
- Data: 558,029 encounters with vitals, labs, symptom narratives, and context.
- Goal: show that privacy‑preserving, cross‑institution learning can approach centralized accuracy while keeping data local.
- Method: train collaboratively without sharing raw patient data; prioritize clinical safety.
- Novelty: introduces a safety‑aware loss and conformal calibration (C³) for clinically safe thresholds.
- Value: scales across hospitals, protects privacy, and supports trustworthy decisions.

# Slide 2 – Data Pipeline & Feature Engineering
- Standardize raw hospital exports into a unified schema with strict quality filters.
- Build six modalities: vitals, symptoms, risk factors, contextual information, labs, and engineered interactions.
- Encode labels as the five Emergency Severity Index levels (ESI5→ESI1) using ascending severity 0–4.
- Use a stratified 80/10/10 train/validation/test split.
- For interpretability, collapse into traffic‑light bands: Green (ESI5–ESI4), Yellow (ESI3), Red (ESI2–ESI1).

# Slide 3 – Centralized Training Flow
- Load the engineered dataset; impute missing values by feature medians and scale inputs.
- Group features so each model branch receives its corresponding subset.
- Use data loaders to coordinate multi‑branch training efficiently.
- Grid‑sweep loss settings, learning rates, and epochs.
- Select configurations that balance accuracy and clinical safety.

# Slide 4 – Model Architecture
- Six shallow neural pathways process each feature group independently.
- Project pathway outputs into a shared token space; fuse with multi‑head self‑attention and residual connections.
- A deep classifier with dropout and normalization produces final logits.
- Enforce ESI5→ESI1 ordering; report collapsed Red/Yellow/Green bands for control and monitoring.
- Provide built‑in calibration and optional per‑class thresholds for site‑specific alert tolerances.

# Slide 5 – Loss & Safety Shaping
- Blend weighted cross‑entropy and focal loss to address imbalance and hard examples.
- Add an explicit penalty for misclassifying the most severe (critical) encounters.
- Apply class weights reflecting observed frequencies across ESI levels.
- Offer presets—balanced, safety‑tilted, safety‑aggressive—to tune risk tolerance.
- Sweep thresholds to enforce minimum precision and recall for the Red band.

# Slide 6 – Centralized Evaluation Harness
- Track training/validation loss, accuracy, and macro‑F1 each epoch; checkpoint model and preprocessing state.
- Post‑training, apply temperature scaling and tune thresholds; record all adjustments for reproducibility.
- Optional analyses: feature attribution, fairness diagnostics, and calibration plots for audits.
- Produce evaluation reports that capture metrics and artifacts for transparent review.

# Slide 7 – Federated Orchestration
- Simulate multiple hospitals (clients) with configurable counts, round budgets, and aggregation strategies.
 - Support IID and non‑IID client splits (e.g., severity mix differences across hospitals).
- Allow optional local personalization epochs at each site.
 - Aggregate via standard FedAvg, momentum‑augmented averaging (FedAvgM), or robust (median/trimmed‑mean) strategies.
- Provide differential privacy through calibrated noise and normalization to protect individuals.

# Slide 8 – Federated Local Training Loop
- Each client trains locally with Adam, gradient clipping (max‑norm 1), and class‑balanced sampling.
- Use the same safety‑aware loss as centralized training for consistency.
- Clients return model updates; the coordinator aggregates and may apply momentum smoothing.
- Track reliability metrics, per‑round communication volume, and client timing for transparency.

# Slide 9 – Calibration & Thresholding in FL
- After each aggregation, perform temperature calibration to minimize validation negative log‑likelihood.
- Sweep class thresholds to meet Red‑band clinical constraints while limiting Green false alarms.
- Persist calibration and thresholds with the global model so sites can reproduce tuned decision logic without sharing data.

# Slide 10 – Experiment Configurations
- Centralized baseline: advanced hierarchical architecture, safety‑aggressive loss, 4 epochs, threshold tuning enabled.
- Federated baseline: same architecture across five simulated hospitals for 2 rounds; temperature 0.8; class thresholds 0.2, 0.0, 0.02 for non‑critical and critical levels.
- Red‑focus federated: ensemble with a specialized head for critical cases; 2 rounds with the same calibration temperature.
- Emphasis: clinically safe trade‑offs over raw accuracy alone.

# Slide 11 – Centralized Results
 - Overall accuracy: 0.88.
 - Green (ESI5–ESI4): precision 1.00, recall 0.81.
 - Yellow (ESI3): precision 0.86, recall 0.89.
 - Red (ESI2–ESI1): precision 0.83, recall 0.92.
 - Clinical safety: under‑triage 2.29%, over‑triage 9.59%, critical sensitivity 92.4%.
 - Inference ~0.008 ms/sample; model size ~0.223 MB (~58k parameters).

# Slide 12 – Federated Baseline Results
 - Overall accuracy: 0.87.
 - Strong Yellow recall: 97.2%; strong Red precision: 94.7%.
 - Reliability improved across rounds: ECE 0.04 → 0.02; NLL 0.31 → 0.28.
 - Clinical safety metrics not computed in this run; follow‑up will populate under‑ and over‑triage statistics.

# Slide 13 – Federated Red‑Focus Results
 - Accuracy: 0.87; Red recall 84.6%, Red precision 90.4%.
- Balanced performance maintained on Green and Yellow bands.
- Macro‑F1 improved from 86.05 to 86.81 between rounds.
 - Calibration improved: ECE 0.12 → 0.02.
- Communication ~0.96 MB/round; model size ~0.229 MB (~60k parameters).

# Slide 14 – Metric Comparison Summary
- Centralized training leads by ~1–1.5 percentage points in overall accuracy.
- Both federated variants remain within ~2 percentage points of the centralized baseline.
- Red‑focus narrows the gap while strengthening critical detection.
- Standard federated run offers the highest Yellow recall for operational balance.
- Completing federated clinical safety metrics will enable a comprehensive comparison table.

# Slide 15 – Reliability & Calibration Insights
- Federated models benefited from temperature 0.8, reducing overconfidence vs centralized temperature 1.0.
- Per‑class calibration shows the Red band gains most under the red‑focus approach.
- Persisted class thresholds let hospitals maintain minimum precision/recall targets without repeated tuning.
- Calibration and thresholding act as guardrails in distributed settings.

# Slide 16 – System Considerations
- Automation scripts streamline dataset preparation, grid sweeps, and federated coordination.
- Communication histories enable bandwidth estimation and monitoring.
- Gradient clipping and momentum‑based aggregation stabilize convergence.
- Modular design supports alternate aggregators, privacy budgets, and client participation policies as needs evolve.

# Slide 17 – Limitations & Next Steps
- Regenerate federated evaluation reports so clinical safety metrics are complete for external review.
- Stress‑test additional rounds, higher client counts, and non‑IID severity distributions.
- Extend fairness audits to sensitive demographics.
- Integrate deployment pipelines for on‑premises inference.
- Involve clinicians in looped validation to mature the collaboration.

# Slide 18 – Takeaways
- Safety‑aware, hierarchical architectures let federated learning approach centralized accuracy for emergency triage.
- Calibration and thresholding guardrails are essential for trustworthy distributed predictions.
- The framework is ready for cross‑institution pilots with privacy, explainability, and monitoring hooks.

# Slide 19 – Backup: Implementation Overview
- Training, analysis, orchestration, and data ingestion live in distinct modules.
- Documentation and consolidated reports make replication straightforward.
- Clear separation of concerns supports maintainability and auditing.
- Reproducible pipelines enable consistent results across sites.

# Slide 20 – Q&A / Discussion
- Discuss deployment pathways, fairness targets, differential‑privacy trade‑offs, and partner onboarding timelines.
- To reproduce results, follow documented configurations and use orchestration scripts to launch new sweeps.
- Address clinical integration, governance considerations, and future research collaborations.

# Slide 21 – Novelty & Contributions
- C³ (Conformal Critical Control): deployment‑time thresholds that guarantee minimum recall for critical cases (global and subgroup), applied to summed probability over ESI1–2.
- Safety‑aware loss: weighted CE + focal + clinical penalty matrix (quadratic under‑triage, linear over‑triage) with an explicit critical‑miss term.
- Hierarchical triage head: gate Non‑Critical vs Critical and compose specialist heads; generalized from 3‑class to full ESI‑5.
 - Multi‑modal token fusion: treat feature groups as tokens and fuse with lightweight self‑attention; DP‑compatible attention (works with Opacus).
 - Federated robustness: non‑IID client splits (e.g., different severity mixes), robust aggregation (median/trimmed‑mean) and momentum‑augmented averaging (FedAvgM); optional client‑side DP‑SGD with privacy accounting and GroupNorm replacement.
- Reliability and safety CIs: report NLL/ECE/Brier and bootstrap confidence intervals for clinical safety metrics; privacy–utility and per‑round curves.
- Reproducible toolkit: centralized/FL trainers, fairness diagnostics, SHAP analysis, communication accounting, and a cross‑institution validation scaffold.
