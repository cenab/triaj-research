# Slide 1 – Collaborative Federated Triage Overview
- Emergency departments need fast, precise triage under heavy demand.
- We analyze 558,029 triage encounters with vitals, labs, symptom narratives, and context.
- Goal: show that privacy-preserving, cross-institutional learning can nearly match centralized accuracy while keeping data local.
- Approach: train collaboratively without sharing raw patient data, emphasizing clinical safety.
- Value: scales across hospitals while protecting privacy and enabling trustworthy decisions.

# Slide 2 – Data Pipeline & Feature Engineering
- Standardize raw hospital exports into a unified schema with strict quality filters.
- Build six feature modalities: vitals, symptoms, risk factors, contextual information, labs, and engineered interactions.
- Encode labels as the five Emergency Severity Index levels (ESI5 to ESI1) using ascending severity 0–4.
- Perform a stratified 80/10/10 split for training, validation, and test sets.
- For interpretability, group classes into traffic-light bands: Green (ESI5–ESI4), Yellow (ESI3), Red (ESI2–ESI1).

# Slide 3 – Centralized Training Flow
- Load the engineered dataset; impute missing values by feature medians and scale inputs.
- Create grouped tensors so each model branch receives its corresponding feature subset.
- Use data loaders to coordinate multi-branch training efficiently.
- Sweep curated grids of loss settings, learning rates, and epochs.
- Select configurations that balance accuracy and clinical safety.

# Slide 4 – Model Architecture
- Six shallow neural pathways process each feature group independently.
- Pathway outputs project into a shared token space and are fused with multi-head self-attention and residual connections.
- A deep classifier with dropout and normalization produces final logits.
- Predictions follow the ESI5→ESI1 ordering and are collapsed into Red/Yellow/Green bands for reporting and control.
- Built-in calibration and optional per-class thresholds support hospital-specific alert tolerances.

# Slide 5 – Loss & Safety Shaping
- Custom loss blends weighted cross-entropy and focal loss to address imbalance and hard examples.
- Adds an explicit penalty for misclassifying the most severe (critical) encounters.
- Class weights restore balance across the five ESI levels based on observed frequencies.
- Preset families (balanced, safety-tilted, safety-aggressive) tune risk tolerance.
- Threshold sweeps enforce minimum precision and recall for the critical Red band.

# Slide 6 – Centralized Evaluation Harness
- Track training and validation loss, accuracy, and macro-F1 each epoch; save checkpoints with model and preprocessing state.
- After training, apply temperature scaling for calibration and tune thresholds; record all adjustments for reproducibility.
- Optional analyses generate feature-attribution, fairness diagnostics, and calibration plots for deeper audits.
- Evaluation reports capture metrics and artifacts for transparent review.

# Slide 7 – Federated Orchestration
- Simulate multiple hospitals (clients) with configurable counts, round budgets, and aggregation strategies.
- Support both IID and non-IID client splits, including Dirichlet-based severity skew.
- Allow optional local personalization epochs at each site.
- Aggregate with standard, momentum-augmented, or robust/trimmed-mean strategies.
- Provide differential privacy via calibrated noise and normalization to protect individuals.

# Slide 8 – Federated Local Training Loop
- Each client trains locally with Adam, gradient clipping (max norm 1), and class-balanced sampling.
- Use the same safety-aware loss as centralized training for consistency.
- Clients return model updates; the coordinator aggregates them and may apply momentum smoothing.
- Track reliability metrics, per-round communication volume, and client timing for operational transparency.

# Slide 9 – Calibration & Thresholding in FL
- After each aggregation, perform temperature calibration to minimize validation negative log-likelihood.
- Sweep class thresholds to meet clinical constraints on the Red band while limiting false alarms in the Green band.
- Store calibration and thresholds with the global model so sites can reproduce tuned decision logic without sharing data.

# Slide 10 – Experiment Configurations
- Centralized baseline: advanced hierarchical architecture, safety-aggressive loss, four epochs, and threshold tuning enabled.
- Federated baseline: same architecture across five simulated hospitals for two rounds; temperature tuned to 0.8; class thresholds of 0.2, 0.0, and 0.02 for noncritical and critical levels.
- Red-focus federated experiment: ensemble with a specialized head for critical cases; two rounds with the same calibration temperature.
- Emphasis on clinically safe trade-offs over raw accuracy alone.

# Slide 11 – Centralized Results
- Overall accuracy: 0.881.
- Green (ESI5–ESI4): precision 0.999, recall 0.814.
- Yellow (ESI3): precision 0.860, recall 0.894.
- Red (ESI2–ESI1): precision 0.830, recall 0.924.
- Clinical safety: under-triage 2.29%, over-triage 9.59%, critical sensitivity 92.4%.
- Inference speed ~0.008 ms per sample; model size ~0.223 MB (~58k parameters).

# Slide 12 – Federated Baseline Results
- Overall accuracy: 0.866.
- Strong Yellow recall: 97.2%; strong Red precision: 94.7%.
- Reliability improved across rounds: expected calibration error 0.044 → 0.016; negative log-likelihood 0.314 → 0.276.
- Clinical safety metrics were not computed in this run; a follow-up analysis will populate under- and over-triage statistics.

# Slide 13 – Federated Red-Focus Results
- Accuracy: 0.870 with Red band recall 84.6% and precision 90.4%.
- Balanced performance maintained on Green and Yellow bands.
- Macro-F1 improved from 86.05 to 86.81 between rounds.
- Calibration improved: expected calibration error 0.121 → 0.019.
- Communication overhead ~0.96 MB per round; model size ~0.229 MB (~60k parameters).

# Slide 14 – Metric Comparison Summary
- Centralized training leads by ~1–1.5 percentage points in overall accuracy.
- Both federated variants remain within ~2 percentage points of the centralized baseline.
- Red-focus configuration narrows the gap while strengthening detection of critical cases.
- Standard federated run offers the highest Yellow recall for operational balance.
- Completing clinical safety metrics for federated runs will enable a comprehensive comparison table.

# Slide 15 – Reliability & Calibration Insights
- Federated models benefited from temperature tuning to 0.8, reducing overconfidence relative to the centralized temperature of 1.0.
- Per-class calibration shows the Red band gains the most under the red-focus approach.
- Persisted class thresholds let hospitals maintain minimum precision/recall targets without repeated manual tuning.
- Calibration and thresholding serve as guardrails in distributed settings.

# Slide 16 – System Considerations
- Automation scripts streamline dataset preparation, grid sweeps, and federated coordination.
- Communication histories enable bandwidth estimation and monitoring.
- Gradient clipping and momentum-based aggregation stabilize convergence.
- Modular design supports alternate aggregators, privacy budgets, and client participation policies as deployment needs evolve.

# Slide 17 – Limitations & Next Steps
- Regenerate federated evaluation reports so clinical safety metrics are complete for external review.
- Stress-test additional rounds, higher client counts, and non-IID severity distributions.
- Extend fairness audits to sensitive demographics.
- Integrate deployment pipelines for on-premises inference.
- Involve clinicians in looped validation to mature the collaboration.

# Slide 18 – Takeaways
- Advanced hierarchical architectures with safety-aware loss enable federated learning to approach centralized accuracy for emergency triage.
- Calibration and thresholding guardrails are essential in distributed settings to deliver trustworthy predictions.
- The framework is ready for cross-institution pilots that respect privacy while providing explainability and monitoring hooks.

# Slide 19 – Backup: Implementation Overview
- Training, analysis, orchestration, and data ingestion are organized into distinct modules within the repository.
- Documentation and consolidated reports make replication straightforward.
- Clear separation of concerns supports maintainability and auditing.
- Reproducible pipelines enable consistent results across sites.

# Slide 20 – Q&A / Discussion
- Discuss deployment pathways, fairness targets, differential privacy trade-offs, and partner onboarding timelines.
- To reproduce results, follow the documented experiment configurations and use the provided orchestration scripts to launch new sweeps.
- Questions welcome on clinical integration, governance considerations, and future research collaborations.

# Slide 21 – Novelty & Contributions
- C³ (Conformal Critical Control): deployment-time thresholds that guarantee minimum recall for critical cases (global and subgroup), applied to the summed probability over ESI1–2.
- Safety‑aware loss: weighted CE + focal + clinical penalty matrix (quadratic under‑triage, linear over‑triage) with an explicit critical‑miss term.
- Hierarchical triage head: gate Non‑Critical vs Critical and compose specialist heads; generalized from 3‑class to full ESI‑5.
- Multi‑modal token fusion: treat feature groups as tokens and fuse with lightweight self‑attention; DP‑compatible attention when Opacus is available.
- Federated robustness: non‑IID client splits (Dirichlet), robust aggregation (median/trim) and FedAvgM; optional client‑side DP‑SGD with privacy accounting and GroupNorm replacement.
- Reliability and safety CIs: report NLL/ECE/Brier and bootstrap confidence intervals for clinical safety metrics; privacy–utility and per‑round curves.
- Reproducible toolkit: centralized/FL trainers, fairness diagnostics, SHAP analysis, communication accounting, and a cross‑institution validation scaffold.
