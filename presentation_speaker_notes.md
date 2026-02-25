# Speaker Notes – Collaborative Federated Triage (End‑to‑End)

## Slide 1 – Collaborative Federated Triage Overview
This work addresses crowded emergency departments where rapid, accurate triage is critical. We leverage 558k encounters spanning vitals, labs, symptoms, and context to train models that keep data local. The aim is near‑centralized accuracy without sharing raw patient data, with clinical safety as a first‑class constraint. The value proposition is scale, privacy, and trust.
Transition: next, show how data is standardized and represented.

## Slide 2 – Data Pipeline & Feature Engineering
We normalize heterogeneous hospital exports into a clean, unified schema with strict quality filters. Six complementary modalities capture physiology and context, and labels follow ESI5→ESI1 with severity encoded 0–4. A stratified 80/10/10 split preserves label balance. Grouping into traffic‑light bands (Green/Yellow/Red) simplifies interpretation and downstream control.
Transition: with data ready, describe centralized training.

## Slide 3 – Centralized Training Flow
We impute by feature medians, scale inputs, and feed grouped feature subsets into dedicated model branches. Efficient data loaders coordinate multi‑branch batches. We sweep curated loss/learning‑rate/epoch grids and pick settings that balance accuracy with safety.
Transition: now open the model’s architecture.

## Slide 4 – Model Architecture
Each modality passes through a shallow pathway; outputs become tokens fused via lightweight self‑attention with residuals. A deep classifier with dropout and normalization produces final logits. We respect the ESI5→ESI1 ordering and report collapsed Red/Yellow/Green bands for governance. Calibration and per‑class thresholds tailor alert tolerance by site.
Transition: explain how the loss emphasizes clinical safety.

## Slide 5 – Loss & Safety Shaping
The loss blends weighted cross‑entropy and focal loss to address imbalance and hard examples. We add explicit penalties for critical misclassifications and set class weights from observed frequencies. Presets (balanced, safety‑tilted, safety‑aggressive) tune risk tolerance. Threshold sweeps enforce minimum precision/recall for the Red band.
Transition: describe how we evaluate centralized training.

## Slide 6 – Centralized Evaluation Harness
We track loss, accuracy, and macro‑F1 per epoch and checkpoint models with preprocessing state. After training, we apply temperature scaling and tune thresholds, logging all adjustments for reproducibility. Optional analyses deliver attribution, fairness diagnostics, and calibration plots. Reports package metrics and artifacts for auditability.
Transition: extend the setup to a federated setting.

## Slide 7 – Federated Orchestration
We simulate multiple hospitals with configurable client counts, rounds, and aggregators. Both IID and non‑IID (Dirichlet‑skewed) splits are supported, and clients can add local personalization epochs. Aggregation can be standard, momentum‑based, or robust (trim/median). Differential privacy adds calibrated noise with normalization safeguards.
Transition: detail the client‑side training loop.

## Slide 8 – Federated Local Training Loop
Each client trains with Adam, gradient clipping (max‑norm 1), and class‑balanced sampling using the same safety‑aware loss for parity with centralized training. Clients return updates; the coordinator aggregates and may apply momentum smoothing. We track reliability, bandwidth, and timing per round for operational transparency.
Transition: calibrate and set thresholds across rounds.

## Slide 9 – Calibration & Thresholding in FL
After each aggregation, we perform temperature calibration to reduce overconfidence (minimize validation NLL). We then sweep decision thresholds to meet Red‑band constraints while limiting Green false alarms. Calibration and thresholds travel with the global model, enabling reproducible, site‑specific decision logic without sharing private data.
Transition: outline the experiment configurations.

## Slide 10 – Experiment Configurations
The centralized baseline uses the hierarchical architecture with a safety‑aggressive loss for 4 epochs plus threshold tuning. The federated baseline mirrors the architecture across 5 clients for 2 rounds; temperature is 0.8 with thresholds (0.2, 0.0, 0.02) for non‑critical/critical levels (ordering documented in configs). A red‑focus variant adds a specialized critical head, maintaining the same calibration temperature.
Transition: present centralized results.

## Slide 11 – Centralized Results
Centralized training achieves 0.881 accuracy with strong Yellow/Red recall and high Green precision. Clinical safety is favorable: under‑triage 2.29%, over‑triage 9.59%, critical sensitivity 92.4%. The model is lightweight (~58k params) and fast (~0.008 ms/sample), supporting real‑time inference.
Transition: compare against federated baselines.

## Slide 12 – Federated Baseline Results
The federated baseline reaches 0.866 accuracy, with standout Yellow recall (97.2%) and Red precision (94.7%). Reliability improves across rounds (ECE 0.044→0.016; NLL 0.314→0.276). Clinical safety metrics were omitted in this run and will be regenerated for completeness.
Transition: show red‑focus federated performance.

## Slide 13 – Federated Red‑Focus Results
Red‑focus achieves 0.870 accuracy with Red recall 84.6% and precision 90.4%, while maintaining balance on Green/Yellow. Macro‑F1 increases from 86.05 to 86.81 between rounds, and calibration improves markedly (ECE 0.121→0.019). Communication overhead remains modest (~0.96 MB/round).
Transition: synthesize the comparison.

## Slide 14 – Metric Comparison Summary
Centralized training leads by ~1–1.5 points in accuracy, but both federated variants remain within ~2 points. Red‑focus narrows the gap while enhancing critical detection; standard FL yields the strongest Yellow recall. Completing federated safety metrics will finalize the comparison table.
Transition: reflect on reliability and calibration.

## Slide 15 – Reliability & Calibration Insights
Federated models benefit from temperature 0.8 (vs 1.0 centralized), reducing overconfidence. Per‑class calibration shows the Red band gains most under red‑focus. Persisted thresholds let hospitals uphold precision/recall targets without repeated tuning. Calibration and thresholds act as safety guardrails in FL.
Transition: highlight system and engineering considerations.

## Slide 16 – System Considerations
Automation supports data prep, sweeps, and coordination; communication histories inform bandwidth planning. Gradient clipping and momentum aggregation stabilize training. Modular components enable swapping aggregators, privacy budgets, and participation policies as deployments evolve.
Transition: articulate limitations and next steps.

## Slide 17 – Limitations & Next Steps
We will regenerate federated evaluation to include clinical safety metrics. Future work stress‑tests more rounds, more clients, and non‑IID severities; expands fairness audits; and integrates on‑prem inference pipelines. Ongoing clinician‑in‑the‑loop validation will mature the collaboration.
Transition: summarize takeaways.

## Slide 18 – Takeaways
Federated learning with safety‑aware, hierarchical models can approach centralized accuracy for triage. Calibration and thresholding guardrails are essential for trustworthy distributed decisions. The toolkit is pilot‑ready with privacy, explainability, and monitoring hooks.
Transition: offer implementation context.

## Slide 19 – Backup: Implementation Overview
The repo separates training, analysis, orchestration, and ingestion, with documentation and consolidated reports for easy replication. Clear separation of concerns aids maintainability and auditing. Reproducible pipelines promote consistent results across sites.
Transition: invite discussion.

## Slide 20 – Q&A / Discussion
We can discuss deployment pathways, fairness targets, differential‑privacy trade‑offs, and partner onboarding. Reproduction follows documented configs and orchestration scripts. We welcome questions on clinical integration, governance, and collaborations.
Transition: close with contributions.

## Slide 21 – Novelty & Contributions
C³ provides deployment‑time thresholds guaranteeing minimum recall for critical cases (global and subgroup) over the ESI1–2 sum. The safety‑aware loss blends weighted CE, focal, and a clinically motivated penalty matrix with an explicit critical‑miss term. The hierarchical head gates Non‑Critical vs Critical and composes specialist heads. Token‑level fusion allows lightweight self‑attention and is compatible with DP (via Opacus). Robust FL supports non‑IID splits, robust aggregation, and optional client DP‑SGD. We report reliability and safety with CIs and provide a reproducible toolkit for end‑to‑end validation.

