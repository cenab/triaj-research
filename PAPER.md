\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{algorithmic}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage[caption=false,font=footnotesize]{subfig} % safe with IEEEtran

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

% ========= CONSISTENCY MACROS (edit here once; reflected everywhere) =========
% Centralized (Advanced model, ESI-5)
\newcommand{\CentralAcc}{0.855}
\newcommand{\CentralAccCI}{[0.852, 0.858]}
\newcommand{\CentralMacroFOne}{0.885}
\newcommand{\CentralMacroFOneCI}{[0.881, 0.888]}
\newcommand{\CentralNLL}{0.302}
\newcommand{\CentralECE}{0.027}
\newcommand{\CentralBrier}{0.033}

% Federated baseline (ESI-5, 5 clients, 10 rounds; representative run)
\newcommand{\FLAcc}{0.575}
\newcommand{\FLAccCI}{[0.572, 0.578]}
\newcommand{\FLMacroFOne}{0.510}
\newcommand{\FLMacroFOneCI}{[0.505, 0.515]}
\newcommand{\FLNLL}{1.382}
\newcommand{\FLECE}{0.249}
\newcommand{\FLBrier}{0.123}

% Model + system
\newcommand{\ModelParams}{67{,}925}
\newcommand{\ModelSizeMB}{0.259}
\newcommand{\NumClients}{5}
\newcommand{\NumRounds}{10}

% ============================== DOCUMENT =====================================
\begin{document}

% Title softened to avoid overclaiming "privacy-preserving" without SA/DP
\title{Federated ESI-5 Multi-Modal Triage with Safety-Aware Calibration}

\author{\IEEEauthorblockN{Cenab Batu Bora}
\IEEEauthorblockA{\textit{Independent Researcher}\\
Edmonton, Canada\\ cenab@ualberta.ca}}

\maketitle

\begin{abstract}
Emergency departments must identify the most urgent patients quickly and safely. We build a practical triage pipeline that can learn in one place (centralized) or across multiple sites without sharing raw data (federated learning). We use the standard 5‑level ESI scale (ESI5 low urgency to ESI1 highest urgency).

Our model reads six groups of information (vitals, symptoms, risks, context, labs, and simple interactions), combines them with a lightweight attention mechanism, and predicts the ESI level. We add a safety‑aware training objective that makes under‑triage (predicting a milder level than the truth) much more costly than over‑triage, and we add an extra penalty for missing truly critical cases (ESI1–2). After training, we calibrate predicted probabilities and set decision thresholds with either a small search or a simple conformal rule that guarantees high recall for critical cases on validation data.

On 558k encounters, the centralized model reaches \CentralAcc{} accuracy (Macro‑F1 \CentralMacroFOne{}), with reliable probabilities (NLL \CentralNLL{}, ECE \CentralECE{}, Brier \CentralBrier{}). In a representative 5‑client, 10‑round federated run, the global model achieves \FLAcc{} accuracy (Macro‑F1 \FLMacroFOne{}). Although federated accuracy is lower under these default settings, it provides a solid baseline to improve with more rounds, personalization, and robust aggregation. We assume an honest‑but‑curious server (no secure aggregation or differential privacy) and report communication cost per round to inform privacy‑enhancing deployments.
\end{abstract}

\begin{IEEEkeywords}
Triage, federated learning, multi-modal modeling, hierarchical classification, clinical safety, privacy-enhancing AI
\end{IEEEkeywords}

\section{Introduction}\label{sec:intro}
Emergency departments must quickly decide which patients need care first. Training one model across hospitals is hard because data cannot be freely shared and hospitals differ. Federated learning (FL)~\cite{mcmahan2017} helps by sharing model updates instead of raw data.

Two needs drive our design:
1) \textbf{Use many signals together.} We combine vitals, symptoms, risks, context, labs, and simple interactions.
2) \textbf{Prefer safety.} Missing a critical case (ESI1–2) is much worse than over‑triaging a non‑critical case.

Our approach follows three simple ideas:
- \textbf{Modular inputs:} encode each feature group and then let them attend to each other.
- \textbf{Safety‑aware training:} use a loss that penalizes under‑triage much more than over‑triage and adds an extra penalty for missed critical cases.
- \textbf{Calibrated decisions:} fix probability calibration on a validation set and set thresholds to meet clinical goals (including a simple conformal rule that guarantees high recall for critical cases on validation).

\textbf{Summary of results.} On the 558{,}029-encounter corpus, the centralized advanced model attains \CentralAcc{} accuracy (Macro-F1 \CentralMacroFOne{}). With \NumClients{} clients and \NumRounds{} rounds, a representative FL baseline reaches \FLAcc{} accuracy (Macro-F1 \FLMacroFOne{}). The backbone has \ModelParams{} parameters (\ModelSizeMB{}\,MB FP32), yielding low-latency inference and modest per-round communication (\ModelSizeMB{}\,MB uplink + \ModelSizeMB{}\,MB downlink per client).

\textbf{Scope and threat model.} We assume an honest-but-curious server and do not apply secure aggregation or DP in these experiments. We therefore report communication costs, calibration quality, and robustness under non-IID clients to support privacy-\emph{enhancing} (not airtight privacy-preserving) deployments; we discuss integration of secure aggregation and client-side DP-SGD as future work.

\textbf{Contributions.}
\begin{itemize}
    \item A multi-modal triage architecture that encodes feature groups as tokens and fuses them via attention, capturing cross-group interactions with a small footprint.
    \item A hierarchical ensemble generalized to ESI-5 and a safety-aware loss combining class-weighted CE, focal~\cite{lin2017}, a clinical cost matrix, and an explicit penalty on critical misses.
    \item Post-hoc calibration with validation-time operating-point selection under clinical constraints, framed as expected-cost minimization under calibrated probabilities.
    \item A federated pipeline mirroring centralized engineering/training with robust aggregation (median/trim/FedAvgM), optional DP-SGD, communication accounting, and non-IID support.
    \item A release-ready toolkit (centralized/FL trainers, conformal thresholding, fairness and SHAP analysis) supporting exact replication and extension.
\end{itemize}

\textbf{Data hygiene.} All features are standardized; missing values are imputed with \emph{training-set medians} (or medians stratified by non-target covariates such as age/sex), avoiding any use of the label at inference. We adopt a 60/20/20 stratified split (seed 42) with a held-out validation set used solely for calibration and operating-point selection to prevent leakage.

\section{Related Work}\label{sec:related}

\textbf{ED triage modeling.} Early triage ML work shows that routinely-available triage signals (vitals, demographics, complaints) can outperform conventional rules such as ESI on critical outcomes and hospitalization; large studies also reported net-benefit gains via decision-curve analysis (DCA) \cite{raita2019,vickers2006}. Subsequent systems integrate structured features with chief-complaint text or notes, improving discrimination and calibration and in some cases benchmarking against emergency physicians \cite{liu2021_sci_rep,chang2024_bmc,gligorijevic2018}. Reviews from 2024–2025 underscore the growing role of ML/NLP in ED triage but also emphasize persistent under-/over-triage and the need for decision-focused metrics beyond AUROC \cite{chang2024_bmc}. Our work follows this line but differs in three respects: (i) we encode \emph{feature groups as tokens} and fuse them with lightweight attention, capturing cross-group interactions; (ii) we optimize a \emph{clinically cost-sensitive} objective aimed at reducing critical under-triage; and (iii) we transpose the full pipeline to \emph{federated} settings while preserving calibration.

\textbf{Safety-aware learning and decision-focused evaluation.} Beyond accuracy/AUROC, clinical adoption benefits from explicit cost trade-offs and operating-point selection. DCA provides a principled way to compare net benefit across thresholds \cite{vickers2006}. Cost-sensitive learning formalizes asymmetric mis-triage penalties via class/instance-weighted objectives or cost matrices; recent surveys for medicine and safety-critical prediction summarize effective designs \cite{araf2024_csl_review,pes2021_cost_sens_med}. Complementary to cost matrices, Neyman–Pearson (NP) classification prioritizes one error type under constraints on the other \cite{tong2013_np}; recent \emph{hierarchical} NP extensions control under-classification errors across multi-class severity ladders, aligning well with triage priorities \cite{wang2024_hnp}. Our \emph{Red-first hierarchical head} is a pragmatic instantiation of this safety-first theme, paired with a clinical penalty matrix and post-hoc thresholding under recall constraints.

\textbf{Calibration and reliability under shift.} Calibrated probabilities matter for threshold selection and expected-cost minimization. Temperature scaling remains a strong post-hoc baseline \cite{guo2017}; however, calibration typically degrades under dataset shift \cite{ovadia2019}. We report NLL/ECE alongside accuracy/F1 and select per-class thresholds on a held-out validation set, which is consistent with best practice for decision-focused use.

\textbf{Federated learning (FL) in healthcare.} FL enables multi-site training without sharing raw encounters; widely-cited clinical studies (e.g., EXAM across 20 hospitals) demonstrate feasibility on vitals, labs, and imaging for outcome prediction \cite{dayan2021}. Systematic reviews through 2024 find hundreds of healthcare FL applications but note that only a small fraction reach real-world deployment and that reporting on calibration and decision thresholds is uncommon \cite{teo2024_fl_survey,rieke2020,sheller2020}. Recent FL research starts to address \emph{calibration} explicitly—e.g., Federated Calibration (FedCal) for local/global calibration and post-hoc FL calibration under heterogeneity/privacy constraints \cite{peng2024_fedcal,yu2022_mdt}. Our pipeline directly incorporates \emph{post-aggregation temperature scaling} and a \emph{validation-time threshold sweep with safety constraints}, closing this gap for triage.

\textbf{Positioning vs.\ closest prior art.} (1) \emph{Dayan et\,al.} showed large-scale healthcare FL for clinical outcomes with multi-modal inputs but did not study hierarchical safety heads or decision-focused threshold selection \cite{dayan2021}. (2) \emph{Raita et\,al.} established triage ML gains at scale and used DCA but trained centrally and did not integrate federated training, attention across feature groups, or Red-first hierarchy \cite{raita2019}. We unify these directions: multi-modal attention, safety-aware hierarchical modeling, and calibrated operating points in a single FL pipeline with end-to-end reproducibility.

\section{Dataset and Feature Engineering}
We use the Kaggle hospital triage dataset (558{,}029 encounters). Feature engineering (shared centrally and under FL) yields six groups: (i) vitals (heart rate, BP, respiration, SpO$_2$, temperature, age), (ii) symptom indicators (derived complaint flags), (iii) risk factors (e.g., comorbidity count, prior visits), (iv) context (gender, arrival timing), (v) labs (glucose, creatinine, hemoglobin, etc.), and (vi) interactions (e.g., age$\times$vitals). All features are standardized; missing values are imputed with \emph{training-set medians} (or medians stratified by non-target covariates such as age/sex), avoiding any use of the label at inference.

\textbf{Splits.} We adopt a 60/20/20 stratified split with fixed seed (42): 60\% train, 20\% validation (for calibration/operating-point selection), and 20\% test. The same policy is applied when training in the federated setting to produce global validation/test sets without data leakage across silos.

\textbf{Feature groups.} The set of features assigned to each group is fixed and saved together with the model checkpoints so that centralized and federated runs use identical input dimensions. The grouping follows the categories described above and is summarized in the supplement.

\textbf{Triage classes.} We use ESI-5 (ESI5--ESI1; ascending severity) as the default encoding; the hierarchical design also supports the legacy 3-class consolidation when required.

\subsection{Terminology and Encodings (Plain Language)}
\begin{itemize}
    \item \textbf{ESI (Emergency Severity Index):} a 5-level triage scale used in EDs. ESI1 means highest urgency; ESI5 means lowest.
    \item \textbf{Label encoding used in code:} we map ESI to integers 0--4 as $y=5-\text{ESI}$ so that 0=ESI5 (lowest) and 4=ESI1 (highest).
    \item \textbf{Critical band:} the top-2 most severe levels, i.e., \{ESI2, ESI1\}. In code this is the set of labels $\{3,4\}$.
    \item \textbf{Feature groups:} we organize inputs into six groups: vitals, symptoms, risks, context, labs, and interactions. The model receives one tensor per group.
\end{itemize}

\subsection{Methodology: Step-by-Step (What we do)}
\begin{enumerate}
    \item \textbf{Load data.} Use the public triage dataset and construct a 5-level target by mapping ESI to integers 0--4 with $y=5-\text{ESI}$ so that 0=ESI5 (lowest) and 4=ESI1 (highest).
    \item \textbf{Feature engineer.} We support two practical modes: (i) an \emph{advanced} routine that derives a richer set of features; (ii) a \emph{condensed} routine that keeps clean vitals, age, a small set of last-available labs, optional pain score, and assigns each column to one of the six groups.
    \item \textbf{Split once.} We stratify into 60\% train, 20\% validation, and 20\% test with a fixed seed. The validation split is used only for calibration and threshold selection.
    \item \textbf{Impute and standardize.} We fill missing values with the \emph{training-set medians} and standardize each feature using \emph{training} mean/scale. We apply the same statistics to validation and test. These statistics are saved with the model checkpoint.
    \item \textbf{Build datasets.} We create one input tensor per feature group in a fixed order and assemble train/validation/test datasets together with labels.
    \item \textbf{Choose a model.} We use either a \emph{single-head} classifier or a \emph{hierarchical ensemble} that first decides Non-Critical vs Critical, then refines within each band.
    \item \textbf{Define the loss.} We combine class-weighted cross-entropy, a focal term that emphasizes hard examples, and a cost matrix that increases the penalty as we under-triage by more levels. We also add an explicit penalty when a critical case (ESI2/ESI1) is assigned a non-critical label.
    \item \textbf{Train.} We train with Adam, clip gradients, and keep the checkpoint with the best validation accuracy. When enabled, federated clients may use class-balanced sampling; differentially private training is supported with group normalization.
    \item \textbf{Calibrate temperature.} On the validation set we pick a single temperature $T$ (by grid search) that improves probability calibration (typically minimizing NLL). This $T$ rescales logits at inference.
    \item \textbf{Set decision thresholds.} We support two options:
    \begin{itemize}
        \item \emph{Conformal (C\textsuperscript{3}).} Choose thresholds that guarantee a minimum recall on the critical band (overall and optionally per subgroup) on validation, then apply them to test.
        \item \emph{Offset sweep.} Adjust per-class logit offsets (with special care for the top-2 classes and, in FL, optionally ESI5) to meet recall/precision targets and improve macro performance.
    \end{itemize}
    \item \textbf{Evaluate and save.} We compute accuracy, per-class precision/recall/F1, clinical safety (under-/over-triage and critical sensitivity), reliability (NLL/ECE/Brier), bootstrap confidence intervals, fairness by subgroups, and save a JSON report and a model checkpoint.
\end{enumerate}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{docs/figures/pipeline_overview.png}
    \caption{End-to-end pipeline: feature engineering, training (centralized/federated), calibration, operating-point selection, and analysis.}
    \label{fig:pipeline}
\end{figure}

\section{Models}
\subsection{MultiPathTriageNetwork}
Each feature group is encoded by a dedicated MLP to yield tokens that self-attend across groups. Let groups be
$g\in\{\text{vital},\text{symptom},\text{risk},\text{context},\text{lab},\text{interaction}\}$.
For group $g$, the encoder maps $x^{(g)}\!\in\!\mathbb{R}^{d_g}$ to $h^{(g)}\!=\!f_g(x^{(g)})\!\in\!\mathbb{R}^{k}$.
We stack tokens $H=[h^{(v)};\dots;h^{(i)}]\in\mathbb{R}^{G\times k}$ with $G{=}6$ and apply multi-head self-attention:
$\tilde{H}=\mathrm{MHA}(H)+H$, followed by a projection and pooling (mean or learned [CLS]) to obtain
$\bar{h}\in\mathbb{R}^{k}$. A compact MLP maps $\bar{h}$ to logits $z\in\mathbb{R}^K$ (default $K{=}5$ for ESI-5).
We apply temperature calibration and optional per-class operating-point offsets at inference.

\subsection{Hierarchical Ensemble (Generalized)}
We generalize the legacy 3-class Red-first design to ESI-5. A gate head learns Non-Critical vs Critical (binary). Specialist heads classify within the non-critical band (e.g., ESI5–3) and the critical band (ESI2–1). Probabilities compose as
\begin{equation}
\Pr(c)=\Pr(\text{NonCrit})\,\Pr(c\,|\,\text{NonCrit})\ \ \text{or}\ \ \Pr(c)=\Pr(\text{Crit})\,\Pr(c\,|\,\text{Crit})\,,
\end{equation}
depending on whether $c$ lies in the non-critical or critical band. The 3-class variant is recovered by using a gate and a non-critical detail head.

\subsection{Implementation Details and Hyperparameters}
We use Adam (typical lr $5\times10^{-3}$ centrally; FL defaults use $5\times10^{-3}$ with 1 local epoch per round), batch size 128–256, gradient norm clipping at 1.0, and dropout (0.1–0.4 across blocks). Class weights are computed via inverse-frequency on the training split. Unless noted, centralized training runs 15–25 epochs with early selection by validation accuracy. The advanced backbone has approximately \ModelParams{} parameters (\ModelSizeMB{}\,MB FP32), yielding low-latency inference and $\approx$\ModelSizeMB{}\,MB per client per uplink and downlink per FL round (thus $\approx 2\times \NumClients{}\times \ModelSizeMB{}$\,MB per round).

\begin{table}[t]
\centering
\caption{Key hyperparameters and defaults.}
\label{tab:hparams}
\small
\begin{tabular}{l l}
\toprule
Component & Setting \\
\midrule
Optimizer & Adam ($\alpha{=}5\times10^{-3}$, default betas) \\
Batch size & 128--256 (all modes) \\
Epochs (centralized) & 15--25 (select best by val acc) \\
Dropout & 0.1--0.4 by block \\
Grad clipping & $\Vert\nabla\Vert_2 \le 1.0$ \\
Loss weights & CE + Focal + Cost + Critical (configurable) \\
Focal $(\alpha,\gamma)$ & $(0.25,\,2.0)$ \\
Temperature search & Val-optimized $T$ (scalar; grid or L-BFGS) \\
Operating-point sweep & Per-class offsets with safety constraints \\
\bottomrule
\end{tabular}
\end{table}

\section{Clinical-Safety Loss and Calibration}
\textbf{Safety-aware loss.} We train with three simple ideas: (i) a standard loss that accounts for class imbalance, (ii) a focal term that pays more attention to hard examples~\cite{lin2017}, and (iii) a cost that makes under‑triage much more expensive than over‑triage (and more expensive the further the miss). We also add a small extra penalty whenever a critical case (ESI1–2) is predicted as non‑critical. For the hierarchical model, we compute the loss on the final class probabilities.

Putting these pieces together, the training objective is
\begin{equation}
\mathcal{L}=\mathcal{L}_{\text{CE}}+\lambda_\text{f}\,\mathcal{L}_{\text{focal}}(\alpha{=}0.25,\,\gamma{=}2.0)+\lambda_\text{s}\,\mathbb{E}[M_{y,\hat{y}}]+\lambda_\text{c}\,\mathbf{1}[y\in\{\text{ESI2,ESI1}\},\ \hat{y}\notin\{\text{ESI2,ESI1}\}]\,.
\end{equation}
For ESI‑5, the cost $M$ follows a simple rule: under‑triage gets a larger penalty the more severe the miss; over‑triage is penalized lightly and grows slowly. This encourages the model to avoid dangerous misses while tolerating some safe over‑triage.

\textbf{Calibration and decisions.} After training, we adjust a single temperature on the validation set to make probabilities reliable~\cite{guo2017}. We then set simple per‑class offsets (with extra attention to the top‑2 critical classes) or use a conformal rule that guarantees high recall for critical cases on validation. One option is to pick the class with the lowest expected clinical cost under the calibrated probabilities:
\[
c^*=\arg\min_{c'} \sum_{y} M_{y,c'}\,p(y\,|\,x),
\]
and we tune practical operating points via per-class logit offsets; optionally, we use conformal thresholds (C³) to provide validation-time guarantees on critical sensitivity with group-specific adjustments.

\section{Training Protocols}
\subsection{Federated Learning: Step-by-Step}
\begin{enumerate}
    \item \textbf{Split the training set across sites.} Either evenly (IID) or with different mixes (non‑IID) across \NumClients{} clients.
    \item \textbf{Train locally.} Each client trains the same model for one short local pass (one epoch) with Adam and gradient clipping. If classes are imbalanced, a balanced sampler can help.
    \item \textbf{Combine models on the server.} We average client parameters (FedAvg) or use robust alternatives (median/trim) or momentum‑based averaging (FedAvgM). We also keep track of the rough communication per round.
    \item \textbf{Calibrate the global model.} After the last round, we calibrate a single temperature on the validation split.
    \item \textbf{Set thresholds.} We set simple per‑class offsets (with extra care for the top‑2 critical classes) or use conformal thresholds from the validation split.
    \item \textbf{Evaluate once.} We evaluate the global model on the held‑out test set and report accuracy, safety, and reliability metrics.
\end{enumerate}
\subsection{Centralized}
We split the dataset into train/val/test (60/20/20), train with Adam (batch 128--256, lr $5\times10^{-3}$ typical), apply temperature calibration and optional thresholding on validation, and evaluate on test. The advanced model attains \CentralAcc{} accuracy (Macro-F1 \CentralMacroFOne{}).

\paragraph*{Client partitioning and aggregation.} IID sharding and non-IID Dirichlet splits are supported. Robust aggregation (median/trim) and FedAvgM are provided. Optional per-client personalization and differentially private training are available; for privacy-aware training, group normalization is used in place of batch normalization.

\section{Evaluation Protocol}
We report overall accuracy; per‑class precision, recall, and F1; and clinical safety: how often the model under‑triages or over‑triages, and how sensitive it is to critical cases (ESI1–2). We also show confusion matrices. When available, we report fairness (e.g., by gender and age groups) and simple SHAP analyses to explain which features matter most.

\subsection{Reliability and Calibration Quality}
To assess reliability beyond accuracy, we report negative log-likelihood (NLL) and expected calibration error (ECE) per class and overall. Temperature scaling~\cite{guo2017} is chosen on the validation set to minimize NLL. Operating-point selection aligns with clinical targets without retraining.

For the centralized model: NLL $=\CentralNLL$, ECE $=\CentralECE$, Brier $=\CentralBrier$, Acc $\CentralAcc{}\,\CentralAccCI$, Macro-F1 $\CentralMacroFOne{}\,\CentralMacroFOneCI$. For a representative FL baseline (Table~\ref{tab:metrics}): NLL $=\FLNLL$, ECE $=\FLECE$, Brier $=\FLBrier$, Acc $\FLAcc{}\,\FLAccCI$, Macro-F1 $\FLMacroFOne{}\,\FLMacroFOneCI$.

\begin{table}[t]
\centering
\caption{Reliability metrics across systems (ESI-5; rounded to 3 decimals).}
\label{tab:reliability}
\small
\begin{tabular}{lcccc}
\toprule
System & NLL & ECE & Brier & Acc / Macro-F1 (95\% CI) \\
\midrule
Centralized & \CentralNLL & \CentralECE & \CentralBrier & \CentralAcc{} \CentralAccCI{} / \CentralMacroFOne{} \CentralMacroFOneCI{} \\
FL Baseline & \FLNLL & \FLECE & \FLBrier & \FLAcc{} \FLAccCI{} / \FLMacroFOne{} \FLMacroFOneCI{} \\
\bottomrule
\end{tabular}
\end{table}

\section{Results}\label{sec:results}
\subsection{Centralized vs Federated (Baseline)}
The centralized model reaches \CentralAcc{} accuracy. Under a \NumClients{}‑client, \NumRounds{}‑round federated baseline, the global model achieves \FLAcc{} accuracy. Federated validation accuracy improves across rounds and then stabilizes.

\begin{table}[t]
\centering
\caption{Core metrics summary (ESI-5; rounded to 3 decimals).}
\label{tab:metrics}
\small
\begin{tabular}{lccccc}
\toprule
System & Acc & Macro-F1 & NLL & ECE & Brier \\
\midrule
Centralized & \CentralAcc{} & \CentralMacroFOne{} & \CentralNLL{} & \CentralECE{} & \CentralBrier{} \\
FL Baseline & \FLAcc{} & \FLMacroFOne{} & \FLNLL{} & \FLECE{} & \FLBrier{} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Red-Focused Variant}
With hierarchical gating and higher safety/critical weights, the red-focused variant prioritizes the critical band (ESI1--2). This improves critical alignment at the cost of overall accuracy under some settings. In FL, constrained threshold searches may not meet extreme targets without additional rounds/personalization. Per-round dynamics are shown in Figure~\ref{fig:redfocus_curves}.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{docs/figures/fl_advanced_red_focus_acc_f1_vs_rounds.png}
    \includegraphics[width=0.95\linewidth]{docs/figures/fl_advanced_red_focus_ece_nll_vs_rounds.png}
    \caption{Red-focused federated model: per-round Acc/Macro-F1 (top) and ECE/NLL (bottom).}
    \label{fig:redfocus_curves}
\end{figure}

\subsection{Fairness and Interpretability}
We report subgroup metrics (gender/age) and optional SHAP explanations. Vitals and labs dominate critical-band attribution; interaction terms capture subtle risk shifts. Fairness outputs include per-group accuracy/F1 and rate differences with violation flags when gaps exceed a threshold (default 0.1). SHAP results present global and class-conditional importances.

\subsection{Baselines (Tabular)}
We compare against calibrated tabular baselines for context (gradient boosting and logistic regression). These baselines confirm the value of calibration and provide a familiar point of reference, but our primary focus is on the centralized and federated neural pipelines.

\section{Ablations and Sensitivity}
We ablate (i) focal/penalty weights, (ii) per-class miss scalers, (iii) hierarchical head, and (iv) calibration/operating points. Results show: (1) focal+penalty consistently reduce under-triage; (2) the hierarchical head improves critical-band margins; (3) temperature and operating points control precision–recall trade-offs without retraining.

\section{Discussion}
\textbf{Clinical value.} The pipeline is designed to avoid dangerous misses (under‑triage) while keeping strong overall accuracy.
\textbf{Federated practicality.} Sites keep data local and still contribute to a useful global model with modest communication per round.
\textbf{Limitations.} Purely tabular signals may not fully separate the most severe cases; adding chief‑complaint text and time‑based features is a natural next step. Very strict recall targets for the critical band may also require stronger features or hybrid models.

\paragraph*{Systems and Communication.} We report per-round wall-time, client time variance, and estimated communication (parameter bytes uploaded/downloaded per round). For our \ModelParams{}-parameter model (\ModelSizeMB{}\,MB), per-round communication is small; under personalization, totals rise proportionally.

\paragraph*{Significance Testing.} We assess differences using paired bootstrap (Acc, Macro-F1) and McNemar tests on matched predictions; significant improvements are reported where two-sided $p<0.05$.

\paragraph*{Interpreting the figures.} Across federated rounds, Accuracy and Macro‑F1 improve and then stabilize; small wiggles are expected from differences between clients. Calibration curves (NLL/ECE) also improve with training. When we tilt thresholds toward the critical band, the model becomes more sensitive to severe cases at the cost of some overall Accuracy; careful temperature tuning helps. Systems summaries show modest communication and fairly uniform client times. Reliability diagrams for the centralized model show well‑calibrated probabilities, which supports simple threshold‑based decisions.

\section{Threat Model and Privacy Considerations}
We adopt an honest-but-curious server: clients share model updates (not raw data) with a central aggregator. These experiments do not apply secure aggregation or DP; gradient leakage or membership inference remain theoretical risks. In deployment, we recommend secure aggregation and client-side DP-SGD, plus audit logging and model-card documentation. No PII leaves client silos in our design.

\section{Reproducibility}
To facilitate faithful replication, we outline the essential steps and settings used in our experiments. This checklist abstracts away any particular codebase.

\textbf{Environment.}
- Python environment with standard scientific computing and machine learning libraries.
- Optional: GPU for faster training; CPU is sufficient for small runs.

\textbf{Data processing.}
- Use the public triage dataset and construct a 5-level ESI target by mapping ESI to integers 0--4 with $y=5-\text{ESI}$.
- Feature engineer into six groups (vitals, symptoms, risks, context, labs, interactions).
- Split once into 60\% train, 20\% validation, 20\% test with a fixed seed and stratification.
- Impute missing values with training-set medians; standardize features using training mean and scale; apply the same transformation to validation and test.

\textbf{Model.}
- Multi-pathway encoders per feature group feeding a compact attention fusion and classifier head, or a hierarchical ensemble with a gate (Non-Critical vs Critical) and band-specific specialists.
- Initialize weights with a standard scheme (e.g., Xavier/Glorot) and apply dropout 0.1--0.4.

\textbf{Optimization.}
- Optimizer: Adam; learning rate $5\times10^{-3}$ (centralized); batch size 128--256; gradient-norm clipping at 1.0.
- Train for 15--25 epochs (centralized); keep the checkpoint with the best validation accuracy.

\textbf{Safety-aware objective.}
- Class-weighted cross-entropy + focal component (e.g., $\alpha{=}0.25,\gamma{=}2.0$).
- Rule-based cost matrix: quadratic penalties for under-triage (by distance), linear penalties for over-triage.
- Explicit penalty when a critical case (ESI1--2) is assigned a non-critical label.

\textbf{Calibration and decision thresholds.}
- Post-hoc temperature scaling on the validation set (minimize NLL).
- Choose thresholds either by (i) conformal rules to guarantee minimum recall for the critical band (optionally per subgroup), or (ii) a small sweep of per-class logit offsets (with emphasis on the top-2 classes).

\textbf{Federated learning.}
- Partition the training set across sites (IID or Dirichlet non-IID with user-chosen concentration).
- Local training: 1 epoch per round, Adam with the same settings as centralized.
- Aggregation: FedAvg (default), or robust alternatives (median/trim) or momentum-based FedAvgM.
- Record estimated communication per round as 2\,$\times$\,(parameter bytes)\,$\times$\,(number of clients).
- Calibrate temperature and set thresholds on the global model using the validation split; evaluate on the held-out test split once at the end.

\textbf{Reporting.}
- Report Accuracy and Macro-F1; per-class precision/recall/F1; clinical safety metrics (under-/over-triage and critical sensitivity); reliability (NLL/ECE/Brier); fairness by subgroups; and bootstrap 95\% confidence intervals.
- Plot learning dynamics (Accuracy/Macro-F1 and ECE/NLL over rounds), systems/communication summaries, and reliability diagrams.

\section{Conclusion}
We present a practical ESI-5 triage pipeline for centralized and federated learning with multi-modal architectures, safety-aware losses, and calibrated decision rules. The hierarchical variant aligns the model with clinical priorities for the critical band. Under minimal rounds and baseline settings, FL underperforms centralized accuracy but provides a solid starting point; future work will expand rounds/personalization, add text embeddings, and integrate secure aggregation and DP in real multi-site deployments.

\begin{thebibliography}{00}
\bibitem{mcmahan2017} H. B. McMahan, E. Moore, D. Ramage, et al., ``Communication-efficient learning of deep networks from decentralized data,'' in \emph{Proc. AISTATS}, 2017.
\bibitem{rieke2020} N. Rieke, et al., ``The future of digital health with federated learning,'' \emph{NPJ Digital Medicine}, 2020.
\bibitem{sheller2020} M. J. Sheller, et al., ``Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data,'' \emph{Scientific Reports}, 2020.
\bibitem{guo2017} C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, ``On calibration of modern neural networks,'' in \emph{Proc. ICML}, 2017.
\bibitem{lin2017} T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Doll\'ar, ``Focal loss for dense object detection,'' in \emph{Proc. ICCV}, 2017.
\bibitem{lundberg2017} S. M. Lundberg, S.-I. Lee, ``A unified approach to interpreting model predictions,'' in \emph{Proc. NeurIPS}, 2017.
\bibitem{hardt2016} M. Hardt, E. Price, N. Srebro, ``Equality of opportunity in supervised learning,'' in \emph{Proc. NeurIPS}, 2016.
\bibitem{dayan2021}
I. Dayan, H. Roth, A. Zhong, et al., ``Federated learning for predicting clinical outcomes in patients with COVID-19,'' \emph{Nature Medicine}, vol. 27, pp. 1735--1743, 2021.
\bibitem{raita2019}
Y. Raita, T. Goto, M. K. Faridi, D. F. M. Brown, C. A. Camargo Jr., and K. Hasegawa, ``Emergency department triage prediction of clinical outcomes using machine learning models,'' \emph{Critical Care}, vol. 23, p. 64, 2019.
\bibitem{liu2021_sci_rep}
Y. Liu, J. Gao, J. Liu, et al., ``Development and validation of a practical machine-learning triage algorithm for the detection of patients in need of critical care in the emergency department,'' \emph{Scientific Reports}, vol. 11, p. 24044, 2021.
\bibitem{chang2024_bmc}
Y.-H. Chang, Y.-C. Lin, F.-W. Huang, et al., ``Using machine learning and natural language processing in triage for prediction of clinical disposition in the emergency department,'' \emph{BMC Emergency Medicine}, vol. 24, p. 237, 2024.
\bibitem{gligorijevic2018}
D. Gligorijevic, J. Stojanovic, W. Satz, et al., ``Deep Attention Model for Triage of Emergency Department Patients,'' arXiv:1804.03240, 2018.
\bibitem{vickers2006}
A. J. Vickers and E. B. Elkin, ``Decision curve analysis: a novel method for evaluating prediction models,'' \emph{Medical Decision Making}, vol. 26, no. 6, pp. 565--574, 2006.
\bibitem{araf2024_csl_review}
I. Araf, S. I. Ayon, and A. Rahman, ``Cost-sensitive learning for imbalanced medical data: a review,'' \emph{Artificial Intelligence Review}, 2024.
\bibitem{pes2021_cost_sens_med}
B. Pes, A. P. Ruiu, G. Cossu, and G. Angioni, ``Cost-sensitive learning strategies for high-dimensional and imbalanced medical data,'' \emph{BMC Medical Informatics and Decision Making}, vol. 21, 2021.
\bibitem{tong2013_np}
X. Tong, ``A Plug-in Approach to Neyman–Pearson Classification,'' \emph{Journal of Machine Learning Research}, vol. 14, pp. 3011--3040, 2013.
\bibitem{wang2024_hnp}
L. Wang, Y. Chen, A. Jiang, et al., ``Hierarchical Neyman–Pearson Classification for Prioritizing Low-False-Negative Decisions,'' \emph{Journal of the American Statistical Association}, 2024.
\bibitem{guo2017}
C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, ``On Calibration of Modern Neural Networks,'' in \emph{Proc. ICML}, 2017.
\bibitem{ovadia2019}
Y. Ovadia, E. Fertig, J. Ren, et al., ``Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift,'' in \emph{Proc. NeurIPS}, 2019.
\bibitem{peng2024_fedcal}
H. Peng, W. Wang, F. Chen, et al., ``FedCal: Achieving Local and Global Calibration in Federated Learning via Aggregated Parameterized Scaler,'' in \emph{Proc. ICML}, PMLR 235, 2024.
\bibitem{yu2022_mdt}
Y. Yu, J. Ma, H. Zhao, and N. Vasconcelos, ``Robust Calibration with Multi-domain Temperature Scaling,'' in \emph{Proc. NeurIPS}, 2022.
\bibitem{teo2024_fl_survey}
Z. L. Teo, L. Jin, S. Li, D. Miao, D. S. W. Ting, ``Federated machine learning in healthcare: A systematic review on clinical applications and technical architecture,'' \emph{Cell Reports Medicine}, vol. 5, no. 2, p. 101419, 2024.
\end{thebibliography}

\end{document}
