## FairTriEdge-FL: Detailed Implementation Plan

#### Phase 1: Data Preparation and Simulation Environment Setup

*   **Objective**: Establish a robust data foundation and a simulated multi-site environment for federated learning.
*   **Detailed Steps**:
    1.  **Data Loading and Initial Cleaning**:
        *   Load the `triaj_data.csv` file into a suitable data structure (e.g., Pandas DataFrame).
        *   Inspect for missing values across all columns. For numerical vital signs (`sistolik kb`, `diastolik kb`, `solunum sayısı`, `nabız`, `ateş`, `saturasyon`), consider imputation strategies (e.g., mean, median, or K-nearest neighbors imputation) or removal of rows if missingness is extensive and random. For textual/categorical fields, treat missing values as "no information" or "not present."
        *   Handle potential data inconsistencies or outliers in numerical fields (e.g., `nabız` = 0 in row 234).
    2.  **Feature Engineering (Structured Data)**:
        *   **Numerical Features**:
            *   Extract `yaş`, `sistolik kb`, `diastolik kb`, `solunum sayısı`, `nabız`, `ateş`, `saturasyon`.
            *   Apply numerical scaling (e.g., StandardScaler or MinMaxScaler) to these features to normalize their ranges.
        *   **Categorical Features**:
            *   `cinsiyet`: One-hot encode the `cinsiyet` (gender) column (e.g., 'Erkek', 'Kadın').
            *   **Textual/Multi-value Features to Boolean**: For columns like `ek hastalıklar` (additional diseases), `semptomlar_non travma_genel 01`, `semptomlar_non travma_genel 02`, `göz`, `göğüs ağrısı`, `karın ağrısı`, `travma_ayak`, `travma_bacak`, `travma_batin`, `travma_boyun`, `travma_el`, `travma_gogus`, `travma_goz`, `travma_kafa`, `travma_kol`, `travma_parmak`, `diğer travmalar`, `dahiliye hastalıklar`, `psikiyatri`, `kardiyoloji`, `göğüs hastalıkları`, `nörolojik hastalıklar`, `beyin cerrahi`, `kalp damar cerrahisi`, `kbb`, `göz hastalıkları`, `İntaniye`, `Üroloji`, `Çevresel ve toksikoloji acilleri`, `kadın ve doğum hastalıkları`, `genel cerrahi hastalıklar`, `deri hastalıkları`, `diğer diyagnoz_travma`, `diğer diyagnoz`:
                *   Parse each cell, splitting by commas or other delimiters if multiple values are present.
                *   Create a comprehensive vocabulary of all unique symptoms, diseases, and trauma types across the entire dataset.
                *   Generate new binary (Boolean) columns for each unique item in the vocabulary, indicating its presence (1) or absence (0) for each patient.
        *   **Temporal Features (from `created` column)**:
            *   Convert the `created` column to datetime objects.
            *   Extract features like hour of day, day of week, month, or season if relevant for modeling patient flow or seasonal illness patterns.
            *   If multi-visit data for the same patient becomes available, calculate temporal trends (e.g., change in vital signs since last visit).
        *   **Target Variable Mapping**:
            *   Map the `doğru triyaj` (correct triage) column to numerical labels: "Kırmızı Alan" -> 2, "Sarı Alan" -> 1, "Yeşil Alan" -> 0. This will be the primary target for classification.
            *   The `triyaj alanı` column can be used for initial exploratory analysis or as a secondary target if `doğru triyaj` is not always available.
    3.  **Simulated Multi-Site Data Generation**:
        *   **Partitioning Strategy**: Implement a function to partition the preprocessed `triaj_data.csv` into `N` virtual client datasets.
            *   **Option A (Random Split)**: Randomly assign a percentage of patients to each virtual client.
            *   **Option B (Demographic Split)**: Partition based on `yaş` (e.g., one client for pediatric patients, another for adults) or `cinsiyet` to simulate different hospital populations.
            *   **Option C (Temporal Split)**: Partition data by `created` date ranges to simulate data from different time periods or sites joining over time.
        *   **Controlled Distribution Shifts**: For stress-testing, implement methods to artificially introduce distribution shifts in synthetic client data (e.g., increase the prevalence of specific `semptomlar` or `ek hastalıklar` in a subset of clients).
    4.  **Domain Adaptation Strategy**:
        *   If significant distribution shifts are introduced or observed, implement domain adaptation techniques.
        *   **Transfer Learning**: Pre-train a base model on a large portion of the data, then fine-tune it on smaller, site-specific synthetic datasets for each virtual client.
        *   **Adversarial Domain Adaptation**: Explore techniques that learn domain-invariant features by using a domain discriminator during training.
    5.  **Continuous Drift Monitoring**:
        *   Develop a module to monitor incoming data streams (simulated new patient data for each client).
        *   **Statistical Tests**: Implement statistical tests for drift detection:
            *   For numerical features (e.g., `yaş`, `sistolik kb`): Kolmogorov-Smirnov test, Jensen-Shannon divergence.
            *   For categorical/Boolean features: Chi-squared test, population stability index (PSI).
        *   **Alerting Mechanism**: Define thresholds for these tests to trigger alerts for potential model recalibration or retraining.

#### Phase 2: Core Model Development and Optimization

*   **Objective**: Design and develop the primary multimodal triage prediction model, optimized for edge deployment.
*   **Detailed Steps**:
    1.  **Multimodal Fusion Architecture**:
        *   **Input Layers**: Define distinct input layers for different data modalities:
            *   Numerical Vitals: A dense input layer for `sistolik kb`, `diastolik kb`, `solunum sayısı`, `nabız`, `ateş`, `saturasyon`.
            *   Demographics: A dense input layer for `yaş` and one-hot encoded `cinsiyet`.
            *   Boolean Symptoms/Diseases/Trauma: A wide input layer for the one-hot encoded binary features derived in Phase 1.
            *   (Optional) Temporal Sequences: If multi-time-point data is available, a recurrent layer (e.g., LSTM or GRU) or 1D CNN for sequences of vital signs.
        *   **Sub-networks**: Design small, efficient neural networks (e.g., 2-3 dense layers with ReLU activations for static features, 1D CNNs for temporal features) for each input modality.
        *   **Fusion Layer**: Concatenate the outputs of the sub-networks.
        *   **Attention Mechanism**: Consider adding a simple attention mechanism to weigh the importance of different modalities before the final classification.
        *   **Output Layer**: A final dense layer with a softmax activation for multi-class classification (Kırmızı, Sarı, Yeşil).
    2.  **Resource Optimization (TinyML)**:
        *   **Model Selection**: Prioritize lightweight deep learning architectures (e.g., MobileNetV2, EfficientNet-Lite) or traditional ML models (e.g., LightGBM, XGBoost) known for efficiency.
        *   **Quantization**: Implement post-training quantization (e.g., to INT8 or even INT4 if feasible) using frameworks like TensorFlow Lite or PyTorch Mobile. This reduces model size and speeds up inference.
        *   **Pruning**: Apply structured or unstructured pruning techniques to remove redundant connections or neurons, further reducing model size and computational load.
        *   **Knowledge Distillation**: Train a smaller, simpler "student" model to mimic the behavior of a larger, more complex "teacher" model, achieving similar performance with fewer resources.
        *   **Benchmarking**: Systematically benchmark inference latency and CPU/memory usage on simulated edge hardware (e.g., Raspberry Pi, NVIDIA Jetson Nano) to ensure sub-100ms inference.
    3.  **Personalization Mechanisms**:
        *   **On-Device Fine-tuning**: Implement a mechanism for each virtual client (simulated hospital) to periodically fine-tune its local model using newly accumulated patient data. This should be done incrementally with regularization to prevent catastrophic forgetting.
        *   **Patient-Specific Features**: If historical patient data is available, engineer features that capture individual patient baselines or trends (e.g., average BP, common comorbidities) and incorporate them into the model input.

#### Phase 3: Federated Learning and Robustness Integration

*   **Objective**: Implement and evaluate the federated learning paradigm, incorporating privacy and robustness features.
*   **Detailed Steps**:
    1.  **FL Framework Selection**: Choose a robust federated learning framework (e.g., Flower, PySyft, TensorFlow Federated) that supports simulated multi-node training.
    2.  **Simulated Multi-Node Training Implementation**:
        *   Develop client-side training scripts for each virtual client, where models are trained locally on their partitioned data.
        *   Develop a central server-side aggregation script that collects model updates (gradients or weights) from clients.
    3.  **Privacy Preservation**:
        *   **Federated Averaging (FedAvg)**: Implement FedAvg as the core aggregation algorithm, ensuring only model parameters (not raw data) are shared.
        *   **Differential Privacy (DP)**: Explore adding differential privacy noise to client updates (gradients) before sending them to the server. Use libraries like Opacus (for PyTorch) or TensorFlow Privacy. Quantify the privacy-utility trade-off.
    4.  **Poisoning Defense and Robust Aggregation**:
        *   **Robust Aggregation Algorithms**: Implement and evaluate robust aggregation techniques on the server-side, such as:
            *   Krum algorithm
            *   Trimmed Mean
            *   Coordinate-wise Median
        *   **Simulated Attacks**: Design scenarios to simulate malicious clients sending poisoned updates (e.g., label flipping attacks, gradient ascent attacks) to test the effectiveness of robust aggregation.
    5.  **Communication Efficiency**:
        *   **Model Update Compression**: Implement techniques to reduce the size of model updates transmitted between clients and server (e.g., sparsification of gradients, quantization of updates).
        *   **Asynchronous Updates**: Explore asynchronous federated learning where clients send updates on their own schedule, and the server aggregates periodically.
    6.  **Fairness in Federated Models**:
        *   **Fairness Metrics**: Define and track fairness metrics across demographic subgroups (e.g., `yaş` groups, `cinsiyet`) for each virtual client and the global model. Examples include:
            *   F1-score parity (equal F1-scores across groups).
            *   Equalized odds (equal true positive rates and false positive rates across groups).
        *   **Fairness-Aware FL Algorithms**: Experiment with federated learning algorithms designed to promote fairness, such as:
            *   FedProx (Federated Proximal Optimization).
            *   q-FedAvg (q-Fair Federated Averaging).
            *   Custom loss functions that incorporate fairness constraints or penalties for disparities.

#### Phase 4: Explainable AI (XAI) and LLM Integration

*   **Objective**: Develop and integrate explainability features, including LLM-enhanced explanations and synthetic case generation.
*   **Detailed Steps**:
    1.  **Built-in XAI**:
        *   Integrate model-agnostic XAI techniques like SHAP (Shapley Additive Explanations) or LIME (Local Interpretable Model-agnostic Explanations) to generate feature importance scores for each individual triage prediction.
        *   For tree-based models, leverage built-in feature importance.
    2.  **LLM-Enhanced Explanation Module**:
        *   **LLM Selection**: Choose a suitable LLM. For edge deployment, consider smaller, fine-tunable medical LLMs or distilled versions of larger models. For server-side explanation generation, larger, more capable LLMs can be used.
        *   **Prompt Engineering**: Design prompts that take the model's prediction, key contributing features (from SHAP/LIME), and patient context (e.g., `yaş`, `ek hastalıklar`, `semptomlar`) as input. The LLM should generate a concise, natural language rationale for the triage decision.
        *   **Medical Soundness Evaluation**: Conduct qualitative evaluations with clinical domain experts to assess the medical accuracy, clarity, and utility of the LLM-generated explanations.
    3.  **Synthetic Rare Case Generation**:
        *   Utilize LLMs (e.g., by fine-tuning a generative model or using advanced prompting with models like GPT-4o) to generate realistic synthetic Boolean patient records.
        *   Focus on generating rare or complex cases (e.g., specific `travma` combinations, unusual `semptomlar` for a given `yaş` group, or critical conditions with subtle vital sign changes) to augment training data and stress-test the model.
        *   Implement a validation process to ensure the clinical plausibility and statistical properties of the generated synthetic data.
    4.  **Real-Time Interpretability of Boolean Rule Chains**:
        *   **Rule Extraction**: Research and implement methods to extract simplified, human-understandable Boolean rule chains that approximate the core model's decision boundary. This could involve:
            *   Training a separate interpretable model (e.g., a decision tree or rule-based classifier) on the predictions of the complex model.
            *   Using techniques like Boolean Rule Column Generation or other rule-mining algorithms.
        *   **UI Presentation**: Develop a user interface component that displays these extracted rules alongside the AI's triage recommendation, allowing clinicians to quickly understand the underlying logic.

#### Phase 5: Comprehensive Evaluation and Open Science

*   **Objective**: Rigorously evaluate the system's clinical efficacy, usability, ethical implications, and ensure transparency and reproducibility.
*   **Detailed Steps**:
    1.  **Clinical Validation (Simulation-based)**:
        *   **Endpoints**: Define clear clinical endpoints for evaluation, such as:
            *   Accuracy in predicting `doğru triyaj` levels (Kırmızı, Sarı, Yeşil).
            *   Sensitivity and specificity for identifying high-acuity (Kırmızı) cases.
            *   Rates of under-triage (assigning lower acuity than `doğru triyaj`) and over-triage (assigning higher acuity).
            *   Simulated impact on ED flow (e.g., hypothetical reduction in time-to-treatment for critical patients).
        *   **Retrospective Replay Evaluation**: Use the `triaj_data.csv` (and augmented synthetic data) to simulate how the model would have triaged historical cases and compare its performance against the `doğru triyaj` labels.
        *   **Prospective Simulation**: Design a simulation environment where clinicians interact with the system using synthetic patient scenarios to evaluate its real-time performance and impact.
    2.  **Usability and Workflow Integration**:
        *   **Mixed-Methods Assessment**: Conduct usability testing sessions with simulated triage scenarios involving target end-users (triage nurses, physicians).
        *   **Data Collection**: Collect subjective feedback through structured surveys (e.g., System Usability Scale - SUS) and semi-structured interviews focusing on:
            *   Ease of use and learnability.
            *   Trust in AI recommendations and explanations.
            *   Impact on cognitive load and decision-making.
            *   Perceived workflow integration and efficiency.
        *   **Iterative UI Improvement**: Continuously refine the user interface based on feedback, focusing on clear presentation of recommendations, explanations, and the ability to drill down into details.
    3.  **Ethical and Bias Analysis**:
        *   **Subgroup Analysis**: Systematically analyze model performance (accuracy, F1-score, under/over-triage rates) across different patient subgroups defined by `yaş` (e.g., pediatric vs. adult vs. elderly) and `cinsiyet`. If additional demographic data is available or can be safely synthesized, include those.
        *   **Bias Detection Metrics**: Employ fairness metrics such as demographic parity, equal opportunity, or predictive equality to identify potential biases.
        *   **Ethics Panel Review**: Convene an ethics panel (simulated or actual, if resources allow) to review the project design, data handling, bias mitigation strategies, and the ethical implications of AI deployment in triage.
        *   **Audit Logging**: Implement comprehensive logging of all AI decisions, inputs, and explanations for auditability and post-hoc analysis.
        *   **Safety Protocols**: Define fail-safe mechanisms and human-in-the-loop protocols for cases where AI recommendations are uncertain or conflict with clinical judgment.
    4.  **Performance Benchmarks and Statistical Rigor**:
        *   **Baselines**: Compare the FairTriEdge-FL system's performance against:
            *   Traditional, rule-based triage scores (if applicable).
            *   Simpler machine learning models (e.g., Logistic Regression, Random Forest) trained centrally on the same data.
        *   **Metrics**: Report standard classification metrics (Accuracy, Precision, Recall, F1-score for each triage class, AUC-ROC for binary classification of high-acuity).
        *   **Validation Techniques**: Use robust validation techniques such as stratified k-fold cross-validation and a separate hold-out test set to ensure reliable performance estimates.
        *   **Statistical Significance**: Where appropriate, use statistical tests (e.g., McNemar's test for paired comparisons, t-tests for mean differences) and confidence intervals to demonstrate the significance of improvements.
    5.  **Open Science Practices**:
        *   **Open-Source Code**: Release the core model code, data preprocessing scripts, and federated learning framework implementation on a public platform (e.g., GitHub) under an open-source license.
        *   **Synthetic Data Release**: Publish the synthetic version of the `triaj_data.csv` dataset (or the generation scripts) to enable reproducibility and further research by others.
        *   **Reporting Guidelines Adherence**: Meticulously follow reporting guidelines such as TRIPOD-ML (for model development) and CONSORT-AI (for clinical trial design, even if simulated) to ensure transparency and completeness in documentation.
        *   **Model Cards**: Publish a detailed Model Card for the triage model, outlining its intended use, architecture, training data scope, performance across subgroups, ethical considerations, and limitations.
        *   **Comprehensive Documentation**: Provide clear documentation on how to set up the simulation environment, train models, deploy on edge devices, and reproduce results.

### Architectural Overview

The architectural diagram provided in `fair_triedge_fl_research_plan.md` (Mermaid diagram) accurately represents the distributed nature of the system, with edge devices at hospitals and a central aggregator. The detailed steps above further specify the internal workings of each component.

```mermaid
graph TD
    subgraph Hospital A (Edge Device)
        A[Triage Data Input (Structured)] --> B(On-Device Preprocessing)
        B --> C(Edge Triage Model - Inference)
        C --> D(XAI Module)
        D --> E(LLM Explanation Generator)
        E --> F(Triage Recommendation + Explanation UI)
        C --> G(On-Device Model Adaptation / Continual Learning)
        G --> H(Data Drift Detector)
        H --> G
        G --> I(Local Model Updates)
    end

    subgraph Hospital B (Edge Device)
        J[Triage Data Input (Structured)] --> K(On-Device Preprocessing)
        K --> L(Edge Triage Model - Inference)
        L --> M(XAI Module)
        M --> N(LLM Explanation Generator)
        N --> O(Triage Recommendation + Explanation UI)
        L --> P(On-Device Model Adaptation / Continual Learning)
        P --> Q(Data Drift Detector)
        Q --> P
        P --> R(Local Model Updates)
    end

    subgraph Central Aggregator (Cloud/Server)
        S(Federated Learning Server)
        I -- Model Updates --> S
        R -- Model Updates --> S
        S -- Global Model --> C
        S -- Global Model --> L
        S --> T(Robust Aggregation)
        T --> U(Fairness Monitoring)
        U --> S
        S --> V(Synthetic Data Generator - LLM/Generative Models)
        V --> W(Rare Case Augmentation)
        W --> S
    end

    subgraph External Validation / Audit
        X(Clinician Validation)
        F --> X
        O --> X
        Y(Ethics Panel / Bias Audit)
        U --> Y
        Z(Open Science Repository)
        S --> Z
        C --> Z
        L --> Z
        V --> Z
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style O fill:#bbf,stroke:#333,stroke-width:2px
    style S fill:#afa,stroke:#333,stroke-width:2px
    style X fill:#ffc,stroke:#333,stroke-width:2px
    style Y fill:#ffc,stroke:#333,stroke-width:2px
    style Z fill:#ccf,stroke:#333,stroke-width:2px