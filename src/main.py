import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time # Import time for latency measurement
from datetime import datetime
import argparse

try:
    from .data_preparation import load_and_clean_data
    from .feature_engineering import feature_engineer_data
    from .data_simulation import simulate_multi_site_data
    from .model_architecture import TriageModel
    from .model_optimization import apply_quantization, apply_pruning, apply_knowledge_distillation
    from .federated_learning import FederatedClient, FederatedServer, apply_domain_adaptation, monitor_data_drift, apply_differential_privacy, apply_robust_aggregation, apply_communication_efficiency, monitor_federated_fairness, get_model_parameters, set_model_parameters, simulate_byzantine_attacks
    from .explainable_ai import generate_feature_importance, generate_llm_explanation, extract_boolean_rules
    from .evaluation_framework import ComprehensiveEvaluator, ClinicalMetrics, FairnessEvaluator, PerformanceBenchmark
    from .kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
except ImportError:
    # Direct imports when running as script
    from data_preparation import load_and_clean_data
    from feature_engineering import feature_engineer_data
    from data_simulation import simulate_multi_site_data
    from model_architecture import TriageModel
    from model_optimization import apply_quantization, apply_pruning, apply_knowledge_distillation
    from federated_learning import FederatedClient, FederatedServer, apply_domain_adaptation, monitor_data_drift, apply_differential_privacy, apply_robust_aggregation, apply_communication_efficiency, monitor_federated_fairness, get_model_parameters, set_model_parameters, simulate_byzantine_attacks
    from explainable_ai import generate_feature_importance, generate_llm_explanation, extract_boolean_rules
    from evaluation_framework import ComprehensiveEvaluator, ClinicalMetrics, FairnessEvaluator, PerformanceBenchmark
    try:
        from kaggle_data import load_kaggle_triage_data, feature_engineer_kaggle_data
    except ImportError:
        load_kaggle_triage_data = None  # type: ignore
        feature_engineer_kaggle_data = None  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="FairTriEdge-FL end-to-end pipeline")
    parser.add_argument("--dataset", choices=["triaj", "kaggle"], default="triaj",
                        help="Dataset to use")
    parser.add_argument("--model", choices=["basic", "advanced"], default="basic",
                        help="Model type: basic=TriageModel, advanced=AdvancedHierarchicalTriageModel")
    args = parser.parse_args()

    use_kaggle = args.dataset == "kaggle"
    use_advanced_model = args.model == "advanced"

    print("--- Phase 1: Data Preparation and Simulation Environment Setup ---")

    if use_kaggle and use_advanced_model:
        # Advanced path – use enhanced feature engineering
        # Attempt relative import first (package execution), fall back to absolute for script execution
        try:
            from .kaggle_data import load_kaggle_triage_data as _load_kag, feature_engineer_kaggle_data as _fe_min
        except ImportError:
            from kaggle_data import load_kaggle_triage_data as _load_kag, feature_engineer_kaggle_data as _fe_min
        # Import advanced Kaggle feature-engineering util
        try:
            from .experimental.kaggle_enhanced_final_fix_v2 import advanced_kaggle_feature_engineering as _fe_adv  # type: ignore
        except Exception:
            try:
                from experimental.kaggle_enhanced_final_fix_v2 import advanced_kaggle_feature_engineering as _fe_adv  # type: ignore
            except Exception:
                _fe_adv = None

        if _fe_adv is None:
            raise ImportError("advanced_kaggle_feature_engineering not available; ensure optional deps installed")

        print("Step 1: Loading Kaggle dataset…")
        kag_df_raw = _load_kag()
        df_engineered, vital_feats, symptom_feats, risk_feats, context_feats, lab_feats, interaction_feats = _fe_adv(kag_df_raw)

        num_classes = df_engineered['esi_3class'].nunique()

        # Import advanced model architecture
        try:
            from .advanced_model_architecture import AdvancedHierarchicalTriageModel  # type: ignore
        except ImportError:
            from advanced_model_architecture import AdvancedHierarchicalTriageModel  # type: ignore

        model = AdvancedHierarchicalTriageModel(
            num_vital_features=len(vital_feats),
            num_symptom_features=len(symptom_feats),
            num_risk_features=len(risk_feats),
            num_context_features=len(context_feats),
            num_lab_features=len(lab_feats),
            num_interaction_features=len(interaction_feats),
            num_classes=num_classes,
        )

        print("\nAdvanced Hierarchical Triage Model instantiated:")
        print(model)
        print("Feature-group sizes:")
        print(f"  vitals           : {len(vital_feats)}")
        print(f"  symptoms         : {len(symptom_feats)}")
        print(f"  risk factors     : {len(risk_feats)}")
        print(f"  context          : {len(context_feats)}")
        print(f"  lab values       : {len(lab_feats)}")
        print(f"  interactions     : {len(interaction_feats)}")

        print("\n▶️  Starting *preview* federated-learning run with the advanced Kaggle model (3 rounds)…")

        # -----------------------------------------------------------------
        # Build PyTorch tensors for six feature groups + target
        # -----------------------------------------------------------------
        all_features = (
            vital_feats + symptom_feats + risk_feats + context_feats + lab_feats + interaction_feats
        )

        # Ensure columns exist and fill NaNs
        df_engineered[all_features] = df_engineered[all_features].fillna(0)

        # -------------------------------------------------------------
        # Sub-sample to manageable size: 5 clients × 3k rows = 15k rows
        # -------------------------------------------------------------
        desired_total = 15000
        if len(df_engineered) > desired_total:
            df_engineered = df_engineered.sample(n=desired_total, random_state=42).reset_index(drop=True)

        import torch
        from torch.utils.data import TensorDataset, DataLoader, random_split

        X_vital = torch.tensor(df_engineered[vital_feats].values, dtype=torch.float32)
        X_symptom = torch.tensor(df_engineered[symptom_feats].values, dtype=torch.float32)
        X_risk = torch.tensor(df_engineered[risk_feats].values, dtype=torch.float32)
        X_context = torch.tensor(df_engineered[context_feats].values, dtype=torch.float32)
        X_lab = torch.tensor(df_engineered[lab_feats].values, dtype=torch.float32)
        X_interaction = torch.tensor(df_engineered[interaction_feats].values, dtype=torch.float32)
        y_tensor = torch.tensor(df_engineered['esi_3class'].values, dtype=torch.long)

        full_dataset_fl = TensorDataset(
            X_vital,
            X_symptom,
            X_risk,
            X_context,
            X_lab,
            X_interaction,
            y_tensor,
        )

        # Split into train / test
        train_size_fl = int(0.8 * len(full_dataset_fl))
        test_size_fl = len(full_dataset_fl) - train_size_fl
        train_dataset_fl, global_test_dataset_fl = random_split(
            full_dataset_fl,
            [train_size_fl, test_size_fl],
            generator=torch.Generator().manual_seed(42),
        )

        # Split train set across clients (3 clients)
        num_clients_fl = 8  # increased client count
        len_per_client = len(train_dataset_fl) // num_clients_fl
        lengths_fl = [len_per_client] * num_clients_fl
        remainder_fl = len(train_dataset_fl) % num_clients_fl
        for i in range(remainder_fl):
            lengths_fl[i] += 1

        client_datasets_fl = random_split(
            train_dataset_fl,
            lengths_fl,
            generator=torch.Generator().manual_seed(42),
        )

        # Ensure each client provides ~3 000 samples per epoch via replacement sampling
        desired_samples_per_client = 3000
        client_data_loaders_fl = []
        from torch.utils.data import RandomSampler

        # Build a class-balanced sampler per client (inverse-frequency weights)
        import numpy as _np
        from torch.utils.data import WeightedRandomSampler

        for ds in client_datasets_fl:
            # extract targets
            labels = _np.array([ds[i][-1].item() for i in range(len(ds))])
            class_sample_count = _np.bincount(labels, minlength=num_classes)
            # avoid div-by-zero
            class_weights = 1.0 / (class_sample_count + 1e-6)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=desired_samples_per_client,
                replacement=True,
            )
            client_data_loaders_fl.append(DataLoader(ds, batch_size=64, sampler=sampler))
        
        print(f"Created {num_clients_fl} clients × {desired_samples_per_client} samples each (with replacement).")
        global_test_loader_fl = DataLoader(global_test_dataset_fl, batch_size=64, shuffle=False)

        try:
            from .federated_learning import (
                FederatedClient,
                FederatedServer,
                get_model_parameters,
                set_model_parameters,
                apply_robust_aggregation,
            )
        except ImportError:
            from federated_learning import (
                FederatedClient,
                FederatedServer,
                get_model_parameters,
                set_model_parameters,
                apply_robust_aggregation,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global_model_fl = AdvancedHierarchicalTriageModel(
            num_vital_features=len(vital_feats),
            num_symptom_features=len(symptom_feats),
            num_risk_features=len(risk_feats),
            num_context_features=len(context_feats),
            num_lab_features=len(lab_feats),
            num_interaction_features=len(interaction_feats),
            num_classes=num_classes,
        ).to(device)

        server_fl = FederatedServer(global_model_fl, device)

        clients_fl = []
        for idx, loader in enumerate(client_data_loaders_fl):
            client_model_fl = AdvancedHierarchicalTriageModel(
                num_vital_features=len(vital_feats),
                num_symptom_features=len(symptom_feats),
                num_risk_features=len(risk_feats),
                num_context_features=len(context_feats),
                num_lab_features=len(lab_feats),
                num_interaction_features=len(interaction_feats),
                num_classes=num_classes,
            ).to(device)

            set_model_parameters(client_model_fl, get_model_parameters(global_model_fl))
            clients_fl.append(
                FederatedClient(
                    f"client_{idx}", client_model_fl, loader, device, privacy_config=None
                )
            )

        num_rounds = 30
        from time import perf_counter

        local_epochs = 2  # keep epochs modest since dataset per client is larger

        # ------------------- FedOpt setup (Adam server) -------------------
        server_opt = torch.optim.Adam(server_fl.global_model.parameters(), lr=0.001, betas=(0.9, 0.99))

        for rnd in range(num_rounds):
            print(f"\n--- Advanced FL Round {rnd + 1}/{num_rounds} ---")
            client_updates = []
            t0 = perf_counter()
            for client in clients_fl:
                client.set_parameters(get_model_parameters(server_fl.global_model))
                params, num_samples, _ = client.train(epochs=local_epochs)
                client_updates.append((params, num_samples))
            aggregated_params = apply_robust_aggregation(client_updates, method="fedavg")

            # FedAdam update: use aggregated_params as target, current params as source
            current_params = list(server_fl.global_model.parameters())
            for p, new_val in zip(current_params, aggregated_params):
                # gradient = current − aggregated (move towards aggregated)
                p.grad = (p.data - torch.tensor(new_val, device=device, dtype=p.dtype))
            server_opt.step()
            server_opt.zero_grad(set_to_none=True)
            server_fl.evaluate_global_model(global_test_loader_fl)
            print(f"Round duration: {perf_counter() - t0:.2f}s")

        print("\n✅  Advanced federated-learning preview complete.")
        return

    if use_kaggle:
        if load_kaggle_triage_data is None:
            raise ImportError("kaggle_data module is not available. Did you install optional deps?")

        print("Step 1: Loading Kaggle dataset…")
        kag_df_raw = load_kaggle_triage_data()
        df_cleaned, feature_cols = feature_engineer_kaggle_data(kag_df_raw)
        print("Kaggle feature-engineering complete.")

    else:
        file_path = 'triaj_data.csv'
        print("Step 1: Loading and Initial Cleaning Data...")
        df_cleaned = load_and_clean_data(file_path)
        print("Initial data cleaning complete.")
        print(f"Shape after cleaning: {df_cleaned.shape}")

    if not use_kaggle and not use_advanced_model:
        print("\nStep 2: Feature Engineering Data...")
        df_engineered = feature_engineer_data(df_cleaned.copy())
        print("Feature engineering complete.")
        print(f"Shape after feature engineering: {df_engineered.shape}")
        print("\nFirst 5 rows of Feature Engineered Data:")
        print(df_engineered.head())
        print("\nTarget variable distribution:")
        print(df_engineered['doğru triyaj_encoded'].value_counts())
    else:
        df_engineered = df_cleaned  # already engineered in Kaggle path
        print(f"Enhanced Kaggle dataset shape: {df_engineered.shape}")
        print("Target distribution:")
        print(df_engineered['doğru triyaj_encoded'].value_counts())

        print("\nℹ️  Kaggle integration is in *preview* mode – downstream FL pipeline still "
              "expects the original feature schema. Skipping the heavier stages for now.")
        return  # Early exit until FL refactor supports Kaggle schema

    print("\nStep 3: Simulated Multi-Site Data Generation...")
    # Example: Random split into 3 clients
    print("\nSimulating random split into 3 clients:")
    clients_random = simulate_multi_site_data(df_engineered.copy(), num_clients=3, strategy="random")
    for client_id, client_df in clients_random.items():
        print(f"  {client_id}: {client_df.shape[0]} samples")

    # Example: Demographic split by age into 2 clients (pediatric vs adult)
    print("\nSimulating demographic split by age into 2 clients (pediatric vs adult):")
    try:
        clients_age = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="demographic_age")
        for client_id, client_df in clients_age.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example: Demographic split by gender
    print("\nSimulating demographic split by gender:")
    try:
        clients_gender = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="demographic_gender")
        for client_id, client_df in clients_gender.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example: Temporal split into 2 clients (if multiple years exist)
    print("\nSimulating temporal split into 2 clients:")
    try:
        clients_temporal = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="temporal")
        for client_id, client_df in clients_temporal.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")

    print("\n--- Phase 1.4: Domain Adaptation Strategy ---")
    # Apply domain adaptation with different methods
    print("\nApplying Domain Adaptation (method: DANN)...")
    clients_adapted_dann = apply_domain_adaptation(clients_random.copy(), method="dann")
    print("Domain adaptation applied (DANN).")

    print("\nApplying Domain Adaptation (method: MMD)...")
    clients_adapted_mmd = apply_domain_adaptation(clients_random.copy(), method="mmd")
    print("Domain adaptation applied (MMD).")

    print("\n--- Phase 1.5: Continuous Drift Monitoring ---")
    # Monitor data drift with different methods
    print("\nMonitoring Data Drift (method: ADWIN)...")
    drift_detected_adwin = monitor_data_drift(clients_random.copy(), method="adwin")
    print(f"Data drift detected (ADWIN): {drift_detected_adwin}.")

    print("\nMonitoring Data Drift (method: KS-Test)...")
    drift_detected_ks = monitor_data_drift(clients_random.copy(), method="ks_test")
    print(f"Data drift detected (KS-Test): {drift_detected_ks}.")
 
    print("\n--- Phase 2: Core Model Development and Optimization ---")
    print("Step 2.1: Initializing Triage Model...")

    # Separate features (X) and target (y)
    X = df_engineered.drop('doğru triyaj_encoded', axis=1)
    y = df_engineered['doğru triyaj_encoded']

    # Identify feature types based on feature_engineering.py logic
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    temporal_cols = ['hour_of_day', 'day_of_week', 'month'] # 'year' is often used for temporal splits, not direct model input
    
    # Boolean columns are all remaining columns after removing numerical, temporal, and target
    boolean_cols = [col for col in X.columns if col not in numerical_cols + temporal_cols + ['year', 'cinsiyet_Male', 'cinsiyet_Female', 'cinsiyet_Unknown', 'yaş_unscaled']]
    
    # Add one-hot encoded gender columns to boolean_cols if they exist
    if 'cinsiyet_Male' in X.columns:
        boolean_cols.append('cinsiyet_Male')
    if 'cinsiyet_Female' in X.columns:
        boolean_cols.append('cinsiyet_Female')
    if 'cinsiyet_Unknown' in X.columns:
        boolean_cols.append('cinsiyet_Unknown')

    num_numerical_features = len(numerical_cols)
    num_boolean_features = len(boolean_cols)
    num_temporal_features = len(temporal_cols)
    num_classes = len(y.unique())

    model = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes)
    print("Triage Model initialized.")
    print(f"Model input dimensions: Numerical={num_numerical_features}, Boolean={num_boolean_features}, Temporal={num_temporal_features}")
    print(f"Number of output classes: {num_classes}")
    print("Model architecture:\n", model)

    print("\nStep 2.2: Resource Optimization (TinyML)...")
    # Demonstrate Quantization
    print("\nApplying Quantization...")
    # Prepare calibration data from the actual dataset
    calibration_numerical = torch.tensor(X[numerical_cols].values[:100], dtype=torch.float32)
    calibration_boolean = torch.tensor(X[boolean_cols].values[:100], dtype=torch.float32)
    calibration_temporal = torch.tensor(X[temporal_cols].values[:100], dtype=torch.float32)
    calibration_data = (calibration_numerical, calibration_boolean, calibration_temporal)
    
    quantized_model = apply_quantization(model, backend='auto', calibration_data=calibration_data)
    print("Quantization applied.")

    # Demonstrate Pruning
    print("\nApplying Pruning...")
    # Create a fresh model instance for pruning to avoid issues with quantized model
    pruning_model = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes)
    pruned_model = apply_pruning(pruning_model, amount=0.5)
    print("Pruning applied.")

    # Demonstrate Knowledge Distillation (requires a training loop and data loader)
    print("\nDemonstrating Knowledge Distillation (requires training data)...")
    # For a full demonstration, you'd need to train the teacher model first.
    # Here, we'll just set up dummy data and a dummy teacher.
    
    # Dummy data for demonstration (same as in model_optimization.py)
    num_samples_kd = 100
    dummy_numerical_data_kd = torch.randn(num_samples_kd, num_numerical_features)
    dummy_boolean_data_kd = torch.randint(0, 2, (num_samples_kd, num_boolean_features)).float()
    dummy_temporal_data_kd = torch.randn(num_samples_kd, num_temporal_features)
    dummy_targets_kd = torch.randint(0, num_classes, (num_samples_kd,))

    dummy_dataset_kd = TensorDataset(dummy_numerical_data_kd, dummy_boolean_data_kd, dummy_temporal_data_kd, dummy_targets_kd)
    dummy_train_loader_kd = DataLoader(dummy_dataset_kd, batch_size=16)

    teacher_model_kd = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes)
    student_model_kd = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes)
    optimizer_kd = optim.Adam(student_model_kd.parameters(), lr=0.001)
    criterion_kd = torch.nn.CrossEntropyLoss()

    trained_student_model_kd = apply_knowledge_distillation(
        teacher_model_kd,
        student_model_kd,
        dummy_train_loader_kd,
        optimizer_kd,
        criterion_kd,
        epochs=2 # Reduced epochs for quick demo
    )
    print("Knowledge Distillation demonstrated.")

    print("\n--- Phase 2.3: Personalization Mechanisms ---")
    print("Personalization based on static patient factors (age, comorbidities) is handled by feature engineering.")
    print("Dynamic personalization (learning from repeated visits/baselines) will be integrated with on-device model adaptation in Phase 3 (Federated Learning).")

    print("\n--- Phase 3: Federated Learning and Robustness Integration ---")
    print("Step 3.1: Simulated Multi-Node Training (Federated Learning Round)...")

    # Convert DataFrame to PyTorch Tensors for FL
    X_numerical_full = torch.tensor(X[numerical_cols].values, dtype=torch.float32)
    X_boolean_full = torch.tensor(X[boolean_cols].values, dtype=torch.float32)
    X_temporal_full = torch.tensor(X[temporal_cols].values, dtype=torch.float32)
    y_tensor_full = torch.tensor(y.values, dtype=torch.long)

    full_dataset_fl = TensorDataset(X_numerical_full, X_boolean_full, X_temporal_full, y_tensor_full)

    # Split into training data for clients and a global test set
    train_size_fl = int(0.8 * len(full_dataset_fl))
    test_size_fl = len(full_dataset_fl) - train_size_fl
    train_dataset_fl, global_test_dataset_fl = random_split(full_dataset_fl, [train_size_fl, test_size_fl], generator=torch.Generator().manual_seed(42))

    # Simulate clients from the train_dataset
    num_clients_fl = 8  # increased client count
    
    # Calculate lengths for random_split to handle remainders
    len_per_client = len(train_dataset_fl) // num_clients_fl
    lengths_fl = [len_per_client] * num_clients_fl
    remainder_fl = len(train_dataset_fl) % num_clients_fl
    for i in range(remainder_fl):
        lengths_fl[i] += 1
    
    client_datasets_fl = random_split(train_dataset_fl, lengths_fl, generator=torch.Generator().manual_seed(42))

    # Create DataLoaders for each client
    client_data_loaders_fl = []
    for i, client_ds in enumerate(client_datasets_fl):
        client_data_loaders_fl.append(DataLoader(client_ds, batch_size=32, shuffle=True))
    
    global_test_loader_fl = DataLoader(global_test_dataset_fl, batch_size=32, shuffle=False)

    # Initialize Models and FL Components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global model (server-side)
    global_model_fl = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes).to(device)
    server_fl = FederatedServer(global_model_fl, device)

    # Client models with privacy configuration
    privacy_config = {
        'enable_dp': True,
        'epsilon': 1.0,
        'delta': 1e-5,
        'max_grad_norm': 1.0,
        'total_epsilon': 10.0,
        'sensitivity': 1.0,
        'method': 'gaussian'
    }
    
    clients_fl = []
    for i, data_loader in enumerate(client_data_loaders_fl):
        client_model_fl = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes).to(device)
        # Initialize client model with global model's parameters
        # Use the standalone set_model_parameters function
        set_model_parameters(client_model_fl, get_model_parameters(global_model_fl))
        clients_fl.append(FederatedClient(f"client_{i}", client_model_fl, data_loader, device, privacy_config))

    # Simulate a few rounds of Federated Learning
    num_communication_rounds = 5
    for round_num in range(num_communication_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1}/{num_communication_rounds} ---")
        
        # Clients train locally with differential privacy
        client_updates_fl = []
        privacy_metrics_round = []
        
        for client in clients_fl:
            # Distribute global model to client before training
            # Use the standalone get_model_parameters function
            client.set_parameters(get_model_parameters(server_fl.global_model))
            
            # Simulate on-device model adaptation with differential privacy
            print(f"Client {client.client_id}: Performing local training with differential privacy...")
            params, num_samples, privacy_metrics = client.train(epochs=local_epochs, apply_dp=True)
            client_updates_fl.append((params, num_samples))
            privacy_metrics_round.append(privacy_metrics)
        
        # Simulate Byzantine attacks (for robustness testing)
        if round_num == 2:  # Apply attack in round 3 for demonstration
            print("\n--- Simulating Byzantine Attack ---")
            attacked_updates = simulate_byzantine_attacks(
                client_updates_fl,
                attack_type="gradient_ascent",
                attack_ratio=0.33,  # Compromise 1 out of 3 clients
                scale_factor=5.0
            )
            client_updates_fl = attacked_updates
        
        # Server aggregates updates with robust aggregation
        print(f"Applying robust aggregation...")
        aggregated_params_fl = apply_robust_aggregation(
            client_updates_fl,
            method="krum",  # Use Krum for Byzantine robustness
            num_malicious=1
        )
        
        # Update global model using the standalone set_model_parameters function
        set_model_parameters(server_fl.global_model, aggregated_params_fl)
        
        # Log privacy metrics
        total_epsilon_consumed = sum(pm.get('epsilon_consumed', 0) for pm in privacy_metrics_round)
        print(f"Round {round_num + 1} privacy consumption: ε={total_epsilon_consumed:.3f}")

        # Evaluate global model
        print(f"\n--- Global Model Evaluation after Round {round_num + 1} ---")
        server_fl.evaluate_global_model(global_test_loader_fl)

    print("\nFederated Learning simulation complete.")

    print("\n--- Phase 3.2: Privacy Preservation ---")
    # Apply differential privacy with enhanced implementation
    print("\nApplying Enhanced Differential Privacy...")
    
    # Use actual model parameters for demonstration
    dummy_gradients = [param.detach().cpu().numpy() for param in global_model_fl.parameters()]
    
    # Test different DP methods
    for method in ['gaussian', 'laplace']:
        print(f"\nTesting DP method: {method}")
        noisy_gradients, dp_metrics = apply_differential_privacy(
            dummy_gradients,
            sensitivity=1.0,
            epsilon=1.0,
            delta=1e-5,
            method=method,
            max_grad_norm=1.0,
            sample_size=len(train_dataset_fl)
        )
        print(f"DP applied: ε={dp_metrics['epsilon_consumed']:.3f}, "
              f"noise_multiplier={dp_metrics['noise_multiplier']:.4f}, "
              f"clipping={dp_metrics['clipping_applied']}")

    print("\n--- Phase 3.3: Poisoning Defense and Robust Aggregation ---")
    # Apply robust aggregation with different methods
    print("\nApplying Robust Aggregation (method: Krum)...")
    robust_aggregated_params_krum = apply_robust_aggregation(client_updates_fl, method="krum")
    print("Robust Aggregation applied (Krum).")

    print("\nApplying Robust Aggregation (method: Trimmed Mean)...")
    robust_aggregated_params_trimmed = apply_robust_aggregation(client_updates_fl, method="trimmed_mean")
    print("Robust Aggregation applied (Trimmed Mean).")

    print("\nApplying Robust Aggregation (method: Median)...")
    robust_aggregated_params_median = apply_robust_aggregation(client_updates_fl, method="median")
    print("Robust Aggregation applied (Median).")

    print("\n--- Phase 3.4: Communication Efficiency ---")
    # Apply communication efficiency with different methods
    print("\nApplying Communication Efficiency (method: Top-k Sparsification)...")
    compressed_updates_topk = apply_communication_efficiency(aggregated_params_fl, compression_ratio=0.1, method="top_k")
    print("Communication Efficiency applied (Top-k Sparsification).")

    print("\nApplying Communication Efficiency (method: Quantization)...")
    compressed_updates_quant = apply_communication_efficiency(aggregated_params_fl, compression_ratio=0.1, method="quantization")
    print("Communication Efficiency applied (Quantization).")

    print("\n--- Phase 3.5: Fairness in Federated Models ---")
    # Monitor fairness with different metrics
    print("\nMonitoring Fairness in Federated Models (metric: F1-score Parity)...")
    fairness_score_f1 = monitor_federated_fairness(global_model_fl, client_data_loaders_fl, device, fairness_metric="f1_score_parity", method="subgroup_evaluation")
    print(f"Fairness Monitoring complete (F1-score Parity). Dummy score: {fairness_score_f1:.2f}")

    print("\nMonitoring Fairness in Federated Models (metric: Demographic Parity)...")
    fairness_score_dp = monitor_federated_fairness(global_model_fl, client_data_loaders_fl, device, fairness_metric="demographic_parity", method="subgroup_evaluation")
    print(f"Fairness Monitoring complete (Demographic Parity). Dummy score: {fairness_score_dp:.2f}")

    print("\nMonitoring Fairness in Federated Models (metric: Equalized Odds)...")
    fairness_score_eo = monitor_federated_fairness(global_model_fl, client_data_loaders_fl, device, fairness_metric="equalized_odds", method="subgroup_evaluation")
    print(f"Fairness Monitoring complete (Equalized Odds). Dummy score: {fairness_score_eo:.2f}")

    print("\n--- Phase 4: Explainable AI (XAI) and LLM Integration ---")
    print("Step 4.1: Built-in XAI (Feature Importance)...")
    
    # Prepare data for XAI (single data point for demonstration)
    # Take the first sample from the global test set for XAI demo
    sample_numerical, sample_boolean, sample_temporal, sample_target = next(iter(global_test_loader_fl))
    sample_numerical = sample_numerical[0].unsqueeze(0) # Take first sample, add batch dim
    sample_boolean = sample_boolean[0].unsqueeze(0)
    sample_temporal = sample_temporal[0].unsqueeze(0)
    sample_target_scalar = sample_target[0].item() # Extract a single scalar target
    
    # Get all feature names for XAI
    all_feature_names = numerical_cols + boolean_cols + temporal_cols # 'year' is not a direct model input feature
    
    print("\nGenerating Feature Importance (method: SHAP)...")
    feature_importance_scores_shap = generate_feature_importance(
        global_model_fl, # Use the trained global model
        torch.cat((sample_numerical, sample_boolean, sample_temporal), dim=1),
        all_feature_names,
        method="shap"
    )
    print("Top 5 Feature Importance Scores (SHAP):")
    for feature, score in feature_importance_scores_shap[:5]:
        print(f"- {feature}: {score:.4f}")

    print("\nGenerating Feature Importance (method: Permutation Importance)...")
    feature_importance_scores_perm = generate_feature_importance(
        global_model_fl, # Use the trained global model
        torch.cat((sample_numerical, sample_boolean, sample_temporal), dim=1),
        all_feature_names,
        method="permutation"
    )
    print("Top 5 Feature Importance Scores (Permutation Importance):")
    for feature, score in feature_importance_scores_perm[:5]:
        print(f"- {feature}: {score:.4f}")

    print("\nStep 4.2: LLM-Enhanced Explanation Module...")
    patient_context = {
        "age": df_engineered['yaş_unscaled'].iloc[0], # Use unscaled age for context
        "gender": df_cleaned['cinsiyet'].iloc[0], # Use original gender for context
        "symptoms": df_cleaned['semptomlar_non travma_genel 01'].iloc[0] + ", " + df_cleaned['semptomlar_non travma_genel 02'].iloc[0],
        "vitals": f"BP {df_cleaned['sistolik kb'].iloc[0]}/{df_cleaned['diastolik kb'].iloc[0]}, HR {df_cleaned['nabız'].iloc[0]}, RR {df_cleaned['solunum sayısı'].iloc[0]}, Temp {df_cleaned['ateş'].iloc[0]}, Sat {df_cleaned['saturasyon'].iloc[0]}%"
    }
    
    print("\nGenerating LLM Explanation (method: OpenAI GPT-4o API)...")
    llm_explanation_gpt = generate_llm_explanation(
        sample_target_scalar, 
        feature_importance_scores_shap[:5], # Using SHAP scores for explanation
        patient_context,
        method="openai_gpt"
    )
    print(llm_explanation_gpt)

    print("\nStep 4.3: Synthetic Rare Case Generation (Conceptual - not implemented here)...")
    print("This would involve using LLMs (e.g., GPT-4o) to generate new synthetic patient records for stress-testing.")

    print("\nStep 4.4: Real-Time Interpretability of Boolean Rule Chains...")
    boolean_rules = extract_boolean_rules(global_model_fl, all_feature_names)
    print("Extracted Boolean Rules:")
    for rule in boolean_rules:
        print(f"- {rule}")

    print("\n--- Phase 5: Comprehensive Evaluation and Open Science ---")
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Prepare sensitive attributes for fairness evaluation
    print("\nPreparing fairness evaluation data...")
    # Simulate sensitive attributes based on the dataset
    test_size = len(global_test_dataset_fl)
    
    # Age groups: 0=Child (<18), 1=Adult (18-65), 2=Elderly (>65)
    # Based on the 'yaş' column from original data
    age_groups = []
    gender_groups = []
    
    for i in range(test_size):
        # Get original age from the dataset (using yaş_unscaled if available)
        if 'yaş_unscaled' in df_engineered.columns:
            age = df_engineered['yaş_unscaled'].iloc[i % len(df_engineered)]
        else:
            age = df_engineered['yaş'].iloc[i % len(df_engineered)]  # Use scaled age as fallback
        
        if age < 18:
            age_groups.append(0)  # Child
        elif age <= 65:
            age_groups.append(1)  # Adult
        else:
            age_groups.append(2)  # Elderly
        
        # Gender: Extract from original data
        if 'cinsiyet_Male' in df_engineered.columns:
            if df_engineered['cinsiyet_Male'].iloc[i % len(df_engineered)] == 1:
                gender_groups.append('Male')
            elif df_engineered['cinsiyet_Female'].iloc[i % len(df_engineered)] == 1:
                gender_groups.append('Female')
            else:
                gender_groups.append('Unknown')
        else:
            gender_groups.append('Unknown')
    
    sensitive_data = {
        'age_group': np.array(age_groups),
        'gender': np.array(gender_groups)
    }
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluation_results = evaluator.evaluate_federated_system(
        global_model=global_model_fl,
        clients=clients_fl,
        server=server_fl,
        test_loader=global_test_loader_fl,
        sensitive_data=sensitive_data,
        device=device
    )
    
    # Display evaluation results
    print("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
    
    # Clinical metrics
    clinical = evaluation_results['clinical_metrics']
    print(f"\n--- Clinical Performance ---")
    print(f"Overall Accuracy: {clinical['overall_accuracy']:.3f}")
    print(f"Class-wise Performance:")
    for class_name, metrics in clinical['class_metrics'].items():
        print(f"  {class_name}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    safety = clinical['clinical_safety']
    print(f"\n--- Clinical Safety ---")
    print(f"Under-triage Rate: {safety['under_triage_rate']:.3f}")
    print(f"Over-triage Rate: {safety['over_triage_rate']:.3f}")
    print(f"Critical Under-triage Rate: {safety['critical_under_triage_rate']:.3f}")
    print(f"Critical Sensitivity: {safety['critical_sensitivity']:.3f}")
    
    # Fairness metrics
    fairness = evaluation_results['fairness_metrics']
    print(f"\n--- Fairness Assessment ---")
    print(f"Overall Fairness Score: {fairness['overall_fairness_score']:.3f}")
    if fairness['fairness_violations']:
        print("Fairness Violations Detected:")
        for violation in fairness['fairness_violations']:
            print(f"  {violation['metric']}: difference={violation['difference']:.3f} "
                  f"(threshold={violation['threshold']:.3f})")
    else:
        print("No significant fairness violations detected.")
    
    # Performance metrics
    performance = evaluation_results['performance_metrics']
    print(f"\n--- Performance Metrics ---")
    print(f"Average Inference Time: {performance['avg_inference_time_ms']:.2f}ms")
    print(f"Throughput: {performance['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Model Size: {performance['model_size_mb']:.2f}MB")
    print(f"Total Parameters: {performance['total_parameters']:,}")
    
    # Federated learning metrics
    fl_metrics = evaluation_results['federated_metrics']
    print(f"\n--- Federated Learning Performance ---")
    print(f"Total Round Time: {fl_metrics['total_round_time']:.2f}s")
    print(f"Average Client Training Time: {fl_metrics['avg_client_training_time']:.2f}s")
    print(f"Aggregation Time: {fl_metrics['aggregation_time']:.3f}s")
    print(f"Communication Overhead: {fl_metrics['communication_overhead']:,} parameters")
    
    # Summary and recommendations
    summary = evaluation_results['summary']
    print(f"\n--- Summary ---")
    print(f"Overall Performance: {summary['overall_performance']}")
    print(f"Risk Assessment: {summary['risk_assessment']}")
    
    if summary['key_findings']:
        print("Key Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
    
    if summary['recommendations']:
        print("Recommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    # Save evaluation report
    import os
    os.makedirs('results', exist_ok=True)
    report_path = f"results/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_evaluation_report(evaluation_results, report_path)
    
    print(f"\n--- Open Science Practices ---")
    print("✅ Comprehensive evaluation completed")
    print("✅ Detailed metrics calculated and logged")
    print("✅ Fairness assessment performed")
    print("✅ Clinical safety metrics evaluated")
    print(f"✅ Evaluation report saved: {report_path}")
    print("✅ Results ready for peer review and publication")
    
    # Performance comparison with baseline
    print(f"\n--- Performance Comparison ---")
    print("Comparing against simple centralized baseline...")
    
    # Simple baseline: Logistic Regression on centralized data
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Prepare centralized data for baseline
    X_baseline = torch.cat([
        X_numerical_full, X_boolean_full, X_temporal_full
    ], dim=1).numpy()
    y_baseline = y_tensor_full.numpy()
    
    # Split for baseline evaluation
    from sklearn.model_selection import train_test_split
    X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
        X_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline
    )
    
    # Train baseline model
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_baseline, y_train_baseline)
    baseline_pred = baseline_model.predict(X_test_baseline)
    baseline_accuracy = accuracy_score(y_test_baseline, baseline_pred)
    
    print(f"Federated Model Accuracy: {clinical['overall_accuracy']:.3f}")
    print(f"Centralized Baseline Accuracy: {baseline_accuracy:.3f}")
    print(f"Performance Difference: {clinical['overall_accuracy'] - baseline_accuracy:+.3f}")
    
    if clinical['overall_accuracy'] >= baseline_accuracy:
        print("✅ Federated model matches or exceeds centralized baseline!")
    else:
        print("⚠️  Federated model underperforms centralized baseline")
    
    print("\n=== PHASE 5 EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()