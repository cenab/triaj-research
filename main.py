import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time # Import time for latency measurement

from data_preparation import load_and_clean_data
from feature_engineering import feature_engineer_data
from data_simulation import simulate_multi_site_data
from model_architecture import TriageModel
from model_optimization import apply_quantization, apply_pruning, apply_knowledge_distillation
from federated_learning import FederatedClient, FederatedServer, apply_domain_adaptation, monitor_data_drift, apply_differential_privacy, apply_robust_aggregation, apply_communication_efficiency, monitor_federated_fairness, get_model_parameters, set_model_parameters
from explainable_ai import generate_feature_importance, generate_llm_explanation, extract_boolean_rules

def main():
    file_path = 'triaj_data.csv'
    
    print("--- Phase 1: Data Preparation and Simulation Environment Setup ---")
    print("Step 1: Loading and Initial Cleaning Data...")
    df_cleaned = load_and_clean_data(file_path)
    print("Initial data cleaning complete.")
    print(f"Shape after cleaning: {df_cleaned.shape}")

    print("\nStep 2: Feature Engineering Data...")
    df_engineered = feature_engineer_data(df_cleaned.copy())
    print("Feature engineering complete.")
    print(f"Shape after feature engineering: {df_engineered.shape}")
    print("\nFirst 5 rows of Feature Engineered Data:")
    print(df_engineered.head())
    print("\nTarget variable distribution:")
    print(df_engineered['doğru triyaj_encoded'].value_counts())

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
    print("\nApplying Quantization (skipped for now due to backend issues)...")
    # quantized_model = apply_quantization(model)
    # print("Quantization applied.")

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
    num_clients_fl = 3
    
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

    # Client models
    clients_fl = []
    for i, data_loader in enumerate(client_data_loaders_fl):
        client_model_fl = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes).to(device)
        # Initialize client model with global model's parameters
        # Use the standalone set_model_parameters function
        set_model_parameters(client_model_fl, get_model_parameters(global_model_fl))
        clients_fl.append(FederatedClient(f"client_{i}", client_model_fl, data_loader, device))

    # Simulate a few rounds of Federated Learning
    num_communication_rounds = 5
    for round_num in range(num_communication_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1}/{num_communication_rounds} ---")
        
        # Clients train locally
        client_updates_fl = []
        for client in clients_fl:
            # Distribute global model to client before training
            # Use the standalone get_model_parameters function
            client.set_parameters(get_model_parameters(server_fl.global_model))
            
            # Simulate on-device model adaptation (Phase 2.3 dynamic personalization)
            print(f"Client {client.client_id}: Performing local training and on-device adaptation...")
            params, num_samples = client.train(epochs=1) # Train for 1 local epoch
            client_updates_fl.append((params, num_samples))
        
        # Server aggregates updates
        aggregated_params_fl = server_fl.aggregate_parameters(client_updates_fl)
        # Update global model using the standalone set_model_parameters function
        set_model_parameters(server_fl.global_model, aggregated_params_fl)

        # Evaluate global model
        print(f"\n--- Global Model Evaluation after Round {round_num + 1} ---")
        server_fl.evaluate_global_model(global_test_loader_fl)

    print("\nFederated Learning simulation complete.")

    print("\n--- Phase 3.2: Privacy Preservation ---")
    # Apply differential privacy with different methods
    print("\nApplying Differential Privacy (method: Opacus/TensorFlow Privacy)...")
    dummy_gradients = [np.random.rand(10, 10).astype(np.float32)] # Example dummy gradients
    noisy_gradients_dp = apply_differential_privacy(dummy_gradients, sensitivity=1.0, epsilon=1.0, method="opacus_tf_privacy")
    print("Differential Privacy applied (Opacus/TensorFlow Privacy).")

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

    print("\n--- Proceeding to Phase 5: Comprehensive Evaluation and Open Science (Not yet implemented) ---")

if __name__ == "__main__":
    main()