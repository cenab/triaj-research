import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def simulate_multi_site_data(df, num_clients=2, strategy="random", random_state=42):
    """
    Simulates multi-site data by partitioning the main DataFrame into virtual client datasets.

    Args:
        df (pd.DataFrame): The preprocessed and feature-engineered DataFrame.
        num_clients (int): The number of virtual clients (hospitals) to simulate.
        strategy (str): The partitioning strategy. Options: "random", "demographic_age", "demographic_gender", "temporal".
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary where keys are client IDs (e.g., "client_0", "client_1")
              and values are pd.DataFrame objects representing each client's local data.
    """
    client_data = {}

    if strategy == "random":
        # Randomly assign a percentage of patients to each virtual client
        # This is a simple split, more complex random splits could be implemented
        if num_clients == 1:
            client_data["client_0"] = df
            return client_data
        
        # Split into initial parts
        df_list = [df]
        # A more robust random split using numpy.array_split
        shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df_splits = np.array_split(shuffled_df, num_clients)
        for i, split_df in enumerate(df_splits):
            client_data[f"client_{i}"] = split_df

    elif strategy == "demographic_age":
        # Partition based on 'yaş' (age)
        # To use original age for splitting, we need to reverse the scaling or pass original age
        # For now, let's assume 'yaş' column in df is the scaled one, and we need to access original age
        # This requires a change in feature_engineering.py to keep original age or pass it.
        # For demonstration, let's assume a temporary 'original_yaş' column is available or re-calculate
        
        # Re-calculate original age for splitting purposes (assuming min/max from original data)
        # Based on triaj_data.csv, min age is 13, max age is 99.
        min_original_age = 13
        max_original_age = 99
        df['yaş_unscaled'] = df['yaş'] * (max_original_age - min_original_age) + min_original_age
        
        if num_clients == 2: # Pediatric (<18) vs Adult (>=18)
            client_data["client_pediatric"] = df[df['yaş_unscaled'] < 18].drop('yaş_unscaled', axis=1)
            client_data["client_adult"] = df[df['yaş_unscaled'] >= 18].drop('yaş_unscaled', axis=1)
        elif num_clients == 3: # Pediatric (<18), Adult (18-65), Elderly (>65)
            client_data["client_pediatric"] = df[df['yaş_unscaled'] < 18].drop('yaş_unscaled', axis=1)
            client_data["client_adult"] = df[(df['yaş_unscaled'] >= 18) & (df['yaş_unscaled'] <= 65)].drop('yaş_unscaled', axis=1)
            client_data["client_elderly"] = df[df['yaş_unscaled'] > 65].drop('yaş_unscaled', axis=1)
        else:
            raise ValueError("For 'demographic_age' strategy, num_clients must be 2 or 3.")
        
        # Remove empty clients if any
        client_data = {k: v for k, v in client_data.items() if not v.empty}

    elif strategy == "demographic_gender":
        # Partition based on 'cinsiyet' (gender)
        # Assuming 'cinsiyet_Male' and 'cinsiyet_Female' columns exist after one-hot encoding
        if 'cinsiyet_Male' in df.columns and 'cinsiyet_Female' in df.columns:
            client_data["client_male"] = df[df['cinsiyet_Male'] == 1]
            client_data["client_female"] = df[df['cinsiyet_Female'] == 1]
            if 'cinsiyet_Unknown' in df.columns:
                client_data["client_unknown_gender"] = df[df['cinsiyet_Unknown'] == 1]
        else:
            raise ValueError("Gender columns (cinsiyet_Male, cinsiyet_Female) not found. Ensure one-hot encoding is applied.")
        
        # Remove empty clients if any
        client_data = {k: v for k, v in client_data.items() if not v.empty}

    elif strategy == "temporal":
        # Partition data by 'created' date ranges (assuming 'year' and 'month' features are present)
        # This strategy requires more specific date ranges or a more dynamic splitting logic
        # For simplicity, let's split by year for now if multiple years are present
        if 'year' not in df.columns:
            raise ValueError("Temporal strategy requires 'year' column in DataFrame.")
        
        unique_years = sorted(df['year'].unique())
        if len(unique_years) < num_clients:
            print(f"Warning: Not enough unique years ({len(unique_years)}) for {num_clients} temporal clients. Falling back to random split.")
            return simulate_multi_site_data(df, num_clients, strategy="random", random_state=random_state)
        
        # Simple split by year, assigning years to clients
        clients_per_year = len(unique_years) // num_clients
        remainder = len(unique_years) % num_clients
        
        start_idx = 0
        for i in range(num_clients):
            client_years = unique_years[start_idx : start_idx + clients_per_year + (1 if i < remainder else 0)]
            client_data[f"client_year_{i}"] = df[df['year'].isin(client_years)]
            start_idx += clients_per_year + (1 if i < remainder else 0)
        
        # Remove empty clients if any
        client_data = {k: v for k, v in client_data.items() if not v.empty}

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from 'random', 'demographic_age', 'demographic_gender', 'temporal'.")

    return client_data

if __name__ == "__main__":
    from data_preparation import load_and_clean_data
    from feature_engineering import feature_engineer_data

    file_path = 'triaj_data.csv'
    df_cleaned = load_and_clean_data(file_path)
    df_engineered = feature_engineer_data(df_cleaned.copy())

    print("\n--- Phase 1.3: Simulated Multi-Site Data Generation ---")

    # Example 1: Random split into 3 clients
    print("\nSimulating random split into 3 clients:")
    clients_random = simulate_multi_site_data(df_engineered.copy(), num_clients=3, strategy="random")
    for client_id, client_df in clients_random.items():
        print(f"  {client_id}: {client_df.shape[0]} samples")

    # Example 2: Demographic split by age into 2 clients (pediatric vs adult)
    print("\nSimulating demographic split by age into 2 clients (pediatric vs adult):")
    try:
        clients_age = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="demographic_age")
        for client_id, client_df in clients_age.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example 3: Demographic split by gender
    print("\nSimulating demographic split by gender:")
    try:
        clients_gender = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="demographic_gender")
        for client_id, client_df in clients_gender.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example 4: Temporal split into 2 clients (if multiple years exist)
    print("\nSimulating temporal split into 2 clients:")
    try:
        clients_temporal = simulate_multi_site_data(df_engineered.copy(), num_clients=2, strategy="temporal")
        for client_id, client_df in clients_temporal.items():
            print(f"  {client_id}: {client_df.shape[0]} samples")
    except ValueError as e:
        print(f"  Error: {e}")