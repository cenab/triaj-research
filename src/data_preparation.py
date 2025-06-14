import pandas as pd

def load_and_clean_data(file_path):
    """
    Loads the triaj data from a CSV file and performs initial cleaning.
    """
    df = pd.read_csv(file_path)

    # Handle missing values for numerical vital signs
    numerical_cols = ["sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    for col in numerical_cols:
        # Replace 0 with NaN for 'nabız' as per plan (row 234 example) and then impute
        if col == "nabız":
            df[col] = df[col].replace(0, pd.NA)
        df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric type
        # Check for empty arrays before computing mean to avoid warnings
        if not df[col].isna().all():
            df[col] = df[col].fillna(df[col].mean()) # Simple mean imputation for now
        else:
            df[col] = df[col].fillna(0) # Fill with 0 if all values are NaN

    # For textual/categorical fields, treat missing values as "not present"
    df = df.fillna("") # Fill all remaining NaN with empty string for text processing

    return df

if __name__ == "__main__":
    file_path = 'triaj_data.csv'
    df = load_and_clean_data(file_path)
    print("Data loaded and initially cleaned. First 5 rows:")
    print(df.head())
    print("\nData Info:")
    df.info()