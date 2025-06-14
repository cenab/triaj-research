import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def feature_engineer_data(df):
    """
    Performs feature engineering on the cleaned triaj DataFrame.
    """
    # 1. Numerical Features
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    
    # Apply StandardScaler for most numerical features
    scaler_standard = StandardScaler()
    # Store original 'yaş' before scaling for potential use in explanations/demographic splits
    # Based on triaj_data.csv, min age is 13, max age is 99.
    # Note: These min/max values should ideally be derived from the full dataset or known domain knowledge
    # For this simulation, we'll use the min/max of the current df for demonstration purposes.
    df['yaş_unscaled'] = df['yaş'].copy() # Store original values before scaling
    
    df[numerical_cols] = scaler_standard.fit_transform(df[numerical_cols])

    # 2. Categorical Features
    # One-hot encode 'cinsiyet'
    df['cinsiyet'] = df['cinsiyet'].replace({'Erkek': 'Male', 'Kadın': 'Female', '': 'Unknown'})
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cinsiyet_encoded = ohe.fit_transform(df[['cinsiyet']])
    cinsiyet_df = pd.DataFrame(cinsiyet_encoded, columns=ohe.get_feature_names_out(['cinsiyet']), index=df.index)
    df = pd.concat([df, cinsiyet_df], axis=1)
    df.drop('cinsiyet', axis=1, inplace=True)

    # 3. Textual/Multi-value Features to Boolean
    text_cols = [
        "ek hastalıklar", "semptomlar_non travma_genel 01", "semptomlar_non travma_genel 02",
        "göz", "göğüs ağrısı", "karın ağrısı", "ayak travması", "bacak travması",
        "batın travması", "boyun travması", "el travması", "göğüs travması",
        "göz travması", "kafa travması", "kol travması", "parmak travması",
        "diğer travmalar", "dahiliye hastalıklar ", "psikiyatri ", "kardiyoloji ",
        "göğüs hastalıkları ", "nörolojik hastalıklar ", "beyin cerrahi  ",
        "kalp damar cerrahisi ", "kbb ", "göz hastalıkları", "İntaniye ", "Üroloji ",
        "Çevresel ve toksikoloji acilleri ", "kadın ve doğum hastalıkları ",
        "genel cerrahi hastalıklar ", "deri hastalıkları ", "travma_ayak",
        "travma_bacak", "travma_batin", "travma_boyun", "travma_el", "travma_gogus",
        "travma_goz", "travma_kafa", "travma_kalca", "travma_kol", "travma_parmak",
        "diğer diyagnoz_travma", "diğer diyagnoz ", "prognoz", "diğer prognoz"
    ]

    # Combine all text into a single string per row for vocabulary extraction
    # Ensure all values are strings before joining
    df['all_text_features'] = df[text_cols].astype(str).agg(','.join, axis=1)

    # Create a comprehensive vocabulary of all unique items
    vocabulary = set()
    for _, row in df.iterrows():
        items = [item.strip() for item in row['all_text_features'].split(',') if item.strip()]
        vocabulary.update(items)

    # Generate new binary (Boolean) columns for each unique item
    boolean_features_data = {}
    for item in sorted(list(vocabulary)): # Sort for consistent column order
        # Sanitize item name for column creation
        col_name = f"feature_{item.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_').replace('.', '').lower()}"
        boolean_features_data[col_name] = df['all_text_features'].apply(lambda x: 1 if item in x.split(',') else 0)
    
    boolean_features_df = pd.DataFrame(boolean_features_data, index=df.index)
    df = pd.concat([df, boolean_features_df], axis=1)

    df.drop(text_cols + ['all_text_features'], axis=1, inplace=True)

    # 4. Temporal Features (from 'created' column)
    df['created'] = pd.to_datetime(df['created'])
    df['hour_of_day'] = df['created'].dt.hour
    df['day_of_week'] = df['created'].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df['created'].dt.month
    df['year'] = df['created'].dt.year # Keep year for potential drift monitoring or time-based splits
    df.drop('created', axis=1, inplace=True)

    # Scale temporal features
    temporal_numerical_cols = ['hour_of_day', 'day_of_week', 'month', 'year']
    scaler_temporal = MinMaxScaler() # MinMaxScaler for cyclical/bounded features
    df[temporal_numerical_cols] = scaler_temporal.fit_transform(df[temporal_numerical_cols])

    # 5. Target Variable Mapping
    triage_mapping = {
        "Kırmızı Alan": 2,
        "Sarı Alan": 1,
        "Yeşil Alan": 0
    }
    df['doğru triyaj_encoded'] = df['doğru triyaj'].map(triage_mapping)
    df.drop(['triyaj alanı', 'doğru triyaj'], axis=1, inplace=True) # Drop original triage columns

    # Drop 'id' and 'protokol' as they are identifiers and not features
    df.drop(['id', 'protokol', 'state'], axis=1, inplace=True)

    return df

if __name__ == "__main__":
    from data_preparation import load_and_clean_data

    file_path = 'triaj_data.csv'
    df_cleaned = load_and_clean_data(file_path)
    df_engineered = feature_engineer_data(df_cleaned.copy()) # Use a copy to avoid modifying original df_cleaned
    print("\nFeature Engineered Data Info:")
    df_engineered.info()
    print("\nFirst 5 rows of Feature Engineered Data:")
    print(df_engineered.head())
    print("\nTarget variable distribution:")
    print(df_engineered['doğru triyaj_encoded'].value_counts())