import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import time
from datetime import datetime
import json
import os
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Download and load Kaggle dataset
def load_kaggle_triage_data():
    """Load the Kaggle hospital triage dataset"""
    print("Loading Kaggle hospital triage dataset...")
    
    # Check if already downloaded
    csv_path = 'src/kaggle_triage_data.csv'
    if os.path.exists(csv_path):
        print(f"Loading existing dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download('maalona/hospital-triage-and-patient-history-data')
        
        # Load R data
        import pyreadr
        rdata_path = os.path.join(path, '5v_cleandf.rdata')
        result = pyreadr.read_r(rdata_path)
        df = result[list(result.keys())[0]]
        
        # Save as CSV for future use
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    
    print(f"Dataset loaded: {df.shape}")
    return df

# Advanced feature engineering for Kaggle dataset
def advanced_kaggle_feature_engineering(df):
    """
    Advanced feature engineering with clinical domain knowledge
    """
    print("Applying advanced feature engineering for Kaggle dataset...")
    
    # 1. Target variable: ESI (Emergency Severity Index)
    df['esi_3class'] = df['esi'].map({
        1: 2,  # Red (Critical)
        2: 2,  # Red (Critical) 
        3: 1,  # Yellow (Urgent)
        4: 0,  # Green (Non-urgent)
        5: 0   # Green (Non-urgent)
    })
    
    # Remove rows with missing target
    df = df.dropna(subset=['esi_3class'])
    
    # 2. Advanced Vital Signs Features
    vital_features = []
    
    # Basic vitals with proper handling
    vital_cols = {
        'triage_vital_hr': 'hr',
        'triage_vital_sbp': 'sbp', 
        'triage_vital_dbp': 'dbp',
        'triage_vital_rr': 'rr',
        'triage_vital_o2': 'spo2',
        'triage_vital_temp': 'temp'
    }
    
    for orig_col, new_col in vital_cols.items():
        if orig_col in df.columns:
            df[new_col] = pd.to_numeric(df[orig_col], errors='coerce')
            # Fill missing with median by ESI class
            for esi_class in df['esi_3class'].unique():
                mask = (df['esi_3class'] == esi_class) & df[new_col].isna()
                median_val = df[df['esi_3class'] == esi_class][new_col].median()
                if pd.notna(median_val):
                    df.loc[mask, new_col] = median_val
            vital_features.append(new_col)
    
    # Age with proper handling
    df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    vital_features.append('age_numeric')
    
    # 3. Advanced Clinical Severity Scores
    def calculate_advanced_severity_score(row):
        """Calculate comprehensive clinical severity score"""
        score = 0
        
        # Heart Rate Score (0-4)
        if pd.notna(row.get('hr')):
            hr = row['hr']
            if hr < 50 or hr > 120: score += 3
            elif hr < 60 or hr > 100: score += 2
            elif hr < 65 or hr > 95: score += 1
        
        # Blood Pressure Score (0-4)
        if pd.notna(row.get('sbp')) and pd.notna(row.get('dbp')):
            sbp, dbp = row['sbp'], row['dbp']
            if sbp < 90 or sbp > 180 or dbp < 60 or dbp > 110: score += 4
            elif sbp < 100 or sbp > 160 or dbp < 65 or dbp > 100: score += 3
            elif sbp < 110 or sbp > 140 or dbp < 70 or dbp > 90: score += 2
        
        # Respiratory Rate Score (0-3)
        if pd.notna(row.get('rr')):
            rr = row['rr']
            if rr < 12 or rr > 25: score += 3
            elif rr < 14 or rr > 22: score += 2
            elif rr < 16 or rr > 20: score += 1
        
        # Oxygen Saturation Score (0-4)
        if pd.notna(row.get('spo2')):
            spo2 = row['spo2']
            if spo2 < 90: score += 4
            elif spo2 < 94: score += 3
            elif spo2 < 96: score += 2
            elif spo2 < 98: score += 1
        
        # Temperature Score (0-3)
        if pd.notna(row.get('temp')):
            temp = row['temp']
            if temp < 95 or temp > 103: score += 3
            elif temp < 96 or temp > 101: score += 2
            elif temp < 97 or temp > 100: score += 1
        
        # Age Score (0-3)
        if pd.notna(row.get('age_numeric')):
            age = row['age_numeric']
            if age > 80: score += 3
            elif age > 65: score += 2
            elif age < 2: score += 2
            elif age < 16: score += 1
        
        return min(score, 20)  # Cap at 20
    
    df['vital_severity_score'] = df.apply(calculate_advanced_severity_score, axis=1)
    vital_features.append('vital_severity_score')
    
    # 4. Advanced Symptom Clustering
    symptom_features = []
    
    # Pain severity
    if 'triage_pain' in df.columns:
        df['pain_score'] = pd.to_numeric(df['triage_pain'], errors='coerce').fillna(0)
        symptom_features.append('pain_score')
    
    # Chief complaint clustering
    if 'cc_reasonforvisit' in df.columns:
        # Create symptom clusters based on chief complaints
        cardiac_keywords = ['chest', 'heart', 'cardiac', 'angina', 'mi', 'infarct']
        respiratory_keywords = ['breath', 'sob', 'dyspnea', 'asthma', 'copd', 'pneumonia']
        neuro_keywords = ['head', 'neuro', 'stroke', 'seizure', 'confusion', 'altered']
        trauma_keywords = ['trauma', 'injury', 'fall', 'accident', 'fracture']
        gi_keywords = ['abdom', 'nausea', 'vomit', 'diarrhea', 'gi', 'gastro']
        
        df['cc_text'] = df['cc_reasonforvisit'].astype(str).str.lower()
        
        df['symptom_cardiac'] = df['cc_text'].str.contains('|'.join(cardiac_keywords), na=False).astype(int)
        df['symptom_respiratory'] = df['cc_text'].str.contains('|'.join(respiratory_keywords), na=False).astype(int)
        df['symptom_neurological'] = df['cc_text'].str.contains('|'.join(neuro_keywords), na=False).astype(int)
        df['symptom_trauma'] = df['cc_text'].str.contains('|'.join(trauma_keywords), na=False).astype(int)
        df['symptom_gi'] = df['cc_text'].str.contains('|'.join(gi_keywords), na=False).astype(int)
        
        symptom_features.extend(['symptom_cardiac', 'symptom_respiratory', 'symptom_neurological', 
                               'symptom_trauma', 'symptom_gi'])
    
    # 5. Risk Factors
    risk_features = []
    
    # Comorbidity count
    comorbidity_cols = [col for col in df.columns if 'previousdx' in col.lower()]
    if comorbidity_cols:
        df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1, skipna=True)
        risk_features.append('comorbidity_count')
    
    # Previous ED visits
    if 'n_edvisits' in df.columns:
        df['prev_ed_visits'] = pd.to_numeric(df['n_edvisits'], errors='coerce').fillna(0)
        risk_features.append('prev_ed_visits')
    
    # 6. Context Features
    context_features = []
    
    # Gender
    if 'gender' in df.columns:
        df['gender_male'] = (df['gender'] == 'Male').astype(int)
        context_features.append('gender_male')
    
    # Arrival timing
    if 'arrivalmonth' in df.columns:
        df['arrival_month'] = pd.to_numeric(df['arrivalmonth'], errors='coerce').fillna(6)
        context_features.append('arrival_month')
    
    if 'arrivalhour_bin' in df.columns:
        df['arrival_hour'] = pd.to_numeric(df['arrivalhour_bin'], errors='coerce').fillna(12)
        # Create shift indicators
        df['shift_night'] = ((df['arrival_hour'] >= 23) | (df['arrival_hour'] <= 7)).astype(int)
        df['shift_weekend'] = 0  # Would need day of week data
        context_features.extend(['arrival_hour', 'shift_night'])
    
    # 7. Lab Values (Enhanced)
    lab_features = []
    
    lab_cols = {
        'glucose_last': 'glucose',
        'creatinine_last': 'creatinine', 
        'bun_last': 'bun',
        'hemoglobin_last': 'hemoglobin',
        'wbc_last': 'wbc',
        'sodium_last': 'sodium',
        'potassium_last': 'potassium'
    }
    
    for orig_col, new_col in lab_cols.items():
        if orig_col in df.columns:
            df[new_col] = pd.to_numeric(df[orig_col], errors='coerce')
            # Create abnormal flags
            if new_col == 'glucose':
                df[f'{new_col}_abnormal'] = ((df[new_col] < 70) | (df[new_col] > 200)).astype(int)
            elif new_col == 'creatinine':
                df[f'{new_col}_abnormal'] = (df[new_col] > 1.5).astype(int)
            elif new_col == 'wbc':
                df[f'{new_col}_abnormal'] = ((df[new_col] < 4) | (df[new_col] > 12)).astype(int)
            
            lab_features.extend([new_col, f'{new_col}_abnormal'])
    
    # Lab availability score
    df['lab_availability'] = df[list(lab_cols.values())].notna().sum(axis=1)
    lab_features.append('lab_availability')
    
    # 8. Interaction Features
    interaction_features = []
    
    # Age-vital interactions
    if 'age_numeric' in df.columns and 'hr' in df.columns:
        df['age_hr_interaction'] = df['age_numeric'] * df['hr'] / 1000  # Scaled
        interaction_features.append('age_hr_interaction')
    
    if 'vital_severity_score' in df.columns and 'pain_score' in df.columns:
        df['severity_pain_interaction'] = df['vital_severity_score'] * df['pain_score']
        interaction_features.append('severity_pain_interaction')
    
    # 9. Final feature preparation
    all_features = vital_features + symptom_features + risk_features + context_features + lab_features + interaction_features
    
    # Ensure all features are numeric and handle missing values
    for feature in all_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            # Fill missing with median for that ESI class
            for esi_class in df['esi_3class'].unique():
                mask = (df['esi_3class'] == esi_class) & df[feature].isna()
                if mask.sum() > 0:
                    median_val = df[df['esi_3class'] == esi_class][feature].median()
                    if pd.notna(median_val):
                        df.loc[mask, feature] = median_val
                    else:
                        df.loc[mask, feature] = 0
        else:
            df[feature] = 0.0
    
    print(f"Advanced feature engineering complete.")
    print(f"Feature groups:")
    print(f"  Vital signs: {len(vital_features)} features")
    print(f"  Symptoms: {len(symptom_features)} features") 
    print(f"  Risk factors: {len(risk_features)} features")
    print(f"  Context: {len(context_features)} features")
    print(f"  Lab values: {len(lab_features)} features")
    print(f"  Interactions: {len(interaction_features)} features")
    print(f"  Total: {len(all_features)} features")
    
    return df, vital_features, symptom_features, risk_features, context_features, lab_features, interaction_features

# Advanced Hierarchical Model
class AdvancedHierarchicalTriageModel(nn.Module):
    """
    Advanced hierarchical model with attention and residual connections
    """
    def __init__(self, num_vital_features=8, num_symptom_features=5, 
                 num_risk_features=2, num_context_features=4, 
                 num_lab_features=8, num_interaction_features=2, num_classes=3):
        super().__init__()
        
        # Enhanced vital signs pathway (most critical)
        self.vital_signs_pathway = nn.Sequential(
            nn.Linear(num_vital_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Enhanced symptom pathway
        self.symptom_pathway = nn.Sequential(
            nn.Linear(num_symptom_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Risk factors pathway
        self.risk_pathway = nn.Sequential(
            nn.Linear(num_risk_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Context pathway
        self.context_pathway = nn.Sequential(
            nn.Linear(num_context_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Lab values pathway
        self.lab_pathway = nn.Sequential(
            nn.Linear(num_lab_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Interaction pathway
        self.interaction_pathway = nn.Sequential(
            nn.Linear(num_interaction_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Multi-head attention for feature fusion
        combined_dim = 32 + 16 + 8 + 8 + 8 + 8  # 80 total
        self.attention = nn.MultiheadAttention(
            embed_dim=combined_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Residual connection
        self.residual_transform = nn.Linear(combined_dim, combined_dim)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data):
        # Process each pathway
        vital_out = self.vital_signs_pathway(vital_data)
        symptom_out = self.symptom_pathway(symptom_data)
        risk_out = self.risk_pathway(risk_data)
        context_out = self.context_pathway(context_data)
        lab_out = self.lab_pathway(lab_data)
        interaction_out = self.interaction_pathway(interaction_data)
        
        # Combine features
        combined = torch.cat([vital_out, symptom_out, risk_out, context_out, lab_out, interaction_out], dim=1)
        
        # Apply attention
        combined_unsqueezed = combined.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(combined_unsqueezed, combined_unsqueezed, combined_unsqueezed)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Residual connection
        residual = self.residual_transform(combined)
        combined = attended + residual
        
        # Final classification
        output = self.classifier(combined)
        return output

# Enhanced Clinical Safety Loss
class AdvancedClinicalSafetyLoss(nn.Module):
    """
    Advanced clinical safety loss with focal loss components
    """
    def __init__(self, class_weights, critical_miss_penalty=50.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.critical_miss_penalty = critical_miss_penalty
        self.alpha = alpha
        self.gamma = gamma
        
        # Enhanced penalty matrix
        self.penalty_matrix = torch.tensor([
            [0.0,  2.0,  5.0],   # Green misclassified
            [8.0,  0.0,  3.0],   # Yellow misclassified  
            [50.0, 20.0, 0.0]    # Red misclassified (MOST DANGEROUS)
        ])
    
    def forward(self, outputs, targets):
        # Standard weighted cross-entropy
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)
        
        # Focal loss component
        ce_loss_raw = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        pt = torch.exp(-ce_loss_raw)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss_raw
        focal_loss = focal_loss.mean()
        
        # Clinical safety penalty
        pred_classes = torch.argmax(outputs, dim=1)
        batch_size = targets.size(0)
        
        safety_penalty = 0.0
        for i in range(batch_size):
            true_class = targets[i].item()
            pred_class = pred_classes[i].item()
            if true_class < len(self.penalty_matrix) and pred_class < len(self.penalty_matrix[0]):
                safety_penalty += self.penalty_matrix[true_class][pred_class]
        
        safety_penalty = safety_penalty / batch_size
        
        # Extra penalty for critical misses
        critical_mask = (targets == 2)
        if critical_mask.sum() > 0:
            critical_preds = pred_classes[critical_mask]
            critical_misses = (critical_preds != 2).sum().float()
            critical_penalty = (critical_misses / critical_mask.sum()) * self.critical_miss_penalty
        else:
            critical_penalty = 0.0
        
        # Combine losses
        total_loss = ce_loss + 0.3 * focal_loss + 0.4 * safety_penalty + 0.6 * critical_penalty
        
        return total_loss

# Enhanced Clinical Metrics (same as before)
class EnhancedClinicalMetrics:
    """Enhanced metrics focused on clinical safety"""
    
    @staticmethod
    def calculate_comprehensive_clinical_metrics(y_true, y_pred, class_names=None):
        if class_names is None:
            class_names = ['Green', 'Yellow', 'Red']
        
        metrics = {}
        
        # Standard classification metrics
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        n_classes = len(class_names)
        if len(precision) < n_classes:
            precision = np.pad(precision, (0, n_classes - len(precision)), 'constant')
            recall = np.pad(recall, (0, n_classes - len(recall)), 'constant')
            f1 = np.pad(f1, (0, n_classes - len(f1)), 'constant')
        
        metrics['class_metrics'] = {}
        for i, class_name in enumerate(class_names):
            metrics['class_metrics'][class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0
            }
        
        # Clinical safety metrics
        safety_metrics = EnhancedClinicalMetrics._calculate_clinical_safety(y_true, y_pred)
        metrics['clinical_safety'] = safety_metrics
        
        return metrics
    
    @staticmethod
    def _calculate_clinical_safety(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Under-triage: Predicting lower acuity than actual
        under_triage = np.sum(y_pred < y_true)
        under_triage_rate = under_triage / len(y_true)
        
        # Over-triage: Predicting higher acuity than actual
        over_triage = np.sum(y_pred > y_true)
        over_triage_rate = over_triage / len(y_true)
        
        # Critical case analysis
        critical_cases = np.sum(y_true == 2)
        if critical_cases > 0:
            critical_detected = np.sum((y_true == 2) & (y_pred == 2))
            critical_sensitivity = critical_detected / critical_cases
            
            critical_missed = np.sum((y_true == 2) & (y_pred < 2))
            critical_miss_rate = critical_missed / critical_cases
            critical_under_triage_rate = critical_miss_rate
        else:
            critical_sensitivity = 0.0
            critical_miss_rate = 0.0
            critical_under_triage_rate = 0.0
        
        return {
            'under_triage_rate': under_triage_rate,
            'over_triage_rate': over_triage_rate,
            'critical_sensitivity': critical_sensitivity,
            'critical_under_triage_rate': critical_under_triage_rate,
            'total_critical_cases': int(critical_cases),
            'critical_correctly_identified': int(critical_detected) if critical_cases > 0 else 0
        }

def run_advanced_kaggle_enhanced_final_fix():
    """
    Advanced enhanced final fix using Kaggle hospital triage dataset
    """
    print("=== ADVANCED ENHANCED TRIAGE MODEL WITH KAGGLE DATASET ===")
    print("Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 1. Load and prepare Kaggle data
    print("\n1. Loading Kaggle dataset...")
    df = load_kaggle_triage_data()
    
    # 2. Advanced feature engineering
    print("\n2. Applying advanced feature engineering...")
    df_enhanced, vital_features, symptom_features, risk_features, context_features, lab_features, interaction_features = advanced_kaggle_feature_engineering(df)
    
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    print(f"Target distribution: {df_enhanced['esi_3class'].value_counts().sort_index()}")
    
    # 3. Prepare features and target
    all_features = vital_features + symptom_features + risk_features + context_features + lab_features + interaction_features
    
    # Ensure all features exist
    for feature in all_features:
        if feature not in df_enhanced.columns:
            df_enhanced[feature] = 0.0
    
    X = df_enhanced[all_features].fillna(0)
    y = df_enhanced['esi_3class'].values.astype(int)
    
    # Sample data for training (use larger subset)
    sample_size = min(100000, len(X))  # Use up to 100k samples
    if len(X) > sample_size:
        print(f"Sampling {sample_size} records for training...")
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[sample_idx]
        y = y[sample_idx]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Final target distribution: {np.bincount(y)}")
    
    # 4. Advanced preprocessing
    print("\n3. Advanced preprocessing...")
    
    # Outlier detection and removal
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_mask = iso_forest.fit_predict(X) == 1
    X = X[outlier_mask]
    y = y[outlier_mask]
    print(f"After outlier removal: {X.shape}")
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Data splits with stratification
    print("\n4. Preparing stratified data splits...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples, classes: {np.bincount(y_train)}")
    print(f"  Validation: {len(X_val)} samples, classes: {np.bincount(y_val)}")
    print(f"  Test: {len(X_test)} samples, classes: {np.bincount(y_test)}")
    
    # 6. Apply SMOTE for class balancing
    print("\n5. Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_balanced.shape}, classes: {np.bincount(y_train_balanced)}")
    
    # 7. Create data loaders with weighted sampling
    def create_advanced_hierarchical_tensors(X, y):
        """Create tensors for advanced hierarchical model"""
        X_vital = torch.tensor(X[vital_features].values, dtype=torch.float32)
        X_symptom = torch.tensor(X[symptom_features].values, dtype=torch.float32)
        X_risk = torch.tensor(X[risk_features].values, dtype=torch.float32)
        X_context = torch.tensor(X[context_features].values, dtype=torch.float32)
        X_lab = torch.tensor(X[lab_features].values, dtype=torch.float32)
        X_interaction = torch.tensor(X[interaction_features].values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_vital, X_symptom, X_risk, X_context, X_lab, X_interaction, y_tensor

    # Create tensors
    train_vital, train_symptom, train_risk, train_context, train_lab, train_interaction, train_y = create_advanced_hierarchical_tensors(pd.DataFrame(X_train_balanced, columns=X_train.columns), y_train_balanced)
    val_vital, val_symptom, val_risk, val_context, val_lab, val_interaction, val_y = create_advanced_hierarchical_tensors(X_val, y_val)
    test_vital, test_symptom, test_risk, test_context, test_lab, test_interaction, test_y = create_advanced_hierarchical_tensors(X_test, y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_vital, train_symptom, train_risk, train_context, train_lab, train_interaction, train_y)
    val_dataset = TensorDataset(val_vital, val_symptom, val_risk, val_context, val_lab, val_interaction, val_y)
    test_dataset = TensorDataset(test_vital, test_symptom, test_risk, test_context, test_lab, test_interaction, test_y)
    
    # Weighted sampling for training
    class_counts = np.bincount(y_train_balanced)
    class_weights_sampling = 1.0 / class_counts
    sample_weights = class_weights_sampling[y_train_balanced]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 8. Initialize advanced model
    print("\n6. Initializing advanced hierarchical model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AdvancedHierarchicalTriageModel(
        num_vital_features=len(vital_features),
        num_symptom_features=len(symptom_features),
        num_risk_features=len(risk_features),
        num_context_features=len(context_features),
        num_lab_features=len(lab_features),
        num_interaction_features=len(interaction_features),
        num_classes=3
    ).to(device)
    
    # 9. Advanced training
    print("\n7. Training advanced model...")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Setup advanced training
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = AdvancedClinicalSafetyLoss(class_weights_tensor, critical_miss_penalty=50.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop with early stopping
    best_critical_sensitivity = 0
    best_model_state = None
    epochs = 50  # More epochs for better training
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    critical_sensitivities = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data, targets in train_loader:
            vital_data = vital_data.to(device)
            symptom_data = symptom_data.to(device)
            risk_data = risk_data.to(device)
            context_data = context_data.to(device)
            lab_data = lab_data.to(device)
            interaction_data = interaction_data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data, targets in val_loader:
                vital_data = vital_data.to(device)
                symptom_data = symptom_data.to(device)
                risk_data = risk_data.to(device)
                context_data = context_data.to(device)
                lab_data = lab_data.to(device)
                interaction_data = interaction_data.to(device)
                targets = targets.to(device)
                
                outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_loss / len(train_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        # Calculate critical sensitivity
        critical_cases = np.sum(np.array(all_targets) == 2)
        if critical_cases > 0:
            critical_detected = np.sum((np.array(all_targets) == 2) & (np.array(all_preds) == 2))
            critical_sens = critical_detected / critical_cases
        else:
            critical_sens = 0.0
        
        # Store metrics
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        critical_sensitivities.append(critical_sens)
        
        # Save best model
        if critical_sens > best_critical_sensitivity:
            best_critical_sensitivity = critical_sens
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.3f}, "
              f"Critical Sens: {critical_sens:.3f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with critical sensitivity: {best_critical_sensitivity:.3f}")
    
    # 10. Final evaluation
    print("\n8. Running final evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data, targets in test_loader:
            start_time = time.time()
            
            vital_data = vital_data.to(device)
            symptom_data = symptom_data.to(device)
            risk_data = risk_data.to(device)
            context_data = context_data.to(device)
            lab_data = lab_data.to(device)
            interaction_data = interaction_data.to(device)
            targets = targets.to(device)
            
            outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data, interaction_data)
            _, predicted = torch.max(outputs, 1)
            
            inference_time = (time.time() - start_time) * 1000 / len(targets)  # ms per sample
            inference_times.append(inference_time)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 11. Calculate comprehensive metrics
    print("\n9. Calculating comprehensive metrics...")
    
    metrics = EnhancedClinicalMetrics.calculate_comprehensive_clinical_metrics(
        all_targets, all_predictions, class_names=['Green', 'Yellow', 'Red']
    )
    
    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    throughput = 1000 / avg_inference_time  # samples per second
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # MB
    total_params = sum(p.numel() for p in model.parameters())
    
    # 12. Generate comprehensive report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'Kaggle Hospital Triage Dataset (Advanced)',
        'model_type': 'AdvancedHierarchicalClinicalTriageModel',
        'clinical_metrics': metrics,
        'performance_metrics': {
            'avg_inference_time_ms': avg_inference_time,
            'throughput_samples_per_sec': throughput,
            'model_size_mb': model_size,
            'total_parameters': total_params,
            'total_samples_tested': len(all_targets)
        },
        'target_comparison': {
            'targets': {
                'overall_accuracy': 0.75,
                'under_triage_rate': 0.15,
                'critical_sensitivity': 0.95
            },
            'actual_metrics': {
                'overall_accuracy': float(metrics['overall_accuracy']),
                'under_triage_rate': float(metrics['clinical_safety']['under_triage_rate']),
                'critical_sensitivity': float(metrics['clinical_safety']['critical_sensitivity'])
            },
            'all_targets_met': bool(
                float(metrics['overall_accuracy']) >= 0.75 and
                float(metrics['clinical_safety']['under_triage_rate']) <= 0.15 and
                float(metrics['clinical_safety']['critical_sensitivity']) >= 0.95
            )
        },
        'model_info': {
            'architecture': 'AdvancedHierarchicalClinicalTriageModel',
            'total_parameters': total_params,
            'model_size_mb': model_size,
            'dataset_size': len(X),
            'training_samples': len(X_train_balanced),
            'feature_groups': {
                'vital_signs': len(vital_features),
                'symptoms': len(symptom_features),
                'risk_factors': len(risk_features),
                'context': len(context_features),
                'lab_values': len(lab_features),
                'interactions': len(interaction_features)
            }
        },
        'training_info': {
            'epochs_trained': len(train_losses),
            'best_critical_sensitivity': best_critical_sensitivity,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
            'used_smote': True,
            'used_outlier_removal': True,
            'used_weighted_sampling': True
        }
    }
    
    # Assessment
    accuracy_met = metrics['overall_accuracy'] >= 0.75
    under_triage_met = metrics['clinical_safety']['under_triage_rate'] <= 0.15
    critical_sens_met = metrics['clinical_safety']['critical_sensitivity'] >= 0.95
    
    if accuracy_met and under_triage_met and critical_sens_met:
        assessment = "✅ ALL TARGETS MET - Ready for deployment"
        risk_level = "Low"
    elif critical_sens_met and accuracy_met:
        assessment = "⚠️ MOSTLY SUCCESSFUL - Minor optimization needed"
        risk_level = "Medium"
    elif critical_sens_met:
        assessment = "⚠️ CRITICAL SAFETY MET - Continue optimization"
        risk_level = "Medium"
    else:
        assessment = "❌ NEEDS IMPROVEMENT - Continue optimization"
        risk_level = "High"
    
    report['summary'] = {
        'assessment': assessment,
        'risk_level': risk_level,
        'all_targets_met': bool(report['target_comparison']['all_targets_met']),
        'key_findings': [
            f"Overall accuracy: {float(metrics['overall_accuracy']):.3f}",
            f"Critical sensitivity: {float(metrics['clinical_safety']['critical_sensitivity']):.3f}",
            f"Under-triage rate: {float(metrics['clinical_safety']['under_triage_rate']):.3f}",
            f"Dataset size: {len(X):,} samples",
            f"Features: {len(all_features)} (vs 972 original)",
            f"Training samples: {len(X_train_balanced):,} (with SMOTE)"
        ]
    }
    
    # Save results
    model_path = f'results/kaggle_advanced_hierarchical_model_{timestamp}.pth'
    report_path = f'results/kaggle_advanced_enhanced_report_{timestamp}.json'
    
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 13. Print results
    print("\n" + "="*80)
    print("ADVANCED ENHANCED TRIAGE MODEL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f} (Target: 0.75)")
    print(f"Under-triage Rate: {metrics['clinical_safety']['under_triage_rate']:.3f} (Target: <0.15)")
    print(f"Critical Sensitivity: {metrics['clinical_safety']['critical_sensitivity']:.3f} (Target: >0.95)")
    print(f"Critical Cases Detected: {metrics['clinical_safety']['critical_correctly_identified']}/{metrics['clinical_safety']['total_critical_cases']}")
    print()
    print("Per-Class Performance:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}: Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1_score']:.3f}")
    print()
    print(f"Model Performance:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model Size: {model_size:.2f} MB")
    print(f"  Inference Time: {avg_inference_time:.2f} ms/sample")
    print(f"  Throughput: {throughput:.0f} samples/second")
    print()
    print(f"Assessment: {assessment}")
    print(f"Risk Level: {risk_level}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print("="*80)
    
    return model, metrics, report

if __name__ == "__main__":
    model, metrics, report = run_advanced_kaggle_enhanced_final_fix()