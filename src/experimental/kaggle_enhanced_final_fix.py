import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import json
import os
import kagglehub

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

# Enhanced feature engineering for Kaggle dataset
def enhanced_kaggle_feature_engineering(df):
    """
    Enhanced feature engineering specifically for Kaggle triage dataset
    """
    print("Applying enhanced feature engineering for Kaggle dataset...")
    
    # 1. Target variable: ESI (Emergency Severity Index)
    # ESI scale: 1 (most urgent) to 5 (least urgent)
    # Convert to our 3-class system: 1-2 -> Red, 3 -> Yellow, 4-5 -> Green
    df['esi_3class'] = df['esi'].map({
        1: 2,  # Red (Critical)
        2: 2,  # Red (Critical) 
        3: 1,  # Yellow (Urgent)
        4: 0,  # Green (Non-urgent)
        5: 0   # Green (Non-urgent)
    })
    
    # 2. Vital Signs Features (Primary pathway)
    vital_features = []
    
    # Triage vital signs
    if 'triage_vital_hr' in df.columns:
        df['hr'] = pd.to_numeric(df['triage_vital_hr'], errors='coerce')
        vital_features.append('hr')
    
    if 'triage_vital_sbp' in df.columns:
        df['sbp'] = pd.to_numeric(df['triage_vital_sbp'], errors='coerce')
        vital_features.append('sbp')
    
    if 'triage_vital_dbp' in df.columns:
        df['dbp'] = pd.to_numeric(df['triage_vital_dbp'], errors='coerce')
        vital_features.append('dbp')
    
    if 'triage_vital_rr' in df.columns:
        df['rr'] = pd.to_numeric(df['triage_vital_rr'], errors='coerce')
        vital_features.append('rr')
    
    if 'triage_vital_o2' in df.columns:
        df['spo2'] = pd.to_numeric(df['triage_vital_o2'], errors='coerce')
        vital_features.append('spo2')
    
    if 'triage_vital_temp' in df.columns:
        df['temp'] = pd.to_numeric(df['triage_vital_temp'], errors='coerce')
        vital_features.append('temp')
    
    # Age
    df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce')
    vital_features.append('age_numeric')
    
    # 3. Vital Signs Severity Score (0-15 scale)
    def calculate_vital_severity_kaggle(row):
        score = 0
        
        # Heart rate severity
        if pd.notna(row.get('hr')):
            hr = row['hr']
            if hr > 120 or hr < 50:
                score += 3
            elif hr > 100 or hr < 60:
                score += 1
        
        # Blood pressure severity
        if pd.notna(row.get('sbp')):
            sbp = row['sbp']
            if sbp > 180 or sbp < 90:
                score += 3
            elif sbp > 160 or sbp < 100:
                score += 2
        
        # Respiratory rate severity
        if pd.notna(row.get('rr')):
            rr = row['rr']
            if rr > 30 or rr < 8:
                score += 3
            elif rr > 24 or rr < 12:
                score += 2
        
        # Temperature severity (convert from Fahrenheit)
        if pd.notna(row.get('temp')):
            temp_f = row['temp']
            temp_c = (temp_f - 32) * 5/9
            if temp_c > 39.5 or temp_c < 35:
                score += 3
            elif temp_c > 38.5:
                score += 2
        
        # Oxygen saturation severity
        if pd.notna(row.get('spo2')):
            spo2 = row['spo2']
            if spo2 < 85:
                score += 3
            elif spo2 < 90:
                score += 2
            elif spo2 < 95:
                score += 1
        
        return score
    
    df['vital_severity_score'] = df.apply(calculate_vital_severity_kaggle, axis=1)
    vital_features.append('vital_severity_score')
    
    # 4. Chief Complaint Clustering (Secondary pathway)
    symptom_features = []
    
    # Cardiac symptoms
    cardiac_cc = [col for col in df.columns if col.startswith('cc_') and any(term in col for term in ['chest', 'cardiac', 'heart', 'palpitation'])]
    if cardiac_cc:
        df['cardiac_symptom_score'] = df[cardiac_cc].sum(axis=1)
        symptom_features.append('cardiac_symptom_score')
    
    # Respiratory symptoms
    respiratory_cc = [col for col in df.columns if col.startswith('cc_') and any(term in col for term in ['breath', 'dyspnea', 'respiratory', 'cough', 'wheez'])]
    if respiratory_cc:
        df['respiratory_symptom_score'] = df[respiratory_cc].sum(axis=1)
        symptom_features.append('respiratory_symptom_score')
    
    # Neurological symptoms
    neuro_cc = [col for col in df.columns if col.startswith('cc_') and any(term in col for term in ['head', 'neuro', 'seizure', 'stroke', 'confusion', 'consciousness'])]
    if neuro_cc:
        df['neurological_symptom_score'] = df[neuro_cc].sum(axis=1)
        symptom_features.append('neurological_symptom_score')
    
    # Trauma symptoms
    trauma_cc = [col for col in df.columns if col.startswith('cc_') and any(term in col for term in ['trauma', 'injury', 'fall', 'crash', 'burn', 'laceration'])]
    if trauma_cc:
        df['trauma_symptom_score'] = df[trauma_cc].sum(axis=1)
        symptom_features.append('trauma_symptom_score')
    
    # Pain symptoms
    pain_cc = [col for col in df.columns if col.startswith('cc_') and 'pain' in col]
    if pain_cc:
        df['pain_symptom_score'] = df[pain_cc].sum(axis=1)
        symptom_features.append('pain_symptom_score')
    
    # 5. Clinical Risk Factors (Tertiary pathway)
    risk_features = []
    
    # Age risk
    df['age_risk_factor'] = np.where(df['age_numeric'] > 65, 2,
                           np.where(df['age_numeric'] < 18, 1, 0))
    risk_features.append('age_risk_factor')
    
    # Comorbidity count (from diagnosis columns)
    diagnosis_cols = [col for col in df.columns if not col.startswith(('cc_', 'meds_', 'triage_', 'n_', 'dep_')) 
                     and col not in ['esi', 'age', 'gender', 'ethnicity', 'race', 'disposition', 'arrivalmode']]
    
    # Count active diagnoses
    df['comorbidity_count'] = 0
    for col in diagnosis_cols[:50]:  # Limit to first 50 to avoid memory issues
        if col in df.columns:
            df['comorbidity_count'] += pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    risk_features.append('comorbidity_count')
    
    # 6. Demographics & Temporal Features (Quaternary pathway)
    context_features = []
    
    # Gender encoding
    if 'gender' in df.columns:
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
        df = pd.concat([df, gender_dummies], axis=1)
        context_features.extend([col for col in gender_dummies.columns])
    
    # Arrival time features
    if 'arrivalmonth' in df.columns:
        df['arrival_month'] = pd.to_numeric(df['arrivalmonth'], errors='coerce').fillna(6)
        context_features.append('arrival_month')
    
    if 'arrivalday' in df.columns:
        df['arrival_day'] = pd.to_numeric(df['arrivalday'], errors='coerce').fillna(1)
        context_features.append('arrival_day')
    
    if 'arrivalhour_bin' in df.columns:
        df['arrival_hour'] = pd.to_numeric(df['arrivalhour_bin'], errors='coerce').fillna(12)
        context_features.append('arrival_hour')
    
    # 7. Lab Values Summary (Additional clinical context)
    lab_features = []
    
    # Key lab values that are commonly available
    key_labs = ['glucose_last', 'creatinine_last', 'bun_last', 'hemoglobin_last', 'wbc_last']
    for lab in key_labs:
        if lab in df.columns:
            df[f'{lab}_norm'] = pd.to_numeric(df[lab], errors='coerce')
            lab_features.append(f'{lab}_norm')
    
    # Lab abnormality score
    if lab_features:
        # Simple abnormality scoring (this would need clinical reference ranges)
        df['lab_abnormality_score'] = df[lab_features].isnull().sum(axis=1)  # Count missing labs as proxy
        lab_features.append('lab_abnormality_score')
    
    # 8. Clean and prepare final feature sets
    all_features = vital_features + symptom_features + risk_features + context_features + lab_features
    
    # Ensure all features are numeric
    for feature in all_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
    
    # Remove rows with missing target
    df = df.dropna(subset=['esi_3class'])
    
    print(f"Enhanced feature engineering complete.")
    print(f"Feature groups:")
    print(f"  Vital signs: {len(vital_features)} features")
    print(f"  Symptoms: {len(symptom_features)} features") 
    print(f"  Risk factors: {len(risk_features)} features")
    print(f"  Context: {len(context_features)} features")
    print(f"  Lab values: {len(lab_features)} features")
    print(f"  Total: {len(all_features)} features")
    
    return df, vital_features, symptom_features, risk_features, context_features, lab_features

# Use the same hierarchical model and training classes from enhanced_final_fix.py
class HierarchicalClinicalTriageModel(nn.Module):
    """
    Hierarchical model that prioritizes vital signs over other features
    """
    def __init__(self, num_vital_features=8, num_symptom_features=5, 
                 num_risk_features=2, num_context_features=6, 
                 num_lab_features=6, num_classes=3):
        super().__init__()
        
        # Primary pathway: Vital Signs (most important)
        self.vital_signs_pathway = nn.Sequential(
            nn.Linear(num_vital_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Secondary pathway: Symptom Clusters
        self.symptom_pathway = nn.Sequential(
            nn.Linear(num_symptom_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        # Tertiary pathway: Risk Factors
        self.risk_pathway = nn.Sequential(
            nn.Linear(num_risk_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Quaternary pathway: Context & Demographics
        self.context_pathway = nn.Sequential(
            nn.Linear(num_context_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Lab pathway: Laboratory values
        self.lab_pathway = nn.Sequential(
            nn.Linear(num_lab_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Attention mechanism for pathway fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=72, num_heads=6, batch_first=True  # 32+16+8+8+8 = 72
        )
        
        # Final classification with clinical hierarchy
        self.classifier = nn.Sequential(
            nn.Linear(72, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, vital_signs, symptoms, risk_factors, context, lab_values):
        # Process each pathway
        vital_features = self.vital_signs_pathway(vital_signs)      # 32 features
        symptom_features = self.symptom_pathway(symptoms)          # 16 features
        risk_features = self.risk_pathway(risk_factors)            # 8 features
        context_features = self.context_pathway(context)           # 8 features
        lab_features = self.lab_pathway(lab_values)                # 8 features
        
        # Combine features
        combined = torch.cat([vital_features, symptom_features, 
                             risk_features, context_features, lab_features], dim=1)  # 72 features
        
        # Apply attention to focus on most relevant features
        attended, _ = self.attention(
            combined.unsqueeze(1), 
            combined.unsqueeze(1), 
            combined.unsqueeze(1)
        )
        
        # Final classification
        output = self.classifier(attended.squeeze(1))
        return output

# Clinical Safety Loss Function
class ClinicalSafetyLoss(nn.Module):
    """
    Loss function that heavily penalizes under-triage (missing critical cases)
    """
    def __init__(self, class_weights=None, under_triage_penalty=10.0, 
                 critical_miss_penalty=20.0):
        super().__init__()
        self.class_weights = class_weights
        self.under_triage_penalty = under_triage_penalty
        self.critical_miss_penalty = critical_miss_penalty
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Clinical error penalty matrix
        # [Green, Yellow, Red]
        self.penalty_matrix = torch.tensor([
            [0.0,  1.0,  2.0],  # Green misclassified
            [5.0,  0.0,  1.0],  # Yellow misclassified  
            [20.0, 10.0, 0.0]   # Red misclassified (most dangerous)
        ])
    
    def forward(self, outputs, targets):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(outputs, targets)
        
        # Get predictions
        pred_classes = torch.argmax(outputs, dim=1)
        
        # Calculate clinical safety penalty
        safety_penalty = 0.0
        batch_size = targets.size(0)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            pred_class = pred_classes[i].item()
            
            # Add penalty based on clinical error severity
            if true_class < len(self.penalty_matrix) and pred_class < len(self.penalty_matrix[0]):
                safety_penalty += self.penalty_matrix[true_class, pred_class]
        
        safety_penalty = safety_penalty / batch_size
        
        # Extra penalty for missing critical cases (Red -> Green/Yellow)
        critical_mask = (targets == 2)  # Red cases
        if critical_mask.sum() > 0:
            critical_preds = pred_classes[critical_mask]
            
            # Heavy penalty for critical misses
            critical_misses = (critical_preds != 2).sum().float()
            critical_penalty = (critical_misses / critical_mask.sum()) * self.critical_miss_penalty
        else:
            critical_penalty = 0.0
        
        # Combine losses
        total_loss = ce_loss + 0.3 * safety_penalty + 0.5 * critical_penalty
        
        return total_loss

# Enhanced Clinical Metrics
class EnhancedClinicalMetrics:
    """
    Enhanced metrics focused on clinical safety
    """
    
    @staticmethod
    def calculate_comprehensive_clinical_metrics(y_true, y_pred, class_names=None):
        """
        Calculate comprehensive clinical metrics
        """
        if class_names is None:
            class_names = ['Green', 'Yellow', 'Red']
        
        metrics = {}
        
        # 1. Standard classification metrics
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Ensure we have metrics for all classes
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
        
        # 3. Clinical safety metrics
        safety_metrics = EnhancedClinicalMetrics._calculate_clinical_safety(y_true, y_pred)
        metrics['clinical_safety'] = safety_metrics
        
        return metrics
    
    @staticmethod
    def _calculate_clinical_safety(y_true, y_pred):
        """Calculate clinical safety metrics"""
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

def run_kaggle_enhanced_final_fix():
    """
    Enhanced final fix using Kaggle hospital triage dataset
    """
    print("=== ENHANCED TRIAGE MODEL WITH KAGGLE DATASET ===")
    print("Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 1. Load and prepare Kaggle data
    print("\n1. Loading Kaggle dataset...")
    df = load_kaggle_triage_data()
    
    # 2. Enhanced feature engineering
    print("\n2. Applying enhanced feature engineering...")
    df_enhanced, vital_features, symptom_features, risk_features, context_features, lab_features = enhanced_kaggle_feature_engineering(df)
    
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    print(f"Target distribution: {df_enhanced['esi_3class'].value_counts().sort_index()}")
    
    # 3. Prepare features and target
    all_features = vital_features + symptom_features + risk_features + context_features + lab_features
    
    # Ensure all features exist
    for feature in all_features:
        if feature not in df_enhanced.columns:
            df_enhanced[feature] = 0.0
    
    X = df_enhanced[all_features].fillna(0)
    y = df_enhanced['esi_3class'].values
    
    # Ensure y is integer type
    y = y.astype(int)
    
    # Sample data for faster training (use subset for demonstration)
    sample_size = min(50000, len(X))  # Use up to 50k samples
    if len(X) > sample_size:
        print(f"Sampling {sample_size} records for faster training...")
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[sample_idx]
        y = y[sample_idx]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Final target distribution: {np.bincount(y)}")
    
    # 4. Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Data splits
    print("\n4. Preparing data splits...")
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
    
    # 6. Create data loaders
    def create_hierarchical_tensors_kaggle(X, y):
        """Create tensors for hierarchical model"""
        X_vital = torch.tensor(X[vital_features].values, dtype=torch.float32)
        X_symptom = torch.tensor(X[symptom_features].values, dtype=torch.float32)
        X_risk = torch.tensor(X[risk_features].values, dtype=torch.float32)
        X_context = torch.tensor(X[context_features].values, dtype=torch.float32)
        X_lab = torch.tensor(X[lab_features].values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_vital, X_symptom, X_risk, X_context, X_lab, y_tensor
    
    # Create tensors
    train_vital, train_symptom, train_risk, train_context, train_lab, train_y = create_hierarchical_tensors_kaggle(X_train, y_train)
    val_vital, val_symptom, val_risk, val_context, val_lab, val_y = create_hierarchical_tensors_kaggle(X_val, y_val)
    test_vital, test_symptom, test_risk, test_context, test_lab, test_y = create_hierarchical_tensors_kaggle(X_test, y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_vital, train_symptom, train_risk, train_context, train_lab, train_y)
    val_dataset = TensorDataset(val_vital, val_symptom, val_risk, val_context, val_lab, val_y)
    test_dataset = TensorDataset(test_vital, test_symptom, test_risk, test_context, test_lab, test_y)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Initialize model
    print("\n5. Initializing hierarchical model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HierarchicalClinicalTriageModel(
        num_vital_features=len(vital_features),
        num_symptom_features=len(symptom_features),
        num_risk_features=len(risk_features),
        num_context_features=len(context_features),
        num_lab_features=len(lab_features),
        num_classes=3
    ).to(device)
    
    # 8. Training
    print("\n6. Training model...")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = ClinicalSafetyLoss(class_weights_tensor, critical_miss_penalty=25.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    # Training loop
    best_critical_sensitivity = 0
    best_model_state = None
    epochs = 20  # Reduced for faster training
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for vital_data, symptom_data, risk_data, context_data, lab_data, targets in train_loader:
            vital_data = vital_data.to(device)
            symptom_data = symptom_data.to(device)
            risk_data = risk_data.to(device)
            context_data = context_data.to(device)
            lab_data = lab_data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data)
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
            for vital_data, symptom_data, risk_data, context_data, lab_data, targets in val_loader:
                vital_data = vital_data.to(device)
                symptom_data = symptom_data.to(device)
                risk_data = risk_data.to(device)
                context_data = context_data.to(device)
                lab_data = lab_data.to(device)
                targets = targets.to(device)
                
                outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data)
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
        
        # Save best model
        if critical_sens > best_critical_sensitivity:
            best_critical_sensitivity = critical_sens
            best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step(critical_sens)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.3f}, "
              f"Critical Sens: {critical_sens:.3f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with critical sensitivity: {best_critical_sensitivity:.3f}")
    
    # 9. Final evaluation
    print("\n7. Running final evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for vital_data, symptom_data, risk_data, context_data, lab_data, targets in test_loader:
            vital_data = vital_data.to(device)
            symptom_data = symptom_data.to(device)
            risk_data = risk_data.to(device)
            context_data = context_data.to(device)
            lab_data = lab_data.to(device)
            targets = targets.to(device)
            
            start_time = time.time()
            outputs = model(vital_data, symptom_data, risk_data, context_data, lab_data)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate enhanced metrics
    enhanced_metrics = EnhancedClinicalMetrics.calculate_comprehensive_clinical_metrics(
        all_targets, all_predictions, class_names=['Green', 'Yellow', 'Red']
    )
    
    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    throughput = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())
    
    performance_metrics = {
        'avg_inference_time_ms': avg_inference_time,
        'throughput_samples_per_sec': throughput,
        'model_size_mb': model_size,
        'total_parameters': total_params,
        'total_samples_tested': len(all_targets)
    }
    
    # Print results
    print("\n=== KAGGLE ENHANCED EVALUATION RESULTS ===")
    
    print(f"\nClinical Performance:")
    print(f"  Overall Accuracy: {enhanced_metrics['overall_accuracy']:.3f}")
    
    for class_name, metrics in enhanced_metrics['class_metrics'].items():
        print(f"  {class_name}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    safety = enhanced_metrics['clinical_safety']
    print(f"\nClinical Safety:")
    print(f"  Under-triage Rate: {safety['under_triage_rate']:.3f}")
    print(f"  Over-triage Rate: {safety['over_triage_rate']:.3f}")
    print(f"  Critical Under-triage Rate: {safety['critical_under_triage_rate']:.3f}")
    print(f"  Critical Sensitivity: {safety['critical_sensitivity']:.3f}")
    print(f"  Total Critical Cases: {safety['total_critical_cases']}")
    print(f"  Critical Cases Correctly Identified: {safety['critical_correctly_identified']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Avg Inference Time: {performance_metrics['avg_inference_time_ms']:.2f}ms")
    print(f"  Throughput: {performance_metrics['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Model Size: {performance_metrics['model_size_mb']:.2f}MB")
    print(f"  Total Parameters: {performance_metrics['total_parameters']:,}")
    
    # Compare against targets
    print(f"\n=== TARGET COMPARISON ===")
    targets = {
        'overall_accuracy': 0.75,
        'under_triage_rate': 0.15,
        'critical_sensitivity': 0.95
    }
    
    actual_metrics = {
        'overall_accuracy': enhanced_metrics['overall_accuracy'],
        'under_triage_rate': safety['under_triage_rate'],
        'critical_sensitivity': safety['critical_sensitivity']
    }
    
    all_targets_met = True
    for metric, target in targets.items():
        actual = actual_metrics[metric]
        
        if metric == 'under_triage_rate':
            meets_target = actual <= target
            status = "✅ PASS" if meets_target else "❌ FAIL"
            gap = target - actual if meets_target else actual - target
        else:
            meets_target = actual >= target
            status = "✅ PASS" if meets_target else "❌ FAIL"
            gap = actual - target if meets_target else target - actual
        
        print(f"{metric}:")
        print(f"  Target: {target:.3f}")
        print(f"  Actual: {actual:.3f}")
        print(f"  Status: {status}")
        print(f"  Gap: {gap:.3f}")
        print()
        
        if not meets_target:
            all_targets_met = False
    
    # Save results
    print("\n8. Saving results...")
    os.makedirs('results', exist_ok=True)
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'Kaggle Hospital Triage Dataset',
        'model_type': 'HierarchicalClinicalTriageModel',
        'clinical_metrics': enhanced_metrics,
        'performance_metrics': performance_metrics,
        'target_comparison': {
            'targets': targets,
            'actual_metrics': actual_metrics,
            'all_targets_met': all_targets_met
        },
        'model_info': {
            'architecture': 'HierarchicalClinicalTriageModel',
            'total_parameters': performance_metrics['total_parameters'],
            'model_size_mb': performance_metrics['model_size_mb'],
            'dataset_size': len(y),
            'feature_groups': {
                'vital_signs': len(vital_features),
                'symptoms': len(symptom_features),
                'risk_factors': len(risk_features),
                'context': len(context_features),
                'lab_values': len(lab_features)
            }
        }
    }
    
    # Assessment
    if all_targets_met:
        assessment = "✅ ALL TARGETS MET - Ready for deployment"
        risk_level = "Low"
    elif actual_metrics['critical_sensitivity'] >= 0.9 and actual_metrics['under_triage_rate'] <= 0.2:
        assessment = "⚠️  MOST TARGETS MET - Good performance"
        risk_level = "Medium"
    else:
        assessment = "❌ NEEDS IMPROVEMENT - Continue optimization"
        risk_level = "High"
    
    final_results['summary'] = {
        'assessment': assessment,
        'risk_level': risk_level,
        'all_targets_met': all_targets_met,
        'key_findings': [
            f"Overall accuracy: {actual_metrics['overall_accuracy']:.3f}",
            f"Critical sensitivity: {actual_metrics['critical_sensitivity']:.3f}",
            f"Under-triage rate: {actual_metrics['under_triage_rate']:.3f}",
            f"Dataset size: {len(y):,} samples",
            f"Features: {len(all_features)} (vs 972 original)"
        ]
    }
    
    # Save report
    report_path = f"results/kaggle_enhanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Results saved to: {report_path}")
    
    # Save model
    model_path = f"results/kaggle_hierarchical_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_vital_features': len(vital_features),
            'num_symptom_features': len(symptom_features),
            'num_risk_features': len(risk_features),
            'num_context_features': len(context_features),
            'num_lab_features': len(lab_features),
            'num_classes': 3
        },
        'scaler': scaler,
        'feature_names': {
            'vital_features': vital_features,
            'symptom_features': symptom_features,
            'risk_features': risk_features,
            'context_features': context_features,
            'lab_features': lab_features
        },
        'final_results': final_results
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Final assessment
    print("\n=== KAGGLE ENHANCED FINAL ASSESSMENT ===")
    print(f"Assessment: {assessment}")
    print(f"Risk Level: {risk_level}")
    
    print(f"\nKey Achievements:")
    print(f"  • Processed {len(y):,} samples from Kaggle dataset")
    print(f"  • Reduced features from 972 → {len(all_features)}")
    print(f"  • Hierarchical model with {performance_metrics['total_parameters']:,} parameters")
    print(f"  • Clinical safety-focused training")
    print(f"  • Enhanced evaluation metrics")
    
    return final_results, model

if __name__ == "__main__":
    results, model = run_kaggle_enhanced_final_fix()