import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import json
import os

try:
    from ..data_preparation import load_and_clean_data
except ImportError:
    from data_preparation import load_and_clean_data

# PHASE 1: Enhanced Feature Engineering
def calculate_vital_severity_score(df):
    """Calculate composite vital signs severity score (0-15 scale)"""
    
    # Blood pressure severity (0-3 scale)
    bp_systolic = df['sistolik kb']
    bp_severity = np.where(bp_systolic > 180, 3,
                  np.where(bp_systolic > 160, 2,
                  np.where(bp_systolic < 90, 2, 0)))
    
    # Heart rate severity (0-3 scale)
    hr = df['nabız']
    hr_severity = np.where(hr > 120, 3,
                  np.where(hr > 100, 1,
                  np.where(hr < 50, 3,
                  np.where(hr < 60, 1, 0))))
    
    # Respiratory rate severity (0-3 scale)
    rr = df['solunum sayısı']
    rr_severity = np.where(rr > 30, 3,
                  np.where(rr > 24, 2,
                  np.where(rr < 8, 3,
                  np.where(rr < 12, 1, 0))))
    
    # Temperature severity (0-3 scale)
    temp = df['ateş']
    temp_severity = np.where(temp > 39.5, 3,
                    np.where(temp > 38.5, 2,
                    np.where(temp < 35, 3, 0)))
    
    # Oxygen saturation severity (0-3 scale)
    spo2 = df['saturasyon']
    spo2_severity = np.where(spo2 < 85, 3,
                    np.where(spo2 < 90, 2,
                    np.where(spo2 < 95, 1, 0)))
    
    # Composite severity score (0-15 scale)
    total_severity = bp_severity + hr_severity + rr_severity + temp_severity + spo2_severity
    
    return total_severity

def create_symptom_clusters(df):
    """Group symptoms into clinically meaningful clusters"""
    
    # Initialize cluster scores as numeric
    df['cardiac_symptom_score'] = 0.0
    df['respiratory_symptom_score'] = 0.0
    df['neurological_symptom_score'] = 0.0
    df['trauma_symptom_score'] = 0.0
    
    # Define symptom clusters based on available columns
    cardiac_keywords = ['göğüs ağrısı', 'kardiyoloji', 'kalp', 'chest pain', 'cardiac']
    respiratory_keywords = ['göğüs hastalıkları', 'solunum', 'nefes', 'respiratory', 'breathing']
    neurological_keywords = ['nörolojik hastalıklar', 'beyin cerrahi', 'baş ağrısı', 'neurological', 'headache']
    trauma_keywords = ['travma_', 'trauma', 'injury']
    
    # Score based on text content in relevant columns
    text_columns = ['semptomlar_non travma_genel 01', 'semptomlar_non travma_genel 02',
                   'göğüs ağrısı', 'kardiyoloji', 'göğüs hastalıkları', 'nörolojik hastalıklar', 'beyin cerrahi']
    
    for col in text_columns:
        if col in df.columns:
            # Convert to string and check for keywords
            col_str = df[col].astype(str).str.lower().fillna('')
            
            # Cardiac symptoms
            for keyword in cardiac_keywords:
                matches = col_str.str.contains(keyword, na=False).astype(int)
                df['cardiac_symptom_score'] = df['cardiac_symptom_score'].astype(float) + matches.astype(float)
            
            # Respiratory symptoms
            for keyword in respiratory_keywords:
                matches = col_str.str.contains(keyword, na=False).astype(int)
                df['respiratory_symptom_score'] = df['respiratory_symptom_score'].astype(float) + matches.astype(float)
            
            # Neurological symptoms
            for keyword in neurological_keywords:
                matches = col_str.str.contains(keyword, na=False).astype(int)
                df['neurological_symptom_score'] = df['neurological_symptom_score'].astype(float) + matches.astype(float)
    
    # Trauma symptoms from trauma columns
    trauma_cols = [col for col in df.columns if 'travma_' in col]
    if trauma_cols:
        # Ensure trauma columns are numeric
        for tcol in trauma_cols:
            if tcol in df.columns:
                # Convert trauma columns to numeric (1 if not empty/null, 0 otherwise)
                df[tcol] = pd.to_numeric(df[tcol], errors='coerce').fillna(0)
                df[tcol] = (df[tcol] > 0).astype(float)
        
        df['trauma_symptom_score'] = df[trauma_cols].sum(axis=1).astype(float)
    
    # Ensure all symptom scores are numeric
    df['cardiac_symptom_score'] = pd.to_numeric(df['cardiac_symptom_score'], errors='coerce').fillna(0.0)
    df['respiratory_symptom_score'] = pd.to_numeric(df['respiratory_symptom_score'], errors='coerce').fillna(0.0)
    df['neurological_symptom_score'] = pd.to_numeric(df['neurological_symptom_score'], errors='coerce').fillna(0.0)
    df['trauma_symptom_score'] = pd.to_numeric(df['trauma_symptom_score'], errors='coerce').fillna(0.0)
    
    return df

def calculate_clinical_risk_factors(df):
    """Calculate clinical risk factor scores"""
    
    # Age risk factor
    age_risk = np.where(df['yaş'] > 65, 2,
               np.where(df['yaş'] < 18, 1, 0))
    
    # Comorbidity risk (from existing conditions)
    comorbidity_risk = 0
    
    # Check for comorbidity indicators in text fields
    if 'ek hastalıklar' in df.columns:
        conditions_text = df['ek hastalıklar'].astype(str).str.lower()
        
        # Common high-risk conditions
        high_risk_conditions = ['hypertension', 'diabetes', 'heart disease', 'cancer', 
                               'hipertansiyon', 'diyabet', 'kalp hastalığı', 'kanser']
        
        for condition in high_risk_conditions:
            comorbidity_risk += conditions_text.str.contains(condition, na=False).astype(int)
    
    df['age_risk_factor'] = age_risk
    df['comorbidity_risk_factor'] = comorbidity_risk
    
    return df

def enhanced_feature_engineer_data(df):
    """
    Enhanced feature engineering for clinical triage
    Reduces features from 278 to ~20 focused clinical features
    """
    print("Applying enhanced feature engineering...")
    
    # 1. Keep original vital signs (7 features)
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    
    # 2. Add vital signs severity score (1 feature)
    df['vital_severity_score'] = calculate_vital_severity_score(df)
    
    # 3. Create symptom clusters (4 features instead of 268)
    df = create_symptom_clusters(df)
    
    # 4. Add clinical risk factors (2 features)
    df = calculate_clinical_risk_factors(df)
    
    # 5. Keep temporal features (3 features)
    if 'created' in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['created']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['created']).dt.dayofweek
        df['month'] = pd.to_datetime(df['created']).dt.month
    else:
        # Default temporal features if created column not available
        df['hour_of_day'] = 12  # Default to noon
        df['day_of_week'] = 1   # Default to Monday
        df['month'] = 6         # Default to June
    
    # 6. Gender encoding (up to 3 features)
    if 'cinsiyet' in df.columns:
        df['cinsiyet'] = df['cinsiyet'].replace({'Erkek': 'Male', 'Kadın': 'Female', '': 'Unknown'})
        gender_dummies = pd.get_dummies(df['cinsiyet'], prefix='gender')
        df = pd.concat([df, gender_dummies], axis=1)
    else:
        # Default gender features
        df['gender_Male'] = 0.5
        df['gender_Female'] = 0.5
        df['gender_Unknown'] = 0
    
    # 7. Encode target variable
    if 'doğru triyaj' in df.columns:
        triage_mapping = {'Yeşil Alan': 0, 'Yeşil': 0, 'Sarı Alan': 1, 'Sarı': 1, 'Kırmızı Alan': 2, 'Kırmızı': 2}
        df['doğru triyaj_encoded'] = df['doğru triyaj'].map(triage_mapping)
        
        # Handle any unmapped values
        df['doğru triyaj_encoded'] = df['doğru triyaj_encoded'].fillna(1)  # Default to Yellow
    
    print(f"Enhanced feature engineering complete. Total features: {len(df.columns)}")
    
    return df

# PHASE 2: Hierarchical Clinical Model
class HierarchicalClinicalTriageModel(nn.Module):
    """
    Hierarchical model that prioritizes vital signs over other features
    """
    def __init__(self, num_vital_features=8, num_symptom_features=4, 
                 num_risk_features=2, num_temporal_features=3, 
                 num_demographic_features=3, num_classes=3):
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
        
        # Quaternary pathway: Demographics & Temporal
        self.context_pathway = nn.Sequential(
            nn.Linear(num_temporal_features + num_demographic_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Attention mechanism for pathway fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )
        
        # Final classification with clinical hierarchy
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
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
    
    def forward(self, vital_signs, symptoms, risk_factors, context):
        # Process each pathway
        vital_features = self.vital_signs_pathway(vital_signs)  # 32 features
        symptom_features = self.symptom_pathway(symptoms)      # 16 features
        risk_features = self.risk_pathway(risk_factors)        # 8 features
        context_features = self.context_pathway(context)       # 8 features
        
        # Combine features with proper weighting
        combined = torch.cat([vital_features, symptom_features, 
                             risk_features, context_features], dim=1)  # 64 features
        
        # Apply attention to focus on most relevant features
        attended, _ = self.attention(
            combined.unsqueeze(1), 
            combined.unsqueeze(1), 
            combined.unsqueeze(1)
        )
        
        # Final classification
        output = self.classifier(attended.squeeze(1))
        return output

# PHASE 2: Clinical Safety Loss Function
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
        # Rows: True class, Columns: Predicted class
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

# PHASE 3: Data Augmentation for Class Imbalance
class ClinicalDataAugmentation:
    """
    Clinical-aware data augmentation to address class imbalance
    """
    def __init__(self, target_ratio=3.0):
        self.target_ratio = target_ratio
    
    def augment_critical_cases(self, X, y):
        """
        Augment critical (Red) cases to improve balance
        """
        # Find critical cases
        critical_indices = np.where(y == 2)[0]
        
        if len(critical_indices) == 0:
            return X, y
        
        # Calculate how many samples we need
        total_samples = len(y)
        current_critical = len(critical_indices)
        target_critical = int(total_samples / (self.target_ratio + 1))
        
        if current_critical >= target_critical:
            return X, y
        
        samples_needed = target_critical - current_critical
        print(f"Augmenting {samples_needed} critical cases to improve balance...")
        
        # Generate augmented samples
        augmented_X = []
        augmented_y = []
        
        for _ in range(samples_needed):
            # Randomly select a critical case
            base_idx = np.random.choice(critical_indices)
            base_sample = X.iloc[base_idx].copy()
            
            # Add realistic noise to vital signs
            vital_cols = ["yaş", "sistolik kb", "diastolik kb", 
                         "solunum sayısı", "nabız", "ateş", "saturasyon"]
            
            for col in vital_cols:
                if col in base_sample:
                    # Add 5% noise
                    noise = np.random.normal(0, 0.05) * base_sample[col]
                    base_sample[col] += noise
            
            # Keep symptom and risk factors unchanged
            # (they define the critical nature)
            
            augmented_X.append(base_sample)
            augmented_y.append(2)  # Red class
        
        # Combine original and augmented data
        if augmented_X:
            augmented_df = pd.DataFrame(augmented_X)
            X_combined = pd.concat([X, augmented_df], ignore_index=True)
            y_combined = np.concatenate([y, augmented_y])
            
            print(f"Data augmentation complete. New shape: {X_combined.shape}")
            return X_combined, y_combined
        
        return X, y

# PHASE 4: Enhanced Clinical Metrics
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
        
        # 4. Triage-specific metrics
        triage_metrics = EnhancedClinicalMetrics._calculate_triage_metrics(y_true, y_pred)
        metrics.update(triage_metrics)
        
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
    
    @staticmethod
    def _calculate_triage_metrics(y_true, y_pred):
        """Calculate triage-specific metrics"""
        
        # Triage efficiency
        correct_triage = np.sum(y_true == y_pred)
        triage_efficiency = correct_triage / len(y_true)
        
        # Resource utilization metrics
        predicted_red = np.sum(y_pred == 2)
        predicted_yellow = np.sum(y_pred == 1)
        predicted_green = np.sum(y_pred == 0)
        
        total_cases = len(y_pred)
        
        return {
            'triage_efficiency': triage_efficiency,
            'predicted_critical_rate': predicted_red / total_cases,
            'predicted_urgent_rate': predicted_yellow / total_cases,
            'predicted_non_urgent_rate': predicted_green / total_cases
        }

# PHASE 3: Multi-Stage Training Protocol
class ClinicalTrainingProtocol:
    """
    Multi-stage training protocol for clinical triage
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'stage_1': {}, 'stage_2': {}, 'stage_3': {}}
        self.best_model_state = None
        self.best_critical_sensitivity = 0.0
    
    def _train_stage(self, train_loader, val_loader, optimizer, criterion, epochs, stage_name):
        """Train a single stage"""
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'critical_sens': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for vital_data, symptom_data, risk_data, context_data, targets in train_loader:
                vital_data = vital_data.to(self.device)
                symptom_data = symptom_data.to(self.device)
                risk_data = risk_data.to(self.device)
                context_data = context_data.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(vital_data, symptom_data, risk_data, context_data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for vital_data, symptom_data, risk_data, context_data, targets in val_loader:
                    vital_data = vital_data.to(self.device)
                    symptom_data = symptom_data.to(self.device)
                    risk_data = risk_data.to(self.device)
                    context_data = context_data.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(vital_data, symptom_data, risk_data, context_data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            train_loss = total_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(all_targets, all_preds)
            
            # Calculate critical sensitivity
            critical_cases = np.sum(np.array(all_targets) == 2)
            if critical_cases > 0:
                critical_detected = np.sum((np.array(all_targets) == 2) & (np.array(all_preds) == 2))
                critical_sens = critical_detected / critical_cases
            else:
                critical_sens = 0.0
            
            # Save best model based on critical sensitivity
            if critical_sens > self.best_critical_sensitivity:
                self.best_critical_sensitivity = critical_sens
                self.best_model_state = self.model.state_dict().copy()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['critical_sens'].append(critical_sens)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"{stage_name} Epoch {epoch+1:2d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.3f}, "
                      f"Critical Sens: {critical_sens:.3f}")
        
        return history
    
    def stage_1_vital_signs_focus(self, train_loader, val_loader, epochs=30):
        """Stage 1: Focus on vital signs patterns"""
        print("Stage 1: Vital Signs Pattern Learning")
        
        # Freeze non-vital pathways
        for param in self.model.symptom_pathway.parameters():
            param.requires_grad = False
        for param in self.model.risk_pathway.parameters():
            param.requires_grad = False
        for param in self.model.context_pathway.parameters():
            param.requires_grad = False
        
        # High learning rate for vital signs pathway
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.003, weight_decay=1e-4
        )
        
        # Focus on critical case detection
        criterion = ClinicalSafetyLoss(critical_miss_penalty=30.0)
        
        # Train for vital signs patterns
        history = self._train_stage(train_loader, val_loader, optimizer, 
                                  criterion, epochs, "Stage1")
        
        self.training_history['stage_1'] = history
        return history
    
    def stage_2_symptom_integration(self, train_loader, val_loader, epochs=20):
        """Stage 2: Integrate symptom information"""
        print("Stage 2: Symptom Integration")
        
        # Unfreeze symptom pathway
        for param in self.model.symptom_pathway.parameters():
            param.requires_grad = True
        
        # Lower learning rate
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001, weight_decay=1e-4
        )
        
        # Balanced loss
        criterion = ClinicalSafetyLoss(critical_miss_penalty=20.0)
        
        history = self._train_stage(train_loader, val_loader, optimizer, 
                                  criterion, epochs, "Stage2")
        
        self.training_history['stage_2'] = history
        return history
    
    def stage_3_safety_refinement(self, train_loader, val_loader, epochs=15):
        """Stage 3: Final safety-focused refinement"""
        print("Stage 3: Clinical Safety Refinement")
        
        # Unfreeze all pathways
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Very low learning rate for fine-tuning
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0005, weight_decay=1e-4
        )
        
        # Maximum safety focus
        criterion = ClinicalSafetyLoss(
            under_triage_penalty=15.0,
            critical_miss_penalty=25.0
        )
        
        history = self._train_stage(train_loader, val_loader, optimizer, 
                                  criterion, epochs, "Stage3")
        
        self.training_history['stage_3'] = history
        return history

# PHASE 5: Enhanced Training Pipeline
def run_enhanced_final_fix():
    """
    Enhanced final fix with all improvements implemented
    """
    print("=== ENHANCED TRIAGE MODEL WITH CLINICAL FIXES ===")
    print("Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 1. Load and prepare data with enhanced feature engineering
    print("\n1. Loading and preparing data...")
    file_path = 'triaj_data.csv'
    
    try:
        df_cleaned = load_and_clean_data(file_path)
        df_enhanced = enhanced_feature_engineer_data(df_cleaned.copy())
        print(f"Data loaded successfully. Shape: {df_enhanced.shape}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Using synthetic data...")
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'yaş': np.random.gamma(2, 20, n_samples).astype(int),
            'sistolik kb': np.random.normal(130, 20, n_samples).clip(80, 200).astype(int),
            'diastolik kb': np.random.normal(80, 15, n_samples).clip(50, 120).astype(int),
            'solunum sayısı': np.random.normal(18, 4, n_samples).clip(10, 40).astype(int),
            'nabız': np.random.normal(80, 15, n_samples).clip(50, 150).astype(int),
            'ateş': np.random.normal(37, 1.5, n_samples).clip(35, 42),
            'saturasyon': np.random.normal(97, 3, n_samples).clip(80, 100).astype(int),
            'cinsiyet': np.random.choice(['Male', 'Female'], n_samples),
            'doğru triyaj': np.random.choice(['Yeşil', 'Sarı', 'Kırmızı'], n_samples, p=[0.3, 0.5, 0.2])
        }
        
        df_cleaned = pd.DataFrame(data)
        df_enhanced = enhanced_feature_engineer_data(df_cleaned.copy())
        print(f"Synthetic data created. Shape: {df_enhanced.shape}")
    
    # 2. Prepare features
    print("\n2. Preparing enhanced feature sets...")
    
    # Define feature groups based on enhanced engineering
    vital_features = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı",
                     "nabız", "ateş", "saturasyon", "vital_severity_score"]
    symptom_features = ["cardiac_symptom_score", "respiratory_symptom_score",
                       "neurological_symptom_score", "trauma_symptom_score"]
    risk_features = ["age_risk_factor", "comorbidity_risk_factor"]
    temporal_features = ["hour_of_day", "day_of_week", "month"]
    demographic_features = [col for col in df_enhanced.columns if col.startswith('gender_')]
    
    # Ensure all feature columns exist
    for feature_list in [vital_features, symptom_features, risk_features, temporal_features]:
        for feature in feature_list:
            if feature not in df_enhanced.columns:
                print(f"Warning: Feature {feature} not found, setting to 0")
                df_enhanced[feature] = 0
    
    # Ensure demographic features exist
    if not demographic_features:
        df_enhanced['gender_Male'] = 0.5
        df_enhanced['gender_Female'] = 0.5
        df_enhanced['gender_Unknown'] = 0
        demographic_features = ['gender_Male', 'gender_Female', 'gender_Unknown']
    
    print(f"Feature groups:")
    print(f"  Vital signs: {len(vital_features)} features")
    print(f"  Symptoms: {len(symptom_features)} features")
    print(f"  Risk factors: {len(risk_features)} features")
    print(f"  Temporal: {len(temporal_features)} features")
    print(f"  Demographics: {len(demographic_features)} features")
    print(f"  Total: {len(vital_features + symptom_features + risk_features + temporal_features + demographic_features)} features")
    
    # Separate features and target
    X = df_enhanced[vital_features + symptom_features + risk_features + temporal_features + demographic_features]
    y = df_enhanced['doğru triyaj_encoded'].values
    
    print(f"Target distribution: {np.bincount(y)}")
    
    # 3. Apply data augmentation for class imbalance
    print("\n3. Applying data augmentation...")
    augmenter = ClinicalDataAugmentation(target_ratio=3.0)
    X_augmented, y_augmented = augmenter.augment_critical_cases(X, y)
    
    print(f"Post-augmentation target distribution: {np.bincount(y_augmented)}")
    
    # 4. Prepare data splits
    print("\n4. Preparing stratified data splits...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_augmented, y_augmented, test_size=0.4, random_state=42, stratify=y_augmented
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples, classes: {np.bincount(y_train)}")
    print(f"  Validation: {len(X_val)} samples, classes: {np.bincount(y_val)}")
    print(f"  Test: {len(X_test)} samples, classes: {np.bincount(y_test)}")
    
    # 5. Create enhanced data loaders
    def create_hierarchical_tensors(X, y):
        """Create tensors for hierarchical model"""
        # Ensure all data is numeric and handle any non-numeric values
        X_vital_data = X[vital_features].astype(float).fillna(0)
        X_symptom_data = X[symptom_features].astype(float).fillna(0)
        X_risk_data = X[risk_features].astype(float).fillna(0)
        X_context_data = X[temporal_features + demographic_features].astype(float).fillna(0)
        
        X_vital = torch.tensor(X_vital_data.values, dtype=torch.float32)
        X_symptom = torch.tensor(X_symptom_data.values, dtype=torch.float32)
        X_risk = torch.tensor(X_risk_data.values, dtype=torch.float32)
        X_context = torch.tensor(X_context_data.values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_vital, X_symptom, X_risk, X_context, y_tensor
    
    # Create tensors
    train_vital, train_symptom, train_risk, train_context, train_y = create_hierarchical_tensors(X_train, y_train)
    val_vital, val_symptom, val_risk, val_context, val_y = create_hierarchical_tensors(X_val, y_val)
    test_vital, test_symptom, test_risk, test_context, test_y = create_hierarchical_tensors(X_test, y_test)
    
    # Create datasets
    train_dataset = TensorDataset(train_vital, train_symptom, train_risk, train_context, train_y)
    val_dataset = TensorDataset(val_vital, val_symptom, val_risk, val_context, val_y)
    test_dataset = TensorDataset(test_vital, test_symptom, test_risk, test_context, test_y)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Initialize hierarchical model
    print("\n5. Initializing hierarchical clinical model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HierarchicalClinicalTriageModel(
        num_vital_features=len(vital_features),
        num_symptom_features=len(symptom_features),
        num_risk_features=len(risk_features),
        num_temporal_features=len(temporal_features),
        num_demographic_features=len(demographic_features),
        num_classes=3
    )
    
    # 7. Multi-stage training
    print("\n6. Starting multi-stage training protocol...")
    trainer = ClinicalTrainingProtocol(model, device)
    
    # Stage 1: Vital signs focus
    trainer.stage_1_vital_signs_focus(train_loader, val_loader, epochs=30)
    
    # Stage 2: Symptom integration
    trainer.stage_2_symptom_integration(train_loader, val_loader, epochs=20)
    
    # Stage 3: Safety refinement
    trainer.stage_3_safety_refinement(train_loader, val_loader, epochs=15)
    
    # Load best model
    if trainer.best_model_state is not None:
        trainer.model.load_state_dict(trainer.best_model_state)
        print(f"Loaded best model with critical sensitivity: {trainer.best_critical_sensitivity:.3f}")
    
    # 8. Enhanced evaluation
    print("\n7. Running enhanced evaluation...")
    
    # Evaluate on test set
    trainer.model.eval()
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for vital_data, symptom_data, risk_data, context_data, targets in test_loader:
            vital_data = vital_data.to(device)
            symptom_data = symptom_data.to(device)
            risk_data = risk_data.to(device)
            context_data = context_data.to(device)
            targets = targets.to(device)
            
            start_time = time.time()
            outputs = trainer.model(vital_data, symptom_data, risk_data, context_data)
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
    model_size = sum(p.numel() * p.element_size() for p in trainer.model.parameters()) / (1024 * 1024)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    
    performance_metrics = {
        'avg_inference_time_ms': avg_inference_time,
        'throughput_samples_per_sec': throughput,
        'model_size_mb': model_size,
        'total_parameters': total_params,
        'total_samples_tested': len(all_targets)
    }
    
    # Print enhanced results
    print("\n=== ENHANCED EVALUATION RESULTS ===")
    
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
    
    # Save enhanced results
    print("\n8. Saving enhanced results...")
    os.makedirs('results', exist_ok=True)
    
    # Create final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'HierarchicalClinicalTriageModel',
        'clinical_metrics': enhanced_metrics,
        'performance_metrics': performance_metrics,
        'training_history': trainer.training_history,
        'target_comparison': {
            'targets': targets,
            'actual_metrics': actual_metrics,
            'all_targets_met': all_targets_met
        },
        'model_info': {
            'architecture': 'HierarchicalClinicalTriageModel',
            'total_parameters': performance_metrics['total_parameters'],
            'model_size_mb': performance_metrics['model_size_mb'],
            'enhancement_features': [
                'Enhanced feature engineering (278→20 features)',
                'Hierarchical model architecture',
                'Clinical safety loss function',
                'Multi-stage training protocol',
                'Data augmentation for class imbalance',
                'Enhanced clinical metrics'
            ]
        },
        'data_info': {
            'total_samples': len(y_augmented),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'class_distribution_original': np.bincount(y).tolist(),
            'class_distribution_augmented': np.bincount(y_augmented).tolist(),
            'feature_dimensions': {
                'vital_signs': len(vital_features),
                'symptoms': len(symptom_features),
                'risk_factors': len(risk_features),
                'temporal': len(temporal_features),
                'demographics': len(demographic_features)
            }
        }
    }
    
    # Assessment and recommendations
    overall_acc = enhanced_metrics['overall_accuracy']
    critical_sens = safety['critical_sensitivity']
    under_triage = safety['under_triage_rate']
    
    if all_targets_met:
        performance_level = "Excellent - All Targets Met"
        risk_level = "Low"
    elif overall_acc >= 0.7 and critical_sens >= 0.9 and under_triage <= 0.2:
        performance_level = "Good - Most Targets Met"
        risk_level = "Medium"
    elif overall_acc >= 0.6 and critical_sens >= 0.8:
        performance_level = "Acceptable - Some Improvement Needed"
        risk_level = "Medium"
    else:
        performance_level = "Poor - Significant Improvement Required"
        risk_level = "High"
    
    recommendations = []
    if not all_targets_met:
        if actual_metrics['overall_accuracy'] < targets['overall_accuracy']:
            recommendations.append("Further enhance feature engineering and model complexity")
        if actual_metrics['critical_sensitivity'] < targets['critical_sensitivity']:
            recommendations.append("Increase critical case focus in training")
        if actual_metrics['under_triage_rate'] > targets['under_triage_rate']:
            recommendations.append("Strengthen clinical safety loss penalties")
    else:
        recommendations.append("Model meets all clinical targets - ready for validation")
    
    final_results['summary'] = {
        'overall_performance': performance_level,
        'key_findings': [
            f"Overall accuracy: {overall_acc:.3f} (Target: {targets['overall_accuracy']:.3f})",
            f"Critical sensitivity: {critical_sens:.3f} (Target: {targets['critical_sensitivity']:.3f})",
            f"Under-triage rate: {under_triage:.3f} (Target: <{targets['under_triage_rate']:.3f})",
            f"Model parameters reduced from 43K to {performance_metrics['total_parameters']:,}",
            f"Features reduced from 278 to {len(vital_features + symptom_features + risk_features + temporal_features + demographic_features)}"
        ],
        'recommendations': recommendations,
        'risk_assessment': risk_level,
        'all_targets_met': all_targets_met
    }
    
    # Save report
    report_path = f"results/enhanced_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Enhanced results saved to: {report_path}")
    
    # Save model
    model_path = f"results/enhanced_hierarchical_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'num_vital_features': len(vital_features),
            'num_symptom_features': len(symptom_features),
            'num_risk_features': len(risk_features),
            'num_temporal_features': len(temporal_features),
            'num_demographic_features': len(demographic_features),
            'num_classes': 3
        },
        'training_history': trainer.training_history,
        'final_results': final_results
    }, model_path)
    
    print(f"Enhanced model saved to: {model_path}")
    
    # Final assessment
    print("\n=== ENHANCED FINAL ASSESSMENT ===")
    if all_targets_met:
        print("🎉 ALL DIAGNOSTIC PLAN TARGETS MET!")
        print("✅ Enhanced model exceeds all clinical requirements")
        print("✅ Ready for clinical validation and deployment")
    elif overall_acc >= 0.7 and critical_sens >= 0.9:
        print("✅ SIGNIFICANT IMPROVEMENT ACHIEVED!")
        print("✅ Enhanced model meets most clinical requirements")
        print("⚠️  Minor optimization needed for full compliance")
    elif overall_acc >= 0.6 and critical_sens >= 0.8:
        print("⚠️  MODERATE IMPROVEMENT ACHIEVED")
        print("⚠️  Enhanced model shows progress but needs more work")
    else:
        print("❌ FURTHER ENHANCEMENT NEEDED")
        print("❌ Continue with diagnostic plan implementation")
    
    print(f"\nKey Enhancements Applied:")
    print(f"  • Enhanced feature engineering: 278 → {len(vital_features + symptom_features + risk_features + temporal_features + demographic_features)} features")
    print(f"  • Hierarchical model architecture with attention")
    print(f"  • Clinical safety loss function")
    print(f"  • Multi-stage training protocol")
    print(f"  • Data augmentation for class imbalance")
    print(f"  • Enhanced clinical metrics evaluation")
    
    return final_results, trainer.model

if __name__ == "__main__":
    results, model = run_enhanced_final_fix()