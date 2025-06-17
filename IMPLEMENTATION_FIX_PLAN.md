# Triage Model Fix Implementation Plan

Based on diagnostic test results, this document provides a structured, step-by-step plan to fix the identified issues in `src/final_fix.py`.

## Current Issues Identified by Diagnostics

### Critical Issues (Must Fix)
1. **Overall Accuracy**: 70.4% (Target: 75%) - Gap: 4.6%
2. **Under-triage Rate**: 23.1% (Target: <15%) - Gap: 8.1%
3. **Critical Sensitivity**: 90.9% (Target: >95%) - Gap: 4.1%

### High Priority Issues
1. **Class Imbalance**: 6:1 ratio (Yeşil:Kırmızı)
2. **Clinical Logic Violations**: 63 cases found
3. **Green Class Precision**: 41.0% (causing over-triage)

### Moderate Issues
1. **Feature Imbalance**: 4.9:1 text:numerical ratio
2. **Missing Clinical Features**: Pain severity, onset time, etc.

---

## PHASE 1: Data Quality & Feature Engineering Fixes (Week 1)

### Step 1.1: Enhanced Feature Engineering
**File to modify**: `src/feature_engineering.py`

**Current Problem**: 268 boolean features overwhelming 7 numerical vital signs

**Fix Implementation**:

```python
def create_clinical_feature_engineer():
    """
    Enhanced feature engineering focused on clinical relevance
    """
    
    # 1. Vital Signs Severity Scoring
    def calculate_vital_severity_score(df):
        """Calculate composite vital signs severity score"""
        severity_score = 0
        
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
    
    # 2. Symptom Clustering (Replace 268 boolean features)
    def create_symptom_clusters(df):
        """Group symptoms into clinically meaningful clusters"""
        
        # Define symptom clusters
        cardiac_symptoms = ['göğüs ağrısı', 'kardiyoloji']
        respiratory_symptoms = ['nefes darlığı', 'göğüs hastalıkları']
        neurological_symptoms = ['baş ağrısı', 'nörolojik hastalıklar', 'beyin cerrahi']
        trauma_symptoms = [col for col in df.columns if 'travma_' in col]
        
        # Create cluster scores
        df['cardiac_symptom_score'] = df[cardiac_symptoms].sum(axis=1)
        df['respiratory_symptom_score'] = df[respiratory_symptoms].sum(axis=1)
        df['neurological_symptom_score'] = df[neurological_symptoms].sum(axis=1)
        df['trauma_symptom_score'] = df[trauma_symptoms].sum(axis=1)
        
        return df
    
    # 3. Clinical Risk Factors
    def calculate_clinical_risk_factors(df):
        """Calculate clinical risk factor scores"""
        
        # Age risk factor
        age_risk = np.where(df['yaş'] > 65, 2,
                   np.where(df['yaş'] < 18, 1, 0))
        
        # Comorbidity risk (from existing conditions)
        comorbidity_conditions = ['hypertension', 'diabetes', 'heart disease', 'cancer']
        comorbidity_risk = 0
        for condition in comorbidity_conditions:
            if f'ek hastalıklar_{condition}' in df.columns:
                comorbidity_risk += df[f'ek hastalıklar_{condition}']
        
        df['age_risk_factor'] = age_risk
        df['comorbidity_risk_factor'] = comorbidity_risk
        
        return df
```

### Step 1.2: Implement Enhanced Feature Engineering
**Action**: Replace current feature engineering with clinical-focused approach

```python
def enhanced_feature_engineer_data(df):
    """
    Enhanced feature engineering for clinical triage
    """
    # 1. Keep original vital signs (7 features)
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    
    # 2. Add vital signs severity score (1 feature)
    df['vital_severity_score'] = calculate_vital_severity_score(df)
    
    # 3. Create symptom clusters (4 features instead of 268)
    df = create_symptom_clusters(df)
    
    # 4. Add clinical risk factors (2 features)
    df = calculate_clinical_risk_factors(df)
    
    # 5. Keep temporal features (3 features)
    df['hour_of_day'] = pd.to_datetime(df['created']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['created']).dt.dayofweek
    df['month'] = pd.to_datetime(df['created']).dt.month
    
    # 6. Gender encoding (3 features)
    df = pd.get_dummies(df, columns=['cinsiyet'], prefix='gender')
    
    # Total: 7 + 1 + 4 + 2 + 3 + 3 = 20 features (vs 278 currently)
    
    return df
```

---

## PHASE 2: Model Architecture Improvements (Week 2)

### Step 2.1: Implement Hierarchical Clinical Model
**File to modify**: `src/final_fix.py`

**Current Problem**: Single pathway processing all features equally

**Fix Implementation**:

```python
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
```

### Step 2.2: Implement Clinical Safety Loss Function
**Current Problem**: Standard cross-entropy doesn't penalize clinical errors appropriately

**Fix Implementation**:

```python
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
            safety_penalty += self.penalty_matrix[true_class, pred_class]
        
        safety_penalty = safety_penalty / batch_size
        
        # Extra penalty for missing critical cases (Red -> Green/Yellow)
        critical_mask = (targets == 2)  # Red cases
        if critical_mask.sum() > 0:
            critical_outputs = outputs[critical_mask]
            critical_targets = targets[critical_mask]
            critical_preds = pred_classes[critical_mask]
            
            # Heavy penalty for critical misses
            critical_misses = (critical_preds != 2).sum().float()
            critical_penalty = (critical_misses / critical_mask.sum()) * self.critical_miss_penalty
        else:
            critical_penalty = 0.0
        
        # Combine losses
        total_loss = ce_loss + 0.3 * safety_penalty + 0.5 * critical_penalty
        
        return total_loss
```

---

## PHASE 3: Training Strategy Improvements (Week 2-3)

### Step 3.1: Multi-Stage Training Protocol
**Current Problem**: Single-stage training doesn't focus on critical case detection

**Fix Implementation**:

```python
class ClinicalTrainingProtocol:
    """
    Multi-stage training protocol for clinical triage
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.training_history = {'stage_1': {}, 'stage_2': {}, 'stage_3': {}}
    
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
```

### Step 3.2: Advanced Data Augmentation for Class Imbalance
**Current Problem**: 6:1 class imbalance causing poor minority class performance

**Fix Implementation**:

```python
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
            
            return X_combined, y_combined
        
        return X, y
    
    def apply_smote_like_augmentation(self, X, y):
        """
        Apply SMOTE-like augmentation for minority classes
        """
        from collections import Counter
        
        class_counts = Counter(y)
        majority_count = max(class_counts.values())
        
        augmented_X = [X]
        augmented_y = [y]
        
        for class_label, count in class_counts.items():
            if count < majority_count / 2:  # Minority class
                samples_needed = int(majority_count / 2) - count
                
                class_indices = np.where(y == class_label)[0]
                
                for _ in range(samples_needed):
                    # Select two random samples from the same class
                    idx1, idx2 = np.random.choice(class_indices, 2, replace=True)
                    
                    # Create synthetic sample (interpolation)
                    alpha = np.random.random()
                    synthetic_sample = alpha * X.iloc[idx1] + (1 - alpha) * X.iloc[idx2]
                    
                    augmented_X.append(synthetic_sample.to_frame().T)
                    augmented_y.append([class_label])
        
        # Combine all data
        X_final = pd.concat(augmented_X, ignore_index=True)
        y_final = np.concatenate(augmented_y)
        
        return X_final, y_final
```

---

## PHASE 4: Evaluation & Validation Improvements (Week 3)

### Step 4.1: Enhanced Clinical Metrics
**Current Problem**: Standard metrics don't capture clinical safety requirements

**Fix Implementation**:

```python
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
        
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name.lower()}_precision'] = precision[i]
            metrics[f'{class_name.lower()}_recall'] = recall[i]
            metrics[f'{class_name.lower()}_f1'] = f1[i]
        
        # 3. Clinical safety metrics
        safety_metrics = EnhancedClinicalMetrics._calculate_clinical_safety(y_true, y_pred)
        metrics.update(safety_metrics)
        
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
        else:
            critical_sensitivity = 0.0
            critical_miss_rate = 0.0
        
        return {
            'under_triage_rate': under_triage_rate,
            'over_triage_rate': over_triage_rate,
            'critical_sensitivity': critical_sensitivity,
            'critical_miss_rate': critical_miss_rate,
            'total_critical_cases': int(critical_cases)
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
```

---

## PHASE 5: Integration & Testing (Week 4)

### Step 5.1: Updated Training Pipeline
**File to create**: `src/enhanced_final_fix.py`

```python
def run_enhanced_final_fix():
    """
    Enhanced final fix with all improvements implemented
    """
    print("=== ENHANCED TRIAGE MODEL WITH CLINICAL FIXES ===")
    
    # 1. Load and prepare data with enhanced feature engineering
    df_cleaned = load_and_clean_data('triaj_data.csv')
    df_enhanced = enhanced_feature_engineer_data(df_cleaned.copy())
    
    # 2. Prepare features
    vital_features = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", 
                     "nabız", "ateş", "saturasyon", "vital_severity_score"]
    symptom_features = ["cardiac_symptom_score", "respiratory_symptom_score", 
                       "neurological_symptom_score", "trauma_symptom_score"]
    risk_features = ["age_risk_factor", "comorbidity_risk_factor"]
    temporal_features = ["hour_of_day", "day_of_week", "month"]
    demographic_features = [col for col in df_enhanced.columns if col.startswith('gender_')]
    
    # 3. Apply data augmentation for class imbalance
    augmenter = ClinicalDataAugmentation(target_ratio=3.0)
    X_augmented, y_augmented = augmenter.augment_critical_cases(X, y)
    
    # 4. Initialize hierarchical model
    model = HierarchicalClinicalTriageModel(
        num_vital_features=len(vital_features),
        num_symptom_features=len(symptom_features),
        num_risk_features=len(risk_features),
        num_temporal_features=len(temporal_features),
        num_demographic_features=len(demographic_features),
        num_classes=3
    )
    
    # 5. Multi-stage training
    trainer = ClinicalTrainingProtocol(model, device)
    
    # Stage 1: Vital signs focus
    trainer.stage_1_vital_signs_focus(train_loader, val_loader, epochs=30)
    
    # Stage 2: Symptom integration
    trainer.stage_2_symptom_integration(train_loader, val_loader, epochs=20)
    
    # Stage 3: Safety refinement
    trainer.stage_3_safety_refinement(train_loader, val_loader, epochs=15)
    
    # 6. Enhanced evaluation
    results = evaluate_with_enhanced_metrics(model, test_loader)
    
    return results, model
```

### Step 5.2: Validation Against Targets
**Expected Improvements**:

| Metric | Current | Target | Expected After Fixes |
|--------|---------|--------|---------------------|
| Overall Accuracy | 70.4% | 75% | 78-82% |
| Under-triage Rate | 23.1% | <15% | 8-12% |
| Critical Sensitivity | 90.9% | >95% | 96-98% |
| Green Precision | 41.0% | >60% | 65-70% |

---

## Implementation Timeline

### Week 1: Data & Features
- **Day 1-2**: Implement enhanced feature engineering
- **Day 3-4**: Create symptom clustering
- **Day 5**: Add clinical risk factors
- **Day 6-7**: Test and validate new features

### Week 2: Model Architecture
- **Day 1-3**: Implement hierarchical model
- **Day 4-5**: Create clinical safety loss
- **Day 6-7**: Implement multi-stage training

### Week 3: Training & Augmentation
- **Day 1-3**: Implement data augmentation
- **Day 4-5**: Multi-stage training protocol
- **Day 6-7**: Enhanced evaluation metrics

### Week 4: Integration & Validation
- **Day 1-3**: Integrate all components
- **Day 4-5**: Comprehensive testing
- **Day 6-7**: Final validation and documentation

---

## Success Criteria

### Minimum Acceptable Performance
- ✅ Overall Accuracy: >75%
- ✅ Under-triage Rate: <15%
- ✅ Critical Sensitivity: >95%

### Target Performance
- 🎯 Overall Accuracy: >80%
- 🎯 Under-triage Rate: <10%
- 🎯 Critical Sensitivity: >97%

### Deployment Ready
- 🚀 Overall Accuracy: >85%
- 🚀 Under-triage Rate: <5%
- 🚀 Critical Sensitivity: >99%

---

## Monitoring & Validation

### Continuous Testing
Run diagnostic tests after each phase:
```bash
python run_comprehensive_diagnostics.py
```

### Key Metrics to Track
1. **Critical Sensitivity** (most important)
2. **Under-triage Rate** (safety critical)
3. **Overall Accuracy** (performance)
4. **Green Precision** (resource efficiency)

### Rollback Plan
If any fix reduces critical sensitivity below 90%:
1. Immediately rollback the change
2. Analyze the issue
3. Implement alternative approach
4. Re-test before proceeding

This structured plan addresses all issues identified by the diagnostic tests and provides a clear path to achieve the target performance metrics.