# Triage Model Performance Diagnostic & Improvement Plan

## Executive Summary

**Current Performance Crisis:**
- Overall Accuracy: 53.7% (Target: >75%)
- Under-triage Rate: 38.9% (Target: <15%) 
- Critical Safety Risk: High

**Root Cause Analysis:** Multiple critical data quality and modeling issues identified.

## Critical Issues Discovered

### 1. **SEVERE DATA QUALITY PROBLEMS**

#### 1.1 Target Variable Inconsistency
```
CRITICAL FINDING: Mismatch between "triyaj alanı" and "doğru triyaj" columns
```

**Evidence from data analysis:**
- Row 12: `triyaj alanı="Sarı Alan"` but `doğru triyaj="Kırmızı Alan"` 
- Row 14: `triyaj alanı="Yeşil Alan"` but `doğru triyaj="Sarı Alan"`
- Row 25: `triyaj alanı="Sarı Alan"` but `doğru triyaj="Kırmızı Alan"`

**Impact:** Model is learning from inconsistent labels, causing confusion matrix chaos.

#### 1.2 Feature Engineering Data Leakage
```python
# PROBLEM: Using both columns creates data leakage
df.drop(['triyaj alanı', 'doğru triyaj'], axis=1, inplace=True)
```

**Critical Issue:** The model may be learning patterns from the initial triage decision (`triyaj alanı`) which should not be available during prediction.

#### 1.3 Text Feature Explosion
- **268 boolean features** created from text fields
- Many features are **post-hoc diagnoses** (e.g., "ST elevasyonlu miyokard enfarktüsü")
- **Massive overfitting** on sparse, noisy features

### 2. **CLINICAL LOGIC VIOLATIONS**

#### 2.1 Vital Signs Patterns Analysis
```
Red Cases with Normal Vitals:
- Row 2: BP=170/95, HR=86, Temp=36°C (Cardiac but stable vitals)
- Row 8: BP=145/86, HR=40 (Bradycardia - critical!)

Green Cases with Abnormal Vitals:  
- Row 20: Solunum=75 (Severe tachypnea but marked Green)
```

**Finding:** Vital signs alone don't predict triage level - clinical context matters more.

#### 2.2 Missing Critical Features
- **Pain severity scores** (only location mentioned)
- **Glasgow Coma Scale** for neurological cases
- **Time since symptom onset** (critical for cardiac cases)
- **Previous medical interventions**

### 3. **MODEL ARCHITECTURE PROBLEMS**

#### 3.1 Feature Imbalance
```
Current Architecture Issues:
- Numerical features: 7 (vital signs)
- Boolean features: 268 (mostly noise)
- Temporal features: 3 (minimal)

Problem: 268 noisy features overwhelm 7 critical vital signs
```

#### 3.2 Class Imbalance Mishandling
```
Class Distribution:
- Green (0): 95 samples (17.6%)
- Yellow (1): 332 samples (61.6%) 
- Red (2): 112 samples (20.8%)

Current approach: Simple class weights
Problem: Doesn't address clinical severity hierarchy
```

## Comprehensive Improvement Strategy

### Phase 1: Data Quality Remediation (Week 1)

#### 1.1 Target Variable Audit
```python
def audit_target_consistency(df):
    """Identify and resolve target variable inconsistencies"""
    inconsistent = df[df['triyaj alanı'] != df['doğru triyaj']]
    
    # Clinical expert review required for:
    # - Cases where initial triage was wrong
    # - Cases where condition deteriorated
    # - Cases with insufficient information
    
    return inconsistent
```

#### 1.2 Feature Leakage Elimination
```python
# REMOVE: Post-hoc diagnostic information
diagnostic_features = [
    'ST elevasyonlu miyokard enfarktüsü',
    'Akut böbrek yetmezliği', 
    'Kalp yetmezliği',
    # ... all diagnostic conclusions
]

# KEEP: Only pre-diagnostic information
clinical_features = [
    'Göğüste baskı hissi',
    'Nefes Darlığı',
    'Bayılma (Senkop)',
    # ... symptoms only
]
```

#### 1.3 Smart Feature Engineering
```python
class ClinicalFeatureEngineer:
    def __init__(self):
        self.vital_signs_processor = VitalSignsProcessor()
        self.symptom_encoder = SymptomEncoder()
        self.risk_calculator = ClinicalRiskCalculator()
    
    def engineer_features(self, df):
        # 1. Vital signs severity scoring
        df['vital_severity_score'] = self.calculate_vital_severity(df)
        
        # 2. Symptom clustering (not individual boolean flags)
        df['symptom_cluster'] = self.cluster_symptoms(df)
        
        # 3. Clinical risk factors
        df['cardiac_risk_score'] = self.calculate_cardiac_risk(df)
        df['respiratory_risk_score'] = self.calculate_respiratory_risk(df)
        
        return df
```

### Phase 2: Advanced Model Architecture (Week 2)

#### 2.1 Hierarchical Clinical Model
```python
class ClinicalTriageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Vital Signs Network (Primary pathway)
        self.vital_signs_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Symptom Network (Secondary pathway)
        self.symptom_net = nn.Sequential(
            nn.Linear(20, 32),  # Reduced from 268!
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        # Demographics Network (Tertiary pathway)
        self.demographics_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Clinical Fusion with Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=40, num_heads=4, batch_first=True
        )
        
        # Final Classification
        self.classifier = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # Green, Yellow, Red
        )
    
    def forward(self, vital_signs, symptoms, demographics):
        # Process each pathway
        vital_features = self.vital_signs_net(vital_signs)
        symptom_features = self.symptom_net(symptoms)
        demo_features = self.demographics_net(demographics)
        
        # Combine features
        combined = torch.cat([vital_features, symptom_features, demo_features], dim=1)
        
        # Apply attention mechanism
        attended, _ = self.attention(
            combined.unsqueeze(1), 
            combined.unsqueeze(1), 
            combined.unsqueeze(1)
        )
        
        # Final classification
        output = self.classifier(attended.squeeze(1))
        return output
```

#### 2.2 Clinical Safety Loss Function
```python
class ClinicalSafetyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Penalty matrix for clinical errors
        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 5.0],  # Green misclassified as Yellow(1x), Red(5x)
            [2.0, 0.0, 3.0],  # Yellow misclassified as Green(2x), Red(3x)  
            [10.0, 5.0, 0.0]  # Red misclassified as Green(10x), Yellow(5x)
        ])
    
    def forward(self, outputs, targets):
        # Standard cross-entropy
        ce = self.ce_loss(outputs, targets)
        
        # Clinical safety penalty
        pred_classes = torch.argmax(outputs, dim=1)
        safety_penalty = 0
        
        for i, (pred, true) in enumerate(zip(pred_classes, targets)):
            safety_penalty += self.penalty_matrix[true, pred]
        
        safety_penalty = safety_penalty / len(targets)
        
        return ce + 0.5 * safety_penalty
```

### Phase 3: Training Strategy Overhaul (Week 2-3)

#### 3.1 Multi-Stage Training Protocol
```python
class ClinicalTrainingProtocol:
    def __init__(self, model):
        self.model = model
        self.stage = 1
    
    def stage_1_vital_signs_focus(self, epochs=20):
        """Focus on learning vital signs patterns first"""
        # Freeze symptom and demographic networks
        # Train only vital signs pathway
        
    def stage_2_symptom_integration(self, epochs=15):
        """Integrate symptom information"""
        # Unfreeze symptom network
        # Lower learning rate
        
    def stage_3_safety_refinement(self, epochs=10):
        """Final safety-focused training"""
        # Use clinical safety loss
        # Focus on reducing under-triage
```

#### 3.2 Advanced Data Augmentation
```python
class ClinicalDataAugmentation:
    def augment_vital_signs(self, vital_signs):
        """Add realistic noise to vital signs"""
        # Simulate measurement variability
        noise = torch.normal(0, 0.05, vital_signs.shape)
        return vital_signs + noise
    
    def augment_symptoms(self, symptoms):
        """Symptom dropout simulation"""
        # Simulate incomplete symptom reporting
        dropout_mask = torch.bernoulli(torch.full(symptoms.shape, 0.9))
        return symptoms * dropout_mask
```

### Phase 4: Validation Framework (Week 3)

#### 4.1 Clinical-First Metrics
```python
class ClinicalValidation:
    def __init__(self):
        self.primary_metrics = [
            'critical_sensitivity',      # >95% (most important)
            'under_triage_rate',        # <10% (safety critical)
            'critical_specificity'      # >80% (resource management)
        ]
        
        self.secondary_metrics = [
            'overall_accuracy',         # >75%
            'yellow_precision',         # >60%
            'green_recall'             # >70%
        ]
    
    def evaluate_clinical_safety(self, y_true, y_pred):
        """Comprehensive clinical safety evaluation"""
        results = {}
        
        # Critical case detection (Red class)
        red_mask = (y_true == 2)
        red_detected = (y_pred == 2) & red_mask
        
        results['critical_sensitivity'] = red_detected.sum() / red_mask.sum()
        results['critical_missed'] = red_mask.sum() - red_detected.sum()
        
        # Under-triage analysis
        under_triage = y_pred < y_true
        results['under_triage_rate'] = under_triage.sum() / len(y_true)
        
        # Detailed error analysis
        results['green_to_red_errors'] = ((y_true == 2) & (y_pred == 0)).sum()
        results['yellow_to_red_errors'] = ((y_true == 2) & (y_pred == 1)).sum()
        
        return results
```

#### 4.2 Temporal Validation Strategy
```python
def temporal_validation_split(df):
    """Split data by time to prevent data leakage"""
    df['created'] = pd.to_datetime(df['created'])
    df = df.sort_values('created')
    
    # Train: First 60% of time period
    # Validation: Next 20% of time period  
    # Test: Last 20% of time period
    
    train_end = int(0.6 * len(df))
    val_end = int(0.8 * len(df))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df
```

## Implementation Roadmap

### Week 1: Data Quality Crisis Resolution
- [ ] **Day 1-2:** Target variable consistency audit
- [ ] **Day 3-4:** Feature leakage elimination  
- [ ] **Day 5-7:** Smart feature engineering implementation

### Week 2: Model Architecture Upgrade
- [ ] **Day 1-3:** Hierarchical model implementation
- [ ] **Day 4-5:** Clinical safety loss function
- [ ] **Day 6-7:** Multi-stage training protocol

### Week 3: Validation & Optimization
- [ ] **Day 1-3:** Clinical validation framework
- [ ] **Day 4-5:** Temporal validation implementation
- [ ] **Day 6-7:** Performance optimization

### Week 4: Clinical Validation & Deployment
- [ ] **Day 1-3:** Clinical expert review
- [ ] **Day 4-5:** Regulatory compliance check
- [ ] **Day 6-7:** Deployment preparation

## Expected Performance Improvements

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| Overall Accuracy | 53.7% | 75-85% | Data quality + architecture |
| Under-triage Rate | 38.9% | <15% | Clinical safety loss |
| Critical Sensitivity | 95.5% | >98% | Hierarchical vital signs focus |
| Green Precision | 29.3% | >60% | Balanced training strategy |
| Yellow F1-Score | 44.4% | >65% | Symptom clustering |

## Risk Mitigation

### Clinical Safety
- **Fail-safe mechanisms:** Always err on side of higher acuity
- **Human oversight:** Maintain clinician review for borderline cases
- **Continuous monitoring:** Real-time performance tracking

### Regulatory Compliance
- **Audit trail:** Complete decision pathway documentation
- **Bias monitoring:** Demographic fairness validation
- **Version control:** Model versioning and rollback capability

## Success Criteria

### Minimum Viable Performance
- Critical sensitivity > 95%
- Under-triage rate < 20%
- Overall accuracy > 70%

### Target Performance
- Critical sensitivity > 98%
- Under-triage rate < 10%
- Overall accuracy > 80%

### Deployment Ready
- Critical sensitivity > 99%
- Under-triage rate < 5%
- Overall accuracy > 85%

## Next Steps

1. **Immediate Action Required:** Data quality audit and target variable resolution
2. **Clinical Expert Consultation:** Review inconsistent cases for ground truth
3. **Feature Engineering Redesign:** Eliminate diagnostic leakage
4. **Model Architecture Implementation:** Hierarchical clinical model
5. **Validation Framework:** Clinical-first metrics implementation

---

**CRITICAL RECOMMENDATION:** Do not deploy current model in clinical setting due to high under-triage rate (38.9%) which poses significant patient safety risk.