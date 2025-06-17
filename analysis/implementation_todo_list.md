# Implementation TODO List: Complete Triage Model Fix

## CRITICAL DATA QUALITY FIXES (Priority 1)

### 1. Data Quality Audit Implementation
- [ ] **Target Variable Consistency Check**
  - Implement function to identify mismatches between "triyaj alanı" and "doğru triyaj"
  - Create audit report showing all inconsistent cases
  - Implement resolution strategy (use "doğru triyaj" as ground truth)

- [ ] **Feature Leakage Detection & Elimination**
  - Identify and remove post-hoc diagnostic features
  - Keep only pre-diagnostic symptoms and vital signs
  - Implement temporal validation to prevent future leakage

- [ ] **Data Preprocessing Overhaul**
  - Fix target encoding to use only "doğru triyaj" 
  - Remove "triyaj alanı" completely from features
  - Implement proper missing value handling

### 2. Smart Feature Engineering (Priority 1)
- [ ] **Reduce Feature Explosion**
  - Replace 268 boolean features with clinical feature groups
  - Implement symptom clustering instead of individual flags
  - Create vital signs severity scoring

- [ ] **Clinical Domain Knowledge Integration**
  - Implement cardiac risk scoring
  - Implement respiratory risk scoring  
  - Create pain severity assessment
  - Add clinical deterioration indicators

## MODEL ARCHITECTURE IMPROVEMENTS (Priority 2)

### 3. Hierarchical Clinical Model
- [ ] **Implement ClinicalTriageModel Class**
  - Vital signs pathway (primary)
  - Symptom pathway (secondary) 
  - Demographics pathway (tertiary)
  - Clinical fusion with attention mechanism

- [ ] **Clinical Safety Loss Function**
  - Implement penalty matrix for clinical errors
  - Heavy penalties for under-triage (especially Red→Green)
  - Moderate penalties for over-triage

### 4. Advanced Training Strategy
- [ ] **Multi-Stage Training Protocol**
  - Stage 1: Vital signs focus
  - Stage 2: Symptom integration
  - Stage 3: Safety refinement

- [ ] **Clinical Data Augmentation**
  - Vital signs noise simulation
  - Symptom dropout simulation
  - Realistic clinical variations

## VALIDATION FRAMEWORK (Priority 2)

### 5. Clinical-First Validation
- [ ] **Implement Clinical Metrics Priority**
  - Primary: Critical sensitivity (>95%)
  - Secondary: Under-triage rate (<15%)
  - Tertiary: Overall accuracy (>75%)

- [ ] **Temporal Validation Strategy**
  - Train on earlier data, test on later data
  - Prevent temporal data leakage
  - Simulate real-world deployment

## IMPLEMENTATION PLAN

### Phase 1: Data Quality Crisis Resolution (Immediate)
```python
# 1. Target Variable Audit
def audit_target_consistency(df):
    inconsistent = df[df['triyaj alanı'] != df['doğru triyaj']]
    return inconsistent

# 2. Feature Leakage Elimination  
def remove_diagnostic_features(df):
    diagnostic_keywords = ['ST elevasyonlu', 'Akut böbrek', 'Kalp yetmezliği']
    # Remove features containing diagnostic conclusions
    
# 3. Smart Feature Reduction
def create_clinical_features(df):
    # Replace 268 boolean with ~20 clinical features
    # Group symptoms by system (cardiac, respiratory, neurological)
```

### Phase 2: Model Architecture Upgrade
```python
class ClinicalTriageModel(nn.Module):
    def __init__(self):
        # Vital signs network (most important)
        self.vital_signs_net = VitalSignsNetwork()
        # Symptom network with attention
        self.symptoms_net = SymptomsAttentionNetwork() 
        # Clinical fusion
        self.clinical_fusion = ClinicalFusionLayer()

class ClinicalSafetyLoss(nn.Module):
    def __init__(self):
        # Penalty matrix: [Green, Yellow, Red] predictions vs true
        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 5.0],   # Green true
            [2.0, 0.0, 3.0],   # Yellow true  
            [10.0, 5.0, 0.0]   # Red true
        ])
```

### Phase 3: Training & Validation
```python
class ClinicalTrainingProtocol:
    def stage_1_vital_signs_focus(self):
        # Train only vital signs pathway
        
    def stage_2_symptom_integration(self):
        # Add symptom information
        
    def stage_3_safety_refinement(self):
        # Focus on reducing under-triage
```

## SPECIFIC CODE CHANGES NEEDED IN final_fix.py

### 1. Add Data Quality Audit Functions
```python
def audit_data_quality(df):
    """Comprehensive data quality audit"""
    
def resolve_target_inconsistency(df):
    """Fix target variable issues"""
    
def eliminate_feature_leakage(df):
    """Remove post-hoc diagnostic features"""
```

### 2. Replace OptimizedTriageModel with ClinicalTriageModel
- Implement hierarchical architecture
- Add attention mechanisms
- Focus on vital signs pathway

### 3. Implement ClinicalSafetyLoss
- Replace current loss function
- Add clinical penalty matrix
- Focus on under-triage prevention

### 4. Update Training Strategy
- Multi-stage training
- Clinical-first early stopping
- Safety-focused validation

## SUCCESS METRICS

### Minimum Viable Performance
- [ ] Critical sensitivity > 95%
- [ ] Under-triage rate < 20% 
- [ ] Overall accuracy > 70%

### Target Performance  
- [ ] Critical sensitivity > 98%
- [ ] Under-triage rate < 10%
- [ ] Overall accuracy > 80%

### Deployment Ready
- [ ] Critical sensitivity > 99%
- [ ] Under-triage rate < 5%
- [ ] Overall accuracy > 85%

## IMPLEMENTATION ORDER

1. **IMMEDIATE (Today)**: Data quality audit and target variable fix
2. **Day 1**: Feature leakage elimination and smart feature engineering
3. **Day 2**: Hierarchical model architecture implementation
4. **Day 3**: Clinical safety loss function and training strategy
5. **Day 4**: Validation framework and testing
6. **Day 5**: Performance optimization and final validation

## CRITICAL SUCCESS FACTORS

1. **Data Quality First**: Fix inconsistent labels before any modeling
2. **Clinical Safety Priority**: Under-triage prevention is most important
3. **Feature Quality over Quantity**: 20 good features > 268 noisy features
4. **Validation Strategy**: Use temporal splits to prevent overfitting
5. **Clinical Expert Review**: Validate results with medical professionals

---

**NEXT ACTION**: Start with data quality audit implementation in final_fix.py