"""
Comprehensive Diagnostic Test Suite for Triage Model Issues
Based on analysis/triage_model_diagnostic_plan.md

This test suite diagnoses the critical issues identified in the diagnostic plan:
1. Data Quality Problems
2. Clinical Logic Violations  
3. Model Architecture Problems
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_preparation import load_and_clean_data
    from feature_engineering import feature_engineer_data
    from final_fix import OptimizedTriageModel, ClinicalMetricsFixed, FinalTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")

class TriageModelDiagnostics:
    """
    Comprehensive diagnostic suite for triage model issues
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df_raw = None
        self.df_cleaned = None
        self.df_engineered = None
        self.diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'data_quality_issues': {},
            'clinical_logic_violations': {},
            'model_architecture_problems': {},
            'performance_issues': {},
            'recommendations': []
        }
    
    def load_data(self):
        """Load and prepare data for diagnostics"""
        print("Loading data for diagnostics...")
        try:
            self.df_raw = pd.read_csv(self.data_path)
            print(f"Raw data loaded: {self.df_raw.shape}")
            
            # Load cleaned data
            self.df_cleaned = load_and_clean_data(self.data_path)
            print(f"Cleaned data: {self.df_cleaned.shape}")
            
            # Load engineered data
            self.df_engineered = feature_engineer_data(self.df_cleaned.copy())
            print(f"Engineered data: {self.df_engineered.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def diagnose_data_quality_issues(self):
        """
        Diagnose critical data quality problems identified in the plan
        """
        print("\n=== DIAGNOSING DATA QUALITY ISSUES ===")
        
        issues = {}
        
        # 1. Target Variable Inconsistency Analysis
        print("1. Analyzing target variable consistency...")
        if 'triyaj alanı' in self.df_raw.columns and 'doğru triyaj' in self.df_raw.columns:
            inconsistent_cases = self.df_raw[self.df_raw['triyaj alanı'] != self.df_raw['doğru triyaj']]
            
            issues['target_inconsistency'] = {
                'total_cases': len(self.df_raw),
                'inconsistent_cases': len(inconsistent_cases),
                'inconsistency_rate': len(inconsistent_cases) / len(self.df_raw),
                'examples': []
            }
            
            # Document specific examples
            for idx, row in inconsistent_cases.head(10).iterrows():
                issues['target_inconsistency']['examples'].append({
                    'row_id': int(idx),
                    'initial_triage': row['triyaj alanı'],
                    'correct_triage': row['doğru triyaj'],
                    'vital_signs': {
                        'bp': f"{row['sistolik kb']}/{row['diastolik kb']}",
                        'hr': row['nabız'],
                        'rr': row['solunum sayısı'],
                        'temp': row['ateş'],
                        'spo2': row['saturasyon']
                    }
                })
            
            print(f"   Found {len(inconsistent_cases)} inconsistent cases ({len(inconsistent_cases)/len(self.df_raw)*100:.1f}%)")
        
        # 2. Feature Engineering Data Leakage Analysis
        print("2. Analyzing feature engineering for data leakage...")
        engineered_cols = list(self.df_engineered.columns)
        
        # Check for diagnostic features that shouldn't be available at triage
        diagnostic_keywords = [
            'miyokard', 'enfarktüs', 'böbrek yetmezliği', 'kalp yetmezliği',
            'cardiac evaluation', 'pulmonary assessment', 'trauma surgery'
        ]
        
        potential_leakage_features = []
        for col in engineered_cols:
            for keyword in diagnostic_keywords:
                if keyword.lower() in col.lower():
                    potential_leakage_features.append(col)
        
        issues['feature_leakage'] = {
            'total_features': len(engineered_cols),
            'potential_leakage_features': len(potential_leakage_features),
            'leakage_examples': potential_leakage_features[:20],  # First 20 examples
            'leakage_rate': len(potential_leakage_features) / len(engineered_cols)
        }
        
        print(f"   Found {len(potential_leakage_features)} potential data leakage features")
        
        # 3. Text Feature Explosion Analysis
        print("3. Analyzing text feature explosion...")
        boolean_features = [col for col in engineered_cols if col.startswith(('semptomlar_', 'travma_', 'dahiliye_', 'kardiyoloji_'))]
        
        # Calculate feature sparsity
        if boolean_features:
            boolean_data = self.df_engineered[boolean_features]
            sparsity = (boolean_data == 0).sum().sum() / (boolean_data.shape[0] * boolean_data.shape[1])
            
            issues['text_feature_explosion'] = {
                'total_boolean_features': len(boolean_features),
                'sparsity_rate': sparsity,
                'feature_density': 1 - sparsity,
                'avg_features_per_sample': boolean_data.sum(axis=1).mean()
            }
            
            print(f"   Found {len(boolean_features)} boolean features with {sparsity*100:.1f}% sparsity")
        
        # 4. Missing Critical Clinical Features
        print("4. Analyzing missing critical clinical features...")
        critical_features = [
            'pain_severity', 'glasgow_coma_scale', 'symptom_onset_time',
            'previous_interventions', 'chief_complaint_severity'
        ]
        
        missing_features = [feat for feat in critical_features if feat not in engineered_cols]
        
        issues['missing_critical_features'] = {
            'expected_critical_features': critical_features,
            'missing_features': missing_features,
            'missing_count': len(missing_features)
        }
        
        print(f"   Missing {len(missing_features)} critical clinical features")
        
        self.diagnostic_results['data_quality_issues'] = issues
        return issues
    
    def diagnose_clinical_logic_violations(self):
        """
        Diagnose clinical logic violations identified in the plan
        """
        print("\n=== DIAGNOSING CLINICAL LOGIC VIOLATIONS ===")
        
        violations = {}
        
        # 1. Vital Signs Pattern Analysis
        print("1. Analyzing vital signs patterns...")
        
        # Define normal ranges
        vital_ranges = {
            'sistolik kb': (90, 140),
            'diastolik kb': (60, 90),
            'nabız': (60, 100),
            'solunum sayısı': (12, 20),
            'ateş': (36.1, 37.2),
            'saturasyon': (95, 100)
        }
        
        # Analyze each triage category
        triage_analysis = {}
        for triage_level in ['Yeşil Alan', 'Sarı Alan', 'Kırmızı Alan']:
            if 'doğru triyaj' in self.df_raw.columns:
                subset = self.df_raw[self.df_raw['doğru triyaj'] == triage_level]
                
                if len(subset) > 0:
                    abnormal_vitals = {}
                    for vital, (low, high) in vital_ranges.items():
                        if vital in subset.columns:
                            abnormal = subset[(subset[vital] < low) | (subset[vital] > high)]
                            abnormal_vitals[vital] = {
                                'count': len(abnormal),
                                'percentage': len(abnormal) / len(subset) * 100,
                                'examples': abnormal[vital].head(5).tolist()
                            }
                    
                    triage_analysis[triage_level] = {
                        'total_cases': len(subset),
                        'abnormal_vitals': abnormal_vitals
                    }
        
        violations['vital_signs_patterns'] = triage_analysis
        
        # 2. Critical Cases with Normal Vitals
        print("2. Identifying critical cases with normal vitals...")
        if 'doğru triyaj' in self.df_raw.columns:
            red_cases = self.df_raw[self.df_raw['doğru triyaj'] == 'Kırmızı Alan']
            
            normal_vital_red_cases = []
            for idx, row in red_cases.iterrows():
                normal_count = 0
                total_vitals = 0
                
                for vital, (low, high) in vital_ranges.items():
                    if vital in row and pd.notna(row[vital]):
                        total_vitals += 1
                        if low <= row[vital] <= high:
                            normal_count += 1
                
                if total_vitals > 0 and normal_count / total_vitals >= 0.7:  # 70% normal vitals
                    normal_vital_red_cases.append({
                        'row_id': int(idx),
                        'normal_vitals_percentage': normal_count / total_vitals,
                        'vitals': {vital: row[vital] for vital in vital_ranges.keys() if vital in row}
                    })
            
            violations['critical_cases_normal_vitals'] = {
                'count': len(normal_vital_red_cases),
                'examples': normal_vital_red_cases[:10]
            }
            
            print(f"   Found {len(normal_vital_red_cases)} critical cases with mostly normal vitals")
        
        # 3. Non-critical Cases with Abnormal Vitals
        print("3. Identifying non-critical cases with abnormal vitals...")
        if 'doğru triyaj' in self.df_raw.columns:
            green_cases = self.df_raw[self.df_raw['doğru triyaj'] == 'Yeşil Alan']
            
            abnormal_vital_green_cases = []
            for idx, row in green_cases.iterrows():
                abnormal_count = 0
                total_vitals = 0
                
                for vital, (low, high) in vital_ranges.items():
                    if vital in row and pd.notna(row[vital]):
                        total_vitals += 1
                        if row[vital] < low or row[vital] > high:
                            abnormal_count += 1
                
                if total_vitals > 0 and abnormal_count / total_vitals >= 0.5:  # 50% abnormal vitals
                    abnormal_vital_green_cases.append({
                        'row_id': int(idx),
                        'abnormal_vitals_percentage': abnormal_count / total_vitals,
                        'vitals': {vital: row[vital] for vital in vital_ranges.keys() if vital in row}
                    })
            
            violations['green_cases_abnormal_vitals'] = {
                'count': len(abnormal_vital_green_cases),
                'examples': abnormal_vital_green_cases[:10]
            }
            
            print(f"   Found {len(abnormal_vital_green_cases)} green cases with abnormal vitals")
        
        self.diagnostic_results['clinical_logic_violations'] = violations
        return violations
    
    def diagnose_model_architecture_problems(self):
        """
        Diagnose model architecture problems identified in the plan
        """
        print("\n=== DIAGNOSING MODEL ARCHITECTURE PROBLEMS ===")
        
        problems = {}
        
        # 1. Feature Imbalance Analysis
        print("1. Analyzing feature imbalance...")
        
        # Define feature categories based on final_fix.py
        numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
        temporal_cols = ['hour_of_day', 'day_of_week', 'month']
        
        all_cols = list(self.df_engineered.columns)
        boolean_cols = [col for col in all_cols if col not in numerical_cols + temporal_cols + ['year', 'yaş_unscaled', 'doğru triyaj_encoded']]
        
        problems['feature_imbalance'] = {
            'numerical_features': len(numerical_cols),
            'boolean_features': len(boolean_cols),
            'temporal_features': len(temporal_cols),
            'total_features': len(all_cols),
            'boolean_to_numerical_ratio': len(boolean_cols) / len(numerical_cols) if len(numerical_cols) > 0 else 0,
            'feature_distribution': {
                'numerical_percentage': len(numerical_cols) / len(all_cols) * 100,
                'boolean_percentage': len(boolean_cols) / len(all_cols) * 100,
                'temporal_percentage': len(temporal_cols) / len(all_cols) * 100
            }
        }
        
        print(f"   Feature imbalance: {len(boolean_cols)} boolean vs {len(numerical_cols)} numerical features")
        print(f"   Ratio: {len(boolean_cols) / len(numerical_cols):.1f}:1 (boolean:numerical)")
        
        # 2. Class Imbalance Analysis
        print("2. Analyzing class imbalance...")
        if 'doğru triyaj_encoded' in self.df_engineered.columns:
            y = self.df_engineered['doğru triyaj_encoded'].values
            unique_classes, class_counts = np.unique(y, return_counts=True)
            
            class_distribution = {}
            for cls, count in zip(unique_classes, class_counts):
                class_distribution[int(cls)] = {
                    'count': int(count),
                    'percentage': float(count / len(y) * 100)
                }
            
            # Calculate imbalance metrics
            max_class_count = max(class_counts)
            min_class_count = min(class_counts)
            imbalance_ratio = max_class_count / min_class_count
            
            problems['class_imbalance'] = {
                'class_distribution': class_distribution,
                'total_samples': len(y),
                'imbalance_ratio': float(imbalance_ratio),
                'most_frequent_class': int(unique_classes[np.argmax(class_counts)]),
                'least_frequent_class': int(unique_classes[np.argmin(class_counts)])
            }
            
            print(f"   Class imbalance ratio: {imbalance_ratio:.1f}:1")
            print(f"   Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # 3. Model Complexity Analysis
        print("3. Analyzing model complexity...")
        
        # Create a sample model to analyze architecture
        try:
            model = OptimizedTriageModel(
                num_numerical_features=len(numerical_cols),
                num_boolean_features=len(boolean_cols),
                num_temporal_features=len(temporal_cols),
                num_classes=3
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Calculate model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            problems['model_complexity'] = {
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'model_size_mb': float(model_size_mb),
                'parameters_per_feature': float(total_params / len(all_cols)) if len(all_cols) > 0 else 0,
                'architecture_layers': self._analyze_model_layers(model)
            }
            
            print(f"   Model parameters: {total_params:,}")
            print(f"   Model size: {model_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"   Error analyzing model complexity: {e}")
            problems['model_complexity'] = {'error': str(e)}
        
        self.diagnostic_results['model_architecture_problems'] = problems
        return problems
    
    def _analyze_model_layers(self, model):
        """Analyze model layer structure"""
        layer_info = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_info[name] = {
                    'type': 'Linear',
                    'input_features': module.in_features,
                    'output_features': module.out_features,
                    'parameters': module.in_features * module.out_features + module.out_features
                }
            elif isinstance(module, nn.BatchNorm1d):
                layer_info[name] = {
                    'type': 'BatchNorm1d',
                    'num_features': module.num_features
                }
            elif isinstance(module, nn.Dropout):
                layer_info[name] = {
                    'type': 'Dropout',
                    'dropout_rate': module.p
                }
        
        return layer_info
    
    def diagnose_performance_issues(self):
        """
        Diagnose performance issues by running the current model
        """
        print("\n=== DIAGNOSING PERFORMANCE ISSUES ===")
        
        performance = {}
        
        try:
            # Prepare data
            X = self.df_engineered.drop('doğru triyaj_encoded', axis=1)
            y = self.df_engineered['doğru triyaj_encoded'].values
            
            # Define feature columns
            numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
            temporal_cols = ['hour_of_day', 'day_of_week', 'month']
            boolean_cols = [col for col in X.columns if col not in numerical_cols + temporal_cols + ['year', 'yaş_unscaled']]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Initialize model
            device = torch.device("cpu")  # Use CPU for diagnostics
            model = OptimizedTriageModel(
                num_numerical_features=len(numerical_cols),
                num_boolean_features=len(boolean_cols),
                num_temporal_features=len(temporal_cols),
                num_classes=len(np.unique(y))
            )
            
            trainer = FinalTrainer(model, device)
            
            # Create data loaders
            train_loader, val_loader, test_loader = trainer.create_data_loaders(
                X_train, X_train[:int(0.2*len(X_train))], X_test, 
                y_train, y_train[:int(0.2*len(y_train))], y_test,
                numerical_cols, boolean_cols, temporal_cols, batch_size=32
            )
            
            # Quick training (just a few epochs for diagnostics)
            print("   Running quick training for performance analysis...")
            class_weights = trainer.calculate_focal_loss_weights(y_train)
            
            # Train for just 5 epochs to get baseline performance
            training_history = trainer.train_with_focus_on_critical(
                train_loader, val_loader,
                epochs=5,
                learning_rate=0.001,
                class_weights=class_weights,
                patience=10
            )
            
            # Evaluate
            results = trainer.evaluate_comprehensive(test_loader)
            
            # Extract key metrics
            clinical_metrics = results['clinical_metrics']
            
            performance['baseline_performance'] = {
                'overall_accuracy': clinical_metrics['overall_accuracy'],
                'class_metrics': clinical_metrics['class_metrics'],
                'clinical_safety': clinical_metrics['clinical_safety'],
                'training_epochs': len(training_history['epochs']),
                'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
                'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None
            }
            
            # Compare against diagnostic plan targets
            targets = {
                'overall_accuracy': 0.75,
                'under_triage_rate': 0.15,
                'critical_sensitivity': 0.95
            }
            
            performance['target_comparison'] = {}
            for metric, target in targets.items():
                if metric == 'under_triage_rate':
                    actual = clinical_metrics['clinical_safety']['under_triage_rate']
                elif metric == 'critical_sensitivity':
                    actual = clinical_metrics['clinical_safety']['critical_sensitivity']
                else:
                    actual = clinical_metrics[metric]
                
                performance['target_comparison'][metric] = {
                    'target': target,
                    'actual': actual,
                    'meets_target': actual >= target if metric != 'under_triage_rate' else actual <= target,
                    'gap': actual - target if metric != 'under_triage_rate' else target - actual
                }
            
            print(f"   Baseline accuracy: {clinical_metrics['overall_accuracy']:.3f}")
            print(f"   Under-triage rate: {clinical_metrics['clinical_safety']['under_triage_rate']:.3f}")
            print(f"   Critical sensitivity: {clinical_metrics['clinical_safety']['critical_sensitivity']:.3f}")
            
        except Exception as e:
            print(f"   Error in performance analysis: {e}")
            performance['error'] = str(e)
        
        self.diagnostic_results['performance_issues'] = performance
        return performance
    
    def generate_recommendations(self):
        """
        Generate specific recommendations based on diagnostic findings
        """
        print("\n=== GENERATING RECOMMENDATIONS ===")
        
        recommendations = []
        
        # Data Quality Recommendations
        data_issues = self.diagnostic_results.get('data_quality_issues', {})
        
        if data_issues.get('target_inconsistency', {}).get('inconsistency_rate', 0) > 0.1:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'Critical',
                'issue': 'High target variable inconsistency',
                'recommendation': 'Conduct clinical expert review of inconsistent cases and establish ground truth',
                'implementation': 'Create target_variable_audit() function and clinical review process'
            })
        
        if data_issues.get('feature_leakage', {}).get('leakage_rate', 0) > 0.1:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'Critical',
                'issue': 'Data leakage from diagnostic features',
                'recommendation': 'Remove post-hoc diagnostic features and keep only pre-diagnostic symptoms',
                'implementation': 'Implement feature filtering based on temporal availability'
            })
        
        if data_issues.get('text_feature_explosion', {}).get('total_boolean_features', 0) > 100:
            recommendations.append({
                'category': 'Feature Engineering',
                'priority': 'High',
                'issue': 'Text feature explosion causing noise',
                'recommendation': 'Implement symptom clustering instead of individual boolean flags',
                'implementation': 'Create ClinicalFeatureEngineer with symptom grouping'
            })
        
        # Model Architecture Recommendations
        arch_problems = self.diagnostic_results.get('model_architecture_problems', {})
        
        feature_imbalance = arch_problems.get('feature_imbalance', {})
        if feature_imbalance.get('boolean_to_numerical_ratio', 0) > 10:
            recommendations.append({
                'category': 'Model Architecture',
                'priority': 'High',
                'issue': 'Severe feature imbalance (too many boolean features)',
                'recommendation': 'Implement hierarchical model with separate pathways for different feature types',
                'implementation': 'Create ClinicalTriageModel with attention mechanism'
            })
        
        class_imbalance = arch_problems.get('class_imbalance', {})
        if class_imbalance.get('imbalance_ratio', 0) > 5:
            recommendations.append({
                'category': 'Training Strategy',
                'priority': 'High',
                'issue': 'Severe class imbalance',
                'recommendation': 'Implement clinical safety loss function with penalty matrix',
                'implementation': 'Create ClinicalSafetyLoss with under-triage penalties'
            })
        
        # Performance Recommendations
        performance = self.diagnostic_results.get('performance_issues', {})
        target_comparison = performance.get('target_comparison', {})
        
        if not target_comparison.get('overall_accuracy', {}).get('meets_target', True):
            recommendations.append({
                'category': 'Performance',
                'priority': 'Critical',
                'issue': 'Overall accuracy below target',
                'recommendation': 'Implement multi-stage training protocol and data quality improvements',
                'implementation': 'Follow Phase 1-3 improvement strategy from diagnostic plan'
            })
        
        if not target_comparison.get('critical_sensitivity', {}).get('meets_target', True):
            recommendations.append({
                'category': 'Clinical Safety',
                'priority': 'Critical',
                'issue': 'Critical sensitivity below safety threshold',
                'recommendation': 'Focus training on critical case detection with enhanced penalties',
                'implementation': 'Implement critical-focused loss function and vital signs pathway'
            })
        
        # Clinical Logic Recommendations
        clinical_violations = self.diagnostic_results.get('clinical_logic_violations', {})
        
        if clinical_violations.get('critical_cases_normal_vitals', {}).get('count', 0) > 5:
            recommendations.append({
                'category': 'Clinical Logic',
                'priority': 'Medium',
                'issue': 'Critical cases with normal vitals indicate missing features',
                'recommendation': 'Add clinical context features (pain severity, symptom onset, etc.)',
                'implementation': 'Expand feature engineering to include clinical severity scores'
            })
        
        self.diagnostic_results['recommendations'] = recommendations
        
        print(f"Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority']}] {rec['issue']}")
        
        return recommendations
    
    def run_full_diagnostics(self):
        """
        Run complete diagnostic suite
        """
        print("=== TRIAGE MODEL COMPREHENSIVE DIAGNOSTICS ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data source: {self.data_path}")
        
        # Load data
        if not self.load_data():
            print("Failed to load data. Exiting diagnostics.")
            return None
        
        # Run all diagnostic tests
        self.diagnose_data_quality_issues()
        self.diagnose_clinical_logic_violations()
        self.diagnose_model_architecture_problems()
        self.diagnose_performance_issues()
        self.generate_recommendations()
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_path = f"results/diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.diagnostic_results, f, indent=2)
        
        print(f"\nDiagnostic results saved to: {results_path}")
        
        # Print summary
        self.print_diagnostic_summary()
        
        return self.diagnostic_results
    
    def print_diagnostic_summary(self):
        """
        Print a summary of diagnostic findings
        """
        print("\n=== DIAGNOSTIC SUMMARY ===")
        
        # Data Quality Issues
        data_issues = self.diagnostic_results.get('data_quality_issues', {})
        print(f"\nData Quality Issues:")
        print(f"  Target inconsistency rate: {data_issues.get('target_inconsistency', {}).get('inconsistency_rate', 0)*100:.1f}%")
        print(f"  Potential leakage features: {data_issues.get('feature_leakage', {}).get('potential_leakage_features', 0)}")
        print(f"  Boolean features: {data_issues.get('text_feature_explosion', {}).get('total_boolean_features', 0)}")
        
        # Model Architecture
        arch_problems = self.diagnostic_results.get('model_architecture_problems', {})
        feature_imbalance = arch_problems.get('feature_imbalance', {})
        print(f"\nModel Architecture:")
        print(f"  Boolean:Numerical ratio: {feature_imbalance.get('boolean_to_numerical_ratio', 0):.1f}:1")
        print(f"  Class imbalance ratio: {arch_problems.get('class_imbalance', {}).get('imbalance_ratio', 0):.1f}:1")
        print(f"  Total parameters: {arch_problems.get('model_complexity', {}).get('total_parameters', 0):,}")
        
        # Performance Issues
        performance = self.diagnostic_results.get('performance_issues', {})
        baseline = performance.get('baseline_performance', {})
        print(f"\nBaseline Performance:")
        print(f"  Overall accuracy: {baseline.get('overall_accuracy', 0):.3f}")
        if 'clinical_safety' in baseline:
            safety = baseline['clinical_safety']
            print(f"  Under-triage rate: {safety.get('under_triage_rate', 0):.3f}")
            print(f"  Critical sensitivity: {safety.get('critical_sensitivity', 0):.3f}")
        
        # Recommendations
        recommendations = self.diagnostic_results.get('recommendations', [])
        critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
        high_recs = [r for r in recommendations if r['priority'] == 'High']
        
        print(f"\nRecommendations:")
        print(f"  Critical priority: {len(critical_recs)}")
        print(f"  High priority: {len(high_recs)}")
        # Overall Assessment
        critical_issues = len(critical_recs)
        if critical_issues >= 3:
            assessment = "CRITICAL - Immediate action required"
        elif critical_issues >= 1:
            assessment = "HIGH RISK - Significant issues found"
        else:
            assessment = "MODERATE - Some improvements needed"
        
        print(f"\nOverall Assessment: {assessment}")


def run_specific_diagnostic_tests():
    """
    Run specific diagnostic tests for individual issues
    """
    print("=== RUNNING SPECIFIC DIAGNOSTIC TESTS ===")
    
    # Test 1: Target Variable Consistency Test
    def test_target_consistency(df_raw):
        """Test for target variable inconsistencies as mentioned in diagnostic plan"""
        print("\nTest 1: Target Variable Consistency")
        
        if 'triyaj alanı' not in df_raw.columns or 'doğru triyaj' not in df_raw.columns:
            print("  SKIP: Required columns not found")
            return False
        
        inconsistent = df_raw[df_raw['triyaj alanı'] != df_raw['doğru triyaj']]
        
        # Check specific examples from diagnostic plan
        specific_issues = []
        for idx, row in inconsistent.iterrows():
            if idx == 12 and row['triyaj alanı'] == 'Sarı Alan' and row['doğru triyaj'] == 'Kırmızı Alan':
                specific_issues.append(f"Row 12: Found expected inconsistency")
            elif idx == 14 and row['triyaj alanı'] == 'Yeşil Alan' and row['doğru triyaj'] == 'Sarı Alan':
                specific_issues.append(f"Row 14: Found expected inconsistency")
            elif idx == 25 and row['triyaj alanı'] == 'Sarı Alan' and row['doğru triyaj'] == 'Kırmızı Alan':
                specific_issues.append(f"Row 25: Found expected inconsistency")
        
        print(f"  Total inconsistent cases: {len(inconsistent)}")
        print(f"  Inconsistency rate: {len(inconsistent)/len(df_raw)*100:.1f}%")
        for issue in specific_issues:
            print(f"  {issue}")
        
        return len(inconsistent) > 0
    
    # Test 2: Feature Leakage Detection
    def test_feature_leakage(df_engineered):
        """Test for data leakage from post-hoc diagnostic features"""
        print("\nTest 2: Feature Leakage Detection")
        
        # Diagnostic terms that shouldn't be available at triage time
        post_hoc_terms = [
            'miyokard', 'enfarktüs', 'böbrek yetmezliği', 'kalp yetmezliği',
            'cardiac evaluation', 'pulmonary assessment', 'trauma surgery',
            'ST elevasyonlu', 'Akut böbrek', 'Kalp yetmezliği'
        ]
        
        leakage_features = []
        for col in df_engineered.columns:
            for term in post_hoc_terms:
                if term.lower() in col.lower():
                    leakage_features.append(col)
                    break
        
        print(f"  Potential leakage features found: {len(leakage_features)}")
        if leakage_features:
            print(f"  Examples: {leakage_features[:5]}")
        
        return len(leakage_features) > 0
    
    # Test 3: Vital Signs Logic Test
    def test_vital_signs_logic(df_raw):
        """Test for clinical logic violations in vital signs"""
        print("\nTest 3: Vital Signs Logic Violations")
        
        violations = []
        
        # Test case from diagnostic plan: Row 20 with severe tachypnea but Green triage
        if len(df_raw) > 20:
            row_20 = df_raw.iloc[19]  # 0-indexed
            if 'solunum sayısı' in row_20 and row_20['solunum sayısı'] > 30:
                if 'doğru triyaj' in row_20 and row_20['doğru triyaj'] == 'Yeşil Alan':
                    violations.append("Row 20: Severe tachypnea (RR>30) but marked Green")
        
        # Test for bradycardia in critical cases (Row 8 from plan)
        if len(df_raw) > 8:
            row_8 = df_raw.iloc[7]  # 0-indexed
            if 'nabız' in row_8 and row_8['nabız'] < 50:
                if 'doğru triyaj' in row_8 and row_8['doğru triyaj'] == 'Kırmızı Alan':
                    violations.append("Row 8: Bradycardia (HR<50) correctly marked Red")
        
        # General logic violations
        for idx, row in df_raw.iterrows():
            if 'doğru triyaj' in row and 'solunum sayısı' in row:
                # Severe tachypnea (>35) should not be Green
                if row['solunum sayısı'] > 35 and row['doğru triyaj'] == 'Yeşil Alan':
                    violations.append(f"Row {idx}: Severe tachypnea (RR={row['solunum sayısı']}) marked Green")
                
                # Severe bradypnea (<8) should be Red
                if row['solunum sayısı'] < 8 and row['doğru triyaj'] != 'Kırmızı Alan':
                    violations.append(f"Row {idx}: Severe bradypnea (RR={row['solunum sayısı']}) not marked Red")
        
        print(f"  Clinical logic violations found: {len(violations)}")
        for violation in violations[:5]:  # Show first 5
            print(f"  {violation}")
        
        return len(violations) > 0
    
    # Test 4: Class Imbalance Severity Test
    def test_class_imbalance(df_engineered):
        """Test severity of class imbalance"""
        print("\nTest 4: Class Imbalance Analysis")
        
        if 'doğru triyaj_encoded' not in df_engineered.columns:
            print("  SKIP: Encoded target not found")
            return False
        
        y = df_engineered['doğru triyaj_encoded'].values
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Calculate imbalance metrics
        max_count = max(class_counts)
        min_count = min(class_counts)
        imbalance_ratio = max_count / min_count
        
        print(f"  Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Check against diagnostic plan findings
        expected_distribution = {0: 95, 1: 332, 2: 112}  # From diagnostic plan
        
        if len(class_counts) == 3:
            actual_percentages = class_counts / len(y) * 100
            expected_percentages = np.array([17.6, 61.6, 20.8])  # From plan
            
            print(f"  Actual percentages: {actual_percentages}")
            print(f"  Expected from plan: {expected_percentages}")
        
        return imbalance_ratio > 3.0  # Significant imbalance
    
    # Test 5: Model Architecture Efficiency Test
    def test_model_architecture():
        """Test model architecture for efficiency issues"""
        print("\nTest 5: Model Architecture Efficiency")
        
        try:
            # Simulate the feature dimensions from diagnostic plan
            numerical_features = 7  # Vital signs
            boolean_features = 268  # Text features (problematic)
            temporal_features = 3   # Time features
            
            model = OptimizedTriageModel(
                num_numerical_features=numerical_features,
                num_boolean_features=boolean_features,
                num_temporal_features=temporal_features,
                num_classes=3
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            
            # Calculate parameter distribution
            boolean_processor_params = sum(p.numel() for p in model.boolean_processor.parameters())
            numerical_processor_params = sum(p.numel() for p in model.numerical_processor.parameters())
            
            boolean_param_ratio = boolean_processor_params / total_params
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Boolean processor params: {boolean_processor_params:,} ({boolean_param_ratio*100:.1f}%)")
            print(f"  Numerical processor params: {numerical_processor_params:,}")
            print(f"  Boolean:Numerical feature ratio: {boolean_features/numerical_features:.1f}:1")
            
            # Check if boolean features dominate (issue from diagnostic plan)
            boolean_dominance = boolean_features / numerical_features > 10
            
            return boolean_dominance
            
        except Exception as e:
            print(f"  Error testing architecture: {e}")
            return False
    
    # Load data and run tests
    data_path = '/Users/batu/Documents/DEVELOPMENT/triaj-research/src/triaj_data.csv'
    
    try:
        df_raw = pd.read_csv(data_path)
        df_cleaned = load_and_clean_data(data_path)
        df_engineered = feature_engineer_data(df_cleaned.copy())
        
        # Run all tests
        test_results = {
            'target_consistency': test_target_consistency(df_raw),
            'feature_leakage': test_feature_leakage(df_engineered),
            'vital_signs_logic': test_vital_signs_logic(df_raw),
            'class_imbalance': test_class_imbalance(df_engineered),
            'model_architecture': test_model_architecture()
        }
        
        # Summary
        issues_found = sum(test_results.values())
        print(f"\n=== TEST SUMMARY ===")
        print(f"Tests run: {len(test_results)}")
        print(f"Issues found: {issues_found}")
        
        for test_name, has_issue in test_results.items():
            status = "ISSUE FOUND" if has_issue else "OK"
            print(f"  {test_name}: {status}")
        
        if issues_found >= 4:
            print("\nOVERALL ASSESSMENT: CRITICAL - Multiple severe issues confirmed")
        elif issues_found >= 2:
            print("\nOVERALL ASSESSMENT: HIGH RISK - Significant issues confirmed")
        else:
            print("\nOVERALL ASSESSMENT: MODERATE - Some issues found")
        
        return test_results
        
    except Exception as e:
        print(f"Error running specific tests: {e}")
        return None


if __name__ == "__main__":
    # Run comprehensive diagnostics
    data_path = '/Users/batu/Documents/DEVELOPMENT/triaj-research/src/triaj_data.csv'
    
    print("Choose diagnostic mode:")
    print("1. Full comprehensive diagnostics")
    print("2. Specific issue tests only")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE DIAGNOSTICS")
        print("="*60)
        
        diagnostics = TriageModelDiagnostics(data_path)
        results = diagnostics.run_full_diagnostics()
    
    if choice in ['2', '3']:
        print("\n" + "="*60)
        print("RUNNING SPECIFIC DIAGNOSTIC TESTS")
        print("="*60)
        
        test_results = run_specific_diagnostic_tests()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the diagnostic report in results/")
    print("2. Address critical issues first")
    print("3. Implement recommendations from the diagnostic plan")
    print("4. Re-run diagnostics to verify improvements")