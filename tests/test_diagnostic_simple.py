"""
Simplified Diagnostic Test Suite for Triage Model Issues
Works directly with the actual data structure without complex feature engineering
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def run_simple_diagnostics(data_path):
    """
    Run simplified diagnostics that work with the actual data structure
    """
    print("=== SIMPLIFIED TRIAGE MODEL DIAGNOSTICS ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source: {data_path}")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'total_samples': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns)
        },
        'issues_found': {}
    }
    
    # Test 1: Target Variable Consistency
    print("\n1. Testing Target Variable Consistency...")
    if 'triyaj alanı' in df.columns and 'doğru triyaj' in df.columns:
        inconsistent = df[df['triyaj alanı'] != df['doğru triyaj']]
        inconsistency_rate = len(inconsistent) / len(df)
        
        print(f"   Total cases: {len(df)}")
        print(f"   Inconsistent cases: {len(inconsistent)}")
        print(f"   Inconsistency rate: {inconsistency_rate*100:.1f}%")
        
        # Find specific examples mentioned in diagnostic plan
        specific_examples = []
        for idx, row in inconsistent.iterrows():
            if len(specific_examples) < 10:  # Show first 10 examples
                specific_examples.append({
                    'row_id': int(idx),
                    'initial_triage': row['triyaj alanı'],
                    'correct_triage': row['doğru triyaj'],
                    'age': row['yaş'],
                    'vitals': {
                        'bp': f"{row['sistolik kb']}/{row['diastolik kb']}",
                        'hr': row['nabız'],
                        'rr': row['solunum sayısı'],
                        'temp': row['ateş'],
                        'spo2': row['saturasyon']
                    }
                })
        
        results['issues_found']['target_inconsistency'] = {
            'severity': 'CRITICAL' if inconsistency_rate > 0.1 else 'MODERATE',
            'inconsistency_rate': inconsistency_rate,
            'total_inconsistent': len(inconsistent),
            'examples': specific_examples
        }
        
        if inconsistency_rate > 0.1:
            print(f"   ❌ CRITICAL: High inconsistency rate ({inconsistency_rate*100:.1f}%)")
        else:
            print(f"   ⚠️  MODERATE: Some inconsistency found ({inconsistency_rate*100:.1f}%)")
    else:
        print("   SKIP: Required columns not found")
    
    # Test 2: Class Distribution Analysis
    print("\n2. Analyzing Class Distribution...")
    if 'doğru triyaj' in df.columns:
        class_counts = df['doğru triyaj'].value_counts()
        total_samples = len(df)
        
        print(f"   Class distribution:")
        for class_name, count in class_counts.items():
            percentage = count / total_samples * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        results['issues_found']['class_imbalance'] = {
            'severity': 'HIGH' if imbalance_ratio > 5 else 'MODERATE',
            'imbalance_ratio': float(imbalance_ratio),
            'class_distribution': class_counts.to_dict(),
            'total_samples': total_samples
        }
        
        if imbalance_ratio > 5:
            print(f"   ❌ HIGH: Severe class imbalance ({imbalance_ratio:.1f}:1)")
        else:
            print(f"   ⚠️  MODERATE: Some class imbalance ({imbalance_ratio:.1f}:1)")
    
    # Test 3: Vital Signs Logic Violations
    print("\n3. Testing Vital Signs Logic...")
    
    # Define normal ranges
    vital_ranges = {
        'sistolik kb': (90, 140),
        'diastolik kb': (60, 90),
        'nabız': (60, 100),
        'solunum sayısı': (12, 20),
        'ateş': (36.1, 37.2),
        'saturasyon': (95, 100)
    }
    
    violations = []
    
    # Check for critical cases with normal vitals
    if 'doğru triyaj' in df.columns:
        red_cases = df[df['doğru triyaj'] == 'Kırmızı Alan']
        
        for idx, row in red_cases.iterrows():
            normal_vitals = 0
            total_vitals = 0
            
            for vital, (low, high) in vital_ranges.items():
                if vital in row and pd.notna(row[vital]):
                    total_vitals += 1
                    if low <= row[vital] <= high:
                        normal_vitals += 1
            
            if total_vitals > 0 and normal_vitals / total_vitals >= 0.7:  # 70% normal
                violations.append({
                    'type': 'critical_with_normal_vitals',
                    'row_id': int(idx),
                    'normal_percentage': normal_vitals / total_vitals,
                    'vitals': {vital: row[vital] for vital in vital_ranges.keys() if vital in row}
                })
        
        # Check for green cases with abnormal vitals
        green_cases = df[df['doğru triyaj'] == 'Yeşil Alan']
        
        for idx, row in green_cases.iterrows():
            abnormal_vitals = 0
            total_vitals = 0
            
            for vital, (low, high) in vital_ranges.items():
                if vital in row and pd.notna(row[vital]):
                    total_vitals += 1
                    if row[vital] < low or row[vital] > high:
                        abnormal_vitals += 1
            
            if total_vitals > 0 and abnormal_vitals / total_vitals >= 0.5:  # 50% abnormal
                violations.append({
                    'type': 'green_with_abnormal_vitals',
                    'row_id': int(idx),
                    'abnormal_percentage': abnormal_vitals / total_vitals,
                    'vitals': {vital: row[vital] for vital in vital_ranges.keys() if vital in row}
                })
    
    print(f"   Clinical logic violations found: {len(violations)}")
    
    results['issues_found']['clinical_logic_violations'] = {
        'severity': 'HIGH' if len(violations) > 10 else 'MODERATE',
        'total_violations': len(violations),
        'examples': violations[:10]  # First 10 examples
    }
    
    if len(violations) > 10:
        print(f"   ❌ HIGH: Many clinical logic violations ({len(violations)})")
    else:
        print(f"   ⚠️  MODERATE: Some clinical logic violations ({len(violations)})")
    
    # Test 4: Feature Analysis
    print("\n4. Analyzing Feature Structure...")
    
    # Count different types of features
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    text_cols = [col for col in df.columns if col not in numerical_cols + ['cinsiyet', 'triyaj alanı', 'doğru triyaj', 'created']]
    
    # Check for potential data leakage features
    diagnostic_keywords = [
        'kardiyoloji', 'dahiliye', 'cerrahi', 'nörolojik', 'psikiyatri'
    ]
    
    potential_leakage = []
    for col in text_cols:
        for keyword in diagnostic_keywords:
            if keyword.lower() in col.lower():
                potential_leakage.append(col)
                break
    
    print(f"   Numerical features: {len(numerical_cols)}")
    print(f"   Text/Boolean features: {len(text_cols)}")
    print(f"   Potential leakage features: {len(potential_leakage)}")
    print(f"   Feature ratio (text:numerical): {len(text_cols)/len(numerical_cols):.1f}:1")
    
    results['issues_found']['feature_structure'] = {
        'severity': 'HIGH' if len(text_cols)/len(numerical_cols) > 10 else 'MODERATE',
        'numerical_features': len(numerical_cols),
        'text_features': len(text_cols),
        'potential_leakage_features': len(potential_leakage),
        'feature_ratio': len(text_cols)/len(numerical_cols),
        'leakage_examples': potential_leakage[:10]
    }
    
    if len(text_cols)/len(numerical_cols) > 10:
        print(f"   ❌ HIGH: Severe feature imbalance ({len(text_cols)/len(numerical_cols):.1f}:1)")
    else:
        print(f"   ⚠️  MODERATE: Some feature imbalance ({len(text_cols)/len(numerical_cols):.1f}:1)")
    
    # Test 5: Data Quality Issues
    print("\n5. Checking Data Quality...")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df) * 100).round(2)
    
    # Empty string values
    empty_strings = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                empty_strings[col] = empty_count
    
    print(f"   Missing values found in {(missing_data > 0).sum()} columns")
    print(f"   Empty strings found in {len(empty_strings)} columns")
    
    results['issues_found']['data_quality'] = {
        'severity': 'MODERATE',
        'missing_values': missing_data[missing_data > 0].to_dict(),
        'empty_strings': empty_strings,
        'total_missing_columns': int((missing_data > 0).sum())
    }
    
    # Overall Assessment
    print("\n=== OVERALL ASSESSMENT ===")
    
    critical_issues = sum(1 for issue in results['issues_found'].values() if issue.get('severity') == 'CRITICAL')
    high_issues = sum(1 for issue in results['issues_found'].values() if issue.get('severity') == 'HIGH')
    
    if critical_issues >= 1:
        overall_severity = "CRITICAL"
        assessment = "❌ CRITICAL ISSUES FOUND - Immediate action required"
    elif high_issues >= 2:
        overall_severity = "HIGH"
        assessment = "⚠️  HIGH RISK - Significant issues found"
    else:
        overall_severity = "MODERATE"
        assessment = "⚠️  MODERATE - Some improvements needed"
    
    results['overall_assessment'] = {
        'severity': overall_severity,
        'critical_issues': critical_issues,
        'high_issues': high_issues,
        'assessment': assessment
    }
    
    print(f"{assessment}")
    print(f"Critical issues: {critical_issues}")
    print(f"High priority issues: {high_issues}")
    
    # Generate Recommendations
    print("\n=== RECOMMENDATIONS ===")
    recommendations = []
    
    if results['issues_found'].get('target_inconsistency', {}).get('severity') == 'CRITICAL':
        recommendations.append("1. CRITICAL: Conduct clinical expert review of target variable inconsistencies")
    
    if results['issues_found'].get('class_imbalance', {}).get('severity') == 'HIGH':
        recommendations.append("2. HIGH: Implement advanced class balancing techniques (focal loss, SMOTE)")
    
    if results['issues_found'].get('clinical_logic_violations', {}).get('severity') == 'HIGH':
        recommendations.append("3. HIGH: Add clinical context features (pain severity, symptom onset)")
    
    if results['issues_found'].get('feature_structure', {}).get('severity') == 'HIGH':
        recommendations.append("4. HIGH: Reduce feature explosion through symptom clustering")
    
    recommendations.append("5. GENERAL: Implement hierarchical model architecture as per diagnostic plan")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    results['recommendations'] = recommendations
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = f"results/simple_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDiagnostic results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    data_path = '/Users/batu/Documents/DEVELOPMENT/triaj-research/src/triaj_data.csv'
    results = run_simple_diagnostics(data_path)