"""
Test suite specifically for diagnosing issues in src/final_fix.py
Based on the diagnostic plan findings
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_final_fix_performance():
    """
    Test the actual performance of final_fix.py against diagnostic plan expectations
    """
    print("=== TESTING FINAL_FIX.PY PERFORMANCE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Import the final fix module
        from final_fix import run_final_fix
        
        print("Running final_fix.py to get baseline performance...")
        
        # Capture the results
        results, model = run_final_fix()
        
        # Extract key metrics
        clinical_metrics = results['clinical_metrics']
        performance_metrics = results['performance_metrics']
        
        # Test against diagnostic plan targets
        targets = {
            'overall_accuracy': 0.75,  # Target from plan
            'under_triage_rate': 0.15,  # Target from plan
            'critical_sensitivity': 0.95  # Target from plan
        }
        
        actual_metrics = {
            'overall_accuracy': clinical_metrics['overall_accuracy'],
            'under_triage_rate': clinical_metrics['clinical_safety']['under_triage_rate'],
            'critical_sensitivity': clinical_metrics['clinical_safety']['critical_sensitivity']
        }
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        
        test_results = {}
        all_passed = True
        
        for metric, target in targets.items():
            actual = actual_metrics[metric]
            
            if metric == 'under_triage_rate':
                passed = actual <= target
                status = "✅ PASS" if passed else "❌ FAIL"
                gap = target - actual if passed else actual - target
            else:
                passed = actual >= target
                status = "✅ PASS" if passed else "❌ FAIL"
                gap = actual - target if passed else target - actual
            
            print(f"{metric}:")
            print(f"  Target: {target:.3f}")
            print(f"  Actual: {actual:.3f}")
            print(f"  Status: {status}")
            print(f"  Gap: {gap:.3f}")
            print()
            
            test_results[metric] = {
                'target': target,
                'actual': actual,
                'passed': passed,
                'gap': gap
            }
            
            if not passed:
                all_passed = False
        
        # Additional diagnostic plan specific tests
        print("=== DIAGNOSTIC PLAN SPECIFIC ISSUES ===")
        
        # Test 1: Class imbalance handling
        class_metrics = clinical_metrics['class_metrics']
        red_recall = class_metrics.get('Red', {}).get('recall', 0)
        green_precision = class_metrics.get('Green', {}).get('precision', 0)
        
        print(f"Critical case detection (Red recall): {red_recall:.3f}")
        print(f"Green precision: {green_precision:.3f}")
        
        # Test 2: Model complexity
        total_params = performance_metrics['total_parameters']
        model_size = performance_metrics['model_size_mb']
        inference_time = performance_metrics['avg_inference_time_ms']
        
        print(f"Model parameters: {total_params:,}")
        print(f"Model size: {model_size:.2f} MB")
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Test 3: Clinical safety metrics
        safety_metrics = clinical_metrics['clinical_safety']
        critical_cases = safety_metrics['total_critical_cases']
        critical_identified = safety_metrics['critical_correctly_identified']
        
        print(f"Total critical cases: {critical_cases}")
        print(f"Critical cases correctly identified: {critical_identified}")
        print(f"Critical under-triage rate: {safety_metrics['critical_under_triage_rate']:.3f}")
        
        # Overall assessment based on diagnostic plan criteria
        print(f"\n=== OVERALL ASSESSMENT ===")
        
        if all_passed:
            assessment = "✅ MEETS DIAGNOSTIC PLAN TARGETS"
            risk_level = "LOW"
        elif actual_metrics['critical_sensitivity'] >= 0.9 and actual_metrics['under_triage_rate'] <= 0.2:
            assessment = "⚠️  PARTIALLY MEETS TARGETS - Acceptable for development"
            risk_level = "MEDIUM"
        else:
            assessment = "❌ FAILS DIAGNOSTIC PLAN TARGETS - Needs improvement"
            risk_level = "HIGH"
        
        print(f"Assessment: {assessment}")
        print(f"Risk Level: {risk_level}")
        
        # Specific recommendations based on failures
        recommendations = []
        
        if not test_results['overall_accuracy']['passed']:
            recommendations.append("Implement data quality improvements from diagnostic plan Phase 1")
        
        if not test_results['critical_sensitivity']['passed']:
            recommendations.append("Enhance critical case detection with focused loss function")
        
        if not test_results['under_triage_rate']['passed']:
            recommendations.append("Implement clinical safety loss with under-triage penalties")
        
        if red_recall < 0.9:
            recommendations.append("Focus training on Red class detection")
        
        if green_precision < 0.6:
            recommendations.append("Improve Green class precision to reduce over-triage")
        
        print(f"\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Save detailed results
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'actual_metrics': actual_metrics,
            'targets': targets,
            'clinical_metrics': clinical_metrics,
            'performance_metrics': performance_metrics,
            'assessment': assessment,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'all_targets_met': all_passed
        }
        
        os.makedirs('results', exist_ok=True)
        results_path = f"results/final_fix_diagnostic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\nDetailed test results saved to: {results_path}")
        
        return detailed_results
        
    except Exception as e:
        print(f"❌ Error testing final_fix.py: {e}")
        print("This indicates issues with the model implementation itself")
        return None

def test_data_consistency():
    """
    Test data consistency issues mentioned in diagnostic plan
    """
    print("\n=== TESTING DATA CONSISTENCY ISSUES ===")
    
    data_path = '/Users/batu/Documents/DEVELOPMENT/triaj-research/src/triaj_data.csv'
    
    try:
        df = pd.read_csv(data_path)
        
        # Test specific examples from diagnostic plan
        issues_found = []
        
        # Check if we have the problematic rows mentioned in the plan
        if len(df) > 25:
            # Row 12: triyaj alanı="Sarı Alan" but doğru triyaj="Kırmızı Alan"
            if len(df) > 12:
                row_12 = df.iloc[11]  # 0-indexed
                if (row_12.get('triyaj alanı') == 'Sarı Alan' and 
                    row_12.get('doğru triyaj') == 'Kırmızı Alan'):
                    issues_found.append("Row 12: Found expected Sarı->Kırmızı inconsistency")
            
            # Row 14: triyaj alanı="Yeşil Alan" but doğru triyaj="Sarı Alan"
            if len(df) > 14:
                row_14 = df.iloc[13]  # 0-indexed
                if (row_14.get('triyaj alanı') == 'Yeşil Alan' and 
                    row_14.get('doğru triyaj') == 'Sarı Alan'):
                    issues_found.append("Row 14: Found expected Yeşil->Sarı inconsistency")
            
            # Row 25: triyaj alanı="Sarı Alan" but doğru triyaj="Kırmızı Alan"
            if len(df) > 25:
                row_25 = df.iloc[24]  # 0-indexed
                if (row_25.get('triyaj alanı') == 'Sarı Alan' and 
                    row_25.get('doğru triyaj') == 'Kırmızı Alan'):
                    issues_found.append("Row 25: Found expected Sarı->Kırmızı inconsistency")
        
        print(f"Specific diagnostic plan issues found: {len(issues_found)}")
        for issue in issues_found:
            print(f"  {issue}")
        
        # General consistency check
        if 'triyaj alanı' in df.columns and 'doğru triyaj' in df.columns:
            total_inconsistent = (df['triyaj alanı'] != df['doğru triyaj']).sum()
            print(f"Total inconsistent cases: {total_inconsistent}")
            print(f"Inconsistency rate: {total_inconsistent/len(df)*100:.1f}%")
        
        return len(issues_found) > 0 or total_inconsistent > 0
        
    except Exception as e:
        print(f"Error testing data consistency: {e}")
        return False

def main():
    """
    Main function to run all diagnostic tests
    """
    print("="*60)
    print("COMPREHENSIVE DIAGNOSTIC TEST FOR FINAL_FIX.PY")
    print("="*60)
    
    # Test 1: Data consistency
    data_issues = test_data_consistency()
    
    # Test 2: Model performance
    performance_results = test_final_fix_performance()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST SUMMARY")
    print("="*60)
    
    if data_issues:
        print("❌ Data consistency issues found")
    else:
        print("✅ No data consistency issues found")
    
    if performance_results:
        if performance_results['all_targets_met']:
            print("✅ All performance targets met")
        else:
            failed_targets = [k for k, v in performance_results['test_results'].items() if not v['passed']]
            print(f"❌ Failed targets: {', '.join(failed_targets)}")
        
        print(f"Risk Level: {performance_results['risk_level']}")
        print(f"Assessment: {performance_results['assessment']}")
    else:
        print("❌ Performance testing failed")
    
    print("\nNext steps:")
    print("1. Review detailed results in results/ directory")
    print("2. Address failed targets using diagnostic plan recommendations")
    print("3. Re-run tests to verify improvements")

if __name__ == "__main__":
    main()