#!/usr/bin/env python3
"""
Comprehensive Diagnostic Test Runner for Triage Model Issues
Combines all diagnostic tests to provide complete analysis
"""

import sys
import os
from datetime import datetime

# Add tests to path
sys.path.append('tests')

def main():
    """
    Run comprehensive diagnostics combining all test suites
    """
    print("="*70)
    print("COMPREHENSIVE TRIAGE MODEL DIAGNOSTIC SUITE")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This suite will run multiple diagnostic tests:")
    print("1. Simple data structure diagnostics")
    print("2. Final fix performance testing")
    print("3. Diagnostic plan issue verification")
    print()
    
    # Test 1: Simple Diagnostics
    print("="*70)
    print("PHASE 1: SIMPLE DATA DIAGNOSTICS")
    print("="*70)
    
    try:
        from test_diagnostic_simple import run_simple_diagnostics
        
        data_path = 'src/triaj_data.csv'
        simple_results = run_simple_diagnostics(data_path)
        
        if simple_results:
            print("✅ Simple diagnostics completed successfully")
            simple_severity = simple_results['overall_assessment']['severity']
            print(f"Simple diagnostics severity: {simple_severity}")
        else:
            print("❌ Simple diagnostics failed")
            simple_severity = "UNKNOWN"
            
    except Exception as e:
        print(f"❌ Error in simple diagnostics: {e}")
        simple_severity = "ERROR"
    
    # Test 2: Final Fix Performance Testing
    print("\n" + "="*70)
    print("PHASE 2: FINAL FIX PERFORMANCE TESTING")
    print("="*70)
    
    try:
        from test_final_fix_issues import test_final_fix_performance, test_data_consistency
        
        # Test data consistency first
        print("Testing data consistency...")
        data_issues = test_data_consistency()
        
        # Test model performance
        print("\nTesting model performance...")
        performance_results = test_final_fix_performance()
        
        if performance_results:
            print("✅ Performance testing completed successfully")
            performance_severity = performance_results['risk_level']
            targets_met = performance_results['all_targets_met']
        else:
            print("❌ Performance testing failed")
            performance_severity = "ERROR"
            targets_met = False
            
    except Exception as e:
        print(f"❌ Error in performance testing: {e}")
        performance_severity = "ERROR"
        targets_met = False
        data_issues = True
    
    # Overall Assessment
    print("\n" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print("="*70)
    
    # Determine overall status
    if simple_severity == "CRITICAL" or performance_severity == "HIGH":
        overall_status = "CRITICAL"
        status_icon = "❌"
    elif simple_severity == "HIGH" or performance_severity == "MEDIUM":
        overall_status = "HIGH RISK"
        status_icon = "⚠️"
    elif simple_severity == "MODERATE" or performance_severity == "LOW":
        overall_status = "MODERATE"
        status_icon = "⚠️"
    else:
        overall_status = "GOOD"
        status_icon = "✅"
    
    print(f"Overall Status: {status_icon} {overall_status}")
    print()
    print("Detailed Results:")
    print(f"  Simple Diagnostics: {simple_severity}")
    print(f"  Performance Testing: {performance_severity}")
    print(f"  Data Consistency Issues: {'Yes' if data_issues else 'No'}")
    print(f"  Performance Targets Met: {'Yes' if targets_met else 'No'}")
    
    # Prioritized Recommendations
    print(f"\n=== PRIORITIZED RECOMMENDATIONS ===")
    
    recommendations = []
    
    if simple_severity == "CRITICAL":
        recommendations.append("🔥 CRITICAL: Address data quality issues immediately")
    
    if not targets_met:
        recommendations.append("🔥 CRITICAL: Model fails performance targets - implement diagnostic plan fixes")
    
    if data_issues:
        recommendations.append("⚠️  HIGH: Resolve data consistency issues")
    
    if simple_severity in ["HIGH", "MODERATE"]:
        recommendations.append("⚠️  HIGH: Implement class balancing and feature engineering improvements")
    
    if performance_severity in ["HIGH", "MEDIUM"]:
        recommendations.append("⚠️  MEDIUM: Optimize model architecture and training strategy")
    
    # Default recommendations from diagnostic plan
    recommendations.extend([
        "📋 GENERAL: Follow diagnostic plan Phase 1-3 implementation",
        "📋 GENERAL: Implement hierarchical clinical model architecture",
        "📋 GENERAL: Add clinical safety loss function",
        "📋 GENERAL: Conduct clinical expert review of edge cases"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    # Next Steps
    print(f"\n=== IMMEDIATE NEXT STEPS ===")
    
    if overall_status == "CRITICAL":
        print("🚨 IMMEDIATE ACTION REQUIRED:")
        print("   1. Do NOT deploy current model in clinical setting")
        print("   2. Address critical data quality issues first")
        print("   3. Implement diagnostic plan Phase 1 fixes")
        print("   4. Re-run diagnostics to verify improvements")
    elif overall_status == "HIGH RISK":
        print("⚠️  HIGH PRIORITY ACTIONS:")
        print("   1. Address high-priority issues before deployment")
        print("   2. Implement diagnostic plan recommendations")
        print("   3. Conduct additional validation testing")
        print("   4. Consider clinical expert consultation")
    else:
        print("✅ MODERATE PRIORITY ACTIONS:")
        print("   1. Continue with planned improvements")
        print("   2. Monitor performance in validation environment")
        print("   3. Prepare for clinical validation phase")
        print("   4. Document all changes and improvements")
    
    # File locations
    print(f"\n=== DIAGNOSTIC REPORTS ===")
    print("Detailed reports saved to:")
    print("  📁 results/simple_diagnostic_report_*.json")
    print("  📁 results/final_fix_diagnostic_test_*.json")
    print("  📁 analysis/triage_model_diagnostic_plan.md")
    
    print(f"\n=== DIAGNOSTIC COMPLETE ===")
    print(f"Overall Assessment: {status_icon} {overall_status}")
    print("Review detailed reports and implement recommendations.")

if __name__ == "__main__":
    main()