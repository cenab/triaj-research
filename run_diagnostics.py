#!/usr/bin/env python3
"""
Simple script to run the triage model diagnostics
"""

import sys
import os

# Add the tests directory to the path
sys.path.append('tests')

from test_diagnostic_issues import TriageModelDiagnostics, run_specific_diagnostic_tests

def main():
    """Main function to run diagnostics"""
    data_path = 'src/triaj_data.csv'
    
    print("=== TRIAGE MODEL DIAGNOSTIC SUITE ===")
    print("This will diagnose the issues identified in the diagnostic plan:")
    print("1. Data Quality Problems")
    print("2. Clinical Logic Violations")
    print("3. Model Architecture Problems")
    print("4. Performance Issues")
    print()
    
    print("Choose diagnostic mode:")
    print("1. Full comprehensive diagnostics (recommended)")
    print("2. Quick specific issue tests only")
    print("3. Both comprehensive and specific tests")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice not in ['1', '2', '3']:
        print("Invalid choice. Running comprehensive diagnostics by default.")
        choice = '1'
    
    try:
        if choice in ['1', '3']:
            print("\n" + "="*60)
            print("RUNNING COMPREHENSIVE DIAGNOSTICS")
            print("="*60)
            
            diagnostics = TriageModelDiagnostics(data_path)
            results = diagnostics.run_full_diagnostics()
            
            if results:
                print("\n✅ Comprehensive diagnostics completed successfully!")
            else:
                print("\n❌ Comprehensive diagnostics failed!")
        
        if choice in ['2', '3']:
            print("\n" + "="*60)
            print("RUNNING SPECIFIC DIAGNOSTIC TESTS")
            print("="*60)
            
            test_results = run_specific_diagnostic_tests()
            
            if test_results:
                issues_found = sum(test_results.values())
                print(f"\n✅ Specific tests completed! Found {issues_found} issues.")
            else:
                print("\n❌ Specific tests failed!")
        
        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYSIS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the diagnostic report in results/")
        print("2. Address critical issues first")
        print("3. Implement recommendations from the diagnostic plan")
        print("4. Re-run diagnostics to verify improvements")
        
    except Exception as e:
        print(f"\n❌ Error running diagnostics: {e}")
        print("Make sure you're running from the project root directory")
        print("and that all dependencies are installed.")

if __name__ == "__main__":
    main()