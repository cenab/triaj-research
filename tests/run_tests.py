"""
Test runner for Phase 2 and Phase 3 implementations.
Runs all unit tests and integration tests with comprehensive reporting.
"""

import unittest
import sys
import time
from io import StringIO
import traceback

def run_test_suite():
    """Run all test suites and provide comprehensive reporting"""
    
    print("=" * 80)
    print("FAIRTRIEDGE-FL PHASE 2 & 3 TEST SUITE")
    print("=" * 80)
    print()
    
    # Test modules to run
    test_modules = [
        'test_robust_aggregation',
        'test_communication_efficiency', 
        'test_domain_adaptation',
        'test_explainable_ai',
        'test_integration'
    ]
    
    # Results tracking
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    module_results = {}
    
    start_time = time.time()
    
    for module_name in test_modules:
        print(f"Running {module_name}...")
        print("-" * 60)
        
        try:
            # Import the test module
            test_module = __import__(module_name)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with custom result handler
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream, 
                verbosity=2,
                buffer=True
            )
            
            module_start_time = time.time()
            result = runner.run(suite)
            module_end_time = time.time()
            
            # Collect results
            module_results[module_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success': result.wasSuccessful(),
                'time': module_end_time - module_start_time,
                'details': stream.getvalue()
            }
            
            # Update totals
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped) if hasattr(result, 'skipped') else 0
            
            # Print summary for this module
            if result.wasSuccessful():
                print(f"✅ {module_name}: {result.testsRun} tests passed")
            else:
                print(f"❌ {module_name}: {len(result.failures)} failures, {len(result.errors)} errors")
                
                # Print failure details
                if result.failures:
                    print("  Failures:")
                    for test, traceback_str in result.failures:
                        print(f"    - {test}: {traceback_str.split('AssertionError:')[-1].strip()}")
                
                if result.errors:
                    print("  Errors:")
                    for test, traceback_str in result.errors:
                        error_msg = traceback_str.split('\n')[-2] if '\n' in traceback_str else traceback_str
                        print(f"    - {test}: {error_msg}")
            
            print(f"  Time: {module_end_time - module_start_time:.2f}s")
            print()
            
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            module_results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'time': 0,
                'details': f"Import error: {e}"
            }
            total_errors += 1
            print()
            
        except Exception as e:
            print(f"❌ Unexpected error in {module_name}: {e}")
            traceback.print_exc()
            module_results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'time': 0,
                'details': f"Unexpected error: {e}"
            }
            total_errors += 1
            print()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print comprehensive summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Successes: {total_tests - total_failures - total_errors}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Total Time: {total_time:.2f}s")
    print()
    
    # Module-by-module breakdown
    print("MODULE BREAKDOWN:")
    print("-" * 40)
    for module_name, results in module_results.items():
        status = "✅ PASS" if results['success'] else "❌ FAIL"
        print(f"{module_name:25} {status:8} {results['tests_run']:3}T {results['time']:6.2f}s")
    print()
    
    # Feature coverage summary
    print("FEATURE COVERAGE:")
    print("-" * 40)
    
    features_tested = {
        "Robust Aggregation": "test_robust_aggregation" in module_results and module_results["test_robust_aggregation"]["success"],
        "Communication Efficiency": "test_communication_efficiency" in module_results and module_results["test_communication_efficiency"]["success"],
        "Domain Adaptation": "test_domain_adaptation" in module_results and module_results["test_domain_adaptation"]["success"],
        "Explainable AI": "test_explainable_ai" in module_results and module_results["test_explainable_ai"]["success"],
        "End-to-End Integration": "test_integration" in module_results and module_results["test_integration"]["success"]
    }
    
    for feature, passed in features_tested.items():
        status = "✅ IMPLEMENTED" if passed else "❌ ISSUES"
        print(f"{feature:25} {status}")
    
    print()
    
    # Implementation status
    phase2_features = ["Robust Aggregation", "Communication Efficiency"]
    phase3_features = ["Domain Adaptation", "Explainable AI", "End-to-End Integration"]
    
    phase2_complete = all(features_tested[f] for f in phase2_features)
    phase3_complete = all(features_tested[f] for f in phase3_features)
    
    print("IMPLEMENTATION STATUS:")
    print("-" * 40)
    print(f"Phase 2 (Robustness): {'✅ COMPLETE' if phase2_complete else '❌ INCOMPLETE'}")
    print(f"Phase 3 (Advanced): {'✅ COMPLETE' if phase3_complete else '❌ INCOMPLETE'}")
    print()
    
    # Recommendations
    if total_failures > 0 or total_errors > 0:
        print("RECOMMENDATIONS:")
        print("-" * 40)
        
        if total_errors > 0:
            print("• Fix import errors and runtime exceptions first")
        if total_failures > 0:
            print("• Review failed assertions and fix implementation logic")
        
        failed_modules = [name for name, results in module_results.items() if not results['success']]
        if failed_modules:
            print(f"• Focus on modules: {', '.join(failed_modules)}")
        print()
    
    # Overall result
    overall_success = total_failures == 0 and total_errors == 0
    
    if overall_success:
        print("🎉 ALL TESTS PASSED! Phase 2 and Phase 3 implementation is complete.")
    else:
        print(f"⚠️  {total_failures + total_errors} issues found. Please review and fix before deployment.")
    
    print("=" * 80)
    
    return overall_success, module_results

def run_specific_test(test_name):
    """Run a specific test module"""
    print(f"Running specific test: {test_name}")
    print("-" * 60)
    
    try:
        test_module = __import__(test_name)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Failed to import {test_name}: {e}")
        return False
    except Exception as e:
        print(f"Error running {test_name}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test runner function"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        success, results = run_test_suite()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()