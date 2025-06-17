# Triage Model Diagnostic Test Suite

This comprehensive diagnostic test suite is designed to identify and analyze the critical issues found in the triage model as outlined in `analysis/triage_model_diagnostic_plan.md`.

## Overview

The diagnostic suite consists of multiple test files that analyze different aspects of the triage model:

1. **Data Quality Issues**
2. **Clinical Logic Violations** 
3. **Model Architecture Problems**
4. **Performance Issues**

## Test Files

### 1. `tests/test_diagnostic_simple.py`
**Purpose**: Quick data structure and basic issue detection
- Tests target variable consistency
- Analyzes class distribution imbalance
- Checks vital signs logic violations
- Examines feature structure problems
- Validates data quality

**Usage**:
```bash
python tests/test_diagnostic_simple.py
```

### 2. `tests/test_final_fix_issues.py`
**Purpose**: Tests the actual performance of `src/final_fix.py` against diagnostic plan targets
- Runs the complete model training and evaluation
- Compares results against target metrics
- Tests specific diagnostic plan criteria
- Provides detailed performance analysis

**Usage**:
```bash
python tests/test_final_fix_issues.py
```

### 3. `tests/test_diagnostic_issues.py`
**Purpose**: Comprehensive diagnostic analysis (requires working feature engineering)
- Deep analysis of all diagnostic plan issues
- Advanced model architecture testing
- Clinical metrics evaluation
- Detailed recommendations generation

**Note**: This test may fail due to feature engineering issues. Use the simple version instead.

## Test Runners

### 1. `run_diagnostics.py`
Simple interactive test runner with options:
```bash
python run_diagnostics.py
```
Choose from:
1. Full comprehensive diagnostics
2. Quick specific issue tests only
3. Both

### 2. `run_comprehensive_diagnostics.py`
**Recommended**: Complete diagnostic suite runner
```bash
python run_comprehensive_diagnostics.py
```

This runs all working diagnostic tests and provides:
- Comprehensive analysis
- Prioritized recommendations
- Overall risk assessment
- Next steps guidance

## Expected Issues (Based on Diagnostic Plan)

### Critical Issues
1. **Target Variable Inconsistency**: Mismatches between initial and correct triage
2. **Class Imbalance**: Severe imbalance (6:1 ratio found)
3. **Clinical Logic Violations**: Cases with abnormal vitals but incorrect triage levels

### High Priority Issues
1. **Feature Imbalance**: Too many text features vs numerical vital signs
2. **Data Leakage**: Post-hoc diagnostic features in training data
3. **Under-triage Risk**: Model missing critical cases

### Performance Targets (from Diagnostic Plan)
- **Overall Accuracy**: >75% (Target)
- **Under-triage Rate**: <15% (Safety Critical)
- **Critical Sensitivity**: >95% (Most Important)

## Diagnostic Results

All diagnostic tests save detailed results to the `results/` directory:

- `simple_diagnostic_report_*.json`: Basic data analysis results
- `final_fix_diagnostic_test_*.json`: Model performance test results
- `diagnostic_report_*.json`: Comprehensive analysis (if working)

## Interpreting Results

### Severity Levels
- **CRITICAL**: Immediate action required, do not deploy
- **HIGH**: Significant issues, address before deployment
- **MODERATE**: Some improvements needed
- **LOW**: Minor issues, acceptable for development

### Key Metrics to Monitor
1. **Critical Sensitivity**: Must be >95% for clinical safety
2. **Under-triage Rate**: Must be <15% to avoid missing critical cases
3. **Class Imbalance Ratio**: Should be <3:1 for stable training
4. **Feature Ratio**: Text:Numerical should be <5:1 for efficiency

## Recommendations Implementation

Based on diagnostic results, implement fixes in this order:

### Phase 1: Data Quality (Week 1)
1. Resolve target variable inconsistencies
2. Remove data leakage features
3. Implement smart feature engineering

### Phase 2: Model Architecture (Week 2)
1. Implement hierarchical clinical model
2. Add clinical safety loss function
3. Use multi-stage training protocol

### Phase 3: Validation & Optimization (Week 3)
1. Clinical validation framework
2. Temporal validation implementation
3. Performance optimization

## Troubleshooting

### Common Issues

1. **Feature Engineering Errors**
   - Use `test_diagnostic_simple.py` instead of full diagnostics
   - Check column names in actual data vs expected

2. **Import Errors**
   - Run from project root directory
   - Ensure all dependencies are installed
   - Check Python path includes src/ directory

3. **Memory Issues**
   - Reduce batch size in model testing
   - Use CPU instead of GPU for diagnostics
   - Limit training epochs for quick testing

### Quick Diagnostic Command
For fastest results:
```bash
python tests/test_diagnostic_simple.py
```

### Full Analysis Command
For complete analysis:
```bash
python run_comprehensive_diagnostics.py
```

## Integration with Development Workflow

1. **Before Making Changes**: Run diagnostics to establish baseline
2. **After Implementing Fixes**: Re-run to verify improvements
3. **Before Deployment**: Ensure all critical issues are resolved
4. **Regular Monitoring**: Run weekly during development

## Clinical Safety Notes

⚠️ **IMPORTANT**: Do not deploy the model in clinical settings until:
- Critical sensitivity >95%
- Under-triage rate <15%
- All critical issues resolved
- Clinical expert validation completed

## Support

For issues with the diagnostic tests:
1. Check the diagnostic plan: `analysis/triage_model_diagnostic_plan.md`
2. Review test output and error messages
3. Ensure data file exists: `src/triaj_data.csv`
4. Verify all dependencies are installed

## Files Created by This Diagnostic Suite

- `tests/test_diagnostic_simple.py`: Simple diagnostic tests
- `tests/test_final_fix_issues.py`: Model performance tests
- `tests/test_diagnostic_issues.py`: Comprehensive diagnostics
- `run_diagnostics.py`: Interactive test runner
- `run_comprehensive_diagnostics.py`: Complete test suite runner
- `DIAGNOSTIC_TESTS_README.md`: This documentation

All tests are designed to work with the existing codebase and identify the specific issues mentioned in the diagnostic plan.