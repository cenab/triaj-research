# FairTriEdge-FL: Federated Learning for Medical Triage - Complete User Guide

## Overview

FairTriEdge-FL is a comprehensive federated learning system designed for medical triage applications. It combines privacy-preserving machine learning with advanced explainable AI, fairness monitoring, and robustness features to provide accurate, fair, and interpretable triage decisions across distributed healthcare environments.

## 🚀 Key Features

### Core Federated Learning
- **Multi-client Training**: Distributed learning across multiple healthcare sites
- **Privacy Preservation**: Differential privacy with Opacus integration
- **Model Aggregation**: FedAvg, Krum, Trimmed Mean, and Median aggregation
- **Communication Efficiency**: Top-k sparsification and quantization (up to 95% bandwidth reduction)

### Advanced AI & Explainability
- **Feature Importance**: SHAP and Permutation Importance analysis
- **LLM Integration**: Multi-provider support (OpenRouter + OpenAI fallback)
- **Medical Context**: Specialized explanations for triage decisions
- **Real-time Interpretability**: Boolean rule extraction and visualization

### Robustness & Fairness
- **Byzantine Fault Tolerance**: Protection against malicious clients
- **Data Drift Detection**: ADWIN and KS-Test monitoring
- **Domain Adaptation**: DANN and MMD for heterogeneous data
- **Fairness Monitoring**: Comprehensive subgroup evaluation across demographics

### Quality Assurance
- **Comprehensive Testing**: 154+ unit and integration tests
- **Performance Benchmarking**: Accuracy, privacy, and fairness metrics
- **Error Handling**: Graceful degradation and fallback mechanisms

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Required Dependencies
```bash
# Install core dependencies
pip install torch torchvision numpy pandas scikit-learn

# Install federated learning and privacy libraries
pip install alibi-detect shap opacus scipy

# Install explainable AI and LLM libraries
pip install requests openai

# Install testing libraries
pip install pytest pytest-cov
```

### Optional Dependencies
```bash
# For enhanced fairness monitoring
pip install aif360

# For advanced model optimization
pip install torchmetrics

# For visualization
pip install matplotlib seaborn
```

## 🗂️ Data Preparation

### Data Format Requirements

Your medical triage data should be in CSV format with the following structure:

```csv
patient_id,age,systolic_bp,diastolic_bp,symptom_1,symptom_2,...,triage_level
1,45,120,80,chest_pain,shortness_of_breath,...,2
2,67,140,90,headache,nausea,...,1
...
```

### Required Columns
- **Patient Demographics**: Age, gender (optional for fairness analysis)
- **Vital Signs**: Blood pressure, heart rate, temperature, etc.
- **Symptoms**: Binary or categorical symptom indicators
- **Target Variable**: Triage level (0=Green/Low, 1=Yellow/Moderate, 2=Red/High)

### Data Preprocessing Steps

1. **Place your data file** in the project directory as `triaj_data.csv`
2. **Update column mappings** in `data_preparation.py` if needed:
   ```python
   # Update these mappings to match your data
   VITAL_SIGNS_COLUMNS = ['yaş', 'sistolik kb', 'diastolik kb', ...]
   TARGET_COLUMN = 'doğru triyaj'
   ```

3. **Configure sensitive attributes** for fairness analysis:
   ```python
   # In data_preparation.py
   SENSITIVE_ATTRIBUTES = ['age_group', 'gender', 'ethnicity']
   ```

## 🚀 Quick Start

### 1. Basic Experiment Run
```bash
# Run the complete federated learning experiment
python main.py
```

### 2. Custom Configuration
Create a configuration file `config.json`:
```json
{
  "federated_learning": {
    "num_rounds": 10,
    "num_clients": 5,
    "aggregation_method": "krum",
    "privacy": {
      "epsilon": 1.0,
      "delta": 1e-5,
      "enable_dp": true
    }
  },
  "explainable_ai": {
    "feature_importance_method": "shap",
    "llm_provider": "openrouter",
    "llm_model": "anthropic/claude-3-sonnet"
  },
  "fairness": {
    "enable_monitoring": true,
    "sensitive_attributes": ["age_group", "gender"],
    "fairness_metrics": ["f1_score_parity", "demographic_parity"]
  }
}
```

### 3. Run with Custom Configuration
```bash
python main.py --config config.json
```

## 🔧 Advanced Configuration

### Federated Learning Parameters

```python
# In main.py or configuration file
FL_CONFIG = {
    "num_rounds": 10,           # Number of federated learning rounds
    "num_clients": 3,           # Number of simulated clients
    "local_epochs": 5,          # Local training epochs per round
    "learning_rate": 0.001,     # Learning rate for local training
    "batch_size": 32,           # Batch size for training
    
    # Aggregation method: 'fedavg', 'krum', 'trimmed_mean', 'median'
    "aggregation_method": "krum",
    
    # Communication efficiency
    "compression_ratio": 0.1,   # For top-k sparsification
    "quantization_bits": 2,     # For quantization (1, 2, 4, 8)
    
    # Privacy settings
    "enable_differential_privacy": True,
    "epsilon": 1.0,             # Privacy budget
    "delta": 1e-5,              # Privacy parameter
    
    # Robustness settings
    "num_malicious_clients": 0, # For Krum aggregation
    "trim_ratio": 0.1,          # For trimmed mean
}
```

### Explainable AI Configuration

```python
XAI_CONFIG = {
    # Feature importance method: 'shap', 'permutation', 'both'
    "feature_importance_method": "shap",
    
    # LLM configuration
    "llm_provider": "openrouter",  # 'openrouter' or 'openai'
    "llm_model": "anthropic/claude-3-sonnet",
    "api_key_env": "OPENROUTER_API_KEY",
    
    # Explanation settings
    "max_features_to_explain": 5,
    "explanation_length": 200,
    "medical_context": True,
}
```

### Fairness Monitoring Configuration

```python
FAIRNESS_CONFIG = {
    "enable_monitoring": True,
    "sensitive_attributes": ["age_group", "gender"],
    "fairness_metrics": [
        "f1_score_parity",
        "demographic_parity", 
        "equalized_odds"
    ],
    "fairness_threshold": 0.1,  # Maximum allowed disparity
    "generate_reports": True,
}
```

## 🔑 API Keys Setup

### OpenRouter API (Recommended)
1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key
3. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

### OpenAI API (Fallback)
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Get your API key
3. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

## 📊 Understanding the Output

### Federated Learning Results
```
--- Federated Learning Round 1/5 ---
Client client_0: Performing local training...
Client client_0 trained. Avg Loss: 0.9474
...
--- Global Model Evaluation after Round 1 ---
Global model evaluation: Accuracy = 68.52%
```

### Privacy Metrics
```
--- Phase 3.2: Privacy Preservation ---
Applying Differential Privacy (epsilon=1.0, delta=1e-05)
Calculated noise multiplier: 0.849609375
Privacy budget consumed: 1.0/10.0 (10.0%)
```

### Fairness Analysis
```
--- Phase 3.5: Fairness Monitoring ---
Fairness Analysis Results:
- Age Group 0-30: F1-Score = 0.72, Demographic Parity = 0.45
- Age Group 31-60: F1-Score = 0.68, Demographic Parity = 0.52
- Age Group 60+: F1-Score = 0.71, Demographic Parity = 0.48
Fairness Violation Detected: Demographic parity disparity = 0.07
```

### Explainable AI Output
```
--- Phase 4: Explainable AI ---
Top 5 Feature Importance Scores (SHAP):
- systolic_bp: 0.1890
- age: 0.1716
- chest_pain: 0.1205
- shortness_of_breath: 0.0987
- heart_rate: 0.0823

LLM Explanation:
Based on the patient's presentation, the AI system has triaged them to the Yellow Area (Urgent) primarily due to elevated systolic blood pressure (189 mmHg) combined with reported chest pain...
```

## 🧪 Testing Your Implementation

### Run All Tests
```bash
# Run comprehensive test suite
python run_tests.py

# Run specific test categories
python -m pytest test_robust_aggregation.py -v
python -m pytest test_explainable_ai.py -v
python -m pytest test_integration.py -v
```

### Performance Benchmarking
```bash
# Run with benchmarking enabled
python main.py --benchmark --output-dir results/
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce batch size and number of clients
   python main.py --batch-size 16 --num-clients 2
   ```

2. **API Rate Limits**
   ```python
   # In explainable_ai.py, increase delay between calls
   time.sleep(1)  # Add delay between API calls
   ```

3. **Missing Dependencies**
   ```bash
   # Install missing libraries
   pip install alibi-detect shap opacus
   ```

4. **CUDA Issues**
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   python main.py
   ```

### Data Issues

1. **Column Name Mismatches**
   - Update column mappings in `data_preparation.py`
   - Ensure target variable is properly encoded (0, 1, 2)

2. **Missing Values**
   - The system handles missing values automatically
   - Check data quality with: `python -c "import pandas as pd; print(pd.read_csv('triaj_data.csv').info())"`

3. **Imbalanced Classes**
   - The system includes class balancing
   - Monitor class distribution in output logs

## 📈 Performance Optimization

### For Large Datasets
```python
# Optimize for memory usage
FL_CONFIG.update({
    "batch_size": 16,           # Smaller batches
    "gradient_accumulation": 4, # Accumulate gradients
    "use_mixed_precision": True # FP16 training
})
```

### For Limited Compute
```python
# Reduce computational complexity
MODEL_CONFIG.update({
    "hidden_size": 64,          # Smaller model
    "num_layers": 2,            # Fewer layers
    "dropout": 0.3              # Higher dropout
})
```

## 🔒 Security Considerations

### Privacy Best Practices
1. **Set appropriate privacy budgets**: ε ≤ 1.0 for strong privacy
2. **Monitor privacy consumption**: Track cumulative privacy loss
3. **Use secure aggregation**: Enable Byzantine fault tolerance
4. **Audit data access**: Log all data interactions

### Production Deployment
1. **Secure API keys**: Use environment variables or secret management
2. **Network security**: Use HTTPS for all communications
3. **Access control**: Implement proper authentication
4. **Audit logging**: Log all system activities

## 📚 Additional Resources

### Research Papers
- [FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [SHAP: A Unified Approach to Explaining Machine Learning Model Predictions](https://arxiv.org/abs/1705.07874)

### Documentation
- [PyTorch Federated Learning](https://pytorch.org/tutorials/intermediate/federated_learning_tutorial.html)
- [Opacus Documentation](https://opacus.ai/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 🤝 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Examine the implementation in `federated_learning.py` and `explainable_ai.py`
4. Run the test suite to verify your setup

## 📄 License

This project is designed for research and educational purposes. Please ensure compliance with healthcare data regulations (HIPAA, GDPR, etc.) when using with real medical data.

---

**Happy Experimenting with FairTriEdge-FL!** 🏥🤖