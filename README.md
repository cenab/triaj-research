# FairTriEdge-FL: Federated Learning for Medical Triage

A comprehensive federated learning system for medical triage with privacy preservation, fairness monitoring, and explainable AI capabilities.

## 📁 Project Structure

```
triaj-research/
├── README.md                 # This file
├── src/                      # Source code
│   ├── main.py              # Main entry point
│   ├── data_preparation.py  # Data preprocessing
│   ├── data_simulation.py   # Multi-site data simulation
│   ├── feature_engineering.py # Feature extraction
│   ├── model_architecture.py # Model definitions
│   ├── model_optimization.py # TinyML optimizations
│   ├── federated_learning.py # FL implementation
│   ├── explainable_ai.py    # XAI features
│   └── setup_and_run.py     # Interactive setup
├── config/                   # Configuration files
│   ├── config.json          # Main configuration
│   ├── config_loader.py     # Config utilities
│   ├── config_templates.py  # Config templates
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
├── tests/                    # Test suite
│   ├── run_tests.py         # Test runner
│   ├── test_integration.py  # Integration tests
│   ├── test_communication_efficiency.py
│   ├── test_domain_adaptation.py
│   ├── test_explainable_ai.py
│   └── test_robust_aggregation.py
├── docs/                     # Documentation
│   ├── fair_triedge_fl_research_plan.md
│   ├── FairTriEdge-FL_Implementation_Plan.md
│   ├── Implementation_Roadmap.md
│   ├── TODO_Implementation_Plan.md
│   └── ... (other documentation files)
├── data/                     # Data files
│   └── triaj_data.csv       # Medical triage dataset
└── logs/                     # Log files
    └── log.txt              # Application logs
```

## � Quick Start

### 1. Install Dependencies
```bash
pip install -r config/requirements.txt
```

### 2. Configure API Keys
Edit the `config/.env` file and add your API keys:

```bash
# For OpenRouter (recommended - supports multiple models)
OPENROUTER_API_KEY=your_actual_api_key_here

# OR for OpenAI (fallback)
OPENAI_API_KEY=your_actual_api_key_here
```

**Getting API Keys:**
- **OpenRouter** (recommended): Sign up at [openrouter.ai](https://openrouter.ai/) - supports Claude, GPT-4, Llama, and more
- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com/)

### 3. Prepare Your Data
Ensure your medical triage data is saved as `data/triaj_data.csv`.

### 4. Run the Experiment

**Option A: Interactive Setup (Recommended)**
```bash
python src/setup_and_run.py
```

**Option B: Direct Run**
```bash
python src/main.py
```

**Option C: Run Tests Only**
```bash
python tests/run_tests.py
```

## 🔧 Configuration System

### Quick Configuration Setup

**Option A: Use Configuration Templates**
```bash
# List available templates
python config_templates.py list

# Create a configuration from template
python config_templates.py basic config.json
python config_templates.py privacy config_privacy.json
python config_templates.py fairness config_fairness.json
```

**Option B: Interactive Setup**
```bash
python setup_and_run.py
# Choose option 4 to create config from template
# Choose option 5 to validate current config
```

### Available Configuration Templates

- **basic**: Simple federated learning experiment
- **privacy**: Privacy-focused with strong differential privacy
- **fairness**: Fairness-focused with bias monitoring
- **robustness**: Robustness-focused with Byzantine fault tolerance
- **xai**: Explainable AI focused with comprehensive explanations
- **performance**: Performance-optimized for resource constraints
- **research**: Comprehensive research experiment with all features

### Custom Configuration

Edit `config.json` to customize:
```json
{
  "federated_learning": {
    "num_rounds": 10,
    "num_clients": 5,
    "aggregation_method": "krum"
  },
  "privacy": {
    "enable_differential_privacy": true,
    "epsilon": 1.0
  },
  "explainable_ai": {
    "enable_xai": true,
    "llm_explanation": {
      "provider": "openrouter",
      "model": "anthropic/claude-3-sonnet"
    }
  }
}
```

## � What This Experiment Does

The system runs a complete 5-phase federated learning experiment:

1. **Phase 1**: Data preparation, feature engineering, and multi-site simulation
2. **Phase 2**: Core model development and optimization (TinyML techniques)
3. **Phase 3**: Federated learning with privacy, robustness, and fairness features
4. **Phase 4**: Explainable AI with feature importance and LLM explanations
5. **Phase 5**: Comprehensive evaluation

## 🔧 Configuration

You can customize the experiment by editing the `.env` file:

```bash
# LLM Settings
LLM_PROVIDER=openrouter                    # or "openai"
OPENROUTER_MODEL=anthropic/claude-3-sonnet # or other models

# Privacy Settings
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5

# Federated Learning Settings
FL_NUM_ROUNDS=5
FL_NUM_CLIENTS=3
FL_AGGREGATION_METHOD=fedavg

# Model Settings
BATCH_SIZE=32
LEARNING_RATE=0.001
```

## 📋 Data Requirements

Your `triaj_data.csv` should contain:
- **Patient Demographics**: Age, gender
- **Vital Signs**: Blood pressure, heart rate, temperature, etc.
- **Symptoms**: Binary or categorical indicators
- **Target Variable**: Triage level (0=Green/Low, 1=Yellow/Moderate, 2=Red/High)

## 🧪 Testing

Run the comprehensive test suite:
```bash
python setup_and_run.py  # Choose option 2
# OR
python run_tests.py
```

## ⚠️ Current Status

**Note**: This is a research prototype. Many advanced features are currently placeholder implementations (marked as TODO in the code). The system will run but with limited functionality for:
- Advanced privacy mechanisms
- Robust aggregation algorithms
- Domain adaptation techniques
- Comprehensive fairness monitoring

## 🔍 Troubleshooting

### Common Issues:

1. **Missing API Keys**: Edit `.env` file with your actual API keys
2. **Missing Data**: Ensure `triaj_data.csv` exists in the project directory
3. **Memory Issues**: Reduce `BATCH_SIZE` and `FL_NUM_CLIENTS` in `.env`
4. **CUDA Issues**: The system will automatically fall back to CPU

### Getting Help:

1. Run `python setup_and_run.py` for interactive setup and configuration
2. Use `python config_loader.py` to validate your configuration
3. Check the console output for specific error messages
4. Review the test results with `python run_tests.py`

### Configuration Management:

```bash
# Validate current configuration
python config_loader.py

# Create configuration from template
python config_templates.py research

# Interactive configuration setup
python setup_and_run.py  # Choose options 4 or 5
```

## 📚 Documentation

- [`FairTriEdge-FL_User_Guide.md`](FairTriEdge-FL_User_Guide.md) - Complete user guide
- [`Implementation_Roadmap.md`](Implementation_Roadmap.md) - Development roadmap
- [`TODO_Implementation_Plan.md`](TODO_Implementation_Plan.md) - Implementation plan

## 🏥 Research Context

This system is designed for research in federated learning for healthcare applications, specifically medical triage. It demonstrates:
- Privacy-preserving machine learning
- Fairness in AI systems
- Explainable AI for medical decisions
- Robust federated learning techniques

## 📄 License

This project is for research and educational purposes. Please ensure compliance with healthcare data regulations (HIPAA, GDPR, etc.) when using with real medical data.

---

**Happy Experimenting! 🚀**