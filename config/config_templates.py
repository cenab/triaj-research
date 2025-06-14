#!/usr/bin/env python3
"""
Configuration Templates for FairTriEdge-FL
Provides pre-configured templates for different experiment scenarios
"""

import json
import os
from typing import Dict, Any

class ConfigTemplates:
    """Pre-configured templates for different experiment scenarios"""
    
    @staticmethod
    def basic_experiment() -> Dict[str, Any]:
        """Basic federated learning experiment with minimal features"""
        return {
            "experiment": {
                "name": "Basic FL Experiment",
                "description": "Simple federated learning for medical triage"
            },
            "federated_learning": {
                "num_rounds": 3,
                "num_clients": 2,
                "local_epochs": 1,
                "aggregation_method": "fedavg"
            },
            "model": {
                "training": {
                    "batch_size": 16,
                    "learning_rate": 0.001
                }
            },
            "privacy": {
                "enable_differential_privacy": False
            },
            "explainable_ai": {
                "enable_xai": False
            },
            "fairness": {
                "enable_monitoring": False
            }
        }
    
    @staticmethod
    def privacy_focused() -> Dict[str, Any]:
        """Privacy-focused experiment with strong differential privacy"""
        return {
            "experiment": {
                "name": "Privacy-Focused FL",
                "description": "Federated learning with strong privacy guarantees"
            },
            "federated_learning": {
                "num_rounds": 10,
                "num_clients": 5,
                "local_epochs": 2,
                "aggregation_method": "fedavg"
            },
            "model": {
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            },
            "privacy": {
                "enable_differential_privacy": True,
                "epsilon": 0.5,  # Strong privacy
                "delta": 1e-6,
                "max_grad_norm": 0.5
            },
            "robustness": {
                "enable_robust_aggregation": True,
                "aggregation_methods": ["fedavg", "krum"]
            },
            "explainable_ai": {
                "enable_xai": True,
                "llm_explanation": {
                    "enable": False  # Disable to avoid data leakage
                }
            }
        }
    
    @staticmethod
    def fairness_focused() -> Dict[str, Any]:
        """Fairness-focused experiment with comprehensive bias monitoring"""
        return {
            "experiment": {
                "name": "Fairness-Focused FL",
                "description": "Federated learning with comprehensive fairness monitoring"
            },
            "federated_learning": {
                "num_rounds": 8,
                "num_clients": 4,
                "local_epochs": 3
            },
            "fairness": {
                "enable_monitoring": True,
                "sensitive_attributes": ["age_group", "gender"],
                "fairness_metrics": [
                    "f1_score_parity",
                    "demographic_parity",
                    "equalized_odds"
                ],
                "fairness_threshold": 0.05,  # Strict fairness
                "subgroup_evaluation": True
            },
            "explainable_ai": {
                "enable_xai": True,
                "feature_importance": {
                    "methods": ["shap", "permutation"],
                    "num_features_to_explain": 10
                }
            }
        }
    
    @staticmethod
    def robustness_focused() -> Dict[str, Any]:
        """Robustness-focused experiment with Byzantine fault tolerance"""
        return {
            "experiment": {
                "name": "Robustness-Focused FL",
                "description": "Federated learning with Byzantine fault tolerance"
            },
            "federated_learning": {
                "num_rounds": 12,
                "num_clients": 6,
                "local_epochs": 2
            },
            "robustness": {
                "enable_robust_aggregation": True,
                "aggregation_methods": ["krum", "trimmed_mean", "median"],
                "byzantine_clients": 1,  # Simulate malicious clients
                "trim_ratio": 0.2,
                "data_drift_detection": {
                    "enable": True,
                    "methods": ["adwin", "ks_test"],
                    "threshold": 0.01
                },
                "domain_adaptation": {
                    "enable": True,
                    "methods": ["dann", "mmd"]
                }
            },
            "communication": {
                "enable_compression": True,
                "compression_methods": ["top_k", "quantization"],
                "compression_ratio": 0.05  # High compression
            }
        }
    
    @staticmethod
    def explainable_ai_focused() -> Dict[str, Any]:
        """XAI-focused experiment with comprehensive explanations"""
        return {
            "experiment": {
                "name": "XAI-Focused FL",
                "description": "Federated learning with comprehensive explainable AI"
            },
            "federated_learning": {
                "num_rounds": 6,
                "num_clients": 3,
                "local_epochs": 2
            },
            "explainable_ai": {
                "enable_xai": True,
                "feature_importance": {
                    "methods": ["shap", "permutation"],
                    "num_features_to_explain": 8,
                    "background_samples": 200
                },
                "llm_explanation": {
                    "enable": True,
                    "provider": "openrouter",
                    "model": "anthropic/claude-3-sonnet",
                    "max_tokens": 300,
                    "medical_context": True,
                    "explanation_style": "clinical"
                },
                "rule_extraction": {
                    "enable": True,
                    "max_rules": 15,
                    "min_support": 0.05
                }
            }
        }
    
    @staticmethod
    def performance_optimized() -> Dict[str, Any]:
        """Performance-optimized experiment for resource-constrained environments"""
        return {
            "experiment": {
                "name": "Performance-Optimized FL",
                "description": "Resource-efficient federated learning"
            },
            "federated_learning": {
                "num_rounds": 5,
                "num_clients": 2,
                "local_epochs": 1
            },
            "model": {
                "hidden_size": 64,  # Smaller model
                "num_layers": 2,
                "training": {
                    "batch_size": 16,  # Smaller batches
                    "learning_rate": 0.002
                }
            },
            "optimization": {
                "enable_tinyml": True,
                "techniques": ["pruning", "quantization"],
                "pruning": {
                    "enable": True,
                    "amount": 0.7  # Aggressive pruning
                },
                "quantization": {
                    "enable": True
                }
            },
            "communication": {
                "enable_compression": True,
                "compression_ratio": 0.01  # Maximum compression
            },
            "hardware": {
                "mixed_precision": True,
                "num_workers": 2
            }
        }
    
    @staticmethod
    def comprehensive_research() -> Dict[str, Any]:
        """Comprehensive research experiment with all features enabled"""
        return {
            "experiment": {
                "name": "Comprehensive Research FL",
                "description": "Full-featured federated learning research experiment"
            },
            "federated_learning": {
                "num_rounds": 15,
                "num_clients": 8,
                "local_epochs": 3,
                "aggregation_method": "krum"
            },
            "privacy": {
                "enable_differential_privacy": True,
                "epsilon": 1.0,
                "delta": 1e-5
            },
            "robustness": {
                "enable_robust_aggregation": True,
                "aggregation_methods": ["fedavg", "krum", "trimmed_mean"],
                "data_drift_detection": {
                    "enable": True,
                    "methods": ["adwin", "ks_test"]
                },
                "domain_adaptation": {
                    "enable": True,
                    "methods": ["dann", "mmd"]
                }
            },
            "fairness": {
                "enable_monitoring": True,
                "sensitive_attributes": ["age_group", "gender"],
                "fairness_metrics": [
                    "f1_score_parity",
                    "demographic_parity",
                    "equalized_odds"
                ]
            },
            "explainable_ai": {
                "enable_xai": True,
                "feature_importance": {
                    "methods": ["shap", "permutation"]
                },
                "llm_explanation": {
                    "enable": True,
                    "provider": "openrouter"
                }
            },
            "communication": {
                "enable_compression": True,
                "compression_methods": ["top_k", "quantization"]
            }
        }

def create_config_file(template_name: str, output_path: str = "config.json"):
    """Create a configuration file from a template"""
    templates = {
        "basic": ConfigTemplates.basic_experiment,
        "privacy": ConfigTemplates.privacy_focused,
        "fairness": ConfigTemplates.fairness_focused,
        "robustness": ConfigTemplates.robustness_focused,
        "xai": ConfigTemplates.explainable_ai_focused,
        "performance": ConfigTemplates.performance_optimized,
        "research": ConfigTemplates.comprehensive_research
    }
    
    if template_name not in templates:
        print(f"❌ Unknown template: {template_name}")
        print(f"Available templates: {list(templates.keys())}")
        return False
    
    config = templates[template_name]()
    
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration template '{template_name}' saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def list_templates():
    """List all available configuration templates"""
    templates = {
        "basic": "Simple federated learning experiment",
        "privacy": "Privacy-focused with strong differential privacy",
        "fairness": "Fairness-focused with bias monitoring",
        "robustness": "Robustness-focused with Byzantine fault tolerance",
        "xai": "Explainable AI focused with comprehensive explanations",
        "performance": "Performance-optimized for resource constraints",
        "research": "Comprehensive research experiment with all features"
    }
    
    print("📋 Available Configuration Templates:")
    print("=" * 50)
    for name, description in templates.items():
        print(f"{name:12} - {description}")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_templates.py <template_name> [output_file]")
        list_templates()
    else:
        template_name = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
        
        if template_name == "list":
            list_templates()
        else:
            create_config_file(template_name, output_file)