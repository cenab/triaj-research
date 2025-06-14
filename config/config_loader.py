#!/usr/bin/env python3
"""
Configuration Loader for FairTriEdge-FL
Handles loading and validation of configuration files
"""

import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigLoader:
    """Configuration loader and validator for FairTriEdge-FL"""
    
    def __init__(self, config_path: str = "config.json", env_path: str = ".env"):
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        self.env_vars = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuration loaded from {self.config_path}")
            return self.config
        except FileNotFoundError:
            print(f"⚠️  Configuration file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {self.config_path}: {e}")
            return self._get_default_config()
    
    def load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env file"""
        load_dotenv(self.env_path)
        
        # Override config with environment variables if they exist
        env_mappings = {
            'OPENROUTER_API_KEY': ('explainable_ai', 'llm_explanation', 'openrouter_api_key'),
            'OPENAI_API_KEY': ('explainable_ai', 'llm_explanation', 'openai_api_key'),
            'LLM_PROVIDER': ('explainable_ai', 'llm_explanation', 'provider'),
            'OPENROUTER_MODEL': ('explainable_ai', 'llm_explanation', 'model'),
            'PRIVACY_EPSILON': ('privacy', 'epsilon'),
            'PRIVACY_DELTA': ('privacy', 'delta'),
            'FL_NUM_ROUNDS': ('federated_learning', 'num_rounds'),
            'FL_NUM_CLIENTS': ('federated_learning', 'num_clients'),
            'FL_AGGREGATION_METHOD': ('federated_learning', 'aggregation_method'),
            'BATCH_SIZE': ('model', 'training', 'batch_size'),
            'LEARNING_RATE': ('model', 'training', 'learning_rate'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_config(config_path, value)
                self.env_vars[env_var] = value
        
        return self.env_vars
    
    def _set_nested_config(self, path: tuple, value: str):
        """Set nested configuration value"""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if path[-1] in ['num_rounds', 'num_clients', 'batch_size']:
            value = int(value)
        elif path[-1] in ['epsilon', 'delta', 'learning_rate']:
            value = float(value)
        elif path[-1] in ['enable_differential_privacy', 'enable_xai']:
            value = value.lower() in ['true', '1', 'yes', 'on']
        
        current[path[-1]] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "federated_learning": {
                "num_rounds": 5,
                "num_clients": 3,
                "aggregation_method": "fedavg"
            },
            "model": {
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            },
            "privacy": {
                "epsilon": 1.0,
                "delta": 1e-5
            },
            "explainable_ai": {
                "llm_explanation": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-3-sonnet"
                }
            }
        }
    
    def get(self, *keys, default=None):
        """Get nested configuration value"""
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required sections
        required_sections = ['federated_learning', 'model', 'privacy']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Check data file
        data_file = self.get('experiment', 'data_file', default='triaj_data.csv')
        if not os.path.exists(data_file):
            errors.append(f"Data file not found: {data_file}")
        
        # Check API keys for XAI
        if self.get('explainable_ai', 'enable_xai', default=True):
            openrouter_key = os.getenv('OPENROUTER_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openrouter_key and not openai_key:
                errors.append("No API keys found for LLM explanations")
        
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("\n📊 Configuration Summary:")
        print("=" * 50)
        
        # Experiment info
        exp_name = self.get('experiment', 'name', default='FairTriEdge-FL')
        print(f"Experiment: {exp_name}")
        
        # Federated Learning
        fl_rounds = self.get('federated_learning', 'num_rounds', default=5)
        fl_clients = self.get('federated_learning', 'num_clients', default=3)
        fl_agg = self.get('federated_learning', 'aggregation_method', default='fedavg')
        print(f"FL Setup: {fl_rounds} rounds, {fl_clients} clients, {fl_agg} aggregation")
        
        # Privacy
        epsilon = self.get('privacy', 'epsilon', default=1.0)
        enable_dp = self.get('privacy', 'enable_differential_privacy', default=True)
        print(f"Privacy: DP={'enabled' if enable_dp else 'disabled'}, ε={epsilon}")
        
        # Model
        batch_size = self.get('model', 'training', 'batch_size', default=32)
        lr = self.get('model', 'training', 'learning_rate', default=0.001)
        print(f"Training: batch_size={batch_size}, lr={lr}")
        
        # XAI
        enable_xai = self.get('explainable_ai', 'enable_xai', default=True)
        llm_provider = self.get('explainable_ai', 'llm_explanation', 'provider', default='openrouter')
        print(f"XAI: {'enabled' if enable_xai else 'disabled'}, LLM provider={llm_provider}")
        
        # API Keys status
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        api_status = "configured" if (openrouter_key or openai_key) else "not configured"
        print(f"API Keys: {api_status}")
        
        print("=" * 50)

def load_experiment_config(config_path: str = "config.json") -> ConfigLoader:
    """Convenience function to load experiment configuration"""
    loader = ConfigLoader(config_path)
    loader.load_config()
    loader.load_env_vars()
    return loader

# Example usage
if __name__ == "__main__":
    # Load and validate configuration
    config_loader = load_experiment_config()
    config_loader.print_config_summary()
    config_loader.validate_config()
    
    # Example of accessing configuration values
    print(f"\nExample config access:")
    print(f"Number of FL rounds: {config_loader.get('federated_learning', 'num_rounds')}")
    print(f"Privacy epsilon: {config_loader.get('privacy', 'epsilon')}")
    print(f"LLM provider: {config_loader.get('explainable_ai', 'llm_explanation', 'provider')}")