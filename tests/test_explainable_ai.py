"""
Unit tests for explainable AI features.
Tests OpenRouter client, LLM explanation engine, and feature importance methods.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Import the modules to test
from src.explainable_ai import (
    OpenRouterClient, 
    LLMExplanationEngine,
    generate_feature_importance,
    generate_llm_explanation
)

class MockTriageModel(nn.Module):
    """Mock triage model for testing"""
    def __init__(self, num_numerical=7, num_boolean=268, num_temporal=3, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(num_numerical + num_boolean + num_temporal, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, numerical, boolean, temporal):
        x = torch.cat([numerical, boolean, temporal], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TestOpenRouterClient(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.api_key = "test_api_key"
        self.client = OpenRouterClient(api_key=self.api_key)
    
    def test_initialization(self):
        """Test OpenRouter client initialization"""
        # Test with API key
        client = OpenRouterClient(api_key="test_key")
        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.model, "anthropic/claude-3-sonnet")
        self.assertEqual(client.request_count, 0)
        self.assertEqual(client.total_cost, 0.0)
        
        # Test with custom model
        client = OpenRouterClient(api_key="test_key", model="openai/gpt-4-turbo")
        self.assertEqual(client.model, "openai/gpt-4-turbo")
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        # Test getting specific model info
        claude_info = self.client.get_model_info("anthropic/claude-3-sonnet")
        self.assertIn("name", claude_info)
        self.assertIn("context_length", claude_info)
        self.assertIn("cost_per_1k_tokens", claude_info)
        
        # Test getting all models
        all_models = self.client.get_model_info()
        self.assertIsInstance(all_models, dict)
        self.assertIn("anthropic/claude-3-sonnet", all_models)
        self.assertIn("openai/gpt-4-turbo", all_models)
    
    def test_select_optimal_model(self):
        """Test optimal model selection"""
        # Test budget-conscious selection
        budget_model = self.client.select_optimal_model(
            prompt_length=100, 
            complexity="low", 
            budget_conscious=True
        )
        budget_cost = self.client.AVAILABLE_MODELS[budget_model]["cost_per_1k_tokens"]
        
        # Should select a cheaper model
        self.assertLessEqual(budget_cost, 0.001)
        
        # Test performance-focused selection
        performance_model = self.client.select_optimal_model(
            prompt_length=100,
            complexity="high",
            budget_conscious=False
        )
        # Should select Claude or GPT-4 for high complexity
        self.assertIn(performance_model, ["anthropic/claude-3-sonnet", "openai/gpt-4-turbo"])
    
    @patch('requests.post')
    def test_generate_explanation_success(self, mock_post):
        """Test successful explanation generation"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test explanation"}}],
            "usage": {"total_tokens": 150}
        }
        mock_post.return_value = mock_response
        
        result = self.client.generate_explanation("Test prompt")
        
        # Check result
        self.assertIn("Test explanation", result)
        self.assertIn("[Generated by", result)
        
        # Check that request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['messages'][1]['content'], "Test prompt")
        
        # Check usage tracking
        self.assertEqual(self.client.request_count, 1)
        self.assertGreater(self.client.total_cost, 0)
    
    @patch('requests.post')
    def test_generate_explanation_fallback(self, mock_post):
        """Test fallback to different models on failure"""
        # Mock rate limit error for first model, success for second
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = Exception("Rate limit")
        mock_response_fail.status_code = 429
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Fallback explanation"}}],
            "usage": {"total_tokens": 100}
        }
        
        # First call fails, second succeeds
        responses = [
            type('MockResponse', (), {'raise_for_status': lambda: None, 'status_code': 429})(),
            mock_response_success
        ]
        responses[0].raise_for_status = Mock(side_effect=Exception("Rate limit"))
        mock_post.side_effect = responses
        
        # Should try fallback and succeed
        result = self.client.generate_explanation("Test prompt")
        self.assertIn("Fallback explanation", result)
    
    def test_get_usage_stats(self):
        """Test usage statistics"""
        # Initially should be zero
        stats = self.client.get_usage_stats()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["total_cost"], 0.0)
        self.assertEqual(stats["average_cost_per_request"], 0.0)
        
        # Simulate some usage
        self.client.request_count = 5
        self.client.total_cost = 0.25
        
        stats = self.client.get_usage_stats()
        self.assertEqual(stats["total_requests"], 5)
        self.assertEqual(stats["total_cost"], 0.25)
        self.assertEqual(stats["average_cost_per_request"], 0.05)
    
    def test_reset_usage_stats(self):
        """Test resetting usage statistics"""
        # Set some usage
        self.client.request_count = 10
        self.client.total_cost = 1.5
        
        # Reset
        self.client.reset_usage_stats()
        
        # Should be back to zero
        self.assertEqual(self.client.request_count, 0)
        self.assertEqual(self.client.total_cost, 0.0)

class TestLLMExplanationEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.engine = LLMExplanationEngine(
            openrouter_key="test_or_key",
            openai_key="test_openai_key"
        )
        
        self.test_prediction = 2  # Emergency
        self.test_features = [
            ("chest_pain", 0.8),
            ("age_65_plus", 0.6),
            ("high_blood_pressure", 0.4),
            ("shortness_of_breath", 0.3),
            ("heart_rate_high", 0.2)
        ]
        self.test_context = {
            "age": 70,
            "gender": "Male",
            "symptoms": "chest pain, difficulty breathing",
            "vitals": "BP 180/100, HR 110, RR 24"
        }
    
    def test_initialization(self):
        """Test LLM explanation engine initialization"""
        # Test with keys
        engine = LLMExplanationEngine(
            openrouter_key="or_key",
            openai_key="openai_key",
            preferred_provider="openai"
        )
        self.assertEqual(engine.openrouter_key, "or_key")
        self.assertEqual(engine.openai_key, "openai_key")
        self.assertEqual(engine.preferred_provider, "openai")
    
    def test_format_patient_context(self):
        """Test patient context formatting"""
        # Test with dictionary
        formatted = self.engine._format_patient_context(self.test_context)
        self.assertIn("Age: 70", formatted)
        self.assertIn("Gender: Male", formatted)
        
        # Test with string
        string_context = "Patient is 70 years old"
        formatted = self.engine._format_patient_context(string_context)
        self.assertEqual(formatted, string_context)
    
    def test_build_medical_prompt(self):
        """Test medical prompt building"""
        # Test high complexity prompt
        prompt = self.engine._build_medical_prompt(
            "Kırmızı Alan (Red Area - Emergency)",
            self.test_features,
            self.test_context,
            "high"
        )
        
        self.assertIn("PATIENT CONTEXT:", prompt)
        self.assertIn("AI TRIAGE DECISION:", prompt)
        self.assertIn("KEY FACTORS IDENTIFIED BY AI:", prompt)
        self.assertIn("chest_pain", prompt)
        self.assertIn("comprehensive medical explanation", prompt)
        
        # Test low complexity prompt
        prompt_low = self.engine._build_medical_prompt(
            "Yeşil Alan (Green Area - Non-urgent)",
            self.test_features[:2],
            self.test_context,
            "low"
        )
        
        self.assertIn("concise explanation", prompt_low)
        self.assertNotIn("comprehensive", prompt_low)
    
    def test_generate_fallback_explanation(self):
        """Test fallback explanation generation"""
        # Test emergency case
        explanation = self.engine._generate_fallback_explanation(
            "Kırmızı Alan (Red Area - Emergency)",
            self.test_features,
            self.test_context
        )
        
        self.assertIn("immediate medical attention", explanation)
        self.assertIn("life-threatening", explanation)
        self.assertIn("Key Clinical Indicators:", explanation)
        self.assertIn("Recommendations:", explanation)
        
        # Test non-urgent case
        explanation_green = self.engine._generate_fallback_explanation(
            "Yeşil Alan (Green Area - Non-urgent)",
            self.test_features,
            self.test_context
        )
        
        self.assertIn("standard priority", explanation_green)
        self.assertIn("stable condition", explanation_green)
    
    @patch.object(LLMExplanationEngine, '_generate_openai_explanation')
    def test_generate_explanation_openai_fallback(self, mock_openai):
        """Test explanation generation with OpenAI fallback"""
        # Mock OpenAI response
        mock_openai.return_value = "OpenAI generated explanation"
        
        # Set up engine with only OpenAI available
        engine = LLMExplanationEngine(openai_key="test_key")
        engine.openai_client = Mock()
        
        result = engine.generate_explanation(
            self.test_prediction,
            self.test_features,
            self.test_context
        )
        
        self.assertIn("AI Triage Decision:", result)
        self.assertIn("Key Contributing Factors:", result)
        self.assertIn("OpenAI generated explanation", result)
    
    def test_get_provider_status(self):
        """Test provider status reporting"""
        status = self.engine.get_provider_status()
        
        self.assertIn("openrouter", status)
        self.assertIn("openai", status)
        
        # Check structure
        for provider in ["openrouter", "openai"]:
            self.assertIn("available", status[provider])
            self.assertIn("key_configured", status[provider])
            self.assertIn("library_available", status[provider])

class TestFeatureImportance(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.model = MockTriageModel()
        self.model.eval()
        
        # Create test data
        self.data_point = torch.randn(1, 278)  # 7 + 268 + 3
        self.feature_names = [f"feature_{i}" for i in range(278)]
        
        # Create test datasets
        self.background_data = torch.randn(10, 278)
        self.test_data = torch.randn(50, 278)
        self.test_labels = torch.randint(0, 3, (50,))
    
    def test_random_feature_importance(self):
        """Test random feature importance generation"""
        result = generate_feature_importance(
            self.model,
            self.data_point,
            self.feature_names,
            method="random"
        )
        
        # Check structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.feature_names))
        
        # Check that it's sorted by importance
        importances = [score for _, score in result]
        self.assertEqual(importances, sorted(importances, key=abs, reverse=True))
        
        # Check that importance scores sum to 1 (normalized)
        total_importance = sum(abs(score) for _, score in result)
        self.assertAlmostEqual(total_importance, 1.0, places=5)
    
    @patch('src.explainable_ai.SHAP_AVAILABLE', True)
    @patch('shap.Explainer')
    def test_shap_feature_importance(self, mock_explainer):
        """Test SHAP feature importance generation"""
        # Mock SHAP explainer
        mock_explainer_instance = Mock()
        mock_shap_values = Mock()
        mock_shap_values.values = np.random.randn(1, 278, 3)  # Multi-class SHAP values
        mock_explainer_instance.return_value = mock_shap_values
        mock_explainer.return_value = mock_explainer_instance
        
        result = generate_feature_importance(
            self.model,
            self.data_point,
            self.feature_names,
            method="shap",
            background_data=self.background_data
        )
        
        # Check that SHAP was called
        mock_explainer.assert_called_once()
        mock_explainer_instance.assert_called_once()
        
        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.feature_names))
    
    @patch('src.explainable_ai.PERMUTATION_AVAILABLE', True)
    @patch('src.explainable_ai.permutation_importance')
    def test_permutation_importance(self, mock_perm_importance):
        """Test permutation importance generation"""
        # Mock permutation importance result
        mock_result = Mock()
        mock_result.importances_mean = np.random.rand(278)
        mock_perm_importance.return_value = mock_result
        
        result = generate_feature_importance(
            self.model,
            self.data_point,
            self.feature_names,
            method="permutation",
            test_data=self.test_data,
            test_labels=self.test_labels
        )
        
        # Check that permutation importance was called
        mock_perm_importance.assert_called_once()
        
        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.feature_names))
    
    def test_unknown_method_fallback(self):
        """Test fallback to random for unknown methods"""
        result = generate_feature_importance(
            self.model,
            self.data_point,
            self.feature_names,
            method="unknown_method"
        )
        
        # Should fallback to random values
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.feature_names))

class TestLegacyLLMExplanation(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_prediction = 1  # Urgent
        self.test_features = [
            ("fever", 0.7),
            ("cough", 0.5),
            ("fatigue", 0.3)
        ]
        self.test_context = {
            "age": 45,
            "symptoms": "fever, cough"
        }
    
    def test_placeholder_explanation(self):
        """Test placeholder explanation generation"""
        result = generate_llm_explanation(
            self.test_prediction,
            self.test_features,
            self.test_context,
            method="placeholder"
        )
        
        self.assertIn("Sarı Alan (Yellow Area - Urgent)", result)
        self.assertIn("Key factors influencing", result)
        self.assertIn("fever", result)
        self.assertIn("placeholder", result)
    
    def test_auto_method(self):
        """Test auto method using new engine"""
        result = generate_llm_explanation(
            self.test_prediction,
            self.test_features,
            self.test_context,
            method="auto"
        )
        
        # Should use the new LLMExplanationEngine
        self.assertIn("AI Triage Decision:", result)
        self.assertIn("Key Contributing Factors:", result)
    
    @patch('src.explainable_ai.REQUESTS_AVAILABLE', True)
    @patch.object(OpenRouterClient, 'generate_explanation')
    def test_openrouter_explanation(self, mock_generate):
        """Test OpenRouter explanation generation"""
        mock_generate.return_value = "OpenRouter generated explanation"
        
        result = generate_llm_explanation(
            self.test_prediction,
            self.test_features,
            self.test_context,
            method="openrouter",
            api_key="test_key"
        )
        
        self.assertIn("OpenRouter generated explanation", result)
        mock_generate.assert_called_once()

if __name__ == '__main__':
    unittest.main()