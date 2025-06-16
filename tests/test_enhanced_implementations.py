import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_architecture import TriageModel
from model_optimization import apply_quantization, _detect_quantization_backend
from federated_learning import (
    apply_differential_privacy, PrivacyAccountant, simulate_byzantine_attacks,
    ByzantineAttackSimulator, FederatedClient, FederatedServer
)
from evaluation_framework import (
    ComprehensiveEvaluator, ClinicalMetrics, FairnessEvaluator, PerformanceBenchmark
)

class TestEnhancedQuantization(unittest.TestCase):
    """Test enhanced quantization implementation."""
    
    def setUp(self):
        self.model = TriageModel(7, 268, 3, 3)
        self.calibration_data = (
            torch.randn(50, 7),
            torch.randint(0, 2, (50, 268)).float(),
            torch.randn(50, 3)
        )
    
    def test_quantization_backend_detection(self):
        """Test automatic backend detection."""
        backend = _detect_quantization_backend()
        self.assertIn(backend, ['fbgemm', 'qnnpack'])
    
    def test_quantization_with_auto_backend(self):
        """Test quantization with automatic backend selection."""
        quantized_model = apply_quantization(
            self.model, 
            backend='auto', 
            calibration_data=self.calibration_data
        )
        self.assertIsNotNone(quantized_model)
    
    def test_quantization_fallback(self):
        """Test quantization fallback behavior."""
        # Test with invalid backend
        quantized_model = apply_quantization(
            self.model, 
            backend='invalid_backend'
        )
        # Should return original model on failure
        self.assertIsNotNone(quantized_model)

class TestEnhancedDifferentialPrivacy(unittest.TestCase):
    """Test enhanced differential privacy implementation."""
    
    def setUp(self):
        self.gradients = [
            np.random.randn(10, 5).astype(np.float32),
            np.random.randn(5).astype(np.float32),
            np.random.randn(3, 10).astype(np.float32)
        ]
        self.epsilon = 1.0
        self.delta = 1e-5
    
    def test_privacy_accountant(self):
        """Test privacy accountant functionality."""
        accountant = PrivacyAccountant(epsilon_total=10.0, delta=1e-5)
        
        # Test budget consumption
        self.assertTrue(accountant.consume_privacy_budget(2.0))
        self.assertEqual(accountant.get_remaining_budget(), 8.0)
        
        # Test budget exceeded
        with self.assertRaises(ValueError):
            accountant.consume_privacy_budget(9.0)
    
    def test_gaussian_dp(self):
        """Test Gaussian differential privacy."""
        noisy_gradients, metrics = apply_differential_privacy(
            self.gradients,
            sensitivity=1.0,
            epsilon=self.epsilon,
            delta=self.delta,
            method='gaussian'
        )
        
        self.assertEqual(len(noisy_gradients), len(self.gradients))
        self.assertTrue(metrics['dp_applied'])
        self.assertGreater(metrics['noise_multiplier'], 0)
    
    def test_laplace_dp(self):
        """Test Laplace differential privacy."""
        noisy_gradients, metrics = apply_differential_privacy(
            self.gradients,
            sensitivity=1.0,
            epsilon=self.epsilon,
            method='laplace'
        )
        
        self.assertEqual(len(noisy_gradients), len(self.gradients))
        self.assertTrue(metrics['dp_applied'])
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        large_gradients = [np.random.randn(10, 5) * 10]  # Large gradients
        
        noisy_gradients, metrics = apply_differential_privacy(
            large_gradients,
            sensitivity=1.0,
            epsilon=self.epsilon,
            method='gaussian',
            max_grad_norm=1.0
        )
        
        self.assertTrue(metrics.get('clipping_applied', False))

class TestByzantineAttacks(unittest.TestCase):
    """Test Byzantine attack simulation."""
    
    def setUp(self):
        # Create dummy client updates
        self.client_updates = []
        for i in range(5):
            params = [
                np.random.randn(10, 5).astype(np.float32),
                np.random.randn(5).astype(np.float32)
            ]
            self.client_updates.append((params, 100))  # (parameters, num_samples)
    
    def test_label_flipping_attack(self):
        """Test label flipping attack simulation."""
        simulator = ByzantineAttackSimulator()
        attacked_updates = simulator.label_flipping_attack(
            self.client_updates, 
            attack_ratio=0.4, 
            flip_probability=0.5
        )
        
        self.assertEqual(len(attacked_updates), len(self.client_updates))
        # Should have attacked 2 out of 5 clients (40%)
        self.assertEqual(len(attacked_updates), 5)
    
    def test_gradient_ascent_attack(self):
        """Test gradient ascent attack simulation."""
        simulator = ByzantineAttackSimulator()
        attacked_updates = simulator.gradient_ascent_attack(
            self.client_updates,
            attack_ratio=0.2,
            scale_factor=10.0
        )
        
        self.assertEqual(len(attacked_updates), len(self.client_updates))
    
    def test_gaussian_noise_attack(self):
        """Test Gaussian noise attack simulation."""
        simulator = ByzantineAttackSimulator()
        attacked_updates = simulator.gaussian_noise_attack(
            self.client_updates,
            attack_ratio=0.2,
            noise_std=1.0
        )
        
        self.assertEqual(len(attacked_updates), len(self.client_updates))
    
    def test_simulate_byzantine_attacks_interface(self):
        """Test the main Byzantine attack simulation interface."""
        # Test different attack types
        for attack_type in ['label_flipping', 'gradient_ascent', 'gaussian_noise']:
            attacked_updates = simulate_byzantine_attacks(
                self.client_updates,
                attack_type=attack_type,
                attack_ratio=0.2
            )
            self.assertEqual(len(attacked_updates), len(self.client_updates))
        
        # Test no attack
        no_attack_updates = simulate_byzantine_attacks(
            self.client_updates,
            attack_type='none'
        )
        self.assertEqual(no_attack_updates, self.client_updates)

class TestEvaluationFramework(unittest.TestCase):
    """Test comprehensive evaluation framework."""
    
    def setUp(self):
        # Create dummy data for testing
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
        self.y_pred = np.array([0, 1, 1, 0, 1, 2, 1, 1, 2, 2])
        self.sensitive_data = {
            'age_group': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1]),
            'gender': np.array(['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'])
        }
    
    def test_clinical_metrics(self):
        """Test clinical metrics calculation."""
        metrics = ClinicalMetrics.calculate_triage_metrics(self.y_true, self.y_pred)
        
        self.assertIn('overall_accuracy', metrics)
        self.assertIn('class_metrics', metrics)
        self.assertIn('clinical_safety', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check clinical safety metrics
        safety = metrics['clinical_safety']
        self.assertIn('under_triage_rate', safety)
        self.assertIn('over_triage_rate', safety)
        self.assertIn('critical_under_triage_rate', safety)
        self.assertIn('critical_sensitivity', safety)
    
    def test_fairness_evaluator(self):
        """Test fairness evaluation."""
        evaluator = FairnessEvaluator()
        fairness_metrics = evaluator.evaluate_fairness(
            self.y_true, self.y_pred, self.sensitive_data
        )
        
        self.assertIn('overall_fairness_score', fairness_metrics)
        self.assertIn('group_metrics', fairness_metrics)
        self.assertIn('fairness_violations', fairness_metrics)
        
        # Check that fairness score is between 0 and 1
        fairness_score = fairness_metrics['overall_fairness_score']
        self.assertGreaterEqual(fairness_score, 0.0)
        self.assertLessEqual(fairness_score, 1.0)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        # Create a simple model and data loader for testing
        model = TriageModel(7, 268, 3, 3)
        
        # Create dummy test data
        numerical_data = torch.randn(20, 7)
        boolean_data = torch.randint(0, 2, (20, 268)).float()
        temporal_data = torch.randn(20, 3)
        targets = torch.randint(0, 3, (20,))
        
        dataset = TensorDataset(numerical_data, boolean_data, temporal_data, targets)
        test_loader = DataLoader(dataset, batch_size=5)
        
        benchmark = PerformanceBenchmark()
        metrics = benchmark.benchmark_model_performance(model, test_loader, 'cpu')
        
        self.assertIn('avg_inference_time_ms', metrics)
        self.assertIn('throughput_samples_per_sec', metrics)
        self.assertIn('model_size_mb', metrics)
        self.assertIn('total_parameters', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['avg_inference_time_ms'], 0)
        self.assertGreater(metrics['throughput_samples_per_sec'], 0)
        self.assertGreater(metrics['model_size_mb'], 0)
        self.assertGreater(metrics['total_parameters'], 0)

class TestFederatedClientWithPrivacy(unittest.TestCase):
    """Test enhanced federated client with privacy features."""
    
    def setUp(self):
        # Create model and data
        self.model = TriageModel(7, 268, 3, 3)
        
        numerical_data = torch.randn(50, 7)
        boolean_data = torch.randint(0, 2, (50, 268)).float()
        temporal_data = torch.randn(50, 3)
        targets = torch.randint(0, 3, (50,))
        
        dataset = TensorDataset(numerical_data, boolean_data, temporal_data, targets)
        self.data_loader = DataLoader(dataset, batch_size=10)
        
        self.privacy_config = {
            'enable_dp': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'total_epsilon': 10.0,
            'sensitivity': 1.0,
            'method': 'gaussian'
        }
    
    def test_client_with_privacy(self):
        """Test federated client with differential privacy."""
        client = FederatedClient(
            "test_client", 
            self.model, 
            self.data_loader, 
            'cpu', 
            self.privacy_config
        )
        
        # Test training with DP
        params, num_samples, privacy_metrics = client.train(epochs=1, apply_dp=True)
        
        self.assertIsNotNone(params)
        self.assertEqual(num_samples, 50)
        self.assertTrue(privacy_metrics['dp_applied'])
        self.assertGreater(privacy_metrics['epsilon_consumed'], 0)
    
    def test_client_without_privacy(self):
        """Test federated client without differential privacy."""
        privacy_config = {'enable_dp': False}
        client = FederatedClient(
            "test_client", 
            self.model, 
            self.data_loader, 
            'cpu', 
            privacy_config
        )
        
        # Test training without DP
        params, num_samples, privacy_metrics = client.train(epochs=1, apply_dp=False)
        
        self.assertIsNotNone(params)
        self.assertEqual(num_samples, 50)
        self.assertFalse(privacy_metrics['dp_applied'])

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnhancedQuantization,
        TestEnhancedDifferentialPrivacy,
        TestByzantineAttacks,
        TestEvaluationFramework,
        TestFederatedClientWithPrivacy
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"ENHANCED IMPLEMENTATIONS TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print(f"{'='*50}")