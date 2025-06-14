"""
Unit tests for robust aggregation methods in federated learning.
Tests Krum, Trimmed Mean, and Median aggregation algorithms.
"""

import unittest
import numpy as np
import torch
from federated_learning import apply_robust_aggregation

class TestRobustAggregation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for robust aggregation tests"""
        # Create mock client updates with different parameter shapes
        np.random.seed(42)
        
        # Simulate 5 clients with 3 parameter layers each
        self.num_clients = 5
        self.client_updates = []
        
        for i in range(self.num_clients):
            # Each client has 3 parameter arrays of different shapes
            params = [
                np.random.randn(10, 5).astype(np.float32),  # Layer 1: 10x5
                np.random.randn(5, 3).astype(np.float32),   # Layer 2: 5x3  
                np.random.randn(3).astype(np.float32)       # Layer 3: bias vector
            ]
            num_samples = 100 + i * 20  # Different sample sizes
            self.client_updates.append((params, num_samples))
        
        # Create malicious client updates (outliers)
        self.malicious_updates = []
        for i in range(2):  # 2 malicious clients
            params = [
                np.random.randn(10, 5).astype(np.float32) * 10,  # 10x larger values
                np.random.randn(5, 3).astype(np.float32) * 10,
                np.random.randn(3).astype(np.float32) * 10
            ]
            num_samples = 50
            self.malicious_updates.append((params, num_samples))
    
    def test_fedavg_baseline(self):
        """Test FedAvg (baseline) aggregation"""
        result = apply_robust_aggregation(self.client_updates, method="fedavg")
        
        # Check that result has correct structure
        self.assertEqual(len(result), 3)  # 3 layers
        self.assertEqual(result[0].shape, (10, 5))
        self.assertEqual(result[1].shape, (5, 3))
        self.assertEqual(result[2].shape, (3,))
        
        # Check that values are reasonable (weighted average)
        self.assertTrue(np.all(np.isfinite(result[0])))
        self.assertTrue(np.all(np.isfinite(result[1])))
        self.assertTrue(np.all(np.isfinite(result[2])))
    
    def test_krum_aggregation(self):
        """Test Krum aggregation method"""
        result = apply_robust_aggregation(self.client_updates, method="krum", num_malicious=1)
        
        # Check structure
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (10, 5))
        self.assertEqual(result[1].shape, (5, 3))
        self.assertEqual(result[2].shape, (3,))
        
        # Krum should select one of the original client updates
        found_match = False
        for client_params, _ in self.client_updates:
            if np.allclose(result[0], client_params[0], rtol=1e-5):
                found_match = True
                break
        self.assertTrue(found_match, "Krum result should match one of the client updates")
    
    def test_krum_with_malicious_clients(self):
        """Test Krum robustness against malicious clients"""
        # Add malicious updates to the mix
        mixed_updates = self.client_updates + self.malicious_updates
        
        result = apply_robust_aggregation(mixed_updates, method="krum", num_malicious=2)
        
        # Krum should still select from honest clients (not malicious ones)
        # Check that result is not from malicious clients (which have 10x larger values)
        max_honest_value = max(np.max(np.abs(params[0])) for params, _ in self.client_updates)
        max_result_value = np.max(np.abs(result[0]))
        
        # Result should be closer to honest clients than malicious ones
        self.assertLess(max_result_value, max_honest_value * 2, 
                       "Krum should reject malicious updates with large values")
    
    def test_trimmed_mean_aggregation(self):
        """Test Trimmed Mean aggregation method"""
        result = apply_robust_aggregation(self.client_updates, method="trimmed_mean", num_malicious=1)
        
        # Check structure
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (10, 5))
        self.assertEqual(result[1].shape, (5, 3))
        self.assertEqual(result[2].shape, (3,))
        
        # Values should be finite and reasonable
        self.assertTrue(np.all(np.isfinite(result[0])))
        self.assertTrue(np.all(np.isfinite(result[1])))
        self.assertTrue(np.all(np.isfinite(result[2])))
        
        # Trimmed mean should be different from simple average
        fedavg_result = apply_robust_aggregation(self.client_updates, method="fedavg")
        self.assertFalse(np.allclose(result[0], fedavg_result[0], rtol=1e-3))
    
    def test_median_aggregation(self):
        """Test Median aggregation method"""
        result = apply_robust_aggregation(self.client_updates, method="median")
        
        # Check structure
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (10, 5))
        self.assertEqual(result[1].shape, (5, 3))
        self.assertEqual(result[2].shape, (3,))
        
        # Values should be finite
        self.assertTrue(np.all(np.isfinite(result[0])))
        self.assertTrue(np.all(np.isfinite(result[1])))
        self.assertTrue(np.all(np.isfinite(result[2])))
    
    def test_median_robustness(self):
        """Test median aggregation robustness against outliers"""
        # Add extreme outliers
        outlier_updates = []
        for i in range(2):
            params = [
                np.full((10, 5), 1000.0, dtype=np.float32),  # Extreme values
                np.full((5, 3), -1000.0, dtype=np.float32),
                np.full(3, 500.0, dtype=np.float32)
            ]
            outlier_updates.append((params, 100))
        
        mixed_updates = self.client_updates + outlier_updates
        
        # Median should be robust against these outliers
        result = apply_robust_aggregation(mixed_updates, method="median")
        
        # Result should not contain extreme values
        self.assertLess(np.max(np.abs(result[0])), 100, 
                       "Median should be robust against extreme outliers")
        self.assertLess(np.max(np.abs(result[1])), 100,
                       "Median should be robust against extreme outliers")
        self.assertLess(np.max(np.abs(result[2])), 100,
                       "Median should be robust against extreme outliers")
    
    def test_empty_updates(self):
        """Test behavior with empty client updates"""
        with self.assertRaises(IndexError):
            apply_robust_aggregation([], method="fedavg")
    
    def test_single_client(self):
        """Test behavior with single client"""
        single_update = [self.client_updates[0]]
        
        for method in ["fedavg", "krum", "trimmed_mean", "median"]:
            result = apply_robust_aggregation(single_update, method=method)
            
            # Should return the single client's parameters
            original_params = single_update[0][0]
            self.assertTrue(np.allclose(result[0], original_params[0]))
            self.assertTrue(np.allclose(result[1], original_params[1]))
            self.assertTrue(np.allclose(result[2], original_params[2]))
    
    def test_unknown_method(self):
        """Test behavior with unknown aggregation method"""
        # Should fallback to FedAvg
        result = apply_robust_aggregation(self.client_updates, method="unknown_method")
        fedavg_result = apply_robust_aggregation(self.client_updates, method="fedavg")
        
        # Should be identical to FedAvg
        self.assertTrue(np.allclose(result[0], fedavg_result[0]))
        self.assertTrue(np.allclose(result[1], fedavg_result[1]))
        self.assertTrue(np.allclose(result[2], fedavg_result[2]))
    
    def test_krum_insufficient_clients(self):
        """Test Krum behavior when there are too many malicious clients"""
        # Only 2 clients, but claiming 2 malicious - should fallback to FedAvg
        small_updates = self.client_updates[:2]
        result = apply_robust_aggregation(small_updates, method="krum", num_malicious=2)
        fedavg_result = apply_robust_aggregation(small_updates, method="fedavg")
        
        # Should fallback to FedAvg
        self.assertTrue(np.allclose(result[0], fedavg_result[0]))

if __name__ == '__main__':
    unittest.main()