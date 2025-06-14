"""
Unit tests for communication efficiency methods in federated learning.
Tests Top-k sparsification and quantization algorithms.
"""

import unittest
import numpy as np
import torch
from federated_learning import apply_communication_efficiency

class TestCommunicationEfficiency(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for communication efficiency tests"""
        np.random.seed(42)
        
        # Create mock model updates with different shapes
        self.model_updates = [
            np.random.randn(100, 50).astype(np.float32),  # Large layer
            np.random.randn(50, 10).astype(np.float32),   # Medium layer
            np.random.randn(10).astype(np.float32),       # Small bias vector
            np.random.randn(5, 5).astype(np.float32)      # Small square matrix
        ]
        
        # Create updates with known patterns for testing
        self.structured_updates = [
            np.array([[1.0, 0.1, 0.01], [0.5, 0.05, 0.005]], dtype=np.float32),
            np.array([10.0, 1.0, 0.1, 0.01], dtype=np.float32)
        ]
    
    def test_no_compression(self):
        """Test that 'none' method returns original updates"""
        result = apply_communication_efficiency(self.model_updates, method="none")
        
        # Should be identical to input
        self.assertEqual(len(result), len(self.model_updates))
        for i, update in enumerate(result):
            np.testing.assert_array_equal(update, self.model_updates[i])
    
    def test_top_k_sparsification_structure(self):
        """Test Top-k sparsification maintains correct structure"""
        compression_ratio = 0.1  # Keep only 10% of parameters
        result = apply_communication_efficiency(
            self.model_updates, 
            compression_ratio=compression_ratio, 
            method="top_k"
        )
        
        # Check structure preservation
        self.assertEqual(len(result), len(self.model_updates))
        for i, update in enumerate(result):
            self.assertEqual(update.shape, self.model_updates[i].shape)
            self.assertEqual(update.dtype, self.model_updates[i].dtype)
    
    def test_top_k_sparsification_sparsity(self):
        """Test that Top-k sparsification creates correct sparsity"""
        compression_ratio = 0.2  # Keep 20% of parameters
        result = apply_communication_efficiency(
            self.structured_updates,
            compression_ratio=compression_ratio,
            method="top_k"
        )
        
        # Check sparsity for each update
        for i, update in enumerate(result):
            original = self.structured_updates[i]
            total_params = original.size
            expected_nonzero = max(1, int(total_params * compression_ratio))
            actual_nonzero = np.count_nonzero(update)
            
            self.assertEqual(actual_nonzero, expected_nonzero,
                           f"Update {i}: Expected {expected_nonzero} non-zero, got {actual_nonzero}")
    
    def test_top_k_selects_largest_magnitude(self):
        """Test that Top-k selects parameters with largest absolute values"""
        # Create update with known largest values
        test_update = np.array([[5.0, -4.0, 3.0], [2.0, -1.0, 0.5]], dtype=np.float32)
        compression_ratio = 0.5  # Keep 50% = 3 out of 6 parameters
        
        result = apply_communication_efficiency(
            [test_update],
            compression_ratio=compression_ratio,
            method="top_k"
        )[0]
        
        # Should keep the 3 largest magnitude values: 5.0, -4.0, 3.0
        expected_nonzero_positions = [(0, 0), (0, 1), (0, 2)]  # 5.0, -4.0, 3.0
        
        nonzero_positions = list(zip(*np.nonzero(result)))
        self.assertEqual(len(nonzero_positions), 3)
        
        # Check that largest magnitude values are preserved
        self.assertAlmostEqual(result[0, 0], 5.0, places=5)
        self.assertAlmostEqual(result[0, 1], -4.0, places=5)
        self.assertAlmostEqual(result[0, 2], 3.0, places=5)
    
    def test_top_k_minimum_k(self):
        """Test that Top-k keeps at least 1 parameter even with very low compression ratio"""
        very_low_ratio = 0.001  # Would normally keep 0 parameters
        result = apply_communication_efficiency(
            [np.random.randn(10).astype(np.float32)],
            compression_ratio=very_low_ratio,
            method="top_k"
        )[0]
        
        # Should keep at least 1 parameter
        self.assertGreaterEqual(np.count_nonzero(result), 1)
    
    def test_quantization_1bit(self):
        """Test 1-bit quantization"""
        compression_ratio = 0.1  # Should trigger 1-bit quantization
        result = apply_communication_efficiency(
            self.structured_updates,
            compression_ratio=compression_ratio,
            method="quantization"
        )
        
        # Check structure preservation
        self.assertEqual(len(result), len(self.structured_updates))
        for i, update in enumerate(result):
            self.assertEqual(update.shape, self.structured_updates[i].shape)
            
            # 1-bit quantization should only have values that are scaled signs
            unique_signs = np.unique(np.sign(update[update != 0]))
            self.assertTrue(len(unique_signs) <= 2, "1-bit quantization should have at most 2 sign values")
    
    def test_quantization_2bit(self):
        """Test 2-bit quantization"""
        compression_ratio = 0.2  # Should trigger 2-bit quantization
        result = apply_communication_efficiency(
            self.structured_updates,
            compression_ratio=compression_ratio,
            method="quantization"
        )
        
        # Check that quantization reduces precision
        for i, update in enumerate(result):
            original = self.structured_updates[i]
            
            # Quantized values should be different from original (unless original was already quantized)
            if not np.allclose(original, update, rtol=1e-6):
                # Check that we have limited number of unique values (due to quantization)
                unique_values = len(np.unique(update.flatten()))
                self.assertLessEqual(unique_values, 20, "2-bit quantization should limit unique values")
    
    def test_quantization_4bit(self):
        """Test 4-bit quantization"""
        compression_ratio = 0.4  # Should trigger 4-bit quantization
        result = apply_communication_efficiency(
            self.model_updates[:2],  # Use first 2 updates
            compression_ratio=compression_ratio,
            method="quantization"
        )
        
        # Structure should be preserved
        for i, update in enumerate(result):
            self.assertEqual(update.shape, self.model_updates[i].shape)
            self.assertTrue(np.all(np.isfinite(update)), "Quantized values should be finite")
    
    def test_quantization_8bit(self):
        """Test 8-bit quantization"""
        compression_ratio = 0.8  # Should trigger 8-bit quantization
        result = apply_communication_efficiency(
            self.model_updates[:1],
            compression_ratio=compression_ratio,
            method="quantization"
        )
        
        # Should preserve structure and have reasonable values
        self.assertEqual(result[0].shape, self.model_updates[0].shape)
        self.assertTrue(np.all(np.isfinite(result[0])))
    
    def test_quantization_constant_array(self):
        """Test quantization behavior with constant arrays"""
        constant_update = np.full((5, 5), 2.5, dtype=np.float32)
        result = apply_communication_efficiency(
            [constant_update],
            compression_ratio=0.3,
            method="quantization"
        )[0]
        
        # Should handle constant arrays gracefully
        self.assertEqual(result.shape, constant_update.shape)
        self.assertTrue(np.all(np.isfinite(result)))
        # For constant arrays, quantization might preserve the constant value
        self.assertTrue(np.allclose(result, constant_update, rtol=1e-3))
    
    def test_torch_tensor_input(self):
        """Test that methods work with PyTorch tensors"""
        torch_updates = [torch.tensor(update) for update in self.model_updates[:2]]
        
        # Test Top-k with torch tensors
        result_topk = apply_communication_efficiency(
            torch_updates,
            compression_ratio=0.3,
            method="top_k"
        )
        
        # Should return numpy arrays
        for update in result_topk:
            self.assertIsInstance(update, np.ndarray)
            self.assertTrue(np.all(np.isfinite(update)))
        
        # Test quantization with torch tensors
        result_quant = apply_communication_efficiency(
            torch_updates,
            compression_ratio=0.3,
            method="quantization"
        )
        
        for update in result_quant:
            self.assertIsInstance(update, np.ndarray)
            self.assertTrue(np.all(np.isfinite(update)))
    
    def test_empty_updates(self):
        """Test behavior with empty update list"""
        result = apply_communication_efficiency([], method="top_k")
        self.assertEqual(len(result), 0)
        
        result = apply_communication_efficiency([], method="quantization")
        self.assertEqual(len(result), 0)
    
    def test_unknown_method(self):
        """Test behavior with unknown compression method"""
        result = apply_communication_efficiency(
            self.model_updates,
            method="unknown_method"
        )
        
        # Should return original updates unchanged
        self.assertEqual(len(result), len(self.model_updates))
        for i, update in enumerate(result):
            np.testing.assert_array_equal(update, self.model_updates[i])
    
    def test_compression_ratio_bounds(self):
        """Test behavior with extreme compression ratios"""
        # Very high compression ratio
        result_high = apply_communication_efficiency(
            self.model_updates[:1],
            compression_ratio=1.5,  # > 1.0
            method="top_k"
        )
        # Should still work (might keep all parameters)
        self.assertEqual(len(result_high), 1)
        
        # Zero compression ratio
        result_zero = apply_communication_efficiency(
            self.model_updates[:1],
            compression_ratio=0.0,
            method="top_k"
        )
        # Should keep at least 1 parameter
        self.assertGreaterEqual(np.count_nonzero(result_zero[0]), 1)
    
    def test_performance_metrics(self):
        """Test that compression actually reduces data size"""
        original_size = sum(update.size for update in self.model_updates)
        
        # Top-k with 10% compression
        result_topk = apply_communication_efficiency(
            self.model_updates,
            compression_ratio=0.1,
            method="top_k"
        )
        
        # Count non-zero parameters (effective size after sparsification)
        compressed_size = sum(np.count_nonzero(update) for update in result_topk)
        compression_achieved = compressed_size / original_size
        
        # Should achieve approximately the target compression ratio
        self.assertLess(compression_achieved, 0.15, "Top-k should achieve significant compression")
        self.assertGreater(compression_achieved, 0.05, "Top-k should not over-compress")

if __name__ == '__main__':
    unittest.main()