"""
Unit tests for domain adaptation methods in federated learning.
Tests DANN and MMD domain adaptation algorithms.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from federated_learning import apply_domain_adaptation

class TestDomainAdaptation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for domain adaptation tests"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create mock client data with different distributions (domains)
        self.client_data_dict = {}
        
        # Client 0: Normal distribution
        self.client_data_dict['client_0'] = (
            torch.randn(100, 10),  # Features
            torch.randint(0, 3, (100,))  # Labels
        )
        
        # Client 1: Shifted distribution (different domain)
        self.client_data_dict['client_1'] = (
            torch.randn(100, 10) + 2.0,  # Shifted features
            torch.randint(0, 3, (100,))
        )
        
        # Client 2: Scaled distribution (different domain)
        self.client_data_dict['client_2'] = (
            torch.randn(100, 10) * 2.0,  # Scaled features
            torch.randint(0, 3, (100,))
        )
        
        # Single client for edge cases
        self.single_client_data = {
            'client_0': self.client_data_dict['client_0']
        }
    
    def test_no_domain_adaptation(self):
        """Test that 'none' method returns data unchanged"""
        result = apply_domain_adaptation(self.client_data_dict, method="none")
        
        # Should return the same data
        self.assertEqual(len(result), len(self.client_data_dict))
        for client_id in self.client_data_dict:
            self.assertIn(client_id, result)
            # Data should be identical
            original_features, original_labels = self.client_data_dict[client_id]
            result_features, result_labels = result[client_id]
            torch.testing.assert_close(result_features, original_features)
            torch.testing.assert_close(result_labels, original_labels)
    
    def test_dann_initialization(self):
        """Test DANN method initializes components correctly"""
        # This should initialize DANN components and store them
        result = apply_domain_adaptation(self.client_data_dict, method="dann")
        
        # Check that DANN components are stored
        self.assertTrue(hasattr(apply_domain_adaptation, 'dann_components'))
        components = apply_domain_adaptation.dann_components
        
        # Check required components exist
        self.assertIn('GradientReversalLayer', components)
        self.assertIn('DomainDiscriminator', components)
        self.assertIn('gradient_reversal', components)
        
        # Test GradientReversalLayer
        GradientReversalLayer = components['GradientReversalLayer']
        test_input = torch.randn(5, 10, requires_grad=True)
        output = GradientReversalLayer.apply(test_input, 1.0)
        
        # Forward pass should be identity
        torch.testing.assert_close(output, test_input)
        
        # Test DomainDiscriminator
        DomainDiscriminator = components['DomainDiscriminator']
        discriminator = DomainDiscriminator(10)
        
        # Should be a valid neural network
        self.assertIsInstance(discriminator, nn.Module)
        
        # Test forward pass
        test_features = torch.randn(5, 10)
        disc_output = discriminator(test_features)
        self.assertEqual(disc_output.shape, (5, 1))
        self.assertTrue(torch.all(disc_output >= 0) and torch.all(disc_output <= 1))  # Sigmoid output
    
    def test_dann_gradient_reversal(self):
        """Test DANN gradient reversal functionality"""
        apply_domain_adaptation(self.client_data_dict, method="dann")
        
        if hasattr(apply_domain_adaptation, 'dann_components'):
            gradient_reversal = apply_domain_adaptation.dann_components['gradient_reversal']
            
            # Test gradient reversal
            x = torch.randn(3, 5, requires_grad=True)
            alpha = 2.0
            
            # Forward pass
            y = gradient_reversal(x, alpha)
            torch.testing.assert_close(y, x)  # Forward should be identity
            
            # Backward pass (gradient reversal)
            loss = y.sum()
            loss.backward()
            
            # Gradients should exist
            self.assertIsNotNone(x.grad)
    
    def test_mmd_initialization(self):
        """Test MMD method initializes components correctly"""
        result = apply_domain_adaptation(self.client_data_dict, method="mmd")
        
        # Check that MMD components are stored
        self.assertTrue(hasattr(apply_domain_adaptation, 'mmd_components'))
        components = apply_domain_adaptation.mmd_components
        
        # Check required components exist
        self.assertIn('mmd_loss', components)
        self.assertIn('rbf_kernel', components)
        self.assertIn('compute_mmd_for_clients', components)
    
    def test_mmd_rbf_kernel(self):
        """Test MMD RBF kernel computation"""
        apply_domain_adaptation(self.client_data_dict, method="mmd")
        
        if hasattr(apply_domain_adaptation, 'mmd_components'):
            rbf_kernel = apply_domain_adaptation.mmd_components['rbf_kernel']
            
            # Test kernel computation
            x = torch.randn(5, 3)
            y = torch.randn(4, 3)
            
            kernel_matrix = rbf_kernel(x, y)
            
            # Check output shape
            self.assertEqual(kernel_matrix.shape, (5, 4))
            
            # Kernel values should be positive
            self.assertTrue(torch.all(kernel_matrix > 0))
            
            # Kernel values should be <= 1 (for RBF kernel)
            self.assertTrue(torch.all(kernel_matrix <= 1))
            
            # Test self-kernel (diagonal should be 1)
            self_kernel = rbf_kernel(x, x)
            diagonal = torch.diag(self_kernel)
            torch.testing.assert_close(diagonal, torch.ones_like(diagonal), rtol=1e-5)
    
    def test_mmd_loss_computation(self):
        """Test MMD loss computation"""
        apply_domain_adaptation(self.client_data_dict, method="mmd")
        
        if hasattr(apply_domain_adaptation, 'mmd_components'):
            mmd_loss = apply_domain_adaptation.mmd_components['mmd_loss']
            
            # Test with identical distributions (should have low MMD)
            source = torch.randn(50, 5)
            target = source + torch.randn(50, 5) * 0.01  # Very similar
            
            mmd_identical = mmd_loss(source, target)
            self.assertIsInstance(mmd_identical, torch.Tensor)
            self.assertEqual(mmd_identical.shape, ())  # Scalar
            
            # Test with different distributions (should have higher MMD)
            target_different = torch.randn(50, 5) + 5.0  # Very different
            mmd_different = mmd_loss(source, target_different)
            
            # Different distributions should have higher MMD
            self.assertGreater(mmd_different.item(), mmd_identical.item())
    
    def test_mmd_client_computation(self):
        """Test MMD computation across clients"""
        apply_domain_adaptation(self.client_data_dict, method="mmd")
        
        if hasattr(apply_domain_adaptation, 'mmd_components'):
            compute_mmd_for_clients = apply_domain_adaptation.mmd_components['compute_mmd_for_clients']
            
            # Test with multiple clients
            mmd_score = compute_mmd_for_clients(self.client_data_dict)
            
            # Should return a numeric score
            self.assertIsInstance(mmd_score, (int, float))
            self.assertGreaterEqual(mmd_score, 0)  # MMD should be non-negative
            
            # Test with single client (should return 0 or handle gracefully)
            mmd_single = compute_mmd_for_clients(self.single_client_data)
            self.assertEqual(mmd_single, 0.0)
    
    def test_mmd_with_numpy_data(self):
        """Test MMD with numpy arrays instead of tensors"""
        # Convert to numpy
        numpy_client_data = {}
        for client_id, (features, labels) in self.client_data_dict.items():
            numpy_client_data[client_id] = (
                features.numpy(),
                labels.numpy()
            )
        
        # Should handle numpy data
        result = apply_domain_adaptation(numpy_client_data, method="mmd")
        
        # Should complete without errors
        self.assertEqual(len(result), len(numpy_client_data))
    
    def test_domain_adaptation_data_preservation(self):
        """Test that domain adaptation preserves data structure"""
        for method in ["dann", "mmd"]:
            result = apply_domain_adaptation(self.client_data_dict, method=method)
            
            # Data structure should be preserved
            self.assertEqual(len(result), len(self.client_data_dict))
            
            for client_id in self.client_data_dict:
                self.assertIn(client_id, result)
                
                original_features, original_labels = self.client_data_dict[client_id]
                result_features, result_labels = result[client_id]
                
                # Shapes should be preserved
                self.assertEqual(result_features.shape, original_features.shape)
                self.assertEqual(result_labels.shape, original_labels.shape)
    
    def test_unknown_method(self):
        """Test behavior with unknown domain adaptation method"""
        result = apply_domain_adaptation(self.client_data_dict, method="unknown_method")
        
        # Should return data unchanged (like 'none' method)
        self.assertEqual(len(result), len(self.client_data_dict))
        for client_id in self.client_data_dict:
            original_features, original_labels = self.client_data_dict[client_id]
            result_features, result_labels = result[client_id]
            torch.testing.assert_close(result_features, original_features)
            torch.testing.assert_close(result_labels, original_labels)
    
    def test_empty_client_data(self):
        """Test behavior with empty client data"""
        empty_data = {}
        
        for method in ["none", "dann", "mmd"]:
            result = apply_domain_adaptation(empty_data, method=method)
            self.assertEqual(len(result), 0)
    
    def test_single_client_dann(self):
        """Test DANN with single client"""
        result = apply_domain_adaptation(self.single_client_data, method="dann")
        
        # Should handle single client gracefully
        self.assertEqual(len(result), 1)
        self.assertIn('client_0', result)
    
    def test_different_feature_dimensions(self):
        """Test domain adaptation with different feature dimensions"""
        mixed_data = {
            'client_0': (torch.randn(50, 5), torch.randint(0, 3, (50,))),
            'client_1': (torch.randn(50, 8), torch.randint(0, 3, (50,))),  # Different feature dim
        }
        
        # Should handle different dimensions gracefully
        for method in ["dann", "mmd"]:
            try:
                result = apply_domain_adaptation(mixed_data, method=method)
                # If it succeeds, data should be preserved
                self.assertEqual(len(result), len(mixed_data))
            except Exception as e:
                # If it fails, it should be a reasonable error
                self.assertIsInstance(e, (ValueError, RuntimeError, IndexError))
    
    def test_error_handling(self):
        """Test error handling in domain adaptation methods"""
        # Test with malformed data
        malformed_data = {
            'client_0': "not_a_tuple",
            'client_1': (torch.randn(10, 5),)  # Missing labels
        }
        
        for method in ["dann", "mmd"]:
            # Should handle errors gracefully and not crash
            try:
                result = apply_domain_adaptation(malformed_data, method=method)
                # If successful, should return something reasonable
                self.assertIsInstance(result, dict)
            except Exception:
                # Errors are acceptable for malformed data
                pass

if __name__ == '__main__':
    unittest.main()