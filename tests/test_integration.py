"""
Integration tests for Phase 2 and Phase 3 features.
Tests end-to-end functionality and integration between components.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
import tempfile
import os

# Import modules to test
from src.federated_learning import (
    FederatedClient, FederatedServer, apply_robust_aggregation,
    apply_communication_efficiency, apply_domain_adaptation,
    monitor_federated_fairness, monitor_data_drift
)
from src.explainable_ai import (
    OpenRouterClient, LLMExplanationEngine, generate_feature_importance
)
from src.model_architecture import TriageModel

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test environment"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Model parameters
        self.num_numerical = 7
        self.num_boolean = 268
        self.num_temporal = 3
        self.num_classes = 3
        self.device = torch.device("cpu")
        
        # Create test data
        self.batch_size = 32
        self.num_samples = 200
        
        # Generate synthetic data
        self.numerical_data = torch.randn(self.num_samples, self.num_numerical)
        self.boolean_data = torch.randint(0, 2, (self.num_samples, self.num_boolean)).float()
        self.temporal_data = torch.randn(self.num_samples, self.num_temporal)
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))
        
        # Create dataset and split for clients
        full_dataset = TensorDataset(
            self.numerical_data, self.boolean_data, 
            self.temporal_data, self.labels
        )
        
        # Split into 3 clients
        client_sizes = [80, 60, 60]  # Different sizes for realistic scenario
        self.client_datasets = torch.utils.data.random_split(
            full_dataset, client_sizes, 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.client_data_loaders = [
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for dataset in self.client_datasets
        ]
        
        # Create global model and server
        self.global_model = TriageModel(
            self.num_numerical, self.num_boolean, 
            self.num_temporal, self.num_classes
        ).to(self.device)
        
        self.server = FederatedServer(self.global_model, self.device)
        
        # Create clients
        self.clients = []
        for i, data_loader in enumerate(self.client_data_loaders):
            client_model = TriageModel(
                self.num_numerical, self.num_boolean,
                self.num_temporal, self.num_classes
            ).to(self.device)
            client_model.load_state_dict(self.global_model.state_dict())
            
            client = FederatedClient(f"client_{i}", client_model, data_loader, self.device)
            self.clients.append(client)
    
    def test_federated_learning_with_robust_aggregation(self):
        """Test complete federated learning pipeline with robust aggregation"""
        print("\n=== Testing Federated Learning with Robust Aggregation ===")
        
        # Test different aggregation methods
        aggregation_methods = ["fedavg", "krum", "trimmed_mean", "median"]
        
        for method in aggregation_methods:
            print(f"\nTesting {method} aggregation...")
            
            # Reset clients to same initial state
            for client in self.clients:
                client.model.load_state_dict(self.global_model.state_dict())
            
            # Perform one round of federated learning
            client_updates = []
            for client in self.clients:
                params, num_samples = client.train(epochs=1)
                client_updates.append((params, num_samples))
            
            # Test robust aggregation
            if method == "krum":
                aggregated_params = apply_robust_aggregation(
                    client_updates, method=method, num_malicious=1
                )
            else:
                aggregated_params = apply_robust_aggregation(
                    client_updates, method=method
                )
            
            # Verify aggregation worked
            self.assertIsInstance(aggregated_params, list)
            self.assertGreater(len(aggregated_params), 0)
            
            # Update global model
            self.server.global_model.load_state_dict(
                dict(zip(self.global_model.state_dict().keys(), 
                        [torch.tensor(p) for p in aggregated_params]))
            )
            
            # Verify model can still make predictions
            test_input = (
                torch.randn(1, self.num_numerical),
                torch.randn(1, self.num_boolean),
                torch.randn(1, self.num_temporal)
            )

            self.global_model.eval()  # Set to evaluation mode for inference
            with torch.no_grad():
                output = self.global_model(*test_input)
                self.assertEqual(output.shape, (1, self.num_classes))
                self.assertTrue(torch.all(torch.isfinite(output)))
            
            print(f"✓ {method} aggregation completed successfully")
    
    def test_communication_efficiency_integration(self):
        """Test communication efficiency with federated learning"""
        print("\n=== Testing Communication Efficiency Integration ===")
        
        # Perform client training
        client_updates = []
        for client in self.clients:
            params, num_samples = client.train(epochs=1)
            client_updates.append((params, num_samples))
        
        # Extract just the parameters for compression
        client_params = [update[0] for update in client_updates]
        
        # Test Top-k sparsification
        print("Testing Top-k sparsification...")
        compressed_topk = apply_communication_efficiency(
            client_params, compression_ratio=0.1, method="top_k"
        )
        
        # Verify compression
        original_nonzero = sum(np.count_nonzero(p) for p in client_params[0])
        compressed_nonzero = sum(np.count_nonzero(p) for p in compressed_topk[0])
        compression_ratio = compressed_nonzero / original_nonzero
        
        self.assertLess(compression_ratio, 0.15)  # Should achieve significant compression
        print(f"✓ Top-k achieved {compression_ratio:.1%} compression")
        
        # Test quantization
        print("Testing quantization...")
        compressed_quant = apply_communication_efficiency(
            client_params, compression_ratio=0.3, method="quantization"
        )
        
        # Verify quantization preserves structure
        for i, (original, quantized) in enumerate(zip(client_params[0], compressed_quant[0])):
            self.assertEqual(original.shape, quantized.shape)
            self.assertTrue(np.all(np.isfinite(quantized)))
        
        print("✓ Quantization completed successfully")
        
        # Test that compressed updates can still be aggregated
        compressed_updates = [(params, client_updates[i][1]) 
                             for i, params in enumerate(compressed_topk)]
        
        aggregated = apply_robust_aggregation(compressed_updates, method="fedavg")
        self.assertIsInstance(aggregated, list)
        print("✓ Compressed updates successfully aggregated")
    
    def test_domain_adaptation_integration(self):
        """Test domain adaptation with client data"""
        print("\n=== Testing Domain Adaptation Integration ===")
        
        # Create client data dictionary for domain adaptation
        client_data_dict = {}
        for i, client in enumerate(self.clients):
            # Extract data from client's data loader
            all_numerical, all_boolean, all_temporal, all_labels = [], [], [], []
            
            for batch in client.data_loader:
                numerical, boolean, temporal, labels = batch
                all_numerical.append(numerical)
                all_boolean.append(boolean)
                all_temporal.append(temporal)
                all_labels.append(labels)
            
            # Concatenate all batches
            client_features = torch.cat([
                torch.cat(all_numerical, dim=0),
                torch.cat(all_boolean, dim=0),
                torch.cat(all_temporal, dim=0)
            ], dim=1)
            client_labels = torch.cat(all_labels, dim=0)
            
            client_data_dict[f"client_{i}"] = (client_features, client_labels)
        
        # Test DANN domain adaptation
        print("Testing DANN domain adaptation...")
        dann_result = apply_domain_adaptation(client_data_dict, method="dann")
        
        # Verify DANN components were created
        self.assertTrue(hasattr(apply_domain_adaptation, 'dann_components'))
        self.assertIn('GradientReversalLayer', apply_domain_adaptation.dann_components)
        print("✓ DANN components initialized successfully")
        
        # Test MMD domain adaptation
        print("Testing MMD domain adaptation...")
        mmd_result = apply_domain_adaptation(client_data_dict, method="mmd")
        
        # Verify MMD components were created
        self.assertTrue(hasattr(apply_domain_adaptation, 'mmd_components'))
        self.assertIn('mmd_loss', apply_domain_adaptation.mmd_components)
        print("✓ MMD components initialized successfully")
        
        # Verify data structure is preserved
        self.assertEqual(len(dann_result), len(client_data_dict))
        self.assertEqual(len(mmd_result), len(client_data_dict))
        print("✓ Domain adaptation preserved data structure")
    
    def test_fairness_monitoring_integration(self):
        """Test fairness monitoring with federated learning"""
        print("\n=== Testing Fairness Monitoring Integration ===")
        
        # Create client data loaders dictionary
        client_data_loaders_dict = {
            f"client_{i}": loader for i, loader in enumerate(self.client_data_loaders)
        }
        
        # Test fairness monitoring
        fairness_metrics = ["f1_score_parity", "demographic_parity", "equalized_odds"]
        
        for metric in fairness_metrics:
            print(f"Testing {metric} fairness monitoring...")
            
            result = monitor_federated_fairness(
                self.global_model,
                client_data_loaders_dict,
                self.device,
                fairness_metric=metric,
                method="subgroup_evaluation"
            )
            
            # Verify result structure
            if isinstance(result, dict):
                self.assertIn('fairness_scores', result)
                self.assertIn('overall_fairness_score', result)
                self.assertIn('metric_type', result)
                print(f"✓ {metric} monitoring returned detailed results")
            else:
                self.assertIsInstance(result, (int, float))
                print(f"✓ {metric} monitoring returned numeric score: {result}")
    
    def test_data_drift_detection_integration(self):
        """Test data drift detection with client data"""
        print("\n=== Testing Data Drift Detection Integration ===")
        
        # Create client data for drift detection
        client_data_dict = {}
        for i, client in enumerate(self.clients):
            # Create different distributions to simulate drift
            if i == 0:
                # Normal distribution
                features = torch.randn(100, 50)
            elif i == 1:
                # Shifted distribution (drift)
                features = torch.randn(100, 50) + 2.0
            else:
                # Scaled distribution (drift)
                features = torch.randn(100, 50) * 2.0
            
            client_data_dict[f"client_{i}"] = features
        
        # Test KS-test drift detection
        print("Testing KS-test drift detection...")
        ks_drift = monitor_data_drift(client_data_dict, method="ks_test")
        self.assertIsInstance(ks_drift, bool)
        print(f"✓ KS-test drift detection result: {ks_drift}")
        
        # Test with single client (should not detect drift)
        single_client_data = {"client_0": client_data_dict["client_0"]}
        single_drift = monitor_data_drift(single_client_data, method="ks_test")
        self.assertFalse(single_drift)
        print("✓ Single client correctly shows no drift")
    
    def test_explainable_ai_integration(self):
        """Test explainable AI integration with trained model"""
        print("\n=== Testing Explainable AI Integration ===")
        
        # Train model for a few epochs to get meaningful predictions
        for epoch in range(3):
            client_updates = []
            for client in self.clients:
                params, num_samples = client.train(epochs=1)
                client_updates.append((params, num_samples))
            
            # Aggregate and update global model
            aggregated_params = apply_robust_aggregation(client_updates, method="fedavg")
            self.server.global_model.load_state_dict(
                dict(zip(self.global_model.state_dict().keys(),
                        [torch.tensor(p) for p in aggregated_params]))
            )
            
            # Distribute to clients
            for client in self.clients:
                client.model.load_state_dict(self.global_model.state_dict())
        
        # Test feature importance generation
        print("Testing feature importance generation...")
        
        # Create test data point
        test_numerical = torch.randn(1, self.num_numerical)
        test_boolean = torch.randint(0, 2, (1, self.num_boolean)).float()
        test_temporal = torch.randn(1, self.num_temporal)
        test_data_point = torch.cat([test_numerical, test_boolean, test_temporal], dim=1)
        
        # Generate feature names
        feature_names = (
            [f"numerical_{i}" for i in range(self.num_numerical)] +
            [f"boolean_{i}" for i in range(self.num_boolean)] +
            [f"temporal_{i}" for i in range(self.num_temporal)]
        )
        
        # Test random feature importance (always available)
        importance_scores = generate_feature_importance(
            self.global_model,
            test_data_point,
            feature_names,
            method="random"
        )
        
        self.assertIsInstance(importance_scores, list)
        self.assertEqual(len(importance_scores), len(feature_names))
        
        # Check that scores are normalized
        total_importance = sum(abs(score) for _, score in importance_scores)
        self.assertAlmostEqual(total_importance, 1.0, places=5)
        print("✓ Feature importance generation successful")
        
        # Test LLM explanation engine (without actual API calls)
        print("Testing LLM explanation engine...")
        
        engine = LLMExplanationEngine()
        
        # Test provider status
        status = engine.get_provider_status()
        self.assertIn("openrouter", status)
        self.assertIn("openai", status)
        print("✓ LLM explanation engine initialized")
        
        # Test fallback explanation
        self.global_model.eval()  # Set to evaluation mode for inference
        with torch.no_grad():
            prediction = torch.argmax(
                self.global_model(test_numerical, test_boolean, test_temporal)
            ).item()
        
        patient_context = {
            "age": 65,
            "symptoms": "chest pain",
            "vitals": "BP 140/90"
        }
        
        # This will use fallback since no real API keys
        explanation = engine.generate_explanation(
            prediction,
            importance_scores[:5],
            patient_context,
            use_fallback=True
        )
        
        self.assertIn("AI Triage Decision:", explanation)
        self.assertIn("Key Contributing Factors:", explanation)
        print("✓ LLM explanation generation successful")
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline with all features"""
        print("\n=== Testing End-to-End Pipeline ===")
        
        # Phase 1: Federated Learning with Robustness
        print("Phase 1: Federated Learning Setup...")
        
        initial_accuracy = self.server.evaluate_global_model(
            DataLoader(
                TensorDataset(self.numerical_data[:50], self.boolean_data[:50],
                             self.temporal_data[:50], self.labels[:50]),
                batch_size=32
            )
        )
        print(f"Initial accuracy: {initial_accuracy:.2f}%")
        
        # Phase 2: Training with Robust Aggregation and Communication Efficiency
        print("Phase 2: Robust Federated Training...")
        
        for round_num in range(3):
            print(f"Round {round_num + 1}/3")
            
            # Client training
            client_updates = []
            for client in self.clients:
                params, num_samples = client.train(epochs=1)
                client_updates.append((params, num_samples))
            
            # Apply communication efficiency
            client_params = [update[0] for update in client_updates]
            compressed_params = apply_communication_efficiency(
                client_params, compression_ratio=0.2, method="top_k"
            )
            
            # Reconstruct updates with compressed parameters
            compressed_updates = [
                (params, client_updates[i][1]) 
                for i, params in enumerate(compressed_params)
            ]
            
            # Robust aggregation
            aggregated_params = apply_robust_aggregation(
                compressed_updates, method="trimmed_mean"
            )
            
            # Update global model
            self.server.global_model.load_state_dict(
                dict(zip(self.global_model.state_dict().keys(),
                        [torch.tensor(p) for p in aggregated_params]))
            )
            
            # Distribute to clients
            for client in self.clients:
                client.model.load_state_dict(self.global_model.state_dict())
        
        # Phase 3: Evaluation and Monitoring
        print("Phase 3: Evaluation and Monitoring...")
        
        final_accuracy = self.server.evaluate_global_model(
            DataLoader(
                TensorDataset(self.numerical_data[:50], self.boolean_data[:50],
                             self.temporal_data[:50], self.labels[:50]),
                batch_size=32
            )
        )
        print(f"Final accuracy: {final_accuracy:.2f}%")
        
        # Fairness monitoring
        client_data_loaders_dict = {
            f"client_{i}": loader for i, loader in enumerate(self.client_data_loaders)
        }
        
        fairness_result = monitor_federated_fairness(
            self.global_model,
            client_data_loaders_dict,
            self.device,
            method="subgroup_evaluation"
        )
        
        print("✓ Fairness monitoring completed")
        
        # Generate explanation for a sample prediction
        test_input = (
            torch.randn(1, self.num_numerical),
            torch.randint(0, 2, (1, self.num_boolean)).float(),
            torch.randn(1, self.num_temporal)
        )
        
        with torch.no_grad():
            prediction = torch.argmax(self.global_model(*test_input)).item()
        
        print(f"Sample prediction: {prediction}")
        print("✓ End-to-end pipeline completed successfully")
        
        # Verify the pipeline maintained model functionality
        self.assertIsInstance(final_accuracy, float)
        self.assertGreaterEqual(final_accuracy, 0.0)
        self.assertLessEqual(final_accuracy, 100.0)

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)