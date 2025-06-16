import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict

class FairnessMitigationEngine:
    """
    Advanced fairness mitigation techniques for federated learning.
    """
    
    def __init__(self, sensitive_attributes=None, fairness_constraints=None):
        self.sensitive_attributes = sensitive_attributes or ['age_group', 'gender']
        self.fairness_constraints = fairness_constraints or {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'calibration': 0.05
        }
        self.mitigation_history = []
    
    def apply_preprocessing_mitigation(self, X, y, sensitive_attrs, method='reweighting'):
        """
        Apply preprocessing fairness mitigation techniques.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_attrs: Sensitive attribute values
            method: Mitigation method ('reweighting', 'resampling', 'synthetic_generation')
        
        Returns:
            tuple: (X_mitigated, y_mitigated, sample_weights)
        """
        print(f"🔧 Applying preprocessing mitigation: {method}")
        
        if method == 'reweighting':
            return self._apply_reweighting(X, y, sensitive_attrs)
        elif method == 'resampling':
            return self._apply_resampling(X, y, sensitive_attrs)
        elif method == 'synthetic_generation':
            return self._apply_synthetic_generation(X, y, sensitive_attrs)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def _apply_reweighting(self, X, y, sensitive_attrs):
        """Apply instance reweighting to balance representation."""
        # Calculate group-wise statistics
        groups = {}
        for i, (attr_val, label) in enumerate(zip(sensitive_attrs, y)):
            key = (attr_val, label)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Calculate weights to balance groups
        total_samples = len(y)
        num_groups = len(groups)
        target_size_per_group = total_samples / num_groups
        
        sample_weights = np.ones(len(y))
        
        for (attr_val, label), indices in groups.items():
            group_size = len(indices)
            weight = target_size_per_group / group_size if group_size > 0 else 1.0
            
            for idx in indices:
                sample_weights[idx] = weight
        
        print(f"  Applied reweighting: {len(groups)} groups balanced")
        return X, y, sample_weights
    
    def _apply_resampling(self, X, y, sensitive_attrs):
        """Apply resampling to balance group representation."""
        # Group samples by sensitive attribute and label
        groups = defaultdict(list)
        for i, (attr_val, label) in enumerate(zip(sensitive_attrs, y)):
            groups[(attr_val, label)].append(i)
        
        # Find target size (size of largest group)
        target_size = max(len(indices) for indices in groups.values())
        
        # Resample each group to target size
        resampled_indices = []
        for (attr_val, label), indices in groups.items():
            if len(indices) < target_size:
                # Oversample minority groups
                resampled = np.random.choice(indices, target_size, replace=True)
            else:
                # Keep majority groups as is (or could undersample)
                resampled = indices
            resampled_indices.extend(resampled)
        
        # Create resampled dataset
        X_resampled = X[resampled_indices]
        y_resampled = y[resampled_indices]
        sample_weights = np.ones(len(y_resampled))
        
        print(f"  Applied resampling: {len(resampled_indices)} samples after balancing")
        return X_resampled, y_resampled, sample_weights
    
    def _apply_synthetic_generation(self, X, y, sensitive_attrs):
        """Generate synthetic samples for underrepresented groups."""
        # This is a simplified version - in practice, you'd use GANs or VAEs
        groups = defaultdict(list)
        for i, (attr_val, label) in enumerate(zip(sensitive_attrs, y)):
            groups[(attr_val, label)].append(i)
        
        # Find target size
        target_size = max(len(indices) for indices in groups.values())
        
        X_augmented = [X]
        y_augmented = [y]
        
        for (attr_val, label), indices in groups.items():
            if len(indices) < target_size:
                # Generate synthetic samples
                num_synthetic = target_size - len(indices)
                group_samples = X[indices]
                
                # Simple synthetic generation using noise around group mean
                group_mean = np.mean(group_samples, axis=0)
                group_std = np.std(group_samples, axis=0)
                
                synthetic_X = np.random.normal(
                    group_mean, 
                    group_std * 0.1,  # Small noise
                    (num_synthetic, X.shape[1])
                )
                synthetic_y = np.full(num_synthetic, label)
                
                X_augmented.append(synthetic_X)
                y_augmented.append(synthetic_y)
        
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        sample_weights = np.ones(len(y_final))
        
        print(f"  Generated synthetic samples: {len(y_final) - len(y)} new samples")
        return X_final, y_final, sample_weights
    
    def apply_inprocessing_mitigation(self, model, train_loader, sensitive_attrs, 
                                   method='adversarial_debiasing', epochs=10):
        """
        Apply in-processing fairness mitigation during training.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            sensitive_attrs: Sensitive attributes for each sample
            method: Mitigation method ('adversarial_debiasing', 'fairness_constraints')
            epochs: Number of training epochs
        
        Returns:
            nn.Module: Fair model
        """
        print(f"🎯 Applying in-processing mitigation: {method}")
        
        if method == 'adversarial_debiasing':
            return self._apply_adversarial_debiasing(model, train_loader, sensitive_attrs, epochs)
        elif method == 'fairness_constraints':
            return self._apply_fairness_constraints(model, train_loader, sensitive_attrs, epochs)
        else:
            raise ValueError(f"Unknown in-processing method: {method}")
    
    def _apply_adversarial_debiasing(self, model, train_loader, sensitive_attrs, epochs):
        """Apply adversarial debiasing during training."""
        
        class AdversarialDiscriminator(nn.Module):
            def __init__(self, input_dim, num_sensitive_attrs):
                super().__init__()
                self.discriminator = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_sensitive_attrs)
                )
            
            def forward(self, x):
                return self.discriminator(x)
        
        # Create adversarial discriminator
        num_sensitive_attrs = len(np.unique(sensitive_attrs))
        discriminator = AdversarialDiscriminator(model.classifier.in_features, num_sensitive_attrs)
        
        # Optimizers
        model_optimizer = optim.Adam(model.parameters(), lr=0.001)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        
        # Loss functions
        task_criterion = nn.CrossEntropyLoss()
        fairness_criterion = nn.CrossEntropyLoss()
        
        model.train()
        discriminator.train()
        
        for epoch in range(epochs):
            total_task_loss = 0
            total_fairness_loss = 0
            
            for batch_idx, (numerical, boolean, temporal, targets) in enumerate(train_loader):
                batch_size = numerical.size(0)
                
                # Get sensitive attributes for this batch
                batch_sensitive = torch.tensor(sensitive_attrs[batch_idx*batch_size:(batch_idx+1)*batch_size])
                
                # Forward pass through main model
                outputs = model(numerical, boolean, temporal)
                
                # Get intermediate representations (before final classifier)
                with torch.no_grad():
                    # Extract features from the model (this would need to be adapted to your model)
                    features = model.fusion_net(torch.cat([
                        model.numerical_net(numerical),
                        model.boolean_net(boolean),
                        model.temporal_net(temporal)
                    ], dim=1))
                
                # Train discriminator to predict sensitive attributes
                disc_optimizer.zero_grad()
                disc_outputs = discriminator(features.detach())
                disc_loss = fairness_criterion(disc_outputs, batch_sensitive)
                disc_loss.backward()
                disc_optimizer.step()
                
                # Train main model
                model_optimizer.zero_grad()
                
                # Task loss
                task_loss = task_criterion(outputs, targets)
                
                # Adversarial loss (fool the discriminator)
                disc_outputs = discriminator(features)
                # Create uniform distribution target to fool discriminator
                uniform_target = torch.full_like(batch_sensitive, num_sensitive_attrs // 2)
                adversarial_loss = -fairness_criterion(disc_outputs, uniform_target)
                
                # Combined loss
                total_loss = task_loss + 0.1 * adversarial_loss  # Lambda = 0.1
                total_loss.backward()
                model_optimizer.step()
                
                total_task_loss += task_loss.item()
                total_fairness_loss += adversarial_loss.item()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Task Loss = {total_task_loss/len(train_loader):.4f}, "
                      f"Fairness Loss = {total_fairness_loss/len(train_loader):.4f}")
        
        print("  Adversarial debiasing completed")
        return model
    
    def _apply_fairness_constraints(self, model, train_loader, sensitive_attrs, epochs):
        """Apply fairness constraints during training."""
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task_criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (numerical, boolean, temporal, targets) in enumerate(train_loader):
                batch_size = numerical.size(0)
                batch_sensitive = torch.tensor(sensitive_attrs[batch_idx*batch_size:(batch_idx+1)*batch_size])
                
                optimizer.zero_grad()
                
                outputs = model(numerical, boolean, temporal)
                predictions = torch.argmax(outputs, dim=1)
                
                # Task loss
                task_loss = task_criterion(outputs, targets)
                
                # Fairness constraint loss
                fairness_loss = self._calculate_fairness_constraint_loss(
                    predictions, targets, batch_sensitive
                )
                
                # Combined loss
                total_loss_batch = task_loss + 0.5 * fairness_loss  # Lambda = 0.5
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Combined Loss = {total_loss/len(train_loader):.4f}")
        
        print("  Fairness constraints training completed")
        return model
    
    def _calculate_fairness_constraint_loss(self, predictions, targets, sensitive_attrs):
        """Calculate fairness constraint loss."""
        # Demographic parity constraint
        unique_attrs = torch.unique(sensitive_attrs)
        
        if len(unique_attrs) < 2:
            return torch.tensor(0.0)
        
        group_positive_rates = []
        
        for attr_val in unique_attrs:
            group_mask = sensitive_attrs == attr_val
            if group_mask.sum() > 0:
                group_predictions = predictions[group_mask]
                positive_rate = (group_predictions > 0).float().mean()
                group_positive_rates.append(positive_rate)
        
        if len(group_positive_rates) < 2:
            return torch.tensor(0.0)
        
        # Minimize variance in positive rates across groups
        rates_tensor = torch.stack(group_positive_rates)
        fairness_loss = torch.var(rates_tensor)
        
        return fairness_loss
    
    def apply_postprocessing_mitigation(self, predictions, targets, sensitive_attrs, 
                                      method='threshold_optimization'):
        """
        Apply post-processing fairness mitigation.
        
        Args:
            predictions: Model predictions
            targets: True labels
            sensitive_attrs: Sensitive attributes
            method: Mitigation method ('threshold_optimization', 'calibration')
        
        Returns:
            np.ndarray: Adjusted predictions
        """
        print(f"📊 Applying post-processing mitigation: {method}")
        
        if method == 'threshold_optimization':
            return self._apply_threshold_optimization(predictions, targets, sensitive_attrs)
        elif method == 'calibration':
            return self._apply_calibration(predictions, targets, sensitive_attrs)
        else:
            raise ValueError(f"Unknown post-processing method: {method}")
    
    def _apply_threshold_optimization(self, predictions, targets, sensitive_attrs):
        """Optimize decision thresholds for each group."""
        unique_attrs = np.unique(sensitive_attrs)
        adjusted_predictions = predictions.copy()
        
        for attr_val in unique_attrs:
            group_mask = sensitive_attrs == attr_val
            group_predictions = predictions[group_mask]
            group_targets = targets[group_mask]
            
            if len(group_predictions) == 0:
                continue
            
            # Find optimal threshold for this group
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.1, 0.9, 0.1):
                binary_preds = (group_predictions > threshold).astype(int)
                f1 = f1_score(group_targets, binary_preds, average='weighted', zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Apply optimal threshold
            adjusted_predictions[group_mask] = (group_predictions > best_threshold).astype(int)
            print(f"  Group {attr_val}: optimal threshold = {best_threshold:.2f}")
        
        return adjusted_predictions
    
    def _apply_calibration(self, predictions, targets, sensitive_attrs):
        """Apply calibration to ensure equal positive predictive value across groups."""
        from sklearn.calibration import CalibratedClassifierCV
        
        unique_attrs = np.unique(sensitive_attrs)
        adjusted_predictions = predictions.copy()
        
        for attr_val in unique_attrs:
            group_mask = sensitive_attrs == attr_val
            group_predictions = predictions[group_mask]
            group_targets = targets[group_mask]
            
            if len(group_predictions) < 10:  # Need minimum samples for calibration
                continue
            
            # Simple calibration using Platt scaling
            # In practice, you'd use more sophisticated calibration methods
            positive_mask = group_targets == 1
            if positive_mask.sum() > 0 and (~positive_mask).sum() > 0:
                pos_mean = group_predictions[positive_mask].mean()
                neg_mean = group_predictions[~positive_mask].mean()
                
                # Adjust predictions to improve calibration
                calibration_factor = 0.5 / pos_mean if pos_mean > 0 else 1.0
                adjusted_predictions[group_mask] = group_predictions * calibration_factor
        
        return adjusted_predictions
    
    def evaluate_fairness_improvement(self, original_predictions, mitigated_predictions, 
                                    targets, sensitive_attrs):
        """
        Evaluate fairness improvement after mitigation.
        
        Returns:
            dict: Fairness metrics before and after mitigation
        """
        print("📈 Evaluating fairness improvement...")
        
        # Calculate metrics for original predictions
        original_metrics = self._calculate_fairness_metrics(
            original_predictions, targets, sensitive_attrs, "Original"
        )
        
        # Calculate metrics for mitigated predictions
        mitigated_metrics = self._calculate_fairness_metrics(
            mitigated_predictions, targets, sensitive_attrs, "Mitigated"
        )
        
        # Calculate improvement
        improvement = {
            'demographic_parity_improvement': (
                mitigated_metrics['demographic_parity_difference'] - 
                original_metrics['demographic_parity_difference']
            ),
            'equalized_odds_improvement': (
                mitigated_metrics['equalized_odds_difference'] - 
                original_metrics['equalized_odds_difference']
            ),
            'accuracy_change': (
                mitigated_metrics['overall_accuracy'] - 
                original_metrics['overall_accuracy']
            )
        }
        
        return {
            'original': original_metrics,
            'mitigated': mitigated_metrics,
            'improvement': improvement
        }
    
    def _calculate_fairness_metrics(self, predictions, targets, sensitive_attrs, label):
        """Calculate comprehensive fairness metrics."""
        unique_attrs = np.unique(sensitive_attrs)
        group_metrics = {}
        
        for attr_val in unique_attrs:
            group_mask = sensitive_attrs == attr_val
            group_preds = predictions[group_mask]
            group_targets = targets[group_mask]
            
            if len(group_preds) > 0:
                group_metrics[attr_val] = {
                    'accuracy': accuracy_score(group_targets, group_preds),
                    'positive_rate': np.mean(group_preds),
                    'true_positive_rate': np.mean(group_preds[group_targets == 1]) if np.sum(group_targets == 1) > 0 else 0,
                    'false_positive_rate': np.mean(group_preds[group_targets == 0]) if np.sum(group_targets == 0) > 0 else 0
                }
        
        # Calculate fairness differences
        positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
        tpr_rates = [metrics['true_positive_rate'] for metrics in group_metrics.values()]
        fpr_rates = [metrics['false_positive_rate'] for metrics in group_metrics.values()]
        
        return {
            'overall_accuracy': accuracy_score(targets, predictions),
            'group_metrics': group_metrics,
            'demographic_parity_difference': max(positive_rates) - min(positive_rates) if positive_rates else 0,
            'equalized_odds_difference': max(tpr_rates) - min(tpr_rates) if tpr_rates else 0,
            'label': label
        }

class FairFederatedLearning:
    """
    Fairness-aware federated learning algorithms.
    """
    
    def __init__(self, fairness_weight=0.1):
        self.fairness_weight = fairness_weight
        self.fairness_history = []
    
    def fair_fedavg(self, client_updates, client_fairness_metrics):
        """
        Fairness-aware FedAvg that weights clients based on fairness.
        
        Args:
            client_updates: List of (parameters, num_samples) tuples
            client_fairness_metrics: Fairness metrics for each client
        
        Returns:
            Aggregated parameters with fairness weighting
        """
        print("🤝 Applying Fair FedAvg aggregation...")
        
        # Calculate fairness weights
        fairness_scores = []
        for metrics in client_fairness_metrics:
            # Higher score for more fair clients
            fairness_score = 1.0 - metrics.get('demographic_parity_difference', 0)
            fairness_scores.append(max(0.1, fairness_score))  # Minimum weight of 0.1
        
        # Combine sample size and fairness weights
        total_weighted_samples = 0
        weighted_updates = []
        
        for i, ((params, num_samples), fairness_score) in enumerate(zip(client_updates, fairness_scores)):
            # Combined weight: sample size * fairness score
            combined_weight = num_samples * fairness_score
            total_weighted_samples += combined_weight
            
            weighted_params = []
            for param in params:
                weighted_param = param * combined_weight
                weighted_params.append(weighted_param)
            
            weighted_updates.append(weighted_params)
            print(f"  Client {i}: samples={num_samples}, fairness={fairness_score:.3f}, weight={combined_weight:.1f}")
        
        # Aggregate weighted parameters
        if not weighted_updates:
            return client_updates[0][0]  # Return first client's parameters as fallback
        
        aggregated_params = []
        for param_idx in range(len(weighted_updates[0])):
            param_sum = sum(update[param_idx] for update in weighted_updates)
            aggregated_param = param_sum / total_weighted_samples
            aggregated_params.append(aggregated_param)
        
        print(f"  Fair FedAvg completed with {len(client_updates)} clients")
        return aggregated_params
    
    def q_fedavg(self, client_updates, q=0.1):
        """
        q-Fair Federated Averaging for fairness-aware aggregation.
        
        Args:
            client_updates: List of (parameters, num_samples) tuples
            q: Fairness parameter (0 = standard FedAvg, higher = more fairness focus)
        
        Returns:
            Aggregated parameters with q-fair weighting
        """
        print(f"⚖️ Applying q-FedAvg (q={q})...")
        
        # Extract sample sizes
        sample_sizes = [num_samples for _, num_samples in client_updates]
        total_samples = sum(sample_sizes)
        
        # Calculate q-fair weights
        if q == 0:
            # Standard FedAvg
            weights = [size / total_samples for size in sample_sizes]
        else:
            # q-fair weights
            weights = []
            for size in sample_sizes:
                if q == 1:
                    weight = 1.0 / len(sample_sizes)  # Equal weighting
                else:
                    weight = (size / total_samples) ** q
                weights.append(weight)
            
            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
        
        # Aggregate with q-fair weights
        aggregated_params = []
        for param_idx in range(len(client_updates[0][0])):
            weighted_sum = sum(
                weights[i] * client_updates[i][0][param_idx] 
                for i in range(len(client_updates))
            )
            aggregated_params.append(weighted_sum)
        
        print(f"  q-FedAvg completed: weights = {[f'{w:.3f}' for w in weights]}")
        return aggregated_params

if __name__ == "__main__":
    print("⚖️ FairTriEdge-FL Fairness Mitigation Framework")
    print("Available mitigation techniques:")
    print("  📋 Preprocessing: reweighting, resampling, synthetic generation")
    print("  🎯 In-processing: adversarial debiasing, fairness constraints")
    print("  📊 Post-processing: threshold optimization, calibration")
    print("  🤝 Fair FL: Fair FedAvg, q-FedAvg")
    print("\n✅ Fairness mitigation framework ready!")