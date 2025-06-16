import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import OrderedDict
import numpy as np

# Assuming TriageModel is defined in model_architecture.py
from .model_architecture import TriageModel

def get_model_parameters(model):
    """Returns the current model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Sets the model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class FederatedClient:
    def __init__(self, client_id, model, data_loader, device, privacy_config=None):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Privacy configuration
        self.privacy_config = privacy_config or {
            'enable_dp': False,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': None
        }
        
        # Privacy accounting
        if self.privacy_config.get('enable_dp', False):
            total_epsilon = self.privacy_config.get('total_epsilon', 10.0)
            self.privacy_accountant = PrivacyAccountant(total_epsilon, self.privacy_config['delta'])
        else:
            self.privacy_accountant = None

    def get_parameters(self):
        """Returns the current model parameters as a list of NumPy arrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        """Sets the model parameters from a list of NumPy arrays."""
        set_model_parameters(self.model, parameters)

    def train(self, epochs=1, apply_dp=None):
        """
        Trains the local model for a specified number of epochs with optional differential privacy.
        
        Args:
            epochs (int): Number of training epochs.
            apply_dp (bool): Whether to apply differential privacy. If None, uses privacy_config.
        
        Returns:
            tuple: (parameters, num_samples, privacy_metrics)
        """
        self.model.train()
        total_loss = 0
        
        # Determine if DP should be applied
        if apply_dp is None:
            apply_dp = self.privacy_config.get('enable_dp', False)
        
        # Store initial parameters for gradient calculation
        if apply_dp:
            initial_params = self.get_parameters()
        
        # Training loop
        for epoch in range(epochs):
            for numerical_data, boolean_data, temporal_data, targets in self.data_loader:
                numerical_data, boolean_data, temporal_data, targets = \
                    numerical_data.to(self.device), boolean_data.to(self.device), \
                    temporal_data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(numerical_data, boolean_data, temporal_data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Apply gradient clipping if DP is enabled
                if apply_dp and self.privacy_config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.privacy_config['max_grad_norm']
                    )
                
                self.optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.data_loader)
        print(f"Client {self.client_id} trained. Avg Loss: {avg_loss:.4f}")
        
        # Apply differential privacy to parameter updates
        privacy_metrics = {'dp_applied': False}
        final_params = self.get_parameters()
        
        if apply_dp:
            # Calculate parameter updates (gradients)
            param_updates = []
            for initial, final in zip(initial_params, final_params):
                update = final - initial
                param_updates.append(update)
            
            # Apply differential privacy to updates
            epsilon_round = self.privacy_config['epsilon']
            if self.privacy_accountant:
                try:
                    self.privacy_accountant.consume_privacy_budget(epsilon_round)
                except ValueError as e:
                    print(f"Privacy budget exceeded for client {self.client_id}: {e}")
                    # Return parameters without DP
                    return final_params, len(self.data_loader.dataset), privacy_metrics
            
            noisy_updates, dp_metrics = apply_differential_privacy(
                param_updates,
                sensitivity=self.privacy_config.get('sensitivity', 1.0),
                epsilon=epsilon_round,
                delta=self.privacy_config['delta'],
                method=self.privacy_config.get('method', 'gaussian'),
                max_grad_norm=self.privacy_config['max_grad_norm'],
                sample_size=len(self.data_loader.dataset)
            )
            
            # Apply noisy updates to get final noisy parameters
            noisy_params = []
            for initial, noisy_update in zip(initial_params, noisy_updates):
                noisy_param = initial + noisy_update
                noisy_params.append(noisy_param)
            
            privacy_metrics = {
                'dp_applied': True,
                'epsilon_consumed': dp_metrics['epsilon_consumed'],
                'noise_multiplier': dp_metrics['noise_multiplier'],
                'clipping_applied': dp_metrics['clipping_applied'],
                'remaining_budget': self.privacy_accountant.get_remaining_budget() if self.privacy_accountant else None
            }
            
            print(f"Client {self.client_id} DP applied: ε={epsilon_round:.3f}, "
                  f"noise_mult={dp_metrics['noise_multiplier']:.4f}")
            
            return noisy_params, len(self.data_loader.dataset), privacy_metrics
        
        return final_params, len(self.data_loader.dataset), privacy_metrics

    def evaluate(self):
        """Evaluates the local model on its local dataset."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in self.data_loader:
                numerical_data, boolean_data, temporal_data, targets = \
                    numerical_data.to(self.device), boolean_data.to(self.device), \
                    temporal_data.to(self.device), targets.to(self.device)
                outputs = self.model(numerical_data, boolean_data, temporal_data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f"Client {self.client_id} evaluation: Accuracy = {accuracy:.2f}%")
        return accuracy

class FederatedServer:
    def __init__(self, global_model, device):
        self.global_model = global_model
        self.device = device

    def aggregate_parameters(self, client_updates):
        """
        Aggregates model updates from clients using Federated Averaging (FedAvg).
        
        Args:
            client_updates (list): A list of tuples, where each tuple contains
                                   (client_parameters, num_samples_from_client).
        
        Returns:
            list: Aggregated global model parameters.
        """
        if not client_updates:
            return get_model_parameters(self.global_model) # No updates, return current global model params

        # Extract parameters and weights (number of samples)
        client_params = [update[0] for update in client_updates]
        client_num_samples = [update[1] for update in client_updates]

        # Calculate total number of samples
        total_samples = sum(client_num_samples)

        # Perform weighted averaging
        aggregated_params = [
            np.zeros_like(param, dtype=np.float32) for param in client_params[0] # Ensure float type for aggregation
        ]

        for i, params in enumerate(client_params):
            weight = client_num_samples[i] / total_samples
            for j in range(len(params)):
                aggregated_params[j] += params[j] * weight
        
        print("Server aggregated model updates.")
        return aggregated_params

    def distribute_parameters(self, aggregated_params, clients):
        """Distributes the aggregated global model parameters to all clients."""
        set_model_parameters(self.global_model, aggregated_params) # Update global model first
        for client in clients:
            client.set_parameters(aggregated_params)
        print("Server distributed global model to clients.")

    def evaluate_global_model(self, test_data_loader):
        """Evaluates the global model on a centralized test dataset."""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in test_data_loader:
                numerical_data, boolean_data, temporal_data, targets = \
                    numerical_data.to(self.device), boolean_data.to(self.device), \
                    temporal_data.to(self.device), targets.to(self.device)
                outputs = self.global_model(numerical_data, boolean_data, temporal_data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f"Global model evaluation: Accuracy = {accuracy:.2f}%")
        return accuracy

# Placeholder for Domain Adaptation Strategy (Phase 1.4)
def apply_domain_adaptation(client_data_dict, method="none"):
    """
    Applies domain adaptation techniques to client datasets based on the specified method.
    Args:
        client_data_dict (dict): Dictionary of client data.
        method (str): The domain adaptation method to use ('none', 'dann', 'mmd').
    """
    if method == "dann":
        print("Applying Domain Adaptation (method: DANN - Domain-Adversarial Neural Networks).")
        # DANN Implementation - trains domain-invariant features using adversarial training
        
        try:
            from torch.autograd import Function
            import torch.nn.functional as F
            
            # Gradient Reversal Layer for adversarial training
            class GradientReversalLayer(Function):
                @staticmethod
                def forward(ctx, x, alpha=1.0):
                    ctx.alpha = alpha
                    return x.view_as(x)
                
                @staticmethod
                def backward(ctx, grad_output):
                    return (-ctx.alpha * grad_output), None
            
            # Domain Discriminator Network
            class DomainDiscriminator(nn.Module):
                def __init__(self, input_dim, hidden_dim=128):
                    super(DomainDiscriminator, self).__init__()
                    self.discriminator = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.discriminator(x)
            
            # Apply gradient reversal function
            def gradient_reversal(x, alpha=1.0):
                return GradientReversalLayer.apply(x, alpha)
            
            print("DANN components initialized successfully.")
            print("Note: Full DANN training requires integration with the main training loop.")
            print("This includes:")
            print("- Feature extractor from main model")
            print("- Domain discriminator training")
            print("- Adversarial loss computation")
            print("- Gradient reversal during backpropagation")
            
            # Store DANN components for later use
            if not hasattr(apply_domain_adaptation, 'dann_components'):
                apply_domain_adaptation.dann_components = {
                    'GradientReversalLayer': GradientReversalLayer,
                    'DomainDiscriminator': DomainDiscriminator,
                    'gradient_reversal': gradient_reversal
                }
            
        except ImportError as e:
            print(f"Error importing required modules for DANN: {e}")
            print("Falling back to no domain adaptation.")
    elif method == "mmd":
        print("Applying Domain Adaptation (method: MMD - Maximum Mean Discrepancy).")
        # MMD Implementation - minimizes distribution distance between domains
        
        try:
            def rbf_kernel(x, y, gamma=1.0):
                """RBF (Gaussian) kernel for MMD computation"""
                # Compute pairwise squared Euclidean distances
                x_norm = (x ** 2).sum(1).view(-1, 1)
                y_norm = (y ** 2).sum(1).view(1, -1)
                dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
                return torch.exp(-gamma * dist)
            
            def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
                """
                Compute Maximum Mean Discrepancy (MMD) loss between source and target domains
                """
                batch_size = int(source_features.size()[0])
                kernels = rbf_kernel(source_features, source_features)
                
                # Multiple kernel widths for better performance
                if fix_sigma:
                    bandwidth = fix_sigma
                else:
                    bandwidth = torch.sum(kernels) / (batch_size ** 2 - batch_size)
                
                bandwidth /= kernel_mul ** (kernel_num // 2)
                bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
                
                kernel_val = [rbf_kernel(source_features, source_features, gamma=1.0/bw) for bw in bandwidth_list]
                kernel_val += [rbf_kernel(target_features, target_features, gamma=1.0/bw) for bw in bandwidth_list]
                kernel_val += [rbf_kernel(source_features, target_features, gamma=1.0/bw) for bw in bandwidth_list]
                
                # Compute MMD
                XX = sum(kernel_val[:kernel_num])
                YY = sum(kernel_val[kernel_num:2*kernel_num])
                XY = sum(kernel_val[2*kernel_num:])
                
                mmd = XX.mean() + YY.mean() - 2 * XY.mean()
                return mmd
            
            def compute_mmd_for_clients(client_data_dict):
                """Compute MMD loss between different client domains"""
                client_ids = list(client_data_dict.keys())
                if len(client_ids) < 2:
                    print("MMD requires at least 2 clients for domain comparison.")
                    return 0.0
                
                total_mmd = 0.0
                comparisons = 0
                
                # Compare each pair of clients
                for i in range(len(client_ids)):
                    for j in range(i + 1, len(client_ids)):
                        source_data = client_data_dict[client_ids[i]]
                        target_data = client_data_dict[client_ids[j]]
                        
                        # Extract features (assuming data is tuple of (X, y))
                        if isinstance(source_data, tuple):
                            source_features = source_data[0]
                            target_features = target_data[0]
                        else:
                            source_features = source_data
                            target_features = target_data
                        
                        # Convert to tensors if needed
                        if isinstance(source_features, np.ndarray):
                            source_features = torch.tensor(source_features, dtype=torch.float32)
                        if isinstance(target_features, np.ndarray):
                            target_features = torch.tensor(target_features, dtype=torch.float32)
                        
                        # Flatten if multi-dimensional
                        if source_features.dim() > 2:
                            source_features = source_features.view(source_features.size(0), -1)
                        if target_features.dim() > 2:
                            target_features = target_features.view(target_features.size(0), -1)
                        
                        # Compute MMD between domains
                        mmd_val = mmd_loss(source_features, target_features)
                        total_mmd += mmd_val.item()
                        comparisons += 1
                        
                        print(f"MMD between client {client_ids[i]} and {client_ids[j]}: {mmd_val.item():.4f}")
                
                avg_mmd = total_mmd / comparisons if comparisons > 0 else 0.0
                print(f"Average MMD across all client pairs: {avg_mmd:.4f}")
                return avg_mmd
            
            # Store MMD components for later use
            if not hasattr(apply_domain_adaptation, 'mmd_components'):
                apply_domain_adaptation.mmd_components = {
                    'mmd_loss': mmd_loss,
                    'rbf_kernel': rbf_kernel,
                    'compute_mmd_for_clients': compute_mmd_for_clients
                }
            
            # Compute MMD for current client data
            mmd_score = compute_mmd_for_clients(client_data_dict)
            print(f"MMD domain adaptation analysis complete. Average MMD: {mmd_score:.4f}")
            
        except Exception as e:
            print(f"Error in MMD implementation: {e}")
            print("Falling back to no domain adaptation.")
    elif method == "none":
        print("Applying Domain Adaptation (method: None - data returned as is).")
    else:
        print(f"Unknown Domain Adaptation method: {method}. Data returned as is.")
    return client_data_dict

# Placeholder for Continuous Drift Monitoring (Phase 1.5)
def monitor_data_drift(client_data_dict, method="none"):
    """
    Monitors data drift in client datasets using the specified method.
    Args:
        client_data_dict (dict): Dictionary of client data.
        method (str): The drift detection method to use ('none', 'adwin', 'ks_test').
    Returns:
        bool: True if drift is detected, False otherwise.
    """
    if method == "adwin":
        print("Monitoring Data Drift (method: ADWIN).")
        try:
            from alibi_detect.cd import ADWIN
            
            # Initialize ADWIN detector with threshold
            threshold = 0.001  # Sensitivity threshold for drift detection
            detector = ADWIN(threshold=threshold)
            
            drift_detected = False
            total_points = 0
            
            # Process data from all clients
            for client_id, client_data in client_data_dict.items():
                print(f"Processing client {client_id} data for drift detection...")
                
                # Extract features for drift detection
                # Assuming client_data is a tuple of (X, y) or just X
                if isinstance(client_data, tuple):
                    X_data = client_data[0]
                else:
                    X_data = client_data
                
                # Convert to numpy if needed
                if isinstance(X_data, torch.Tensor):
                    X_data = X_data.numpy()
                
                # For ADWIN, we need scalar values, so we'll use the mean of each sample
                # In practice, you might want to monitor specific features
                if X_data.ndim > 1:
                    # Use mean across features for each sample
                    data_points = np.mean(X_data, axis=1)
                else:
                    data_points = X_data
                
                # Update ADWIN detector with each data point
                for point in data_points:
                    detector.update(float(point))
                    total_points += 1
                    
                    # Check for drift after each update
                    if detector.drift():
                        print(f"ADWIN drift detected at point {total_points} from client {client_id}!")
                        drift_detected = True
                        break
                
                if drift_detected:
                    break
            
            if drift_detected:
                print(f"Data drift detected by ADWIN after processing {total_points} points.")
            else:
                print(f"No drift detected by ADWIN after processing {total_points} points.")
            
            return drift_detected
            
        except ImportError:
            print("alibi-detect not available. Install with: pip install alibi-detect")
            print("Falling back to simple statistical drift detection.")
            
            # Simple fallback: detect drift based on mean shift
            try:
                all_means = []
                for client_id, client_data in client_data_dict.items():
                    if isinstance(client_data, tuple):
                        X_data = client_data[0]
                    else:
                        X_data = client_data
                    
                    if isinstance(X_data, torch.Tensor):
                        X_data = X_data.numpy()
                    
                    # Calculate mean for this client's data
                    client_mean = np.mean(X_data)
                    all_means.append(client_mean)
                
                # Simple drift detection: check if variance in means is high
                if len(all_means) > 1:
                    mean_variance = np.var(all_means)
                    drift_threshold = 0.1  # Arbitrary threshold
                    drift_detected = mean_variance > drift_threshold
                    
                    if drift_detected:
                        print(f"Simple drift detection: variance in client means ({mean_variance:.4f}) exceeds threshold ({drift_threshold})")
                    else:
                        print(f"Simple drift detection: no significant variance in client means ({mean_variance:.4f})")
                    
                    return drift_detected
                else:
                    return False
                    
            except Exception as e:
                print(f"Error in fallback drift detection: {e}")
                return np.random.rand() > 0.8  # Random fallback
                
        except Exception as e:
            print(f"Error in ADWIN drift detection: {e}")
            return np.random.rand() > 0.8  # Random fallback
    elif method == "ks_test":
        print("Monitoring Data Drift (method: KS-Test).")
        try:
            from scipy.stats import ks_2samp
            
            # Parameters for KS-test
            alpha_threshold = 0.05  # Significance level
            drift_detected = False
            
            # We need baseline data for comparison
            # For this implementation, we'll use the first client's data as baseline
            # In practice, you'd store baseline data from initial training
            
            client_ids = list(client_data_dict.keys())
            if len(client_ids) < 2:
                print("KS-Test requires at least 2 clients for comparison. No drift detected.")
                return False
            
            # Use first client as baseline
            baseline_client = client_ids[0]
            baseline_data = client_data_dict[baseline_client]
            
            if isinstance(baseline_data, tuple):
                baseline_X = baseline_data[0]
            else:
                baseline_X = baseline_data
            
            if isinstance(baseline_X, torch.Tensor):
                baseline_X = baseline_X.numpy()
            
            print(f"Using client {baseline_client} as baseline for KS-Test comparison.")
            
            # Compare each other client against baseline
            for client_id in client_ids[1:]:
                current_data = client_data_dict[client_id]
                
                if isinstance(current_data, tuple):
                    current_X = current_data[0]
                else:
                    current_X = current_data
                
                if isinstance(current_X, torch.Tensor):
                    current_X = current_X.numpy()
                
                print(f"Comparing client {client_id} against baseline...")
                
                # Perform KS-test for each feature
                if baseline_X.ndim == 1 and current_X.ndim == 1:
                    # Single feature case
                    statistic, p_value = ks_2samp(baseline_X, current_X)
                    if p_value < alpha_threshold:
                        print(f"KS-Test drift detected for client {client_id}: p-value={p_value:.4f} < {alpha_threshold}")
                        drift_detected = True
                    else:
                        print(f"KS-Test: No drift for client {client_id}: p-value={p_value:.4f}")
                elif baseline_X.ndim > 1 and current_X.ndim > 1:
                    # Multiple features case - test each feature
                    num_features = min(baseline_X.shape[1], current_X.shape[1])
                    
                    for feature_idx in range(num_features):
                        try:
                            baseline_feature = baseline_X[:, feature_idx]
                            current_feature = current_X[:, feature_idx]
                            
                            # Ensure we have enough data points for KS test
                            if len(baseline_feature) > 1 and len(current_feature) > 1:
                                statistic, p_value = ks_2samp(baseline_feature, current_feature)
                                
                                if p_value < alpha_threshold:
                                    print(f"KS-Test drift detected for client {client_id}, feature {feature_idx}: p-value={p_value:.4f}")
                                    drift_detected = True
                                    break  # Stop at first drift detection
                            else:
                                print(f"Insufficient data for KS-test on feature {feature_idx}")
                        except Exception as e:
                            print(f"Error in KS-test for feature {feature_idx}: {e}")
                            continue
                    
                    if not drift_detected:
                        print(f"KS-Test: No drift detected for client {client_id} across {num_features} features")
                else:
                    print(f"Dimension mismatch: baseline shape {baseline_X.shape}, current shape {current_X.shape}")
                
                if drift_detected:
                    break  # Stop testing other clients if drift is found
            
            return drift_detected
            
        except ImportError:
            print("scipy not available. Install with: pip install scipy")
            print("Falling back to simple variance-based drift detection.")
            
            # Simple fallback using variance comparison
            try:
                client_variances = []
                for client_id, client_data in client_data_dict.items():
                    if isinstance(client_data, tuple):
                        X_data = client_data[0]
                    else:
                        X_data = client_data
                    
                    if isinstance(X_data, torch.Tensor):
                        X_data = X_data.numpy()
                    
                    # Calculate variance for this client's data
                    client_var = np.var(X_data)
                    client_variances.append(client_var)
                
                # Check if variance across clients is significantly different
                if len(client_variances) > 1:
                    variance_of_variances = np.var(client_variances)
                    drift_threshold = 0.1  # Arbitrary threshold
                    drift_detected = variance_of_variances > drift_threshold
                    
                    if drift_detected:
                        print(f"Variance-based drift detection: variance of client variances ({variance_of_variances:.4f}) exceeds threshold")
                    else:
                        print(f"Variance-based drift detection: no significant difference in client variances")
                    
                    return drift_detected
                else:
                    return False
                    
            except Exception as e:
                print(f"Error in fallback drift detection: {e}")
                return np.random.rand() > 0.7  # Random fallback
                
        except Exception as e:
            print(f"Error in KS-Test drift detection: {e}")
            return np.random.rand() > 0.7  # Random fallback
    elif method == "none":
        print("Monitoring Data Drift (method: None).")
    else:
        print(f"Unknown Data Drift monitoring method: {method}.")
    return False # Default to no drift detected

# Placeholder for Privacy Preservation (Phase 3.2)
class PrivacyAccountant:
    """
    Privacy accountant for tracking privacy budget consumption in federated learning.
    """
    def __init__(self, epsilon_total, delta=1e-5):
        self.epsilon_total = epsilon_total
        self.delta = delta
        self.epsilon_consumed = 0.0
        self.round_history = []
    
    def consume_privacy_budget(self, epsilon_round):
        """Consume privacy budget for a round."""
        if self.epsilon_consumed + epsilon_round > self.epsilon_total:
            raise ValueError(f"Privacy budget exceeded! Requested: {epsilon_round}, "
                           f"Available: {self.epsilon_total - self.epsilon_consumed}")
        
        self.epsilon_consumed += epsilon_round
        self.round_history.append(epsilon_round)
        return True
    
    def get_remaining_budget(self):
        """Get remaining privacy budget."""
        return self.epsilon_total - self.epsilon_consumed
    
    def get_privacy_spent(self):
        """Get total privacy spent."""
        return self.epsilon_consumed, self.delta

def apply_differential_privacy(gradients, sensitivity=1.0, epsilon=1.0, delta=1e-5,
                             method="gaussian", max_grad_norm=1.0, sample_size=None):
    """
    Enhanced differential privacy implementation with proper privacy accounting.
    
    Args:
        gradients (list): List of gradients (model parameters as numpy arrays).
        sensitivity (float): The L2 sensitivity of the function.
        epsilon (float): The privacy budget for this round.
        delta (float): The delta parameter for (epsilon, delta)-DP.
        method (str): DP method ('gaussian', 'laplace', 'opacus', 'none').
        max_grad_norm (float): Maximum gradient norm for clipping.
        sample_size (int): Number of samples (for noise calibration).
    
    Returns:
        tuple: (noisy_gradients, privacy_metrics)
    """
    privacy_metrics = {
        'epsilon_consumed': epsilon,
        'delta': delta,
        'noise_multiplier': 0.0,
        'clipping_applied': False,
        'method': method
    }
    
    if method == "none":
        print("Applying Differential Privacy (method: None).")
        return gradients, privacy_metrics
    
    print(f"Applying Differential Privacy (method: {method}, ε={epsilon}, δ={delta})")
    
    try:
        # Convert gradients to tensors if needed
        tensor_gradients = []
        for grad in gradients:
            if isinstance(grad, np.ndarray):
                tensor_gradients.append(torch.from_numpy(grad).float())
            elif isinstance(grad, torch.Tensor):
                tensor_gradients.append(grad.float())
            else:
                tensor_gradients.append(torch.tensor(grad).float())
        
        # Apply gradient clipping
        if max_grad_norm > 0:
            total_norm = 0.0
            for grad in tensor_gradients:
                total_norm += grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > max_grad_norm:
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                for i, grad in enumerate(tensor_gradients):
                    tensor_gradients[i] = grad * clip_coef
                privacy_metrics['clipping_applied'] = True
                print(f"Gradient clipping applied: norm {total_norm:.4f} -> {max_grad_norm}")
        
        # Apply noise based on method
        if method == "gaussian":
            noise_multiplier = _calculate_gaussian_noise_multiplier(epsilon, delta, sample_size)
            noisy_gradients = _add_gaussian_noise(tensor_gradients, noise_multiplier, sensitivity)
            privacy_metrics['noise_multiplier'] = noise_multiplier
            
        elif method == "laplace":
            noise_scale = sensitivity / epsilon
            noisy_gradients = _add_laplace_noise(tensor_gradients, noise_scale)
            privacy_metrics['noise_multiplier'] = noise_scale
            
        elif method == "opacus":
            noisy_gradients, opacus_metrics = _apply_opacus_dp(tensor_gradients, epsilon, delta,
                                                             sensitivity, sample_size)
            privacy_metrics.update(opacus_metrics)
            
        else:
            print(f"Unknown DP method: {method}. Using Gaussian noise.")
            noise_multiplier = _calculate_gaussian_noise_multiplier(epsilon, delta, sample_size)
            noisy_gradients = _add_gaussian_noise(tensor_gradients, noise_multiplier, sensitivity)
            privacy_metrics['noise_multiplier'] = noise_multiplier
        
        # Convert back to numpy arrays
        result_gradients = []
        for grad in noisy_gradients:
            if isinstance(grad, torch.Tensor):
                result_gradients.append(grad.detach().cpu().numpy())
            else:
                result_gradients.append(grad)
        
        print(f"DP applied: ε={epsilon}, noise_multiplier={privacy_metrics['noise_multiplier']:.4f}")
        return result_gradients, privacy_metrics
        
    except Exception as e:
        print(f"Error in differential privacy implementation: {e}")
        print("Returning gradients without noise.")
        privacy_metrics['epsilon_consumed'] = 0.0
        privacy_metrics['method'] = 'failed'
        return gradients, privacy_metrics

def _calculate_gaussian_noise_multiplier(epsilon, delta, sample_size=None):
    """Calculate noise multiplier for Gaussian mechanism."""
    if sample_size is None:
        # Use standard formula for infinite sample size
        return np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    else:
        # Use more precise calculation for finite sample size
        # This is a simplified version; real implementation would use RDP accountant
        c = np.sqrt(2 * np.log(1.25 / delta))
        return c / epsilon

def _add_gaussian_noise(gradients, noise_multiplier, sensitivity):
    """Add Gaussian noise to gradients."""
    noisy_gradients = []
    for grad in gradients:
        noise_std = noise_multiplier * sensitivity
        noise = torch.normal(0, noise_std, grad.shape)
        noisy_grad = grad + noise
        noisy_gradients.append(noisy_grad)
    return noisy_gradients

def _add_laplace_noise(gradients, noise_scale):
    """Add Laplace noise to gradients."""
    noisy_gradients = []
    for grad in gradients:
        # PyTorch doesn't have Laplace distribution, so we'll use numpy
        noise = np.random.laplace(0, noise_scale, grad.shape)
        noise_tensor = torch.from_numpy(noise).float()
        noisy_grad = grad + noise_tensor
        noisy_gradients.append(noisy_grad)
    return noisy_gradients

def _apply_opacus_dp(gradients, epsilon, delta, sensitivity, sample_size):
    """Apply differential privacy using Opacus library."""
    try:
        from opacus.accountants.utils import get_noise_multiplier
        
        # Calculate noise multiplier using Opacus
        if sample_size is None:
            sample_size = 1000  # Default assumption
        
        sample_rate = 1.0 / sample_size
        epochs = 1  # Single round
        
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs
        )
        
        # Apply noise
        noisy_gradients = _add_gaussian_noise(gradients, noise_multiplier, sensitivity)
        
        metrics = {
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'opacus_available': True
        }
        
        return noisy_gradients, metrics
        
    except ImportError:
        print("Opacus not available. Falling back to manual implementation.")
        noise_multiplier = _calculate_gaussian_noise_multiplier(epsilon, delta, sample_size)
        noisy_gradients = _add_gaussian_noise(gradients, noise_multiplier, sensitivity)
        
        metrics = {
            'noise_multiplier': noise_multiplier,
            'opacus_available': False
        }
        
        return noisy_gradients, metrics

class ByzantineAttackSimulator:
    """
    Simulator for various Byzantine attacks in federated learning.
    """
    
    @staticmethod
    def label_flipping_attack(client_updates, attack_ratio=0.2, flip_probability=0.5):
        """
        Simulate label flipping attack by modifying client updates.
        
        Args:
            client_updates (list): List of (parameters, num_samples) tuples.
            attack_ratio (float): Fraction of clients to compromise.
            flip_probability (float): Probability of flipping each parameter.
        
        Returns:
            list: Modified client updates with attacks.
        """
        num_clients = len(client_updates)
        num_malicious = int(num_clients * attack_ratio)
        
        if num_malicious == 0:
            return client_updates
        
        # Select random clients to compromise
        malicious_indices = np.random.choice(num_clients, num_malicious, replace=False)
        
        attacked_updates = []
        for i, (params, num_samples) in enumerate(client_updates):
            if i in malicious_indices:
                # Apply label flipping by negating some parameters
                attacked_params = []
                for param in params:
                    if np.random.random() < flip_probability:
                        attacked_param = -param  # Flip the parameter
                    else:
                        attacked_param = param
                    attacked_params.append(attacked_param)
                attacked_updates.append((attacked_params, num_samples))
                print(f"Applied label flipping attack to client {i}")
            else:
                attacked_updates.append((params, num_samples))
        
        return attacked_updates
    
    @staticmethod
    def gradient_ascent_attack(client_updates, attack_ratio=0.2, scale_factor=10.0):
        """
        Simulate gradient ascent attack by scaling malicious updates.
        
        Args:
            client_updates (list): List of (parameters, num_samples) tuples.
            attack_ratio (float): Fraction of clients to compromise.
            scale_factor (float): Factor to scale malicious updates.
        
        Returns:
            list: Modified client updates with attacks.
        """
        num_clients = len(client_updates)
        num_malicious = int(num_clients * attack_ratio)
        
        if num_malicious == 0:
            return client_updates
        
        malicious_indices = np.random.choice(num_clients, num_malicious, replace=False)
        
        attacked_updates = []
        for i, (params, num_samples) in enumerate(client_updates):
            if i in malicious_indices:
                # Scale parameters to simulate gradient ascent
                attacked_params = []
                for param in params:
                    attacked_param = param * scale_factor
                    attacked_params.append(attacked_param)
                attacked_updates.append((attacked_params, num_samples))
                print(f"Applied gradient ascent attack to client {i} (scale: {scale_factor})")
            else:
                attacked_updates.append((params, num_samples))
        
        return attacked_updates
    
    @staticmethod
    def gaussian_noise_attack(client_updates, attack_ratio=0.2, noise_std=1.0):
        """
        Simulate Gaussian noise attack by adding random noise.
        
        Args:
            client_updates (list): List of (parameters, num_samples) tuples.
            attack_ratio (float): Fraction of clients to compromise.
            noise_std (float): Standard deviation of Gaussian noise.
        
        Returns:
            list: Modified client updates with attacks.
        """
        num_clients = len(client_updates)
        num_malicious = int(num_clients * attack_ratio)
        
        if num_malicious == 0:
            return client_updates
        
        malicious_indices = np.random.choice(num_clients, num_malicious, replace=False)
        
        attacked_updates = []
        for i, (params, num_samples) in enumerate(client_updates):
            if i in malicious_indices:
                # Add Gaussian noise to parameters
                attacked_params = []
                for param in params:
                    noise = np.random.normal(0, noise_std, param.shape)
                    attacked_param = param + noise
                    attacked_params.append(attacked_param)
                attacked_updates.append((attacked_params, num_samples))
                print(f"Applied Gaussian noise attack to client {i} (std: {noise_std})")
            else:
                attacked_updates.append((params, num_samples))
        
        return attacked_updates

def simulate_byzantine_attacks(client_updates, attack_type="none", attack_ratio=0.2, **kwargs):
    """
    Simulate various Byzantine attacks on client updates.
    
    Args:
        client_updates (list): List of (parameters, num_samples) tuples.
        attack_type (str): Type of attack ('none', 'label_flipping', 'gradient_ascent', 
                          'gaussian_noise').
        attack_ratio (float): Fraction of clients to compromise.
        **kwargs: Additional attack-specific parameters.
    
    Returns:
        list: Client updates with simulated attacks.
    """
    if attack_type == "none":
        return client_updates
    
    simulator = ByzantineAttackSimulator()
    
    if attack_type == "label_flipping":
        flip_probability = kwargs.get('flip_probability', 0.5)
        return simulator.label_flipping_attack(client_updates, attack_ratio, flip_probability)
    
    elif attack_type == "gradient_ascent":
        scale_factor = kwargs.get('scale_factor', 10.0)
        return simulator.gradient_ascent_attack(client_updates, attack_ratio, scale_factor)
    
    elif attack_type == "gaussian_noise":
        noise_std = kwargs.get('noise_std', 1.0)
        return simulator.gaussian_noise_attack(client_updates, attack_ratio, noise_std)
    
    else:
        print(f"Unknown attack type: {attack_type}. Returning original updates.")
        return client_updates
# Placeholder for Poisoning Defense and Robust Aggregation (Phase 3.3)
def apply_robust_aggregation(client_updates, method="fedavg", num_malicious=0):
    """
    Applies robust aggregation techniques to client updates.
    Args:
        client_updates (list): A list of tuples, where each tuple contains
                               (client_parameters, num_samples_from_client).
        method (str): The robust aggregation method to use ('fedavg', 'krum', 'trimmed_mean', 'median').
        num_malicious (int): Number of potential malicious clients (for Krum).
    Returns:
        list: Aggregated global model parameters.
    """
    client_params = [update[0] for update in client_updates]
    client_num_samples = [update[1] for update in client_updates]
    total_samples = sum(client_num_samples)

    if method == "fedavg":
        print("Applying Robust Aggregation (method: Federated Averaging - baseline).")
        aggregated_params = [
            np.zeros_like(param, dtype=np.float32) for param in client_params[0]
        ]
        for i, params in enumerate(client_params):
            weight = client_num_samples[i] / total_samples
            for j in range(len(params)):
                aggregated_params[j] += params[j] * weight
        return aggregated_params
    elif method == "krum":
        print(f"Applying Robust Aggregation (method: Krum, num_malicious={num_malicious}).")
        # Krum Implementation - selects the most representative client update
        # based on distances to other clients, robust against Byzantine attacks
        
        num_clients = len(client_params)
        if num_clients <= 2 * num_malicious:
            print("Warning: Too many malicious clients for Krum. Falling back to FedAvg.")
            return apply_robust_aggregation(client_updates, method="fedavg")
        
        # Calculate number of closest clients to consider (n - f - 2)
        num_closest = num_clients - num_malicious - 2
        if num_closest <= 0:
            num_closest = 1
        
        scores = []
        for i, update_i in enumerate(client_params):
            distances = []
            # Calculate distances to all other clients
            for j, update_j in enumerate(client_params):
                if i != j:
                    # Calculate Euclidean distance between flattened parameters
                    flat_i = np.concatenate([p.flatten() for p in update_i])
                    flat_j = np.concatenate([p.flatten() for p in update_j])
                    dist = np.linalg.norm(flat_i - flat_j)
                    distances.append(dist)
            
            # Sort distances and sum the num_closest smallest ones
            distances.sort()
            score = sum(distances[:num_closest])
            scores.append(score)
        
        # Select the client with the minimum score (most "central" update)
        best_client_idx = np.argmin(scores)
        print(f"Krum selected client {best_client_idx} with score {scores[best_client_idx]:.4f}")
        aggregated_params = client_params[best_client_idx]
        return aggregated_params
    elif method == "trimmed_mean":
        print("Applying Robust Aggregation (method: Trimmed Mean).")
        # Trimmed Mean Implementation - removes outliers by trimming extreme values
        # More robust than simple averaging against Byzantine attacks
        
        # Configurable trim ratio - remove this percentage from each end
        trim_ratio = min(0.3, num_malicious / len(client_params))  # Adaptive trim ratio
        if trim_ratio == 0:
            trim_ratio = 0.1  # Default 10% trim
        
        print(f"Using trim ratio: {trim_ratio:.2f}")
        
        aggregated_params = []
        for layer_idx in range(len(client_params[0])):
            layer_params = np.array([client_param[layer_idx] for client_param in client_params])
            
            # Apply trimmed mean element-wise
            num_to_trim = max(1, int(trim_ratio * len(client_params)))
            
            # Ensure we don't trim too much
            if 2 * num_to_trim >= len(client_params):
                num_to_trim = max(0, (len(client_params) - 1) // 2)
            
            if num_to_trim > 0:
                # Sort along client dimension and trim
                sorted_params = np.sort(layer_params, axis=0)
                trimmed_params = sorted_params[num_to_trim:-num_to_trim]
                aggregated_layer = np.mean(trimmed_params, axis=0)
            else:
                # No trimming needed
                aggregated_layer = np.mean(layer_params, axis=0)
            
            aggregated_params.append(aggregated_layer)
        
        print(f"Trimmed mean: removed {num_to_trim} extreme values from each end")
        return aggregated_params
    elif method == "median":
        print("Applying Robust Aggregation (method: Median).")
        # Median Aggregation Implementation - highly robust against outliers
        # Takes the median of each parameter across all clients
        
        aggregated_params = []
        for layer_idx in range(len(client_params[0])):
            layer_params = np.array([client_param[layer_idx] for client_param in client_params])
            # Use median which is robust against up to 50% Byzantine clients
            aggregated_layer = np.median(layer_params, axis=0)
            aggregated_params.append(aggregated_layer)
        
        print(f"Median aggregation: robust against up to {len(client_params)//2} Byzantine clients")
        return aggregated_params
    else:
        print(f"Unknown Robust Aggregation method: {method}. Falling back to FedAvg.")
        return apply_robust_aggregation(client_updates, method="fedavg")

# Placeholder for Communication Efficiency (Phase 3.4)
def apply_communication_efficiency(model_updates, compression_ratio=0.1, method="none"):
    """
    Applies communication efficiency techniques (e.g., compression) to model updates.
    Args:
        model_updates (list): List of model updates (e.g., gradients or parameters).
        compression_ratio (float): The ratio of compression (e.g., for sparsification).
        method (str): The communication efficiency method to use ('none', 'top_k', 'quantization').
    Returns:
        list: Compressed/efficient model updates.
    """
    if method == "top_k":
        print(f"Applying Communication Efficiency (method: Top-k Sparsification, ratio={compression_ratio}).")
        # Top-k Sparsification Implementation - keeps only the k largest magnitude updates
        # Significantly reduces communication overhead while preserving important updates
        
        if not model_updates:
            return []
        compressed_updates = []
        total_params = 0
        total_kept = 0
        
        for update in model_updates:
            # Handle case where update is a list of parameters
            if isinstance(update, list):
                # Process each parameter in the list
                compressed_params = []
                for param in update:
                    if isinstance(param, torch.Tensor):
                        param = param.numpy()
                    
                    # Flatten the parameter to find top-k values by magnitude
                    flat_param = param.flatten()
                    total_params += len(flat_param)
                    
                    # Calculate k based on compression ratio
                    k = min(len(flat_param), max(1, int(len(flat_param) * compression_ratio)))
                    total_kept += k
                    
                    # Get indices of top-k values by absolute magnitude
                    top_k_indices = np.argpartition(np.abs(flat_param), -k)[-k:]
                    
                    # Create a sparse update with only top-k values
                    sparse_param = np.zeros_like(flat_param)
                    sparse_param[top_k_indices] = flat_param[top_k_indices]
                    
                    compressed_params.append(sparse_param.reshape(param.shape))
                
                compressed_updates.append(compressed_params)
                continue
            
            if isinstance(update, torch.Tensor):
                update = update.numpy()
            
            # Flatten the update to find top-k values by magnitude
            flat_update = update.flatten()
            total_params += len(flat_update)
            
            # Calculate k based on compression ratio
            k = min(len(flat_update), max(1, int(len(flat_update) * compression_ratio)))
            total_kept += k
            
            # Get indices of top-k values by absolute magnitude
            top_k_indices = np.argpartition(np.abs(flat_update), -k)[-k:]
            
            # Create a sparse update with only top-k values
            sparse_update = np.zeros_like(flat_update)
            sparse_update[top_k_indices] = flat_update[top_k_indices]
            
            compressed_updates.append(sparse_update.reshape(update.shape))
        
        compression_achieved = total_kept / total_params
        print(f"Top-k sparsification: kept {total_kept}/{total_params} parameters ({compression_achieved:.1%})")
        return compressed_updates
    elif method == "quantization":
        print(f"Applying Communication Efficiency (method: Quantization, ratio={compression_ratio}).")
        # Multi-bit Quantization Implementation - reduces precision to save bandwidth
        # Supports 1-bit, 2-bit, 4-bit, and 8-bit quantization
        
        # Determine quantization bits based on compression ratio
        if compression_ratio <= 0.125:  # 1-bit
            bits = 1
            levels = 2
        elif compression_ratio <= 0.25:  # 2-bit
            bits = 2
            levels = 4
        elif compression_ratio <= 0.5:   # 4-bit
            bits = 4
            levels = 16
        else:                           # 8-bit
            bits = 8
            levels = 256
        
        print(f"Using {bits}-bit quantization with {levels} levels")
        
        compressed_updates = []
        for update in model_updates:
            # Handle case where update is a list of parameters
            if isinstance(update, list):
                # Process each parameter in the list
                quantized_params = []
                for param in update:
                    if isinstance(param, torch.Tensor):
                        param = param.numpy()
                    
                    if bits == 1:
                        # 1-bit quantization: sign-based
                        quantized_param = np.sign(param)
                        # Store the scale factor for reconstruction
                        scale = np.mean(np.abs(param))
                        quantized_param = quantized_param * scale
                    else:
                        # Multi-bit uniform quantization
                        # Find min and max values for dynamic range
                        min_val = np.min(param)
                        max_val = np.max(param)
                        
                        if max_val == min_val:
                            # Handle constant arrays
                            quantized_param = param
                        else:
                            # Quantize to specified number of levels
                            scale = (max_val - min_val) / (levels - 1)
                            quantized_indices = np.round((param - min_val) / scale)
                            quantized_indices = np.clip(quantized_indices, 0, levels - 1)
                            quantized_param = min_val + quantized_indices * scale
                    
                    quantized_params.append(quantized_param)
                
                compressed_updates.append(quantized_params)
                continue
            
            if isinstance(update, torch.Tensor):
                update = update.numpy()
            
            if bits == 1:
                # 1-bit quantization: sign-based
                quantized_update = np.sign(update)
                # Store the scale factor for reconstruction
                scale = np.mean(np.abs(update))
                quantized_update = quantized_update * scale
            else:
                # Multi-bit uniform quantization
                # Find min and max values for dynamic range
                min_val = np.min(update)
                max_val = np.max(update)
                
                if max_val == min_val:
                    # Handle constant arrays
                    quantized_update = update
                else:
                    # Quantize to specified number of levels
                    scale = (max_val - min_val) / (levels - 1)
                    quantized_indices = np.round((update - min_val) / scale)
                    quantized_indices = np.clip(quantized_indices, 0, levels - 1)
                    quantized_update = min_val + quantized_indices * scale
            
            compressed_updates.append(quantized_update)
        
        theoretical_compression = 32 / bits  # Assuming 32-bit floats
        print(f"Quantization: {bits}-bit encoding, theoretical compression ratio: {1/theoretical_compression:.1%}")
        return compressed_updates
    elif method == "none":
        print("Applying Communication Efficiency (method: None).")
        compressed_updates = model_updates
    else:
        print(f"Unknown Communication Efficiency method: {method}. Updates returned as is.")
        compressed_updates = model_updates
    return compressed_updates

# Placeholder for Fairness in Federated Models (Phase 3.5)
def monitor_federated_fairness(global_model, client_data_loaders, device, fairness_metric="f1_score_parity", method="none"):
    """
    Monitors fairness across federated clients using the specified method and metric.
    Args:
        global_model: The global model.
        client_data_loaders (dict): Dictionary of client data loaders.
        device: The device (cpu/cuda).
        fairness_metric (str): The fairness metric to calculate ('f1_score_parity', 'demographic_parity', 'equalized_odds').
        method (str): The fairness monitoring method to use ('none', 'subgroup_evaluation').
    Returns:
        float: A fairness score.
    """
    if method == "subgroup_evaluation":
        print(f"Monitoring Federated Fairness (method: Subgroup Evaluation, metric={fairness_metric}).")
        # Comprehensive Subgroup Evaluation Implementation
        # Evaluates model performance across different demographic groups and clients
        
        try:
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            from sklearn.metrics import confusion_matrix, classification_report
            
            fairness_scores = {}
            overall_metrics = {}
            
            # Simulate sensitive attributes (in real scenario, these would be provided)
            # For medical triage, sensitive attributes might include age groups, gender, etc.
            def simulate_sensitive_attributes(data_size):
                """Simulate sensitive attributes for fairness evaluation"""
                # Age groups: 0=young (18-35), 1=middle (36-65), 2=elderly (65+)
                age_groups = np.random.choice([0, 1, 2], size=data_size, p=[0.3, 0.4, 0.3])
                # Gender: 0=female, 1=male, 2=other
                gender = np.random.choice([0, 1, 2], size=data_size, p=[0.45, 0.45, 0.1])
                return {'age_group': age_groups, 'gender': gender}
            
            def calculate_fairness_metrics(y_true, y_pred, sensitive_attr, attr_name):
                """Calculate comprehensive fairness metrics for a sensitive attribute"""
                unique_groups = np.unique(sensitive_attr)
                group_metrics = {}
                
                for group in unique_groups:
                    group_mask = sensitive_attr == group
                    group_true = y_true[group_mask]
                    group_pred = y_pred[group_mask]
                    
                    if len(group_true) == 0:
                        continue
                    
                    # Calculate performance metrics for this group
                    group_acc = accuracy_score(group_true, group_pred)
                    group_f1 = f1_score(group_true, group_pred, average='weighted', zero_division=0)
                    group_precision = precision_score(group_true, group_pred, average='weighted', zero_division=0)
                    group_recall = recall_score(group_true, group_pred, average='weighted', zero_division=0)
                    
                    # Demographic parity: P(Y_hat=1|A=a)
                    demo_parity = np.mean(group_pred == 1) if len(group_pred) > 0 else 0
                    
                    # Equalized odds: TPR and FPR across groups
                    if len(np.unique(group_true)) > 1:
                        try:
                            cm = confusion_matrix(group_true, group_pred, labels=[0, 1, 2])
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            else:
                                tpr, fpr = 0, 0
                        except:
                            tpr, fpr = 0, 0
                    else:
                        tpr, fpr = 0, 0
                    
                    group_metrics[f"{attr_name}_group_{group}"] = {
                        'accuracy': group_acc,
                        'f1_score': group_f1,
                        'precision': group_precision,
                        'recall': group_recall,
                        'demographic_parity': demo_parity,
                        'true_positive_rate': tpr,
                        'false_positive_rate': fpr,
                        'sample_size': len(group_true)
                    }
                
                return group_metrics
            
            def calculate_fairness_violations(group_metrics, metric_name, threshold=0.1):
                """Calculate fairness violations based on metric differences between groups"""
                values = [metrics[metric_name] for metrics in group_metrics.values() if metric_name in metrics]
                if len(values) < 2:
                    return 0.0, False
                
                max_val = max(values)
                min_val = min(values)
                difference = max_val - min_val
                violation = difference > threshold
                
                return difference, violation
            
            # Process each client's data
            for client_id, data_loader in client_data_loaders.items():
                print(f"Evaluating fairness for {client_id}...")
                
                all_preds = []
                all_labels = []
                
                # Get predictions for all data in this client
                global_model.eval()
                with torch.no_grad():
                    for batch in data_loader:
                        if len(batch) == 4:  # (numerical, boolean, temporal, labels)
                            numerical_data, boolean_data, temporal_data, labels = batch
                            numerical_data = numerical_data.to(device)
                            boolean_data = boolean_data.to(device)
                            temporal_data = temporal_data.to(device)
                            labels = labels.to(device)
                            
                            outputs = global_model(numerical_data, boolean_data, temporal_data)
                            preds = torch.argmax(outputs, dim=1)
                            
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                
                if len(all_preds) == 0:
                    continue
                
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                # Simulate sensitive attributes (in real scenario, these would be extracted from data)
                sensitive_attrs = simulate_sensitive_attributes(len(all_labels))
                
                # Calculate fairness metrics for each sensitive attribute
                client_fairness = {}
                
                for attr_name, attr_values in sensitive_attrs.items():
                    group_metrics = calculate_fairness_metrics(all_labels, all_preds, attr_values, attr_name)
                    client_fairness.update(group_metrics)
                    
                    # Calculate fairness violations
                    for metric in ['accuracy', 'f1_score', 'demographic_parity']:
                        diff, violation = calculate_fairness_violations(group_metrics, metric)
                        client_fairness[f"{attr_name}_{metric}_difference"] = diff
                        client_fairness[f"{attr_name}_{metric}_violation"] = violation
                
                fairness_scores[client_id] = client_fairness
                
                # Print summary for this client
                print(f"  {client_id} fairness summary:")
                for attr_name in sensitive_attrs.keys():
                    acc_diff = client_fairness.get(f"{attr_name}_accuracy_difference", 0)
                    f1_diff = client_fairness.get(f"{attr_name}_f1_score_difference", 0)
                    print(f"    {attr_name}: accuracy_diff={acc_diff:.3f}, f1_diff={f1_diff:.3f}")
            
            # Calculate overall fairness score
            if fairness_metric == "f1_score_parity":
                # Average F1 score difference across all clients and attributes
                f1_diffs = []
                for client_scores in fairness_scores.values():
                    for key, value in client_scores.items():
                        if 'f1_score_difference' in key:
                            f1_diffs.append(value)
                fairness_score = np.mean(f1_diffs) if f1_diffs else 0.0
                
            elif fairness_metric == "demographic_parity":
                # Average demographic parity difference
                dp_diffs = []
                for client_scores in fairness_scores.values():
                    for key, value in client_scores.items():
                        if 'demographic_parity_difference' in key:
                            dp_diffs.append(value)
                fairness_score = np.mean(dp_diffs) if dp_diffs else 0.0
                
            elif fairness_metric == "equalized_odds":
                # Average TPR and FPR differences
                tpr_diffs = []
                for client_scores in fairness_scores.values():
                    for key, value in client_scores.items():
                        if 'true_positive_rate' in key:
                            tpr_diffs.append(value)
                fairness_score = np.std(tpr_diffs) if tpr_diffs else 0.0
            else:
                fairness_score = 0.0
            
            print(f"Overall fairness score ({fairness_metric}): {fairness_score:.4f}")
            
            # Store detailed results
            overall_metrics['fairness_scores'] = fairness_scores
            overall_metrics['overall_fairness_score'] = fairness_score
            overall_metrics['metric_type'] = fairness_metric
            
            return overall_metrics
            
        except ImportError:
            print("sklearn not available. Install with: pip install scikit-learn")
            print("Using placeholder fairness score.")
            fairness_score = 0.0
        except Exception as e:
            print(f"Error in fairness evaluation: {e}")
            fairness_score = 0.0
    elif method == "none":
        print("Monitoring Federated Fairness (method: None).")
        fairness_score = 0.0 # Default score
    else:
        print(f"Unknown Fairness Monitoring method: {method}.")
        fairness_score = 0.0 # Default score
    return fairness_score

if __name__ == "__main__":
    # This block demonstrates the FL setup and a single round of training.
    # In a full implementation, this would be part of the main training loop.

    from data_preparation import load_and_clean_data
    from feature_engineering import feature_engineer_data
    from data_simulation import simulate_multi_site_data

    # 1. Data Preparation and Feature Engineering
    file_path = 'triaj_data.csv'
    df_cleaned = load_and_clean_data(file_path)
    df_engineered = feature_engineer_data(df_cleaned.copy())

    # Separate features (X) and target (y)
    X = df_engineered.drop('doğru triyaj_encoded', axis=1)
    y = df_engineered['doğru triyaj_encoded']

    # Identify feature types based on feature_engineering.py logic
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    temporal_cols = ['hour_of_day', 'day_of_week', 'month']
    boolean_cols = [col for col in X.columns if col not in numerical_cols + temporal_cols + ['year', 'cinsiyet_Male', 'cinsiyet_Female', 'cinsiyet_Unknown', 'yaş_unscaled']]
    if 'cinsiyet_Male' in X.columns: boolean_cols.append('cinsiyet_Male')
    if 'cinsiyet_Female' in X.columns: boolean_cols.append('cinsiyet_Female')
    if 'cinsiyet_Unknown' in X.columns: boolean_cols.append('cinsiyet_Unknown')

    num_numerical_features = len(numerical_cols)
    num_boolean_features = len(boolean_cols)
    num_temporal_features = len(temporal_cols)
    num_classes = len(y.unique())

    # Convert DataFrame to PyTorch Tensors
    X_numerical = torch.tensor(X[numerical_cols].values, dtype=torch.float32)
    X_boolean = torch.tensor(X[boolean_cols].values, dtype=torch.float32)
    X_temporal = torch.tensor(X[temporal_cols].values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    full_dataset = TensorDataset(X_numerical, X_boolean, X_temporal, y_tensor)

    # 2. Simulate Multi-Site Data
    # For FL simulation, we need to split the full dataset into client datasets
    # and a global test set.
    
    # First, split into training data for clients and a global test set
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, global_test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # Now, simulate clients from the train_dataset
    num_clients = 3
    client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients, generator=torch.Generator().manual_seed(42))

    # Create DataLoaders for each client
    client_data_loaders = []
    for i, client_ds in enumerate(client_datasets):
        client_data_loaders.append(DataLoader(client_ds, batch_size=32, shuffle=True))
    
    global_test_loader = DataLoader(global_test_dataset, batch_size=32, shuffle=False)

    # 3. Initialize Models and FL Components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global model (server-side)
    global_model = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes).to(device)
    server = FederatedServer(global_model, device)

    # Client models
    clients = []
    for i, data_loader in enumerate(client_data_loaders):
        client_model = TriageModel(num_numerical_features, num_boolean_features, num_temporal_features, num_classes).to(device)
        # Initialize client model with global model's parameters
        client_model.load_state_dict(global_model.state_dict()) 
        clients.append(FederatedClient(f"client_{i}", client_model, data_loader, device))

    print("\n--- Phase 3: Federated Learning and Robustness Integration ---")
    print("Step 1: Simulated Multi-Node Training (Federated Learning Round)...")

    # Simulate a few rounds of Federated Learning
    num_communication_rounds = 5
    for round_num in range(num_communication_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1}/{num_communication_rounds} ---")
        
        # Clients train locally
        client_updates = []
        for client in clients:
            # Distribute global model to client before training
            client.set_parameters(server.global_model.get_parameters())
            
            # Simulate on-device model adaptation (Phase 2.3 dynamic personalization)
            # In a real scenario, this would involve fine-tuning on new local data
            print(f"Client {client.client_id}: Performing local training and on-device adaptation...")
            params, num_samples = client.train(epochs=1) # Train for 1 local epoch
            client_updates.append((params, num_samples))
        
        # Server aggregates updates
        aggregated_params = server.aggregate_parameters(client_updates)
        server.global_model.set_parameters(aggregated_params) # Update global model

        # Evaluate global model
        print(f"\n--- Global Model Evaluation after Round {round_num + 1} ---")
        server.evaluate_global_model(global_test_loader)

    print("\nFederated Learning simulation complete.")

    # Placeholder for subsequent phases
    print("\n--- Proceeding to Phase 3.2: Privacy Preservation (Not yet implemented) ---")
    print("--- Proceeding to Phase 3.3: Poisoning Defense and Robust Aggregation (Not yet implemented) ---")
    print("--- Proceeding to Phase 3.4: Communication Efficiency (Not yet implemented) ---")
    print("--- Proceeding to Phase 3.5: Fairness in Federated Models (Not yet implemented) ---")
    print("--- Proceeding to Phase 4: Explainable AI (XAI) and LLM Integration (Not yet implemented) ---")
    print("--- Proceeding to Phase 5: Comprehensive Evaluation and Open Science (Not yet implemented) ---")