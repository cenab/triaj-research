import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import time
from typing import Dict, List, Tuple, Any
from numpy.random import RandomState
import json
from datetime import datetime

class ClinicalMetrics:
    """
    Clinical evaluation metrics specific to medical triage.
    """
    
    @staticmethod
    def calculate_triage_metrics(y_true, y_pred, class_names=None):
        """
        Calculate triage-specific metrics.
        
        Args:
            y_true (array): True triage labels encoded with ascending severity (e.g., 0..K-1)
            y_pred (array): Predicted triage labels
            class_names (list): Optional names for classes
        
        Returns:
            dict: Comprehensive triage metrics
        """
        if class_names is None:
            # Infer number of classes from data
            k = int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 0
            if k == 5:
                class_names = ['ESI5', 'ESI4', 'ESI3', 'ESI2', 'ESI1']
            elif k == 3:
                class_names = ['Green', 'Yellow', 'Red']
            else:
                # ESI-only configuration: fall back to generic class names for non-5-class setups
                class_names = [f'Class {i}' for i in range(k)]
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Triage-specific metrics
        metrics = {
            'overall_accuracy': accuracy,
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'clinical_safety': {}
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            metrics['class_metrics'][class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0
            }
        
        # Clinical safety metrics
        metrics['clinical_safety'] = ClinicalMetrics._calculate_safety_metrics(y_true, y_pred, cm)
        
        return metrics
    
    @staticmethod
    def _calculate_safety_metrics(y_true, y_pred, cm):
        """Calculate clinical safety metrics."""
        total_samples = len(y_true)
        
        # Under-triage: Assigning lower priority than correct
        under_triage = 0
        # Over-triage: Assigning higher priority than correct
        over_triage = 0
        
        for true_label, pred_label in zip(y_true, y_pred):
            if pred_label < true_label:  # Under-triage (more dangerous)
                under_triage += 1
            elif pred_label > true_label:  # Over-triage (less dangerous but resource waste)
                over_triage += 1
        
        # Critical under-triage/sensitivity: ESI top-2 band by default
        k = int(np.max(y_true)) + 1 if len(y_true) else 0
        critical_threshold = max(1, k - 2)
        critical_mask = y_true >= critical_threshold
        critical_cases = int(np.sum(critical_mask))
        if critical_cases > 0:
            critical_detected = int(np.sum((y_pred >= critical_threshold) & critical_mask))
            critical_under = int(np.sum((y_pred < critical_threshold) & critical_mask))
            critical_under_rate = critical_under / critical_cases
            critical_sens = critical_detected / critical_cases
        else:
            critical_under_rate = 0.0
            critical_sens = 0.0

        return {
            'under_triage_rate': under_triage / total_samples if total_samples > 0 else 0.0,
            'over_triage_rate': over_triage / total_samples if total_samples > 0 else 0.0,
            'critical_under_triage_rate': critical_under_rate,
            'critical_sensitivity': critical_sens,
        }

    @staticmethod
    def _critical_rate(y_pred: np.ndarray) -> float:
        """Rate of critical predictions under ascending-severity encoding.

        For K=3, critical=class 2. For K>=5, critical=top-2 classes.
        """
        if y_pred.size == 0:
            return 0.0
        k = int(np.max(y_pred)) + 1
        if k == 3:
            return float(np.mean(y_pred == 2))
        threshold = max(1, k - 2)
        return float(np.mean(y_pred >= threshold))

class FairnessEvaluator:
    """
    Comprehensive fairness evaluation for medical AI systems.
    """
    
    def __init__(self, sensitive_attributes=None):
        self.sensitive_attributes = sensitive_attributes or ['age_group', 'gender']
    
    def evaluate_fairness(self, y_true, y_pred, sensitive_data, threshold=0.1):
        """
        Evaluate fairness across demographic groups.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            sensitive_data (dict): Dictionary with sensitive attribute arrays
            threshold (float): Fairness violation threshold
        
        Returns:
            dict: Comprehensive fairness metrics
        """
        fairness_metrics = {
            'overall_fairness_score': 0.0,
            'group_metrics': {},
            'fairness_violations': [],
            'bias_summary': {}
        }
        
        for attr_name, attr_values in sensitive_data.items():
            if attr_name not in self.sensitive_attributes:
                continue
            
            group_metrics = self._calculate_group_metrics(y_true, y_pred, attr_values, attr_name)
            fairness_metrics['group_metrics'][attr_name] = group_metrics
            
            # Check for fairness violations
            violations = self._detect_fairness_violations(group_metrics, threshold)
            if violations:
                fairness_metrics['fairness_violations'].extend(violations)
        
        # Calculate overall fairness score
        fairness_metrics['overall_fairness_score'] = self._calculate_overall_fairness_score(
            fairness_metrics['group_metrics']
        )
        
        return fairness_metrics
    
    def _calculate_group_metrics(self, y_true, y_pred, group_attr, attr_name):
        """Calculate metrics for each demographic group."""
        unique_groups = np.unique(group_attr)
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = group_attr == group
            if np.sum(group_mask) == 0:
                continue
            
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate metrics for this group
            group_metrics[str(group)] = {
                'sample_size': np.sum(group_mask),
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'precision': precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                'recall': recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                'positive_rate': np.mean(group_y_pred > 0),  # Non-green predictions (works for 3/5-class)
                'critical_rate': ClinicalMetrics._critical_rate(group_y_pred)  # Rate of critical predictions
            }
        
        return group_metrics
    
    def _detect_fairness_violations(self, group_metrics, threshold):
        """Detect fairness violations between groups."""
        violations = []
        metrics_to_check = ['accuracy', 'f1_score', 'positive_rate', 'critical_rate']
        
        for metric in metrics_to_check:
            values = [group[metric] for group in group_metrics.values() if metric in group]
            if len(values) < 2:
                continue
            
            max_val = max(values)
            min_val = min(values)
            difference = max_val - min_val
            
            if difference > threshold:
                violations.append({
                    'metric': metric,
                    'max_value': max_val,
                    'min_value': min_val,
                    'difference': difference,
                    'threshold': threshold
                })
        
        return violations
    
    def _calculate_overall_fairness_score(self, group_metrics):
        """Calculate overall fairness score (0-1, higher is more fair)."""
        if not group_metrics:
            return 1.0
        
        fairness_scores = []
        for attr_metrics in group_metrics.values():
            # Calculate coefficient of variation for key metrics
            metrics_to_evaluate = ['accuracy', 'f1_score']
            attr_fairness = []
            
            for metric in metrics_to_evaluate:
                values = [group[metric] for group in attr_metrics.values() if metric in group]
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / (mean_val + 1e-8)  # Coefficient of variation
                    fairness = max(0, 1 - cv)  # Higher CV = lower fairness
                    attr_fairness.append(fairness)
            
            if attr_fairness:
                fairness_scores.append(np.mean(attr_fairness))
        
        return np.mean(fairness_scores) if fairness_scores else 1.0

    @staticmethod
    def _critical_rate(y_pred: np.ndarray) -> float:
        if y_pred.size == 0:
            return 0.0
        k = int(np.max(y_pred)) + 1
        if k == 3:
            return float(np.mean(y_pred == 2))
        threshold = max(1, k - 2)
        return float(np.mean(y_pred >= threshold))


class ReliabilityMetrics:
    """
    Reliability and calibration utilities: NLL, ECE, Brier score, and bootstrap CIs.
    """

    @staticmethod
    def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        p = np.clip(y_prob[np.arange(len(y_true)), y_true], eps, 1.0)
        return float(-np.log(p).mean())

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n = y_true.shape[0]
        k = y_prob.shape[1]
        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(n), y_true] = 1.0
        return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1) / k))

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        pred = y_prob.argmax(axis=1)
        conf = y_prob.max(axis=1)
        correct = (pred == y_true).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if not np.any(mask):
                continue
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
        return float(ece)

    @staticmethod
    def expected_calibration_error_per_class(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[int, float]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        k = y_prob.shape[1]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        out: Dict[int, float] = {}
        for c in range(k):
            p_c = y_prob[:, c]
            is_c = (y_true == c).astype(float)
            ece_c = 0.0
            n = len(y_true)
            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (p_c > lo) & (p_c <= hi) if i > 0 else (p_c >= lo) & (p_c <= hi)
                if not np.any(mask):
                    continue
                acc_bin = is_c[mask].mean()
                conf_bin = p_c[mask].mean()
                ece_c += (mask.sum() / n) * abs(acc_bin - conf_bin)
            out[c] = float(ece_c)
        return out

    @staticmethod
    def bootstrap_ci(
        y_true: np.ndarray,
        *,
        y_pred: np.ndarray | None = None,
        y_prob: np.ndarray | None = None,
        n_boot: int = 1000,
        alpha: float = 0.05,
        stratified: bool = True,
        random_state: int = 42,
        metric: str = "accuracy",
    ) -> Tuple[float, float, float]:
        rng = RandomState(random_state)
        y_true = np.asarray(y_true, dtype=int)
        if metric in ("accuracy", "macro_f1", "critical_sensitivity", "under_triage_rate", "critical_under_triage_rate", "over_triage_rate"):
            assert y_pred is not None, "y_pred required for accuracy/f1/safety metrics"
        if metric in ("nll", "ece"):
            assert y_prob is not None, "y_prob required for nll/ece"

        def _metric(idx: np.ndarray) -> float:
            if metric == "accuracy":
                return float(accuracy_score(y_true[idx], y_pred[idx]))
            if metric == "macro_f1":
                return float(f1_score(y_true[idx], y_pred[idx], average="macro"))
            if metric == "nll":
                return ReliabilityMetrics.negative_log_likelihood(y_true[idx], y_prob[idx])
            if metric == "ece":
                return ReliabilityMetrics.expected_calibration_error(y_true[idx], y_prob[idx])
            if metric in {"critical_sensitivity", "under_triage_rate", "critical_under_triage_rate", "over_triage_rate"}:
                cm = confusion_matrix(y_true[idx], y_pred[idx])
                safety = ClinicalMetrics._calculate_safety_metrics(y_true[idx], y_pred[idx], cm)
                return float(safety[metric])
            raise ValueError(f"Unsupported metric: {metric}")

        n = len(y_true)
        if stratified:
            unique = np.unique(y_true)
            class_indices = {c: np.where(y_true == c)[0] for c in unique}
            class_counts = {c: len(class_indices[c]) for c in unique}
            def sample_idx() -> np.ndarray:
                parts = [rng.choice(class_indices[c], size=class_counts[c], replace=True) for c in unique]
                return np.concatenate(parts)
        else:
            def sample_idx() -> np.ndarray:
                return rng.choice(np.arange(n), size=n, replace=True)

        samples = []
        for _ in range(n_boot):
            idx = sample_idx()
            samples.append(_metric(idx))
        samples = np.array(samples, dtype=float)
        mean = float(samples.mean())
        lower = float(np.quantile(samples, alpha / 2))
        upper = float(np.quantile(samples, 1 - alpha / 2))
        return mean, lower, upper

class PerformanceBenchmark:
    """
    Performance benchmarking for federated learning systems.
    """
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_model_performance(self, model, test_loader, device):
        """
        Benchmark model inference performance.
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            device: Computing device
        
        Returns:
            dict: Performance metrics
        """
        model.eval()
        
        # Measure inference time
        inference_times = []
        total_samples = 0
        
        # Warm-up one pass (excluded from timing)
        with torch.no_grad():
            warmup_done = False
            for numerical_data, boolean_data, temporal_data, targets in test_loader:
                numerical_data = numerical_data.to(device)
                boolean_data = boolean_data.to(device)
                temporal_data = temporal_data.to(device)
                batch_size = numerical_data.size(0)
                if not warmup_done:
                    # Warm-up (no timing)
                    _ = model(numerical_data, boolean_data, temporal_data)
                    warmup_done = True
                    total_samples += batch_size
                    continue
                # Timed pass (model-only)
                if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(numerical_data, boolean_data, temporal_data)
                if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                inference_times.append(batch_time / max(1, batch_size))
                total_samples += batch_size
        
        # Calculate performance metrics
        avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
        throughput = (1.0 / avg_inference_time) if avg_inference_time > 0 else 0.0
        
        # Model size estimation
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'model_size_mb': model_size_mb,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'total_samples_tested': total_samples
        }
    
    def benchmark_federated_round(self, clients, server, test_loader, device):
        """
        Benchmark a complete federated learning round.
        
        Args:
            clients: List of FederatedClient objects
            server: FederatedServer object
            test_loader: Test data loader
            device: Computing device
        
        Returns:
            dict: Round performance metrics
        """
        round_start_time = time.time()
        
        # Client training phase
        client_times = []
        client_updates = []
        
        for client in clients:
            client_start_time = time.time()
            params, num_samples, privacy_metrics = client.train(epochs=1, return_metrics=True)
            client_end_time = time.time()
            
            client_times.append(client_end_time - client_start_time)
            client_updates.append((params, num_samples))
        
        # Server aggregation phase
        aggregation_start_time = time.time()
        aggregated_params = server.aggregate_parameters(client_updates)
        aggregation_end_time = time.time()
        
        # Global model evaluation
        eval_start_time = time.time()
        server.evaluate_global_model(test_loader)
        eval_end_time = time.time()
        
        round_end_time = time.time()
        
        return {
            'total_round_time': round_end_time - round_start_time,
            'avg_client_training_time': np.mean(client_times),
            'aggregation_time': aggregation_end_time - aggregation_start_time,
            'evaluation_time': eval_end_time - eval_start_time,
            'num_clients': len(clients),
            'communication_overhead': len(client_updates) * len(client_updates[0][0]) if client_updates else 0
        }

class ComprehensiveEvaluator:
    """
    Main evaluation framework combining all evaluation components.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.clinical_metrics = ClinicalMetrics()
        self.fairness_evaluator = FairnessEvaluator()
        self.performance_benchmark = PerformanceBenchmark()
        self.evaluation_history = []
    
    def evaluate_federated_system(self, global_model, clients, server, test_loader, 
                                sensitive_data=None, device='cpu'):
        """
        Comprehensive evaluation of the federated learning system.
        
        Args:
            global_model: Trained global model
            clients: List of federated clients
            server: Federated server
            test_loader: Test data loader
            sensitive_data: Dictionary with sensitive attributes for fairness evaluation
            device: Computing device
        
        Returns:
            dict: Comprehensive evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'clinical_metrics': {},
            'fairness_metrics': {},
            'performance_metrics': {},
            'federated_metrics': {},
            'summary': {}
        }
        
        # Get predictions from global model
        y_true, y_pred, y_prob = self._get_model_predictions(global_model, test_loader, device)
        
        # Clinical evaluation
        print("Evaluating clinical metrics...")
        evaluation_results['clinical_metrics'] = self.clinical_metrics.calculate_triage_metrics(
            y_true, y_pred
        )
        
        # Fairness evaluation
        if sensitive_data is not None:
            print("Evaluating fairness metrics...")
            evaluation_results['fairness_metrics'] = self.fairness_evaluator.evaluate_fairness(
                y_true, y_pred, sensitive_data
            )
        
        # Performance benchmarking
        print("Benchmarking performance...")
        evaluation_results['performance_metrics'] = self.performance_benchmark.benchmark_model_performance(
            global_model, test_loader, device
        )
        
        # Federated learning specific metrics
        print("Evaluating federated learning metrics...")
        evaluation_results['federated_metrics'] = self.performance_benchmark.benchmark_federated_round(
            clients, server, test_loader, device
        )
        
        # Generate summary
        evaluation_results['summary'] = self._generate_evaluation_summary(evaluation_results)
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        
        print("Comprehensive evaluation completed.")
        return evaluation_results
    
    def _get_model_predictions(self, model, test_loader, device):
        """Get model predictions on test data."""
        model.eval()
        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in test_loader:
                numerical_data = numerical_data.to(device)
                boolean_data = boolean_data.to(device)
                temporal_data = temporal_data.to(device)
                
                outputs = model(numerical_data, boolean_data, temporal_data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                y_true_list.extend(targets.cpu().numpy())
                y_pred_list.extend(predictions.cpu().numpy())
                y_prob_list.extend(probabilities.cpu().numpy())
        
        return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)
    
    def _generate_evaluation_summary(self, results):
        """Generate a summary of evaluation results."""
        summary = {
            'overall_performance': 'Good',
            'key_findings': [],
            'recommendations': [],
            'risk_assessment': 'Low'
        }
        
        # Clinical performance assessment
        clinical = results['clinical_metrics']
        overall_accuracy = clinical.get('overall_accuracy', 0)
        critical_sensitivity = clinical.get('clinical_safety', {}).get('critical_sensitivity', 0)
        
        if overall_accuracy > 0.85 and critical_sensitivity > 0.90:
            summary['overall_performance'] = 'Excellent'
        elif overall_accuracy > 0.75 and critical_sensitivity > 0.80:
            summary['overall_performance'] = 'Good'
        elif overall_accuracy > 0.65:
            summary['overall_performance'] = 'Acceptable'
        else:
            summary['overall_performance'] = 'Poor'
        
        # Key findings
        summary['key_findings'].append(f"Overall accuracy: {overall_accuracy:.3f}")
        summary['key_findings'].append(f"Critical case sensitivity: {critical_sensitivity:.3f}")
        
        # Fairness assessment
        if 'fairness_metrics' in results and results['fairness_metrics']:
            fairness_score = results['fairness_metrics'].get('overall_fairness_score', 1.0)
            summary['key_findings'].append(f"Fairness score: {fairness_score:.3f}")
            
            if fairness_score < 0.8:
                summary['recommendations'].append("Address fairness concerns across demographic groups")
                summary['risk_assessment'] = 'Medium'
        
        # Performance assessment
        if 'performance_metrics' in results:
            inference_time = results['performance_metrics'].get('avg_inference_time_ms', 0)
            summary['key_findings'].append(f"Average inference time: {inference_time:.2f}ms")
            
            if inference_time > 100:  # 100ms threshold for real-time triage
                summary['recommendations'].append("Optimize model for faster inference")
        
        # Safety assessment
        under_triage_rate = clinical.get('clinical_safety', {}).get('under_triage_rate', 0)
        if under_triage_rate > 0.1:  # 10% threshold
            summary['risk_assessment'] = 'High'
            summary['recommendations'].append("Critical: Reduce under-triage rate for patient safety")
        
        return summary
    
    def save_evaluation_report(self, results, filepath):
        """Save evaluation results to file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Evaluation report saved to {filepath}")
    
    def generate_comparison_report(self, baseline_results=None):
        """Generate a comparison report against baseline or previous evaluations."""
        if not self.evaluation_history:
            return "No evaluation history available for comparison."
        
        if baseline_results is None and len(self.evaluation_history) > 1:
            baseline_results = self.evaluation_history[-2]  # Previous evaluation
        
        if baseline_results is None:
            return "No baseline results available for comparison."
        
        current_results = self.evaluation_history[-1]
        
        comparison = {
            'accuracy_change': (
                current_results['clinical_metrics']['overall_accuracy'] - 
                baseline_results['clinical_metrics']['overall_accuracy']
            ),
            'fairness_change': (
                current_results.get('fairness_metrics', {}).get('overall_fairness_score', 1.0) - 
                baseline_results.get('fairness_metrics', {}).get('overall_fairness_score', 1.0)
            ),
            'performance_change': (
                baseline_results['performance_metrics']['avg_inference_time_ms'] - 
                current_results['performance_metrics']['avg_inference_time_ms']
            )  # Negative change means improvement (faster)
        }
        
        return comparison
