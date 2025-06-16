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
            y_true (array): True triage labels (0=Green, 1=Yellow, 2=Red)
            y_pred (array): Predicted triage labels
            class_names (list): Names of triage classes
        
        Returns:
            dict: Comprehensive triage metrics
        """
        if class_names is None:
            class_names = ['Green', 'Yellow', 'Red']
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Triage-specific metrics
        metrics = {
            'overall_accuracy': accuracy,
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
        
        # Critical under-triage: Missing Red (critical) cases
        critical_under_triage = 0
        red_cases = np.sum(y_true == 2)
        if red_cases > 0:
            # Cases where true=Red but predicted=Green or Yellow
            critical_under_triage = np.sum((y_true == 2) & (y_pred < 2))
        
        return {
            'under_triage_rate': under_triage / total_samples,
            'over_triage_rate': over_triage / total_samples,
            'critical_under_triage_rate': critical_under_triage / red_cases if red_cases > 0 else 0.0,
            'critical_sensitivity': np.sum((y_true == 2) & (y_pred == 2)) / red_cases if red_cases > 0 else 0.0
        }

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
                'positive_rate': np.mean(group_y_pred > 0),  # Rate of non-Green predictions
                'critical_rate': np.mean(group_y_pred == 2)  # Rate of Red predictions
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
        
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in test_loader:
                numerical_data = numerical_data.to(device)
                boolean_data = boolean_data.to(device)
                temporal_data = temporal_data.to(device)
                
                batch_size = numerical_data.size(0)
                
                # Measure single batch inference time
                start_time = time.time()
                outputs = model(numerical_data, boolean_data, temporal_data)
                end_time = time.time()
                
                batch_time = end_time - start_time
                inference_times.append(batch_time / batch_size)  # Per-sample time
                total_samples += batch_size
        
        # Calculate performance metrics
        avg_inference_time = np.mean(inference_times)
        throughput = 1.0 / avg_inference_time  # Samples per second
        
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
            params, num_samples, privacy_metrics = client.train(epochs=1)
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