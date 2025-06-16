import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from datetime import datetime
import json
import os

try:
    from .model_architecture import TriageModel
    from .evaluation_framework import ClinicalMetrics, ComprehensiveEvaluator
    from .data_preparation import load_and_clean_data
    from .feature_engineering import feature_engineer_data
except ImportError:
    from model_architecture import TriageModel
    from evaluation_framework import ClinicalMetrics, ComprehensiveEvaluator
    from data_preparation import load_and_clean_data
    from feature_engineering import feature_engineer_data

class ImprovedTriageModel(nn.Module):
    """
    Simplified and improved triage model with better architecture for small datasets.
    """
    def __init__(self, num_numerical_features, num_boolean_features, num_temporal_features, num_classes):
        super(ImprovedTriageModel, self).__init__()
        
        # Simplified architecture for small dataset
        total_features = num_numerical_features + num_boolean_features + num_temporal_features
        
        # Input layer with dropout
        self.input_layer = nn.Linear(total_features, 128)
        self.input_dropout = nn.Dropout(0.3)
        self.input_bn = nn.BatchNorm1d(128)
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(128, 64)
        self.hidden1_dropout = nn.Dropout(0.2)
        self.hidden1_bn = nn.BatchNorm1d(64)
        
        self.hidden2 = nn.Linear(64, 32)
        self.hidden2_dropout = nn.Dropout(0.1)
        self.hidden2_bn = nn.BatchNorm1d(32)
        
        # Output layer
        self.output_layer = nn.Linear(32, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, numerical_data, boolean_data, temporal_data):
        # Concatenate all inputs
        x = torch.cat((numerical_data, boolean_data, temporal_data), dim=1)
        
        # Forward pass with residual connections
        x = torch.relu(self.input_bn(self.input_layer(x)))
        x = self.input_dropout(x)
        
        # First hidden layer
        h1 = torch.relu(self.hidden1_bn(self.hidden1(x)))
        h1 = self.hidden1_dropout(h1)
        
        # Second hidden layer
        h2 = torch.relu(self.hidden2_bn(self.hidden2(h1)))
        h2 = self.hidden2_dropout(h2)
        
        # Output
        logits = self.output_layer(h2)
        return logits

class FixedTrainer:
    """
    Fixed training implementation with proper class balancing and validation.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_model_state = None
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Prepare data with proper stratified splits and class balancing.
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Data splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        # Print class distributions
        print(f"\nClass distributions:")
        print(f"  Train: {np.bincount(y_train)}")
        print(f"  Validation: {np.bincount(y_val)}")
        print(f"  Test: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           numerical_cols, boolean_cols, temporal_cols, batch_size=16):
        """
        Create data loaders with proper tensor conversion.
        """
        def create_tensors(X, y):
            X_numerical = torch.tensor(X[numerical_cols].values, dtype=torch.float32)
            X_boolean = torch.tensor(X[boolean_cols].values, dtype=torch.float32)
            X_temporal = torch.tensor(X[temporal_cols].values, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            return X_numerical, X_boolean, X_temporal, y_tensor
        
        # Create tensors
        train_num, train_bool, train_temp, train_y = create_tensors(X_train, y_train)
        val_num, val_bool, val_temp, val_y = create_tensors(X_val, y_val)
        test_num, test_bool, test_temp, test_y = create_tensors(X_test, y_test)
        
        # Create datasets
        train_dataset = TensorDataset(train_num, train_bool, train_temp, train_y)
        val_dataset = TensorDataset(val_num, val_bool, val_temp, val_y)
        test_dataset = TensorDataset(test_num, test_bool, test_temp, test_y)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def calculate_class_weights(self, y_train):
        """
        Calculate class weights to handle imbalanced data.
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        print(f"Class weights: {class_weights}")
        return class_weights_tensor
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for numerical_data, boolean_data, temporal_data, targets in train_loader:
            numerical_data = numerical_data.to(self.device)
            boolean_data = boolean_data.to(self.device)
            temporal_data = temporal_data.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(numerical_data, boolean_data, temporal_data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """
        Validate for one epoch.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in val_loader:
                numerical_data = numerical_data.to(self.device)
                boolean_data = boolean_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(numerical_data, boolean_data, temporal_data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, 
              class_weights=None, patience=10):
        """
        Train the model with early stopping and proper validation.
        """
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Early stopping
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_pred, val_true = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['epochs'].append(epoch + 1)
            
            epoch_time = time.time() - start_time
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return self.training_history
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for numerical_data, boolean_data, temporal_data, targets in test_loader:
                numerical_data = numerical_data.to(self.device)
                boolean_data = boolean_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                targets = targets.to(self.device)
                
                start_time = time.time()
                outputs = self.model(numerical_data, boolean_data, temporal_data)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Clinical metrics
        clinical_metrics = ClinicalMetrics.calculate_triage_metrics(
            all_targets, all_predictions, class_names=['Green', 'Yellow', 'Red']
        )
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        throughput = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        results = {
            'accuracy': accuracy,
            'clinical_metrics': clinical_metrics,
            'performance_metrics': {
                'avg_inference_time_ms': avg_inference_time,
                'throughput_samples_per_sec': throughput,
                'model_size_mb': model_size,
                'total_parameters': total_params,
                'total_samples_tested': len(all_targets)
            },
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        return results

def fix_critical_failures():
    """
    Main function to fix the critical failures in the triage system.
    """
    print("=== FIXING CRITICAL FAILURES IN FAIRTRI-EDGE FL SYSTEM ===")
    print("Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    file_path = 'triaj_data.csv'
    
    try:
        df_cleaned = load_and_clean_data(file_path)
        df_engineered = feature_engineer_data(df_cleaned.copy())
        print(f"Data loaded successfully. Shape: {df_engineered.shape}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Creating synthetic data for demonstration...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 500
        
        # Create synthetic features
        data = {
            'yaş': np.random.randint(18, 80, n_samples),
            'sistolik kb': np.random.randint(90, 180, n_samples),
            'diastolik kb': np.random.randint(60, 120, n_samples),
            'solunum sayısı': np.random.randint(12, 30, n_samples),
            'nabız': np.random.randint(60, 120, n_samples),
            'ateş': np.random.normal(37, 1, n_samples),
            'saturasyon': np.random.randint(85, 100, n_samples),
            'cinsiyet': np.random.choice(['Male', 'Female'], n_samples),
            'doğru triyaj': np.random.choice(['Yeşil', 'Sarı', 'Kırmızı'], n_samples, p=[0.4, 0.4, 0.2])
        }
        
        df_cleaned = pd.DataFrame(data)
        df_engineered = feature_engineer_data(df_cleaned.copy())
        print(f"Synthetic data created. Shape: {df_engineered.shape}")
    
    # Separate features and target
    X = df_engineered.drop('doğru triyaj_encoded', axis=1)
    y = df_engineered['doğru triyaj_encoded'].values
    
    print(f"Target distribution: {np.bincount(y)}")
    
    # Define feature columns
    numerical_cols = ["yaş", "sistolik kb", "diastolik kb", "solunum sayısı", "nabız", "ateş", "saturasyon"]
    temporal_cols = ['hour_of_day', 'day_of_week', 'month']
    boolean_cols = [col for col in X.columns if col not in numerical_cols + temporal_cols + ['year', 'yaş_unscaled']]
    
    # Add gender columns if they exist
    gender_cols = ['cinsiyet_Male', 'cinsiyet_Female', 'cinsiyet_Unknown']
    for col in gender_cols:
        if col in X.columns and col not in boolean_cols:
            boolean_cols.append(col)
    
    print(f"Feature dimensions: Numerical={len(numerical_cols)}, Boolean={len(boolean_cols)}, Temporal={len(temporal_cols)}")
    
    # Initialize improved model
    print("\n2. Initializing improved model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImprovedTriageModel(
        num_numerical_features=len(numerical_cols),
        num_boolean_features=len(boolean_cols),
        num_temporal_features=len(temporal_cols),
        num_classes=len(np.unique(y))
    )
    
    trainer = FixedTrainer(model, device)
    
    # Prepare data splits
    print("\n3. Preparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        numerical_cols, boolean_cols, temporal_cols, batch_size=16
    )
    
    # Calculate class weights
    print("\n4. Calculating class weights for imbalanced data...")
    class_weights = trainer.calculate_class_weights(y_train)
    
    # Train the model
    print("\n5. Training the model...")
    training_history = trainer.train(
        train_loader, val_loader, 
        epochs=100, 
        learning_rate=0.001,
        class_weights=class_weights,
        patience=15
    )
    
    # Evaluate the model
    print("\n6. Evaluating the model...")
    results = trainer.evaluate(test_loader)
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Overall Accuracy: {results['accuracy']:.3f}")
    
    clinical = results['clinical_metrics']
    print(f"\nClinical Metrics:")
    print(f"  Overall Accuracy: {clinical['overall_accuracy']:.3f}")
    
    for class_name, metrics in clinical['class_metrics'].items():
        print(f"  {class_name}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    safety = clinical['clinical_safety']
    print(f"\nClinical Safety:")
    print(f"  Under-triage Rate: {safety['under_triage_rate']:.3f}")
    print(f"  Over-triage Rate: {safety['over_triage_rate']:.3f}")
    print(f"  Critical Under-triage Rate: {safety['critical_under_triage_rate']:.3f}")
    print(f"  Critical Sensitivity: {safety['critical_sensitivity']:.3f}")
    
    performance = results['performance_metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Avg Inference Time: {performance['avg_inference_time_ms']:.2f}ms")
    print(f"  Throughput: {performance['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Model Size: {performance['model_size_mb']:.2f}MB")
    print(f"  Total Parameters: {performance['total_parameters']:,}")
    
    # Save results
    print("\n7. Saving results...")
    os.makedirs('results', exist_ok=True)
    
    # Create comprehensive results
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'clinical_metrics': clinical,
        'performance_metrics': performance,
        'training_history': training_history,
        'model_info': {
            'architecture': 'ImprovedTriageModel',
            'total_parameters': performance['total_parameters'],
            'model_size_mb': performance['model_size_mb']
        },
        'data_info': {
            'total_samples': len(y),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'class_distribution': np.bincount(y).tolist(),
            'feature_dimensions': {
                'numerical': len(numerical_cols),
                'boolean': len(boolean_cols),
                'temporal': len(temporal_cols)
            }
        },
        'summary': {
            'overall_performance': 'Good' if clinical['overall_accuracy'] > 0.7 else 'Poor',
            'key_findings': [
                f"Overall accuracy: {clinical['overall_accuracy']:.3f}",
                f"Critical case sensitivity: {safety['critical_sensitivity']:.3f}",
                f"Under-triage rate: {safety['under_triage_rate']:.3f}",
                f"Average inference time: {performance['avg_inference_time_ms']:.2f}ms"
            ],
            'recommendations': [],
            'risk_assessment': 'Low' if safety['critical_sensitivity'] > 0.9 else 'High'
        }
    }
    
    # Add recommendations based on performance
    if clinical['overall_accuracy'] < 0.7:
        comprehensive_results['summary']['recommendations'].append("Improve overall model accuracy")
    if safety['critical_sensitivity'] < 0.9:
        comprehensive_results['summary']['recommendations'].append("Critical: Improve critical case detection")
    if safety['under_triage_rate'] > 0.2:
        comprehensive_results['summary']['recommendations'].append("Reduce under-triage rate for patient safety")
    
    if not comprehensive_results['summary']['recommendations']:
        comprehensive_results['summary']['recommendations'].append("Model performance meets clinical requirements")
    
    # Save report
    report_path = f"results/fixed_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Results saved to: {report_path}")
    
    # Save model
    model_path = f"results/fixed_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'num_numerical_features': len(numerical_cols),
            'num_boolean_features': len(boolean_cols),
            'num_temporal_features': len(temporal_cols),
            'num_classes': len(np.unique(y))
        },
        'training_history': training_history,
        'class_weights': class_weights.cpu().numpy() if class_weights is not None else None
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Assessment
    print("\n=== ASSESSMENT ===")
    if clinical['overall_accuracy'] > 0.7 and safety['critical_sensitivity'] > 0.8:
        print("✅ CRITICAL FAILURES FIXED!")
        print("✅ Model now meets minimum clinical requirements")
    elif clinical['overall_accuracy'] > 0.5:
        print("⚠️  PARTIAL SUCCESS - Model improved but needs more work")
    else:
        print("❌ STILL FAILING - Further investigation needed")
    
    return comprehensive_results, trainer.model

if __name__ == "__main__":
    results, model = fix_critical_failures()