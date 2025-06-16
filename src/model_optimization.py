import torch
import torch.nn as nn
import torch.nn.functional as F # Added import for F
import torch.nn.utils.prune as prune
import torch.quantization

def apply_quantization(model, backend='auto', calibration_data=None):
    """
    Applies post-training static quantization to the model with improved backend detection.
    
    Args:
        model (nn.Module): The PyTorch model to quantize.
        backend (str): Quantization backend ('auto', 'qnnpack', 'fbgemm', 'x86').
                       'auto' will automatically detect the best backend.
        calibration_data (tuple): Optional calibration data (numerical, boolean, temporal).
    
    Returns:
        nn.Module: The quantized model, or original model if quantization fails.
    """
    try:
        import copy
        
        # Create a copy to avoid modifying the original model
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        model_copy.to('cpu')
        
        # Auto-detect best backend
        if backend == 'auto':
            backend = _detect_quantization_backend()
        
        print(f"Applying quantization with backend: {backend}")
        
        # Check if backend is supported
        if not torch.backends.quantized.is_available():
            print("Warning: Quantization backend not available. Returning original model.")
            return model
        
        # Set quantization configuration
        try:
            model_copy.qconfig = torch.quantization.get_default_qconfig(backend)
        except Exception as e:
            print(f"Warning: Failed to set qconfig for backend {backend}: {e}")
            # Fallback to default qconfig
            model_copy.qconfig = torch.quantization.default_qconfig
        
        # Prepare the model for quantization
        torch.quantization.prepare(model_copy, inplace=True)
        
        # Calibrate the model
        print("Calibrating model for quantization...")
        if calibration_data is not None:
            numerical_data, boolean_data, temporal_data = calibration_data
        else:
            # Generate dummy calibration data
            num_samples = 50  # Increased for better calibration
            num_numerical = 7
            num_boolean = 268
            num_temporal = 3
            numerical_data = torch.randn(num_samples, num_numerical)
            boolean_data = torch.randint(0, 2, (num_samples, num_boolean)).float()
            temporal_data = torch.randn(num_samples, num_temporal)
        
        # Run calibration
        with torch.no_grad():
            for i in range(0, len(numerical_data), 10):  # Process in batches
                batch_num = numerical_data[i:i+10]
                batch_bool = boolean_data[i:i+10]
                batch_temp = temporal_data[i:i+10]
                try:
                    _ = model_copy(batch_num, batch_bool, batch_temp)
                except Exception as e:
                    print(f"Warning: Calibration batch {i//10} failed: {e}")
                    continue
        
        print("Calibration complete.")
        
        # Convert the model to quantized version
        quantized_model = torch.quantization.convert(model_copy, inplace=False)
        
        # Verify the quantized model works
        try:
            with torch.no_grad():
                test_num = torch.randn(1, 7)
                test_bool = torch.randint(0, 2, (1, 268)).float()
                test_temp = torch.randn(1, 3)
                _ = quantized_model(test_num, test_bool, test_temp)
            print("Quantization successful!")
            return quantized_model
        except Exception as e:
            print(f"Warning: Quantized model verification failed: {e}")
            return model
            
    except Exception as e:
        print(f"Error during quantization: {e}")
        print("Returning original model.")
        return model

def _detect_quantization_backend():
    """
    Automatically detect the best quantization backend for the current system.
    
    Returns:
        str: The best available backend.
    """
    import platform
    
    # Check available backends
    available_backends = []
    
    try:
        if torch.backends.quantized.engine == 'fbgemm' or hasattr(torch.backends.quantized, 'fbgemm'):
            available_backends.append('fbgemm')
    except:
        pass
    
    try:
        if torch.backends.quantized.engine == 'qnnpack' or hasattr(torch.backends.quantized, 'qnnpack'):
            available_backends.append('qnnpack')
    except:
        pass
    
    # Platform-specific backend selection
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if 'arm' in machine or 'aarch64' in machine:
        # ARM processors (including Apple Silicon, Raspberry Pi)
        if 'qnnpack' in available_backends:
            return 'qnnpack'
    elif 'x86' in machine or 'amd64' in machine:
        # x86/x64 processors
        if 'fbgemm' in available_backends:
            return 'fbgemm'
        elif 'qnnpack' in available_backends:
            return 'qnnpack'
    
    # Fallback
    if available_backends:
        return available_backends[0]
    else:
        print("Warning: No quantization backends detected. Using default.")
        return 'fbgemm'  # Default fallback

def apply_pruning(model, amount=0.5):
    """
    Applies global unstructured pruning to the model's linear layers.
    
    Args:
        model (nn.Module): The PyTorch model to prune.
        amount (float): Percentage of connections to prune (e.g., 0.5 for 50%).
    
    Returns:
        nn.Module: The pruned model.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            # Optionally prune bias as well: parameters_to_prune.append((module, 'bias'))
            
    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        # Remove pruning re-parametrization and make the pruned weights permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        print(f"Pruning applied: {amount*100}% of connections removed from linear layers.")
    else:
        print("No linear layers found for pruning.")
        
    return model

def apply_knowledge_distillation(teacher_model, student_model, train_loader, optimizer, criterion, temperature=2.0, alpha=0.5, epochs=10):
    """
    Appains knowledge distillation to train a smaller student model using a larger teacher model.
    
    Args:
        teacher_model (nn.Module): The larger, pre-trained teacher model.
        student_model (nn.Module): The smaller student model to train.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the student model.
        criterion (nn.Module): Loss function for hard targets (e.g., CrossEntropyLoss).
        temperature (float): Temperature for softening softmax probabilities.
        alpha (float): Weight for the soft target loss (1 - alpha for hard target loss).
        epochs (int): Number of epochs for distillation training.
    
    Returns:
        nn.Module: The trained student model.
    """
    teacher_model.eval() # Set teacher to evaluation mode
    student_model.train() # Set student to training mode
    
    print(f"Starting knowledge distillation for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (numerical_data, boolean_data, temporal_data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Teacher model outputs (soft targets)
            with torch.no_grad():
                teacher_logits = teacher_model(numerical_data, boolean_data, temporal_data)
            
            # Student model outputs
            student_logits = student_model(numerical_data, boolean_data, temporal_data)
            
            # Hard target loss (standard cross-entropy)
            hard_loss = criterion(student_logits, targets)
            
            # Soft target loss (KL divergence between softened teacher and student outputs)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature * temperature) # Scale by T^2 as per Hinton et al.
            
            # Combined loss
            loss = alpha * soft_loss + (1. - alpha) * hard_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    print("Knowledge distillation complete.")
    return student_model

if __name__ == "__main__":
    from model_architecture import TriageModel
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data for demonstration (same as in model_architecture.py)
    num_samples = 100
    num_numerical = 7
    num_boolean = 268
    num_temporal = 3
    num_classes = 3

    dummy_numerical_data = torch.randn(num_samples, num_numerical)
    dummy_boolean_data = torch.randint(0, 2, (num_samples, num_boolean)).float()
    dummy_temporal_data = torch.randn(num_samples, num_temporal)
    dummy_targets = torch.randint(0, num_classes, (num_samples,))

    # Create a dummy dataset and DataLoader for distillation example
    dummy_dataset = TensorDataset(dummy_numerical_data, dummy_boolean_data, dummy_temporal_data, dummy_targets)
    dummy_train_loader = DataLoader(dummy_dataset, batch_size=16)

    # Instantiate a teacher model (larger/more complex)
    teacher_model = TriageModel(num_numerical, num_boolean, num_temporal, num_classes)
    # For a real scenario, teacher_model would be pre-trained and performant.

    # Instantiate a student model (smaller/simpler, e.g., fewer hidden units)
    # For this example, let's just use the same TriageModel but imagine it's smaller
    # In a real case, you'd define a simpler architecture for the student.
    student_model = TriageModel(num_numerical, num_boolean, num_temporal, num_classes) 

    # --- Demonstrate Quantization ---
    print("\n--- Demonstrating Quantization ---")
    quantized_model = apply_quantization(teacher_model) # Using teacher_model for demo
    print("Quantized model size (example, actual size reduction depends on saving format):")
    # To truly see size reduction, you'd save and load the model
    # torch.save(quantized_model.state_dict(), "quantized_model.pth")
    # print(f"Quantized model saved to quantized_model.pth")

    # --- Demonstrate Pruning ---
    print("\n--- Demonstrating Pruning ---")
    pruned_model = apply_pruning(student_model.copy(), amount=0.7) # Prune 70% of student model
    # Note: .copy() is used to avoid modifying student_model in-place for distillation demo later
    print("Pruned model architecture:\n", pruned_model)

    # --- Demonstrate Knowledge Distillation ---
    print("\n--- Demonstrating Knowledge Distillation ---")
    # Reset student model for distillation demo
    student_model_for_distillation = TriageModel(num_numerical, num_boolean, num_temporal, num_classes)
    optimizer = optim.Adam(student_model_for_distillation.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trained_student_model = apply_knowledge_distillation(
        teacher_model, 
        student_model_for_distillation, 
        dummy_train_loader, 
        optimizer, 
        criterion
    )
    print("Distilled student model trained.")