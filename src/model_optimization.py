import torch
import torch.nn as nn
import torch.nn.functional as F # Added import for F
import torch.nn.utils.prune as prune
import torch.quantization

def apply_quantization(model, backend='qnnpack'): # Reverted to 'qnnpack' for ARM CPUs
    """
    Applies post-training static quantization to the model.
    
    Args:
        model (nn.Module): The PyTorch model to quantize.
        backend (str): Quantization backend (e.g., 'qnnpack', 'fbgemm', 'x86').
                       'qnnpack' is generally recommended for ARM CPUs (like Raspberry Pi).
    
    Returns:
        nn.Module: The quantized model.
    """
    model.eval() # Set model to evaluation mode
    model.to('cpu') # Ensure model is on CPU for quantization
    
    # Fuse modules where applicable (e.g., Conv + ReLU, Linear + ReLU)
    # This is crucial for effective quantization
    # For our current model, we have BatchNorm after Linear, so fusion might be less direct
    # but we can still prepare for it.
    
    # Example fusion for Linear + BatchNorm + ReLU (if applicable)
    # For simplicity, we'll just prepare for quantization directly on the model
    
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare the model for quantization. This inserts observers and fake_quant modules.
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate the model (run inference on a representative dataset)
    # In a real scenario, you would pass a calibration dataset here.
    # For demonstration, we'll just call a dummy forward pass.
    # You need to replace this with actual data calibration.
    print("Calibrating model for quantization (using dummy data)...")
    # Assuming dummy data has the same structure as model_architecture.py example
    num_samples = 10
    num_numerical = 7
    num_boolean = 268 # Adjusted based on main.py output
    num_temporal = 3
    dummy_numerical_data = torch.randn(num_samples, num_numerical)
    dummy_boolean_data = torch.randint(0, 2, (num_samples, num_boolean)).float()
    dummy_temporal_data = torch.randn(num_samples, num_temporal)
    
    with torch.no_grad():
        model(dummy_numerical_data, dummy_boolean_data, dummy_temporal_data)
    print("Calibration complete.")
    
    # Convert the model to a quantized version
    torch.quantization.convert(model, inplace=True)
    
    return model

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