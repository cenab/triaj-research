import torch
import torch.nn as nn
import torch.nn.functional as F

class TriageModel(nn.Module):
    def __init__(self, num_numerical_features, num_boolean_features, num_temporal_features, num_classes):
        super(TriageModel, self).__init__()
        
        # Numerical features sub-network
        self.numerical_fc1 = nn.Linear(num_numerical_features, 64)
        self.numerical_bn1 = nn.BatchNorm1d(64)
        self.numerical_fc2 = nn.Linear(64, 32)
        self.numerical_bn2 = nn.BatchNorm1d(32)

        # Boolean features sub-network
        self.boolean_fc1 = nn.Linear(num_boolean_features, 128)
        self.boolean_bn1 = nn.BatchNorm1d(128)
        self.boolean_fc2 = nn.Linear(128, 64)
        self.boolean_bn2 = nn.BatchNorm1d(64)

        # Temporal features sub-network (simple dense for now, can be LSTM/GRU for sequences)
        self.temporal_fc1 = nn.Linear(num_temporal_features, 32)
        self.temporal_bn1 = nn.BatchNorm1d(32)

        # Combined features fusion
        # Sum of output dimensions from sub-networks
        combined_features_dim = 32 + 64 + 32 # numerical_fc2 + boolean_fc2 + temporal_fc1
        self.fusion_fc1 = nn.Linear(combined_features_dim, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_bn2 = nn.BatchNorm1d(64)

        # Output layer
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, numerical_data, boolean_data, temporal_data):
        # Numerical pathway
        numerical_out = F.relu(self.numerical_bn1(self.numerical_fc1(numerical_data)))
        numerical_out = F.relu(self.numerical_bn2(self.numerical_fc2(numerical_out)))

        # Boolean pathway
        boolean_out = F.relu(self.boolean_bn1(self.boolean_fc1(boolean_data)))
        boolean_out = F.relu(self.boolean_bn2(self.boolean_fc2(boolean_out)))

        # Temporal pathway
        temporal_out = F.relu(self.temporal_bn1(self.temporal_fc1(temporal_data)))

        # Concatenate all outputs
        combined_out = torch.cat((numerical_out, boolean_out, temporal_out), dim=1)

        # Fusion layers
        fusion_out = F.relu(self.fusion_bn1(self.fusion_fc1(combined_out)))
        fusion_out = F.relu(self.fusion_bn2(self.fusion_fc2(fusion_out)))

        # Final output
        logits = self.output_layer(fusion_out)
        return logits

if __name__ == "__main__":
    # Example usage:
    # Assuming feature_engineered_df is available from feature_engineering.py
    # and target variable is 'doğru triyaj_encoded'

    # Dummy data for demonstration
    num_samples = 10
    num_numerical = 7 # yaş, sistolik kb, diastolik kb, solunum sayısı, nabız, ateş, saturasyon
    num_boolean = 269 # Example from feature_engineering.py output (280 total - 7 numerical - 3 temporal - 1 target)
    num_temporal = 3 # hour_of_day, day_of_week, month (excluding year for simplicity in model input)
    num_classes = 3 # Kırmızı, Sarı, Yeşil

    # Create dummy tensors
    dummy_numerical_data = torch.randn(num_samples, num_numerical)
    dummy_boolean_data = torch.randint(0, 2, (num_samples, num_boolean)).float()
    dummy_temporal_data = torch.randn(num_samples, num_temporal)

    # Instantiate the model
    model = TriageModel(num_numerical, num_boolean, num_temporal, num_classes)

    # Forward pass
    outputs = model(dummy_numerical_data, dummy_boolean_data, dummy_temporal_data)
    print("Model output shape:", outputs.shape)
    print("Model architecture:\n", model)