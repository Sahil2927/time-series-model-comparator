import torch
import torch.nn as nn

class FairLSTM(nn.Module):
    """
    A Fair LSTM wrapper designed to provide a baseline for time-series forecasting.
    It maps input sequences to a hidden representation and projects the final state 
    across the entire prediction horizon.
    """
    def __init__(self, num_features, d_model=512, num_layers=2, pred_len=96):
        super(FairLSTM, self).__init__()
        
        self.pred_len = pred_len
        self.num_features = num_features
        
        # 1. Feature Projection: Projects input features to the internal embedding size (d_model)
        self.feature_projection = nn.Linear(num_features, d_model)
        
        # 2. LSTM Mechanism: Processes the sequence
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 3. Output Projection: Projects the final hidden state to the future target shape
        # We output pred_len * num_features to get the full forecast window
        self.output_projection = nn.Linear(d_model, pred_len * num_features)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size = x.shape[0]
        
        # Project features
        x = self.feature_projection(x)
        
        # Run LSTM
        # out: [batch, seq_len, d_model], (hn, cn): [num_layers, batch, d_model]
        out, (hn, cn) = self.lstm(x)
        
        # Take the last hidden state of the top LSTM layer
        last_hidden = hn[-1] # shape: [batch, d_model]
        
        # Predict the future window
        predictions = self.output_projection(last_hidden)
        
        # Reshape to match target: [batch, pred_len, features]
        return predictions.view(batch_size, self.pred_len, self.num_features)
# Test block
if __name__ == "__main__":
    # Mock data: [batch=32, seq_len=96, features=7]
    test_input = torch.randn(32, 96, 7)
    
    # Initialize model
    model = FairLSTM(num_features=7, d_model=64, num_layers=2, pred_len=96)
    
    # Forward pass
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [32, 96, 7]
    print("LSTM forward pass successful!")