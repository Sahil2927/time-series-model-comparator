import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FairTransformer(nn.Module):
    """
    Standard Transformer architecture adapted for LTSF (Long-Term Time-Series Forecasting).
    """
    def __init__(self, num_features, d_model=512, nhead=8, num_layers=2, pred_len=96):
        super(FairTransformer, self).__init__()
        
        self.pred_len = pred_len
        self.num_features = num_features
        
        # 1. Feature Projection
        self.feature_projection = nn.Linear(num_features, d_model)
        # ADDED LayerNorm for stability
        self.norm_emb = nn.LayerNorm(d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # ADDED LayerNorm after encoder
        self.norm_enc = nn.LayerNorm(d_model)
        
        # 4. Output Projection
        self.output_projection = nn.Linear(d_model, pred_len * num_features)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size = x.shape[0]
        
        # Project features to model dimension
        x = self.feature_projection(x)
        x = self.norm_emb(x) # Normalize projection
        
        # Add positional info
        x = self.pos_encoder(x)
        
        # Process through Transformer
        x = self.transformer_encoder(x)
        x = self.norm_enc(x) # Normalize output
        
        # Flatten last hidden representation or take last index
        # For fair comparison with LSTM baseline, we take the last hidden state
        last_hidden = x[:, -1, :] 
        
        # Project to prediction horizon
        predictions = self.output_projection(last_hidden)
        
        # Reshape to [batch, pred_len, features]
        return predictions.view(batch_size, self.pred_len, self.num_features)

if __name__ == "__main__":
    # Mock data: [batch=32, seq_len=96, features=7]
    test_input = torch.randn(32, 96, 7)
    
    # Initialize model
    model = FairTransformer(num_features=7, d_model=64, nhead=4, num_layers=2, pred_len=96)
    
    # Forward pass
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [32, 96, 7]
    print("Transformer forward pass successful!")