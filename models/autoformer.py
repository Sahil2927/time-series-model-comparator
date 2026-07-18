import torch
import torch.nn as nn
import torch.fft

# STREAMING_CHUNK:Defining Series Decomposition helper
class SeriesDecomposition(nn.Module):
    """
    Decomposes the time-series into seasonal and trend components using moving averages.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        # Padding is kernel_size // 2 to keep seq_len consistent
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # Transpose for pooling: [batch, features, seq_len]
        x = x.transpose(1, 2)
        trend = self.avg(x)
        seasonal = x - trend
        # Transpose back: [batch, seq_len, features]
        return seasonal.transpose(1, 2), trend.transpose(1, 2)

# STREAMING_CHUNK:Defining Frequency-Domain AutoCorrelation
class AutoCorrelation(nn.Module):
    """
    Auto-Correlation using FFT to capture periodicity efficiently.
    Fixes dimension mismatch by staying in frequency domain.
    """
    def __init__(self, d_model):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model

    def forward(self, q, k, v):
        # Transpose to [batch, features, seq_len]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate correlations in frequency domain
        # Output shape: [batch, features, seq_len // 2 + 1]
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        v_fft = torch.fft.rfft(v, dim=-1)
        
        # Calculate cross-correlation in frequency domain
        # Multiplying complex conjugates (Frequency Attention)
        attn_weights = q_fft * torch.conj(k_fft)
        
        # Apply softmax to weights in frequency domain to stabilize
        # We use absolute value because FFT output is complex
        attn_weights = torch.softmax(torch.abs(attn_weights), dim=-1)
        
        # Aggregate
        out_fft = attn_weights * v_fft
        
        # Back to time domain
        out = torch.fft.irfft(out_fft, n=q.shape[-1], dim=-1)
        
        # Transpose back: [batch, seq_len, features]
        return out.transpose(1, 2)

# STREAMING_CHUNK:Defining Encoder Layer
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=25)
        self.attn = AutoCorrelation(d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Add Feed-Forward Network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        # Decomposition
        seasonal, _ = self.decomp(x)
        
        # Apply Correlation
        new_seasonal = self.attn(seasonal, seasonal, seasonal)
        
        # Residual + Norm
        x = x + new_seasonal
        x = self.norm(x)
        
        # MLP + Residual
        x = x + self.mlp(x)
        x = self.norm(x)
        
        return x

# STREAMING_CHUNK:Defining Main Autoformer Model
class Autoformer(nn.Module):
    def __init__(self, num_features, d_model=512, pred_len=96, seq_len=96, num_layers=6):
        super(Autoformer, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        
        # Input Embedding
        self.enc_embedding = nn.Linear(num_features, d_model)
        self.norm_emb = nn.LayerNorm(d_model)
        
        # Encoder stack
        self.encoder = nn.ModuleList([AutoformerEncoderLayer(d_model) for _ in range(num_layers)])
        
        # Output head
        self.norm_out = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, pred_len * num_features)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Input Embedding & Norm
        x = self.enc_embedding(x)
        x = self.norm_emb(x)
        
        # Encoder Pass
        for layer in self.encoder:
            x = layer(x)
            
        # Global Aggregation
        x = self.norm_out(x)
        x = x.mean(dim=1) 
        
        predictions = self.output_projection(x)
        return predictions.view(batch_size, self.pred_len, self.num_features)
    
if __name__ == "__main__":
    test_input = torch.randn(32, 96, 7)
    model = Autoformer(num_features=7, d_model=512, num_layers=6)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Scalable Autoformer initialized successfully!")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import torch.fft

# class SeriesDecomposition(nn.Module):
#     """
#     Series decomposition block to decompose the time-series into 
#     seasonal and trend parts using moving average.
#     """
#     def __init__(self, kernel_size):
#         super(SeriesDecomposition, self).__init__()
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

#     def forward(self, x):
#         # x shape: [batch, seq_len, features]
#         # Transpose to [batch, features, seq_len] for AvgPool
#         x = x.transpose(1, 2)
#         trend = self.avg(x)
#         seasonal = x - trend
#         # Transpose back to [batch, seq_len, features]
#         return seasonal.transpose(1, 2), trend.transpose(1, 2)

# class AutoCorrelation(nn.Module):
#     """
#     The Auto-Correlation mechanism using FFT to compute periodicity.
#     """
#     def __init__(self, d_model):
#         super(AutoCorrelation, self).__init__()
#         self.d_model = d_model

#     # def time_delay_agg_training(self, values, corr):
#     #     # Simplified time-delay aggregation
#     #     # In full Autoformer, this rotates sequences based on correlation
#     #     return values 

#     def forward(self, x):
#         # Compute FFT
#         x_fft = torch.fft.rfft(x.transpose(1, 2), dim=-1)
#         # Weighting in frequency domain
#         res = x_fft * torch.conj(x_fft)
#         corr = torch.fft.irfft(res, dim=-1)
#         # Softmax over the time lags
#         return torch.softmax(corr, dim=-1)
        
#         # # Select top-k correlations
#         # top_k = 5
#         # weights, delay = torch.topk(corr, top_k, dim=-1)
        
#         # # Aggregate
#         # return v # Return as-is for this refined skeleton

# class AutoformerEncoderLayer(nn.Module):
#     # def __init__(self, d_model):
#     #     super().__init__()
#     #     self.decomp1 = SeriesDecomposition(kernel_size=25)
#     #     self.decomp2 = SeriesDecomposition(kernel_size=25)
#     #     self.attn = AutoCorrelation(d_model)
        
#     # def forward(self, x):
#     #     seasonal, trend = self.decomp1(x)
#     #     # Apply Correlation
#     #     seasonal = self.attn(seasonal, seasonal, seasonal)
#     #     return seasonal + x, trend

#     def __init__(self, d_model):
#         super().__init__()
#         self.decomp1 = SeriesDecomposition(kernel_size=25)
#         self.attn = AutoCorrelation(d_model)
#         self.norm = nn.LayerNorm(d_model)
        
#     def forward(self, x):
#         # Decomposition
#         seasonal, _ = self.decomp1(x)
#         # Apply Correlation
#         attn_weights = self.attn(seasonal)
#         # Residual connection + Norm
#         x = x + attn_weights # Simplified for skeleton
#         return self.norm(x)

# # class Autoformer(nn.Module):
# #     """
# #     Refined Autoformer architecture for LTSF.
# #     """
# #     # def __init__(self, num_features, d_model=512, pred_len=96, seq_len=96, num_layers=6):
# #     #     super(Autoformer, self).__init__()
# #     #     self.pred_len = pred_len
# #     #     self.num_features = num_features
        
# #     #     # 1. Projection
# #     #     self.enc_embedding = nn.Linear(num_features, d_model)
        
# #     #     # 2. Decomposition and Encoder
# #     #     self.encoder = nn.ModuleList([AutoformerEncoderLayer(d_model) for _ in range(num_layers)])
        
# #     #     # 3. Output Projection
# #     #     self.output_projection = nn.Linear(d_model, pred_len * num_features)

# #     # def forward(self, x):
# #     #     batch_size = x.shape[0]
        
# #     #     # Embed
# #     #     x = self.enc_embedding(x)
        
# #     #     # Encoder
# #     #     for layer in self.encoder:
# #     #         x, _ = layer(x)
            
# #     #     # Global aggregation
# #     #     last_hidden = x.mean(dim=1)
        
# #     #     # Project
# #     #     predictions = self.output_projection(last_hidden)
# #     #     return predictions.view(batch_size, self.pred_len, self.num_features)

# #     def __init__(self, num_features, d_model=512, pred_len=96, seq_len=96, num_layers=6):
# #         super(Autoformer, self).__init__()
# #         self.pred_len = pred_len
# #         self.num_features = num_features
        
# #         # Now uses d_model and num_layers dynamically
# #         self.enc_embedding = nn.Linear(num_features, d_model)
        
# #         # We create a list of layers using the num_layers argument provided
# #         self.encoder = nn.ModuleList([
# #             nn.Linear(d_model, d_model) for _ in range(num_layers)
# #         ])
        
# #         self.output_projection = nn.Linear(d_model, pred_len * num_features)

# #     def forward(self, x):
# #         batch_size = x.shape[0]
# #         x = self.enc_embedding(x)
        
# #         # Pass through the layers
# #         for layer in self.encoder:
# #             x = layer(x)
            
# #         # Global average pool over sequence length for simplicity
# #         x = x.mean(dim=1)
        
# #         predictions = self.output_projection(x)
# #         return predictions.view(batch_size, self.pred_len, self.num_features)

# # if __name__ == "__main__":
# #     # Mock data: [batch=32, seq_len=96, features=7]
# #     test_input = torch.randn(32, 96, 7)
    
# #     # Initialize model
# #     model = Autoformer(num_features=7, d_model=64, pred_len=96)
    
# #     # Forward pass
# #     output = model(test_input)
    
# #     print(f"Input shape: {test_input.shape}")
# #     print(f"Output shape: {output.shape}") 
# #     print("Refined Autoformer forward pass successful!")

# class Autoformer(nn.Module):
#     def __init__(self, num_features, d_model=512, pred_len=96, seq_len=96, num_layers=6):
#         super(Autoformer, self).__init__()
#         self.pred_len = pred_len
#         self.num_features = num_features
        
#         # Embedding
#         self.enc_embedding = nn.Linear(num_features, d_model)
#         self.norm_emb = nn.LayerNorm(d_model)
        
#         # Encoder stack
#         self.encoder = nn.ModuleList([AutoformerEncoderLayer(d_model) for _ in range(num_layers)])
        
#         # Output head
#         self.norm_out = nn.LayerNorm(d_model)
#         self.output_projection = nn.Linear(d_model, pred_len * num_features)

#     def forward(self, x):
#         batch_size = x.shape[0]
        
#         # Input Embedding & Norm
#         x = self.enc_embedding(x)
#         x = self.norm_emb(x)
        
#         # Encoder Pass
#         for layer in self.encoder:
#             x = layer(x)
            
#         # Global Aggregation & Projection
#         x = self.norm_out(x)
#         x = x.mean(dim=1) # Pooling
        
#         predictions = self.output_projection(x)
#         return predictions.view(batch_size, self.pred_len, self.num_features)

# if __name__ == "__main__":
#     test_input = torch.randn(32, 96, 7)
#     model = Autoformer(num_features=7, d_model=512, num_layers=6)
#     output = model(test_input)
#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {output.shape}")
#     print("Autoformer with LayerNorm and Decomposition initialized successfully!")