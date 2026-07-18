import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os

from data_loader.dataset import TimeSeriesDataset
from models.rnn_lstm import FairLSTM
from models.transformer import FairTransformer
from models.autoformer import Autoformer

HORIZONS = [96, 192, 336]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "datasets/ETTh1.csv"
EPOCHS = 5 # Rapid benchmark cycles

def get_model(name, pred_len, num_features=7):
    """Initializes the correct model architecture with the specific prediction length."""
    if name == "LSTM":
        return FairLSTM(num_features=num_features, d_model=512, pred_len=pred_len)
    elif name == "Transformer":
        return FairTransformer(num_features=num_features, d_model=512, pred_len=pred_len)
    elif name == "Autoformer":
        return Autoformer(num_features=num_features, d_model=512, pred_len=pred_len)
    return None

def train_and_eval(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    results = []
    os.makedirs('results', exist_ok=True)
    
    for horizon in HORIZONS:
        print(f"\n{'='*20} Horizon: {horizon} {'='*20}")
        
        # Prepare Data
        dataset = TimeSeriesDataset(DATA_PATH, seq_len=96, pred_len=horizon)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        for model_name in ["LSTM", "Transformer", "Autoformer"]:
            print(f"Benchmarking {model_name}...")
            model = get_model(model_name, horizon).to(DEVICE)
            
            mse = train_and_eval(model, train_loader, val_loader, DEVICE)
            
            results.append({
                "Horizon": horizon,
                "Model": model_name,
                "MSE": mse
            })
            print(f" -> MSE: {mse:.4f}")
            
    # Saving Results
    df = pd.DataFrame(results)
    df.to_csv("results/benchmark_results.csv", index=False)
    print("\nBenchmark Complete! Results saved to results/benchmark_results.csv")
    print(df)