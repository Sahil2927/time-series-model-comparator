import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import time # Added for Latency tracking

from data_loader.dataset import TimeSeriesDataset
from models.rnn_lstm import FairLSTM
from models.transformer import FairTransformer
from models.autoformer import Autoformer
from utils.tools import plot_training_history

# Configuration
DATASETS = {
    "ETTh1": "datasets/ETTh1.csv",
    "Traffic": "datasets/traffic.csv"
}
HORIZONS = [96, 192, 336]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5 

def get_model(name, pred_len, num_features):
    """Initializes the correct model architecture."""
    if name == "LSTM":
        return FairLSTM(num_features=num_features, d_model=512, num_layers=2, pred_len=pred_len)
    elif name == "Transformer":
        return FairTransformer(num_features=num_features, d_model=512, num_layers=2, pred_len=pred_len)
    elif name == "Autoformer":
        return Autoformer(num_features=num_features, d_model=512, num_layers=6, pred_len=pred_len)
    return None

def train_and_evaluate_robust(model, train_loader, val_loader, device, epochs, plot_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    train_history = []
    val_history = []
    
    best_val_mse = float('inf')
    best_val_mae = float('inf')
    peak_vram = 0
    avg_latency = 0
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"    [Batch {batch_idx}/{len(train_loader)}] Computing...")
                
        avg_train_loss = total_train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        # --- VALIDATION & BENCHMARKING PHASE ---
        model.eval()
        total_val_mse = 0
        total_val_mae = 0
        
        # Reset VRAM tracker before validation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        start_time = time.time()
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                
                # Calculate both MSE and MAE
                total_val_mse += criterion(output, y).item()
                total_val_mae += torch.mean(torch.abs(output - y)).item()
                
        end_time = time.time()
        
        # Calculate Metrics
        epoch_val_mse = total_val_mse / len(val_loader)
        epoch_val_mae = total_val_mae / len(val_loader)
        val_history.append(epoch_val_mse)
        
        # Hardware Metrics
        epoch_latency = ((end_time - start_time) / len(val_loader)) * 1000 # milliseconds per batch
        epoch_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
        
        scheduler.step(epoch_val_mse)
        
        print(f"  -> Epoch {epoch+1}/{epochs} | Val MSE: {epoch_val_mse:.4f} | Val MAE: {epoch_val_mae:.4f} | VRAM: {epoch_vram:.1f}MB | Latency: {epoch_latency:.1f}ms/batch")
        
        # Save best metrics for final report
        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            best_val_mae = epoch_val_mae
            peak_vram = epoch_vram
            avg_latency = epoch_latency
            
    plot_training_history(plot_name, train_history, val_history)
    
    # Return a dictionary of all tools
    return {
        "MSE": best_val_mse,
        "MAE": best_val_mae,
        "VRAM_MB": peak_vram,
        "Latency_ms": avg_latency
    }

if __name__ == "__main__":
    results = []
    os.makedirs('results', exist_ok=True)
    
    for ds_name, path in DATASETS.items():
        print(f"\n{'#'*20} Dataset: {ds_name} {'#'*20}")
        
        # Prevent OOM on Traffic dataset by shrinking batch size
        batch_size = 8 if ds_name == "Traffic" else 32
        
        for horizon in HORIZONS:
            print(f"\n{'='*20} Horizon: {horizon} {'='*20}")
            
            dataset = TimeSeriesDataset(path, seq_len=96, pred_len=horizon)
            num_features = dataset.df_raw.shape[1] 
            
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for model_name in ["LSTM", "Transformer", "Autoformer"]:
                print(f"\nBenchmarking {model_name}...")
                model = get_model(model_name, horizon, num_features).to(DEVICE)
                
                plot_name = f"{ds_name}_{model_name}_H{horizon}"
                
                # Get the full dictionary of metrics
                metrics = train_and_evaluate_robust(model, train_loader, val_loader, DEVICE, EPOCHS, plot_name)
                
                # Save ALL tools to the final results
                results.append({
                    "Dataset": ds_name,
                    "Horizon": horizon,
                    "Model": model_name,
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                    "Peak_VRAM_MB": metrics["VRAM_MB"],
                    "Latency_ms": metrics["Latency_ms"]
                })
            
    df = pd.DataFrame(results)
    df.to_csv("results/final_benchmark_results.csv", index=False)
    print("\nBenchmark Complete! Full metrics saved to results/final_benchmark_results.csv")