import torch

def calculate_mse(pred, true):
    return torch.mean((pred - true) ** 2).item()

def calculate_mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def print_metrics(model_name, mse, mae, latency_ms, vram_mb):
    print(f"[{model_name}]")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Latency: {latency_ms:.2f} ms/batch")
    print(f"  Peak VRAM: {vram_mb:.2f} MB")
    print("-" * 30)