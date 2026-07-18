# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import time

# from data_loader.dataset import TimeSeriesDataset
# from models.rnn_lstm import FairLSTM
# from models.transformer import FairTransformer
# from models.autoformer import Autoformer
# from utils.metrics import calculate_mse, calculate_mae
# from utils.tools import plot_training_history

# def count_parameters(model):
#     """Counts the number of trainable parameters in the model."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def train_and_evaluate(model, train_loader, val_loader, device, epochs=10):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Scheduler: Reduces learning rate if val loss plateaus
#     # Verbose argument removed for compatibility with recent PyTorch versions
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=2
#     )
    
#     model.to(device)
    
#     train_history = []
#     val_history = []
    
#     for epoch in range(epochs):
#         model.train()
#         total_train_loss = 0
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             output = model(x)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
        
#         avg_train_loss = total_train_loss / len(train_loader)
#         train_history.append(avg_train_loss)
        
#         # Validation
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device), y.to(device)
#                 output = model(x)
#                 total_val_loss += criterion(output, y).item()
        
#         avg_val_loss = total_val_loss / len(val_loader)
#         val_history.append(avg_val_loss)
        
#         # Step the scheduler
#         scheduler.step(avg_val_loss)
        
#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
#     return train_history, val_history

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataset = TimeSeriesDataset("datasets/ETTh1.csv")
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
#     # Initialize models
#     # Note: Keep hidden dimensions similar for parameter fairness
#     models = {
#         "LSTM": FairLSTM(num_features=7, d_model=512, num_layers=2),
#         "Transformer": FairTransformer(num_features=7, d_model=512, num_layers=2),
#         "Autoformer": Autoformer(num_features=7, d_model=512, num_layers=6)
#     }
    
#     for name, model in models.items():
#         print(f"\nTraining {name} on {device}...")
        
#         # Fair comparison check
#         param_count = count_parameters(model)
#         print(f"Total trainable parameters for {name}: {param_count:,}")
        
#         t_hist, v_hist = train_and_evaluate(model, train_loader, val_loader, device)
#         plot_training_history(name, t_hist, v_hist)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader.dataset import TimeSeriesDataset
from models.rnn_lstm import FairLSTM
from models.transformer import FairTransformer
from models.autoformer import Autoformer
from utils.tools import plot_training_history

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler: Reduces learning rate if val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    model.to(device)
    
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                total_val_loss += criterion(output, y).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_history.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    return train_history, val_history

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TimeSeriesDataset("datasets/ETTh1.csv")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    # Increased Autoformer capacity to match others
    models = {
        "LSTM": FairLSTM(num_features=7, d_model=512, num_layers=2),
        "Transformer": FairTransformer(num_features=7, d_model=512, num_layers=2),
        "Autoformer": Autoformer(num_features=7, d_model=512, num_layers=6)
    }
    
    # 1. Print all counts first
    print("\n--- Model Fairness Check ---")
    for name, model in models.items():
        print(f"{name} parameters: {count_parameters(model):,}")
    
    # 2. Run Training
    for name, model in models.items():
        print(f"\nTraining {name} on {device}...")
        t_hist, v_hist = train_and_evaluate(model, train_loader, val_loader, device)
        plot_training_history(name, t_hist, v_hist)