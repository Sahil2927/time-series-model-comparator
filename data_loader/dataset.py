import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset class for LTSF datasets (like ETTh1).
    It reads a CSV, scales the data, and generates input/target windows.
    """
    def __init__(self, file_path, seq_len=96, pred_len=96, target_col='OT'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Load the raw data
        print(f"Loading data from {file_path}...")
        self.df_raw = pd.read_csv(file_path)
        
        # Drop the date column if it exists
        if 'date' in self.df_raw.columns:
            self.df_raw = self.df_raw.drop(columns=['date'])
            
        # 2. Scale the data
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.df_raw.values)
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Test block
if __name__ == "__main__":
    # Point to the file you just downloaded
    dataset = TimeSeriesDataset("datasets/ETTh1.csv")
    x, y = dataset[0]
    print(f"Sample X shape: {x.shape}") # Should be [96, 7]
    print(f"Sample Y shape: {y.shape}") # Should be [96, 7]
    print("Dataset loaded successfully!")