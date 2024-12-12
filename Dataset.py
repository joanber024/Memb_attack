from torch.utils.data import Dataset
import torch
import numpy as np


class Standard_Dataset(Dataset):
    def __init__(self, X, Y, transformation=None):
        super().__init__()
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(Y, dtype = torch.float32)
        self.transformation = transformation
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return self.X[idx], self.y[idx]

