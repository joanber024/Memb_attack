from torch.utils.data import Dataset
import torch
import numpy as np


class Standard_Dataset(Dataset):
    def __init__(self, X, Y, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return torch.from_numpy(np.array(self.X[idx])).float(), torch.from_numpy(np.array(self.y[idx])).float()

