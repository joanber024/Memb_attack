import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x))

