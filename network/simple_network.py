import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 8)
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x