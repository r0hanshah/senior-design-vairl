import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), # Takes in state
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs deltas
        )

    def forward(self, state):
        return self.net(state)
