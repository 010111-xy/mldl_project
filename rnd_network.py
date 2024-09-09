# rnd_network.py
import torch
import torch.nn as nn

class RNDNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RNDNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.net(x)
