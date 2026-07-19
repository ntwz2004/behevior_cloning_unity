import torch
import torch.nn as nn

class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: [Batch, 3, 200, 240]
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 13, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 3) # Output: 3 คลาส (Forward, Left, Right)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x