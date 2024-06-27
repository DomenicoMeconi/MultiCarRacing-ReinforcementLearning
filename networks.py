import torch.nn as nn
import torch.optim as optim
from typing import Optional
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, ACTION_DIM, LEARNING_RATE_POLICY):
        super(PolicyNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=3),
            nn.Tanh(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_DIM),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_POLICY)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=20)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, algo, ACTION_DIM, LEARNING_RATE_CRITIC):
        super(CriticNetwork, self).__init__()
        self.algo = algo
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Flatten()
        )
        if self.algo == 'svg0':
            self.fc_layers = nn.Sequential(
                nn.Linear(32 * 2 * 2 + ACTION_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif self.algo == 'svg1':
            self.fc_layers = nn.Sequential(
                nn.Linear(32 * 2 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_CRITIC)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=40)

    def forward(self, state, action: Optional[torch.Tensor] = None):
        state = self.conv_layers(state)
        if self.algo == 'svg0':
            state = self.fc_layers(torch.cat([state, action], dim=1))
        elif self.algo == 'svg1':
            state = self.fc_layers(state)
        return state
    

"""
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 12 * 12, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_DIM),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_POLICY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.0001)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, algo):
        super(CriticNetwork, self).__init__()
        self.algo = algo
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        if self.algo == 'svg0':
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 12 * 12 + ACTION_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        elif self.algo == 'svg1':
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 12 * 12 , 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_CRITIC)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.0001)

    def forward(self, state, action: Optional[torch.Tensor] = None):
        state = self.conv_layers(state)
        if self.algo == 'svg0':
            state = self.fc_layers(torch.cat([state, action], dim=1))
        elif self.algo == 'svg1':
            state = self.fc_layers(state)
        return state
"""