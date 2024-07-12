import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNetwork(nn.Module):
    def __init__(self, ACTION_DIM, LEARNING_RATE_POLICY, latent_dim=3):
        super(PolicyNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, kernel_size=5, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=3)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = layer_init(nn.Linear(32 * 2 * 2 , 64))
        self.fc2 = layer_init(nn.Linear(64, ACTION_DIM))

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_POLICY)

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.ReLU()(self.fc1(x))
        steering = torch.tanh(self.fc2(x)[:, 0])  # steering in range [-1, 1]
        gas = torch.sigmoid(self.fc2(x)[:, 1])    # gas in range [0, 1]
        brake = torch.sigmoid(self.fc2(x)[:, 2])  # brake in range [0, 1]
        return torch.stack((steering, gas, brake), dim=1)
            

class CriticNetwork(nn.Module):
    def __init__(self, ACTION_DIM, LEARNING_RATE_CRITIC, LSTM_HIDDEN_SIZE= 32 , LSTM_NUM_LAYERS=2):
        super(CriticNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, kernel_size=5, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=3)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = layer_init(nn.Linear(32 * 2 * 2 + ACTION_DIM, 64))
        self.fc2 = layer_init(nn.Linear(64, 32))
        self.fc3 = layer_init(nn.Linear(32, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE_CRITIC)

    def forward(self, state, action):
        x = self.conv_layers(state)
        x = torch.cat([x, action], dim=1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
                 

