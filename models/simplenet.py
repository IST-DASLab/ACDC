"""
Just a mock net for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import get_2d_conv

class SimpleNet(nn.Module):
    def __init__(self, dataset, use_butterfly):
        super().__init__()
        self.conv1 = get_2d_conv(1, 2**5, 3, 1, use_butterfly=use_butterfly)
        self.conv2 = get_2d_conv(2**5, 2**6, 3, 1, cardinality=2, use_butterfly=use_butterfly)
        self.fc1 = nn.Linear(1600, 500)
        if dataset == 'mnist':
            self.fc2 = nn.Linear(500, 10)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleNet_cifar(nn.Module):
    def __init__(self):
        super(SimpleNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MnistMLP(nn.Module):
    def __init__(self, hidden_size=500):
        super(MnistMLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
