import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        output, _ = self.rnn(x)

        x = self.linear_relu_stack(output)
        return x

# import torch
# import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self, input_size, n_filters, filter_size,drop_frac,num_classes):
        super(MyNetwork, self).__init__()

        # Define layers
        self.dropout = nn.Dropout(drop_frac)
        self.conv1 = nn.Conv1d(1,1, 1)
        nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(filter_size)
        self.conv2 = nn.Conv1d(1,1, 1)
        self.maxpool2 = nn.MaxPool1d(filter_size)
        self.conv3 = nn.Conv1d(1,1,1)
        self.global_maxpool = nn.AdaptiveMaxPool1d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters,n_filters)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_filters, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        print(x.shape)
        x = self.conv1(x.unsqueeze(1))
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.global_maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        logits = x

        return logits
