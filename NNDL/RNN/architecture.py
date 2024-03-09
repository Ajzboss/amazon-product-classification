import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        output, _ = self.rnn(x)
        
        # Use the output from the last time step for each sequence in the batch
        # print(output)
        # last_output = output[:, -1, :]
        
        # # If the last_output tensor is 3D, you might need to flatten it
        # last_output = last_output.view(last_output.size(0), -1)

        x = self.linear_relu_stack(output)
        return x
