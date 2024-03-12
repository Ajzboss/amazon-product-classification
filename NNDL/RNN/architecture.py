import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchtext.vocab import GloVe

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,embed_size,n_layers=1):
        super(RNN, self).__init__()
        self.n_layers= n_layers
        self.hidden_dim = hidden_size
        glove_weights = torch.load(f".vector_cache/glove.6B.{embed_size}d.txt.pt")
        #print(glove_weights[2].shape)
        self.embedding = nn.Embedding.from_pretrained(glove_weights[2], freeze=True)
        self.LSTM = nn.LSTM(embed_size, hidden_size,n_layers, batch_first=True)
        # self.flatten = nn.Flatten()
        self.fc= nn.Linear(embed_size, num_classes)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        x=self.embedding(x)
        x= torch.transpose(x,2,1)
        #print(x.shape)
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        x, h_n = self.LSTM(x)
        x = x[:, -1, :]
        print(x.shape)
        x = self.fc(x)
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return x

# import torch
# import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self, input_size, n_filters, filter_size,drop_frac,num_classes,embed_dim):
        super(MyNetwork, self).__init__()
        # Define layers
        glove_weights = torch.load(f".vector_cache/glove.6B.{embed_dim}d.txt.pt")
        #print(glove_weights[2].shape)
        self.embedding = nn.Embedding.from_pretrained(glove_weights[2], freeze=True)
        self.conv1 = nn.Conv1d(input_size,n_filters, filter_size,padding="same")
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(filter_size)
        self.conv2 = nn.Conv1d(n_filters,n_filters, filter_size,padding="same")
        self.maxpool2 = nn.MaxPool1d(filter_size)
        self.conv3 = nn.Conv1d(n_filters,n_filters, filter_size,padding="same")
        self.global_maxpool = nn.AdaptiveMaxPool1d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters*128,n_filters*128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(n_filters*128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #print(f"input SHAOE: {x.shape}")
        x=self.embedding(x)
        x= torch.transpose(x,2,1)
        #x = self.dropout(x)
        #print(x.shape)
        x = self.conv1(x)
        x= self.relu(x)
        #print(x.shape)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.conv2(x)
        x= self.relu(x)
        x = self.relu2(x)
        #print(x.shape)
        x = self.maxpool2(x)
        #print(x.shape)
        x = self.conv3(x)
        x= self.relu(x)
        x = self.global_maxpool(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc1(x)
        x= self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        logits = x

        return logits

