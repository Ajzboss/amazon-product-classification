import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
class TransformerModel(nn.Module):
    def __init__(self,input_size,embed_size,num_classes, nhead=5, num_encoder_layers=6):
        super(TransformerModel, self).__init__()
        glove_weights = torch.load(f".vector_cache/glove.6B.{embed_size}d.txt.pt")
        #print(glove_weights[2].shape)
        self.embedding = nn.Embedding.from_pretrained(glove_weights[2], freeze=True)
        self.layer = nn.TransformerEncoderLayer(embed_size,nhead,dim_feedforward=100)
        self.transformer = nn.TransformerEncoder(self.layer,num_encoder_layers)
        self.fc= nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        #print(x.shape)
        x = x.permute(1, 0, 2)  # Change to (sequence_length, batch_size, d_model) for transformer input
        x = self.transformer(x)
        #print(x.shape)
        x = x.permute(1, 0, 2)
        x = x.mean(axis=1)
        x = self.fc(x)
        return x