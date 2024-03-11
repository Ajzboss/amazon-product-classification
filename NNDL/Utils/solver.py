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
#device="cpu"
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if torch.isnan(X).any() or torch.isinf(X).any():
                raise ValueError("Input data contains NaN or infinite values.")
        # Compute prediction error
        pred = model(X)
        #print(y)
        loss = loss_fn(pred, y)
        loss = loss.clamp(min=1e-4)
        # Backpropagation
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Param.grad that is nan or inf:{param.grad} ")
        torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite =True)
        optimizer.step()
        optimizer.zero_grad()
        #print(loss)
        loss, current = loss.item(), (batch + 1) * len(X)
        #print(f"loss: {loss}  [{current}/{size}]")

        if batch % 64 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")