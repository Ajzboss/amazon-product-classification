import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here

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
        #print(X)
        X, y = X.to(device), y.to(device)
        if torch.isnan(X).any() or torch.isinf(X).any():
                raise ValueError("Input data contains NaN or infinite values.")
        # Compute prediction error
        pred = model(X)
        #print(pred.shape)
        #print(y.shape)
        loss = loss_fn(pred, y)
        #loss = loss.clamp(min=1e-4)
        # cm = confusion_matrix(y, pred)
        # ConfusionMatrixDisplay(cm, model.classes_).plot()

        # Backpropagation
        loss.backward()
        # for param in model.parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             print(f"Param.grad that is nan or inf:{param.grad} ")
        #torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite =True)
        optimizer.step()
        optimizer.zero_grad()
        #print(loss)
        loss, current = loss.item(), (batch + 1) * len(X)
        #print(f"loss: {loss}  [{current}/{size}]")

        if batch % 128 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Using GPU:", torch.cuda.get_device_name(device))
                print("GPU Memory Usage:")
                print("Allocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
                print("Cached:   ", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
        del loss,current,pred
    #print("Batching Complete")


def test(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        iter=0
        for X, y in dataloader:
            #print("Testing Prediction")
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(pred)
            #print(y)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            iter+= 1 
            if iter % 100==0:
                print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
    #print("completed inference")
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
