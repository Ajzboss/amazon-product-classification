import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
#for confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#APply this if you want to make your model cpu or gpu
device="cpu" 
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,X_img, y) in enumerate(dataloader):
        #print(X)
        X, y = X.to(device), y.to(device)
        if torch.isnan(X).any() or torch.isinf(X).any():
                raise ValueError("Input data contains NaN or infinite values.")
        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)
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
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     print("Using GPU:", torch.cuda.get_device_name(device))
            #     print("GPU Memory Usage:")
            #     print("Allocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
            #     print("Cached:   ", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
        del loss,current,pred
    #print("Batching Complete")


def test(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        iter=0
        all_pred=None
        all_labels=None
        i = None
        for X,X_img,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if i == None:
                all_pred=np.argmax(np.array(pred), axis=1)
                all_labels =np.array(y)
                i=True
            else:
                all_pred= np.concatenate((all_pred,np.argmax(np.array(pred), axis=1)),axis=0)
                all_labels = np.concatenate((all_labels,np.array(y)),axis=0)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            iter+= 1 
            print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
            if iter % 100==0:
                print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
        test_loss /= num_batches
        correct /= size
        cm = confusion_matrix(all_labels,all_pred)
        class_names=['arts, crafts & sewing','books', 'clothing, shoes & jewelry','electronics', 'grocery & gourmet food','health & personal care', 'musical instruments', 'patio, lawn & garden','sports & outdoors','toys & games']
        confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        confusion_df.index.name = 'True Label'
        confusion_df.columns.name = 'Predicted Label'
        sns.heatmap(confusion_df, annot=True, fmt='3g')
        plt.show()
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_ensemble(dataloader, model_list, loss_fn,t_model,weights=[0.5,0.5]):
    for model in model_list:
        model.eval()
    print("Beginnning inference")
    all_pred=None
    all_labels=None
    i = None
    with torch.no_grad():
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        iter=0
        for X_text,X_img,y in dataloader:
            X_img=torch.transpose(X_img,1,3)
            X_img = X_img.numpy()
            X_img = tf.convert_to_tensor(X_img)
            X_img=tf.cast(X_img,tf.float32)
            X_img=tf.keras.applications.inception_v3.preprocess_input(X_img)
            print("Testing Prediction")
            X_text, y = X_text.to(device),y.to(device)
            pred_by_model = []
            #print(X_text)
            for model in model_list:
                out = model(X_text)
                pred_by_model.append(out)
            t_out=t_model.predict(X_img)
            t_out=torch.tensor(t_out)
            pred_by_model.append(t_out)
            for it,weight in enumerate(weights):
                pred_by_model[it]*=weight
            stacked_tensor = torch.stack(pred_by_model)
            pred = torch.sum(stacked_tensor,axis=0)
            if i == None:
                all_pred=np.argmax(np.array(pred), axis=1)
                all_labels =np.array(y)
                i=True
            else:
                all_pred= np.concatenate((all_pred,np.argmax(np.array(pred), axis=1)),axis=0)
                all_labels = np.concatenate((all_labels,np.array(y)),axis=0)
            #print(pred)
            #print(y)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            iter+= 1 
            print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")

            if iter % 100==0:
                print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
    #print("completed inference")
        test_loss /= num_batches
        correct /= size
        cm = confusion_matrix(all_labels,all_pred)
        class_names=['arts, crafts & sewing','books', 'clothing, shoes & jewelry','electronics', 'grocery & gourmet food','health & personal care', 'musical instruments', 'patio, lawn & garden','sports & outdoors','toys & games']
        confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        confusion_df.index.name = 'True Label'
        confusion_df.columns.name = 'Predicted Label'
        #print(confusion_df)
        sns.heatmap(confusion_df, annot=True, fmt='3g')
        plt.show()
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_stack(dataloader, model_list,loss_fn, optimizer,meta_model,t_model):
    print("Training Meta Model")
    size = len(dataloader.dataset)
    for model in model_list:
        model.eval()
    meta_model.train()
    for batch, (X_text,X_img, y) in enumerate(dataloader):
        #print(X)
        X_text, y = X_text.to(device),y.to(device)
        X_img=torch.transpose(X_img,1,3)
        X_img = X_img.numpy()
        X_img = tf.convert_to_tensor(X_img)
        X_img=tf.cast(X_img,tf.float32)
        X_img=tf.keras.applications.inception_v3.preprocess_input(X_img)
        # Compute prediction error
        pred_by_model = []
        for model in model_list:
            pred_by_model.append(model(X_text))
        t_out=t_model.predict(X_img)
        t_out=torch.tensor(t_out)
        pred_by_model.append(t_out)
        stacked_tensor = torch.stack(pred_by_model)
        pred_by_model=torch.transpose(stacked_tensor,1,0)
        pred = meta_model(pred_by_model)
        #pred=torch.transpose(pred)
        loss = loss_fn(pred, y)
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
        loss, current = loss.item(), (batch + 1) * (len(X_text))
        #print(f"loss: {loss}  [{current}/{size}]")
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % 128 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     print("Using GPU:", torch.cuda.get_device_name(device))
            #     print("GPU Memory Usage:")
            #     print("Allocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
            #     print("Cached:   ", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
        del loss,current,pred
    #print("Batching Complete")

def test_stack(dataloader, model_list, loss_fn,meta_model,t_model):
    for model in model_list:
        model.eval()
    meta_model.eval()
    print("Beginnning inference")
    with torch.no_grad():
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        iter=0
        all_pred=None
        all_labels=None
        i = None
        for X_text,X_img,y in dataloader:
            X_img=torch.transpose(X_img,1,3)
            X_img = X_img.numpy()
            X_img = tf.convert_to_tensor(X_img)
            X_img=tf.cast(X_img,tf.float32)
            X_img=tf.keras.applications.inception_v3.preprocess_input(X_img)
            X_text, y = X_text.to(device),y.to(device)
            pred_by_model = []
            #print(X_text)
            for model in model_list:
                out = model(X_text)
                pred_by_model.append(out)
            t_out=t_model.predict(X_img)
            t_out=torch.tensor(t_out)
            pred_by_model.append(t_out)
            stacked_tensor = torch.stack(pred_by_model)
            pred_by_model=torch.transpose(stacked_tensor,1,0)
            pred = meta_model(pred_by_model)
            if i == None:
                all_pred=np.argmax(np.array(pred), axis=1)
                all_labels =np.array(y)
                i=True
            else:
                all_pred= np.concatenate((all_pred,np.argmax(np.array(pred), axis=1)),axis=0)
                all_labels = np.concatenate((all_labels,np.array(y)),axis=0)
            #print(pred)
            #print(y)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            iter+= 1 
            print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
            if iter % 100==0:
                print(f"Progress: {iter}/{num_batches},accuracy:{(100*(correct/size)):>0.1f},test_loss:{(test_loss/iter):>8f}")
    #print("completed inference")
        test_loss /= num_batches
        correct /= size
        cm = confusion_matrix(all_labels,all_pred)
        class_names=['arts, crafts & sewing','books', 'clothing, shoes & jewelry','electronics', 'grocery & gourmet food','health & personal care', 'musical instruments', 'patio, lawn & garden','sports & outdoors','toys & games']
        confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        confusion_df.index.name = 'True Label'
        confusion_df.columns.name = 'Predicted Label'
        #print(confusion_df)
        sns.heatmap(confusion_df, annot=True, fmt='3g')
        plt.show()
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

