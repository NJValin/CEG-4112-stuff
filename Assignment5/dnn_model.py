from logging import error
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer   
import pandas as pd
from umap import UMAP

def preprocess(X,y):
    feature_extraction = TfidfVectorizer(stop_words="english", max_features=1_000)
    umaper = UMAP(n_neighbors=15, min_dist=0.1, n_components=512, n_jobs=-1)
    X_transformed = feature_extraction.fit_transform(X).toarray()
    X_Reduced = umaper.fit_transform(X_transformed)
    X_tensor = torch.tensor(X_Reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

class TextDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1),

        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits

def train(dataloader, model, loss_func, optimizer):
    if loss_func is None or optimizer is None:
        error("loss_func or optimizer is None")
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_func(y_pred.squeeze(1), y)

        # Back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")

def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_func(y_pred.squeeze(1), y).item()
            correct += ((torch.sigmoid(y_pred) > 0.5).float() == y).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate = 1e-3

    ds_train = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    X_train, y_train = preprocess(ds_train["text"], ds_train["label"])
    train_dataset = TextDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    ds_test = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    X_test, y_test = preprocess(ds_test["text"], ds_test["label"])
    print(f"X:{X_test}")
    print(f"y:{y_test}")
    test_dataset = TextDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    #ds_valid = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
    #X_valid, y_valid = preprocess(ds_valid["text"], ds_valid["label"])
    #valid_dataset = TextDataset(X_valid, y_valid)
    #valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        print("Success")
    except Exception as e:
        print(".pth file cannot be found at ./model.pth")

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    epochs = 150
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------------")
        train(train_dataloader, model, loss_func, optimizer)
        test(test_dataloader, model, loss_func)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
