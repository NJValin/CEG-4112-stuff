import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

def preprocess(X, y):
    X = np.array(X)
    X = X/255.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

class Iterative_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train(mode=True)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_prediction = model(X)
        loss = loss_func(y_prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Batch:{batch}|loss: {loss:8f} [{current}|{size}]")

def validate(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_prediction = model(X)
            test_loss += loss_func(y_prediction, y).item()
            correct += (y_prediction.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test(X, y, model):
    model.eval()
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    y_pred = y_pred.argmax(1)
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    # Show classification evaluation
    print("\nClassification Report:\n", classification_report(y_np, y_pred_np))

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate = 1e-4

    dataset = load_dataset("mnist")
    ds_train = dataset["train"]
    ds_test = dataset["test"]
    X_train, y_train = ds_train['image'], ds_train['label']
    X_train, y_train = preprocess(X_train, y_train)
    train_dataset = Iterative_Dataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test, X_validate, y_test, y_validate = train_test_split(ds_test['image'], ds_test['label'], test_size=0.25)
    X_test, y_test = preprocess(X_test, y_test)
    test_dataset = Iterative_Dataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    X_validate, y_validate = preprocess(X_validate, y_validate)
    validate_dataset = Iterative_Dataset(X_validate, y_validate)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    try:
        model.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
        print("Success")
    except Exception as e:
        print("\n\npth file cannot be found at ./mnist_model.pth\n\n")
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------------")
        train(train_dataloader, model, loss_func, optimizer)
        validate(validate_dataloader, model, loss_func)



    print("Testing\n--------------------------------------------")
    test(X_test, y_test, model)
    print("Done!")
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Saved PyTorch Model State to mnist_model.pth")
