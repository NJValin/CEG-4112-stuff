import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer   
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import re
import string
import matplotlib.pyplot as plt

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

def preprocess(X, y, split_name="train"):
    save_path = f"preprocessed_{split_name}.pt"

    # If preprocessed data exists, load it
    if os.path.exists(save_path):
        print(f"Loading preprocessed data from {save_path}...")
        data = torch.load(save_path)
        return data["X"], data["y"]

    print(f"Preprocessing data for {split_name}...")

    feature_extraction = TfidfVectorizer(stop_words="english", lowercase=True, max_features=10_000)
    reducer = PCA(n_components=512)
    
    X_cleaned = [clean_text(text) for text in X]
    X_transformed = feature_extraction.fit_transform(X_cleaned)
    X_Reduced = reducer.fit_transform(X_transformed)
    
    X_tensor = torch.tensor(X_Reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Ensure float labels

    # Save processed data for future use
    #torch.save({"X": X_tensor, "y": y_tensor}, save_path)
    print(f"Saved preprocessed data to {save_path}")

    return X_tensor, y_tensor


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()


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
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits

def train(dataloader, model, loss_func, optimizer):
    if loss_func is None or optimizer is None:
        print("loss_func or optimizer is None")
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X).squeeze(1)
        loss = loss_func(y_pred, y)

        # Back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")

def validate(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze(1)
            test_loss += loss_func(y_pred, y).item()
            correct += ((torch.sigmoid(y_pred) > 0.5).float() == y).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def test(X, y, model):
    model.eval()
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    y_pred = torch.sigmoid(y_pred).squeeze(1)
    y_pred = torch.round(y_pred)
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    # Show classification evaluation
    print("\nClassification Report:\n", classification_report(y_np, y_pred_np))

    # Confusion matrix
    plot_confusion_matrix(y_np, y_pred_np)
    

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate = 5e-4

    ds_train = load_dataset("imdb", split="train")
    X_train, y_train = preprocess(ds_train["text"], ds_train["label"], "train")
    train_dataset = TextDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ds_test_valid = load_dataset("imdb", split="test")
    X_test, X_valid, y_test, y_valid = train_test_split(ds_test_valid["text"], ds_test_valid["label"], test_size=0.4, random_state=42)

    X_test, y_test = preprocess(X_test, y_test, "test")
    test_dataset = TextDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    X_valid, y_valid = preprocess(X_valid, y_valid, "valid")
    valid_dataset = TextDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    try:
        try:
            model.load_state_dict(torch.load("best_model.pt", weights_only=True))
        except Exception as e:
            model.load_state_dict(torch.load("model.pth", weights_only=True))
        print("Success")
    except Exception as e:
        print("\n\npth file cannot be found at ./model.pth\n\n")
        model.apply(init_weights)
        print(e)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)



    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    epochs = 150
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------------")
        train(train_dataloader, model, loss_func, optimizer)
        val_loss = validate(valid_dataloader, model, loss_func)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Stopping early at Epoch {t+1}")
            break



    print("Testing\n--------------------------------------------")
    test(X_test, y_test, model)


    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
