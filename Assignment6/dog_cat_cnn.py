from threading import main_thread
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import time
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def preprocess(X, y):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    y_tensor = []
    if isinstance(X, list):
        X_tensor = []
        for i in range(len(X)):
            img = X[i]
            img_tensor = transform(img)
            if img_tensor.shape[0] != 3:  # Ensure it's a 3-channel image
                continue
            X_tensor.append(img_tensor)
            y_tensor.append(y[i])
        X_tensor = torch.stack(X_tensor)
        y_tensor = torch.tensor(y_tensor, dtype=torch.float32)
    else:
        X_tensor = transform(X)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if X_tensor.shape[0] != 3:  # Ensure it's a 3-channel image
            return None, None  # Skip invalid images
    return X_tensor, y_tensor

def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=["Actual 0", "Actual 1"], 
                     columns=["Predicted 0", "Predicted 1"])
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()


class Iterative_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Convolution Layers
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # reduce spactial dimension by 2


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # reduce spactial dimension by 2

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # reduce spactial dimension by 2
        )


        # DNN layers
        self.relu_stack = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(128,1)
        )

    def forward(self, X):
        X = self.convolution_layers(X)
        #print("post conv layer X:", X.shape)
        X = X.view(X.size(0), -1)
        #print("flattened layer X:", X.shape)
        logits = self.relu_stack(X)
        return logits

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train(mode=True)

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_prediction = model(X)
        loss = loss_func(y_prediction.squeeze(1), y)

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
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_prediction = model(X)
            test_loss += loss_func(y_prediction.squeeze(1), y).item()
            correct += ((torch.sigmoid(y_prediction.squeeze(1)) > 0.5).float() == y).sum().item()
            print(correct)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test(X,y, model):
    model.eval()
    X, y = X.to(device), y.to(device)

    y_pred = model(X)
    y_pred = torch.sigmoid(y_pred).squeeze(1)
    y_pred = torch.round(y_pred)
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    print("\nClassification Report:\n", classification_report(y_np, y_pred_np))

    #Confusion Matrix
    plot_confusion_matrix(y_np, y_pred_np)

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 64
    learning_rate=1e-5

    dataset = load_dataset("microsoft/cats_vs_dogs", split="train", keep_in_memory=True)
    X_train, X_other, y_train, y_other = train_test_split(dataset["image"], dataset["labels"], train_size=0.7)

    X_test, X_validate, y_test, y_validate = train_test_split(X_other, y_other, train_size=0.5)

    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)
    X_validate, y_validate = preprocess(X_validate, y_validate)

    ds_train = Iterative_Dataset(X_train, y_train)
    ds_test = Iterative_Dataset(X_test, y_test)
    ds_validate = Iterative_Dataset(X_validate, y_validate)

    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(ds_validate, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        print("model loaded")
    except:
        print("model not found")

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------")
        train(train_dataloader, model, loss_func, optimizer)
        validate(validate_dataloader, model, loss_func)


    print("TESTING\n---------------------------")
    test(X_test, y_test, model)

    torch.save(model.state_dict(), "model.pth")
    print("Saved model to model.pth")
