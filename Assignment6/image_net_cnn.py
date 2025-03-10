import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def preprocess(X, y):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    X = [transform(img) for img in X]
    X_tensor = torch.stack(X)
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
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # reduce spactial dimension by 2
        )

        # Flatten
        nn.Flatten()

        # DNN layers
        self.relu_stack = nn.Sequential(
            nn.Linear(128*8*8, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(512,200)
        )

    def forward(self, X):
        X  = self.convolution_layers(X)
        logits = self.relu_stack(X)
        return logits

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

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate=1e-5

    train_dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    X_train, X_validate, y_train, y_validate = train_test_split(train_dataset["image"], train_dataset["label"],  test_size=0.1)

    X_train, y_train = preprocess(X_train, y_train)
    train_ds = Iterative_Dataset(X_train, y_train)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    validate_ds = Iterative_Dataset(X_validate, y_validate)
    validate_dataloader = DataLoader(validate_ds, batch_size=batch_size, shuffle=True)

    test_dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
    X_test, y_test = preprocess(test_dataset["image"], test_dataset["label"])
    test_ds = Iterative_Dataset(X_test, y_test)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

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

