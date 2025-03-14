import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def preprocess(X, y):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    y_tensor = []
    if isinstance(X, list):
        X_tensor = []
        for i in range(len(X)):
            img = X[i]
            img_tensor = transform(img)
            X_tensor.append(img_tensor)
            y_tensor.append(y[i])
        X_tensor = torch.stack(X_tensor)
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
    else:
        X_tensor = transform(X)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if X_tensor.shape[0] != 3:  # Ensure it's a 3-channel image
            return None, None  # Skip invalid images
    print("\ny_tensor:", y_tensor)
    print("\ny_tensor.shape:", y_tensor.shape)
    return X_tensor, y_tensor

def augment(X, transform_composition):
    return torch.stack([transform_composition(X_i) for X_i in X])

def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    size = cm.shape[0]
    index_str = [f"Actual {i}" for i in range(size)]
    column_str = [f"Predicted {i}" for i in range(size)]
    cm_df = pd.DataFrame(cm,
                         index=index_str,
                         columns=column_str)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()


class Iterative_Dataset(Dataset):
    def __init__(self, X, y, transformation=None):
        self.X = X
        self.y = y
        self.transformation = transformation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transformation:
            x = self.transformation(x)
        return x, self.y[index]

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Convolution Layers
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # reduce spactial dimension by 2


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # reduce spactial dimension by 2

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        )


        # DNN layers
        self.relu_stack = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, X):
        X = self.convolution_layers(X)
        X = X.reshape(X.size(0), -1)
        logits = self.relu_stack(X)
        return logits

def train(dataloader, model, loss_func, optimizer, scaler=None):
    size = len(dataloader.dataset)
    model.train(mode=True)

    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if scaler:
            with torch.amp.autocast('cuda'):
                y_prediction = model(X).squeeze(1)
                loss = loss_func(y_prediction, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            y_prediction = model(X).squeeze(1)
            loss = loss_func(y_prediction, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        correct += (y_prediction.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()*X.size(0)

        

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Batch:{batch}|loss: {loss:8f} [{current}|{size}]")
    accuracy = correct/size
    return accuracy, train_loss

def validate(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_prediction = model(X).squeeze(1)
            test_loss += loss_func(y_prediction, y).item()
            correct += (y_prediction.argmax(1) == y).type(torch.float).sum().item()

    total_loss = test_loss/num_batches
    accuracy = correct/size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, total_loss

def apply_pseudo_labels(dataloader, model, confidence_threshold=0.8):
    size = len(dataloader.dataset)
    correct = 0
    label = []
    X_tensors = []
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y_pred = model(X).squeeze(1)
            
            probabilities = nn.functional.softmax(y_pred, dim=1)

            # Get the highest probability and its index (class)
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            
            # Only use predictions with high confidence
            if max_prob.item() >= confidence_threshold or predicted_class==6:
                if predicted_class.item() == y.item():
                    correct += 1
                label.append(predicted_class.item())
                X_tensors.append(X.cpu())
    X_tensors = torch.cat(X_tensors, dim=0)  # Concatenate batches correctly
    label = torch.tensor(label, dtype=torch.long)  # Convert list to tensor
    print(f"Accuracy: {correct/size} : [{correct}|{size}]")
    return X_tensors, label


def test(X,y, model, train_accuracy=None, valid_accuracy=None, train_loss=None, valid_loss=None, epochs=10):
    model.eval()
    X, y = X.to(device), y.to(device)

    y_pred = model(X).squeeze(1)
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.argmax(1)
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    print("\nClassification Report:\n", classification_report(y_np, y_pred_np))

    #Confusion Matrix
    plot_confusion_matrix(y_np, y_pred_np)

    # training and validation Accuracy/loss curves
    if (train_accuracy is not None) or (valid_accuracy is not None) or (train_loss is not None) or (valid_loss is not None):
        fig = plt.figure(figsize=(8, 6))

        plt1 = fig.add_subplot(1, 2, 1)
        plt2 = fig.add_subplot(1, 2, 2)

        plt1.plot(range(epochs), train_accuracy)
        plt1.plot(range(epochs), valid_accuracy)
        plt1.set_title("model accuracy")
        plt1.set_ylabel("accuracy")
        plt1.set_xlabel("epoch")
        plt1.legend(["train", "validation"], loc="upper left")

        plt2.plot(range(epochs), train_loss)
        plt2.plot(range(epochs), valid_loss)
        plt2.set_title("model loss")
        plt2.set_ylabel("accuracy")
        plt2.set_xlabel("epoch")
        plt2.legend(["train", "validation"], loc="upper left")

        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    batch_size = 64
    learning_rate=1e-5
    training_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530]),  # Fashion MNIST-specific
    ])

    labled_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
    unlabelled_dataset = load_dataset("zalando-datasets/fashion_mnist", split="train")

    X_train, X_validate, y_train, y_validate = train_test_split(labled_dataset["image"], labled_dataset["label"], test_size=0.2)

    X_unlabelled, X_test, y_unlabelled, y_test = train_test_split(unlabelled_dataset["image"], unlabelled_dataset["label"], test_size=0.1)

    X_train, y_train = preprocess(X_train, y_train)
    X_validate, y_validate = preprocess(X_validate, y_validate)
    X_test, y_test = preprocess(X_test, y_test)
    X_unlabelled, y_unlabelled = preprocess(X_unlabelled, y_unlabelled)

    train_ds = Iterative_Dataset(X_train, y_train, training_transform)
    validate_ds = Iterative_Dataset(X_validate, y_validate)
    test_ds = Iterative_Dataset(X_test, y_test)
    unlabelled_ds = Iterative_Dataset(X_unlabelled, y_unlabelled)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    validate_dataloader = DataLoader(validate_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    unlabelled_dataloader = DataLoader(unlabelled_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    model = NeuralNetwork().to(device)
    try:
        #model.load_state_dict(torch.load("model.pth", weights_only=True))
        print("model loaded")
    except:
        print("model not found")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Do initial training
    init_train_accuracy = []
    init_train_loss = []
    init_valid_accuracy = []
    init_valid_loss = []
    epochs = 100
    for i in range(epochs):
        print(f"Epoch {i+1}\n--------------------------")
        train_accuracy, train_loss = train(train_dataloader, model, loss_func, optimizer, scaler)
        valid_accuracy, valid_loss = validate(validate_dataloader, model, loss_func)
        init_train_accuracy.append(train_accuracy)
        init_train_loss.append(train_loss)
        init_valid_accuracy.append(valid_accuracy)
        init_valid_loss.append(valid_loss)


    print("TESTING\n---------------------------")
    test(X_test, y_test, model, init_train_accuracy, init_valid_accuracy, init_train_loss, init_valid_loss, epochs)

    print("Applying Psuedo-Labels\n--------------------------")
    X_pseudo_labelled, pseudo_labels = apply_pseudo_labels(unlabelled_dataloader, model);

    pseudo_labelled_ds = Iterative_Dataset(X_pseudo_labelled, pseudo_labels)
    pseudo_labeled_dataloader = DataLoader(pseudo_labelled_ds, batch_size=batch_size, shuffle=True)

    print("\nTraining on Psuedo-Labelled data\n--------------------------")
    epoch_2 = 50
    for i in range(epoch_2):
        print(f"Epoch {i+1}\n--------------------------")
        train(pseudo_labeled_dataloader, model, loss_func, optimizer, scaler)
        validate(validate_dataloader, model, loss_func)

    print("TESTING\n---------------------------")
    test(X_test, y_test, model)

    torch.save(model.state_dict(), "model.pth")
    print("Saved model to model.pth")

