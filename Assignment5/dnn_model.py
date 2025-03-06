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
    umaper = UMAP(n_neighbors=15, min_dist=0.1, n_components=512, random_state=42)
    X_transformed = feature_extraction.fit_transform(X).toarray()
    X_Reduced = umaper.fit_transform(X_transformed)
    X_tensor = torch.tensor(X_Reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y)
    return X_tensor, nn.functional.one_hot(y_tensor, 2)

class TextDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim)->None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
        )
if __name__=='__main__':
    batch_size = 64
    ds_train = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    X_train, y_train = preprocess(ds_train["text"], ds_train["label"])
    train_dataset = TextDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    for i in train_dataloader:
        print(i[0].shape)
        print(i)
        break

    ds_test = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    X_test, y_test = preprocess(ds_test["text"], ds_test["label"])
    test_dataset = TextDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    ds_valid = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
    X_valid, y_valid = preprocess(ds_valid["text"], ds_valid["label"])
    valid_dataset = TextDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True)
