import torch
from torch import nn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer   
import pandas as pd

def preprocess(df_train, df_test, df_valid):
    feature_extract = TfidfVectorizer(stop_words='english', max_features=10_000)
    X_tr = df_train["text"].values
    X_te = df_test["text"].values
    X_va = df_valid["text"].values
    y_train =  df_train["label"].values
    y_test =  df_test["label"].values
    y_valid =  df_valid["label"].values
    X_train = feature_extract.fit_transform(X_tr)
    X_test = feature_extract.fit_transform(X_te)
    X_valid = feature_extract.fit_transform(X_va)
    return X_train, X_test, X_valid, y_train, y_test, y_valid


class NeuralNetwork(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(),
        )
if __name__=='__main__':
    ds_train = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    df_train = pd.DataFrame(ds_train)
    ds_test = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    df_test = pd.DataFrame(ds_test)
    ds_valid = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
    df_valid = pd.DataFrame(ds_valid)

    X_train, X_test, X_valid, y_train, y_test, y_valid = preprocess(df_train, df_test, df_valid)
    batch_size = 64

    

