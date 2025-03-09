import torch
import torch.nn as nn
from datasets import load_dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def preprocess(X):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    X = transform(X)
    return X

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Convolution Layers

if __name__=='__main__':

    index = 5
    train_dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    test_dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")

