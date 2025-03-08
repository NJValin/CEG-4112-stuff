import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

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
    torch.save({"X": X_tensor, "y": y_tensor}, save_path)
    print(f"Saved preprocessed data to {save_path}")

    return X_tensor, y_tensor

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model

# Example Usage:
# Assuming you have your text data in a list called 'texts' and binary labels in 'labels'

# Example data (replace with your actual data)
texts = ["This is a positive review.", "This is a negative review.", "I loved this movie.", "I hated this movie.", "Great product!", "Terrible service."]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

X, y = preprocess(texts, labels)
input_size = X.shape[1]
model = BinaryClassifier(input_size)

trained_model = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Evaluate on test set
trained_model.eval()
with torch.no_grad():
    test_outputs = trained_model(X_test_tensor).squeeze()
    test_preds = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())
    print(f"Test Accuracy: {test_accuracy:.4f}")
