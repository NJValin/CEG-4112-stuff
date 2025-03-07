import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Text preprocessing function
def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class TextDataProcessor:
    def __init__(self, max_features=10000, min_df=5, max_df=0.9):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        self.label_encoder = LabelEncoder()
        
    def fit(self, texts, labels):
        """Fit the vectorizer and label encoder"""
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Fit the TF-IDF vectorizer
        self.vectorizer.fit(processed_texts)
        
        # Fit the label encoder
        self.label_encoder.fit(labels)
        
        return self
    
    def transform_texts(self, texts):
        """Transform texts to TF-IDF features"""
        processed_texts = [preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def transform_labels(self, labels):
        """Transform labels to numerical form"""
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, encoded_labels):
        """Convert numerical labels back to original form"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    @property
    def vocab_size(self):
        return len(self.vectorizer.get_feature_names_out())
    
    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)


class TfidfDataset(Dataset):
    def __init__(self, texts, labels, processor):
        # Transform texts to TF-IDF features
        self.features = processor.transform_texts(texts)
        
        # Transform labels
        self.labels = processor.transform_labels(labels)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        # Convert sparse matrix row to dense numpy array
        feature_row = self.features[idx].toarray().flatten().astype(np.float32)
        return torch.tensor(feature_row), torch.tensor(self.labels[idx], dtype=torch.long)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(SimpleClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


def prepare_data_loaders(texts, labels, processor, batch_size=16, test_size=0.2, random_state=42):
    """Prepare train and validation DataLoaders"""
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Create datasets
    train_dataset = TfidfDataset(train_texts, train_labels, processor)
    val_dataset = TfidfDataset(val_texts, val_labels, processor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, lr=0.001, epochs=10, device='cpu'):
    """Train the model"""
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return model


def predict(model, texts, processor, device='cpu'):
    """Make predictions for new texts"""
    # Transform texts
    features = processor.transform_texts(texts)
    
    # Convert to tensor
    inputs = torch.tensor(features.toarray(), dtype=torch.float32).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to original labels
    return processor.inverse_transform_labels(predicted.cpu().numpy())


# Complete example usage
def run_complete_example():
    # Example data (IMDB-like sentiment analysis)
    texts = [
        "This movie was fantastic! I really enjoyed every moment of it.",
        "The acting was terrible and the plot made no sense whatsoever.",
        "I loved the characters and the storyline was incredibly engaging.",
        "Worst film I've seen all year, complete waste of time and money.",
        "The special effects were amazing, but the dialogue was weak.",
        "A masterpiece of modern cinema, thoughtful and beautifully shot.",
        "I fell asleep halfway through, it was that boring.",
        "The director's vision really shines through in every scene.",
        "Predictable plot with cardboard characters, don't bother watching.",
        "A thrilling ride from start to finish, couldn't look away."
    ]
    
    labels = [
        "positive", "negative", "positive", "negative", "neutral",
        "positive", "negative", "positive", "negative", "positive"
    ]
    
    # Initialize processor
    processor = TextDataProcessor(max_features=1000, min_df=1)
    
    # Fit processor
    processor.fit(texts, labels)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(texts, labels, processor, batch_size=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SimpleClassifier(
        input_dim=processor.vocab_size,
        hidden_dim=64,
        output_dim=processor.num_classes
    )
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        epochs=5,
        device=device
    )
    
    # Make predictions on new texts
    new_texts = [
        "I thought this was a really good movie overall.",
        "The worst experience I've had in a theater."
    ]
    
    predictions = predict(model, new_texts, processor, device)
    
    for text, prediction in zip(new_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print()
    
    return model, processor

if __name__ == "__main__":
    run_complete_example()
