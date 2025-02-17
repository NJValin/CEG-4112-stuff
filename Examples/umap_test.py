import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from faker import Faker

def create_synthetic_dataset():
    seed=42
    """
    make_classification generates a n-class dataset for classification models

    """
    X, y = make_classification(
        n_samples=10_000, # number of samples
        n_features=20, # number of features within a datapoint
        n_informative=10, # number of informative features
        n_redundant=5, # number of irrelevant features.
        n_clusters_per_class=1,
        n_classes=5, # The number of output classes (or labels) 
        random_state=seed,
        hypercube=True
    )
    fake = Faker()
    feature_names = [f'Visit Frequency: {fake.company()}' for i in range(X.shape[1])]
    input_scalar = MinMaxScaler()
    return pd.DataFrame(np.abs(input_scalar.fit_transform(X)), columns=feature_names), pd.Series(y, name="Activity_Label")

def preprocess_data(X):
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def apply_umap(X, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(X)
    return pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

def visualize_umap(embedding, labels):
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(
        embedding["UMAP_1"],
        embedding["UMAP_2"],
        c=labels,
        s=10,
        cmap="Spectral",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Activity Label")
    plt.title("UMAP Embedding of Human Activity Data")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()


X, y = create_synthetic_dataset()

X_scaled = preprocess_data(X)

umap_embedding = apply_umap(X_scaled)

visualize_umap(umap_embedding, y)

X["Activity_Label"]=y
X.head()
