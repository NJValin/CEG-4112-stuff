import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from umap import UMAP

def apply_umap(X,y):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    embedding = reducer.fit_transform(X)
    df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(
        df['UMAP_1'],
        df['UMAP_2'],
        c=y,
        cmap=['r', 'g', 'b'],
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label="Activity Label")
    plt.title("UMAP Embedding of Human Activity Data")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()

def preprocess(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    apply_umap(X,y)
    return X, y


    

ds = load_dataset("katossky/wine-recognition")
df = pd.DataFrame(ds['train'])
X = df.drop(columns=["label"])
y = df['label']
X, y = preprocess(X,y)

accuracy = []

for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    accuracy.append(accuracy_score(y, y_kmeans))


    print(accuracy)
plt.figure(figsize=(8,5)) 
plt.plot(range(1,21), accuracy, marker='o', linestyle='--')
plt.show()
