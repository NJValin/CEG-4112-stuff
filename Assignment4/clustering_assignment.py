import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from umap import UMAP

def preprocess(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_umap(X):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)
    return pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

def visualize_data(X,y):
    twoD_df = apply_umap(X)
    plt.scatter(twoD_df.iloc[:,0], twoD_df.iloc[:,1], c=y, cmap='viridis')
    plt.show()



ds = load_dataset("lvwerra/red-wine")

df = pd.DataFrame(ds['train'])
print(df.head())

X = df.drop(columns=['quality'])
y = df['quality']
#X = preprocess(X)
visualize_data(X, y.values)
accuracies = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    accuracy = accuracy_score(y, y_kmeans)
    accuracies.append(accuracy)
    print(accuracy)

plt.plot(range(1, 10), accuracies, marker='o')
plt.ylabel("Accuracies")
plt.xlabel("minimum samples to be a cluster")
plt.grid()
plt.show()
