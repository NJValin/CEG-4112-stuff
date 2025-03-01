import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from icecream import ic

def preprocess(X):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5_000)
    X_transformed = tfidf.fit_transform(X)
    ic(X_transformed)
    return X_transformed

def apply_umap(X, target_components):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=target_components, random_state=42)
    embedding = reducer.fit_transform(X)
    return embedding

def visualize_data(X,y):
    twoD_df = apply_umap(X, 3)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(twoD_df.iloc[:,0], twoD_df.iloc[:,1], twoD_df.iloc[:,2], c=y, cmap='viridis')
    plt.show()

def preprocess(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    apply_umap(X,y)
    return X, y


ds = load_dataset("dair-ai/emotion", "split")
df = pd.DataFrame(ds['train'])
ic(df.head())

X = df['text'].values
y = df['label'].values
X = preprocess(X)
ic(X)
X_1 = apply_umap(X, 500)

#visualize_data(X, y)
accuracies = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(X_1)
    y_kmeans = kmeans.predict(X_1)
    accuracy = accuracy_score(y, y_kmeans)
    accuracies.append(accuracy)
    print(accuracy)


    print(accuracy)
plt.figure(figsize=(8,5)) 
plt.plot(range(1,21), accuracy, marker='o', linestyle='--')
plt.show()
