import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, explained_variance_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from icecream import ic

def preprocess(X):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1_000)
    minmax = MaxAbsScaler()
    reducer = PCA(n_components=2)
    X_transformed = tfidf.fit_transform(X)
    X_tr = minmax.fit_transform(X_transformed)
    X_reduced = reducer.fit_transform(X_tr)
    return X_reduced

def apply_umap(X, target_components):
    #reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=target_components, random_state=42)
    reducer = PCA(n_components=target_components)
    embedding = reducer.fit_transform(X)
    return embedding

def visualize_data(X,y):
    twoD_df = apply_umap(X, 2)
    plt.scatter(
        twoD_df[:,0],
        twoD_df[:,1],
        c=y,
        cmap='viridis',
        alpha=0.6
    )
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.colorbar()
    plt.show()



ds = load_dataset("dair-ai/emotion", split="train")
X = ds["text"]
y = ds["label"]
X= preprocess(X)

visualize_data(X, y)
#kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, random_state=0)
#y_kmeans = kmeans.fit_predict(X)
dbs = DBSCAN(eps=0.001, min_samples=50, n_jobs=-1)
y_db = dbs.fit_predict(X)
visualize_data(X,y_db)

accuracies = []
k_values = range(1,10)
for i in k_values:
    #kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    #y_kmeans = kmeans.fit_predict(X)
    dbs = DBSCAN(eps=0.5, min_samples=i, n_jobs=-1)
    y_db = dbs.fit_predict(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y_db, test_size=0.2, random_state=42)
    
    classifier = RandomForestClassifier(max_depth=10)
    classifier.fit(X_train, y_train)
    y_label_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_label_pred)
    accuracies.append(accuracy)
    print(accuracy)

plt.figure(figsize=(8,5)) 
plt.plot(k_values, accuracies, marker='o', linestyle='--')
plt.ylabel("Classifier accuracies")
plt.xlabel("number of clusters")
plt.show()
