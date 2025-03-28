import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import time

def preprocess(X : pd.DataFrame, y : pd.DataFrame):
    #minmax = StandardScaler()
    reducer = PCA(n_components=2)
    encoder = LabelEncoder()
    X["explicit"] = X["explicit"].astype(int)
    #X_transformed = minmax.fit_transform(X)
    y_transformed = encoder.fit_transform(y)
    #X_tr = minmax.fit_transform(X_transformed)
    print(time.time())
    X_reduced = reducer.fit_transform(X)
    print(time.time())
    return X_reduced, y_transformed

def apply_umap(X, target_components):
    #reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=target_components, random_state=42)
    reducer = PCA(n_components=target_components)
    embedding = reducer.fit_transform(X)
    return embedding

def visualize_data(X,y, centres=None):
    twoD_df = apply_umap(X, 2)
    plt.scatter(
        twoD_df[:,0],
        twoD_df[:,1],
        c=y,
        cmap='viridis',
        alpha=0.6
    )
    if centres is not None:
        plt.scatter(centres[:, 0], centres[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    time_start = time.time()
    print(time_start)
    ds = load_dataset("maharshipandya/spotify-tracks-dataset")
    df = pd.DataFrame(ds["train"])

    print(f"pre define X {time.time()-time_start}")
    X = df[["explicit", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "time_signature"]]
    print(f"post define X {time.time()-time_start}")
    y = df["track_genre"]
    print(f"pre preprocess {time.time()-time_start}")
    X, y= preprocess(X,y)
    print(f"post preprocess {time.time()-time_start}")

    visualize_data(X, y)
    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    centres = kmeans.cluster_centers_
    #dbs = DBSCAN(eps=0.5, min_samples=5)
    #y_db = dbs.fit_predict(X)
    visualize_data(X,y_kmeans, centres)

    accuracies = []
    k_values = range(1,10)
    for i in k_values:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
        y_kmeans = kmeans.fit_predict(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y_kmeans, test_size=0.2, random_state=42)
        
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
