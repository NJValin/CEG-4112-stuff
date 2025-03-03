import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
dataset = load_dataset("emotion", split="train")
texts = dataset["text"]
labels = dataset["label"]  # We won't use these for clustering, only for supervised learning

# Step 2: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(texts)

# Step 3: Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Step 4: Determine optimal clusters using accuracy
k_values = range(2, 10)
accuracy_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Train a supervised model to predict clusters
    X_train, X_test, y_train, y_test = train_test_split(X, clusters, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot accuracy vs. number of clusters
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Accuracy")
plt.title("Optimal K using Accuracy")
plt.show()

# Step 5: Apply K-Means with optimal clusters (choosing K with highest accuracy)
best_k = k_values[np.argmax(accuracy_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Step 6: Visualize Clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title(f"K-Means Clustering (K={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

# Step 7: Train a supervised model to predict clusters
X_train, X_test, y_train, y_test = train_test_split(X, clusters, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Step 8: Evaluate the classification model
print("Optimal K:", best_k)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
