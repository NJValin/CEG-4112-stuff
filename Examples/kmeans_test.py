import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from icecream import ic

seed = 41

X, y = make_blobs(n_samples=300, centers=6, cluster_std=0.7, random_state=seed)

kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=seed)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)
centres = kmeans.cluster_centers_
ic(centres)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centres[:, 0], centres[:, 1], c='black', s=200, alpha=0.5);
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
