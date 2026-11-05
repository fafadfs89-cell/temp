
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# PART 1: K-Means Clustering
X, y = make_blobs(n_samples=300, centers=3,
                  cluster_std=0.6, random_state=42)
print("K-Means Dataset:\n", pd.DataFrame(X).head())

# 1. Elbow Method
wcss = [KMeans(n_clusters=i, n_init=10, random_state=42)
        .fit(X).inertia_ for i in range(1, 11)]

plt.plot(range(1, 11), wcss, 'o--')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 2. K-Means Clustering
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='viridis')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
            c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()

print("Silhouette Score:", round(silhouette_score(X, labels), 3))

# PART 2: PCA
X, y = make_blobs(n_samples=500, n_features=4,
                  centers=4, random_state=25)

df = pd.DataFrame(X, columns=['F1','F2','F3','F4'])
df['True_Cluster'] = y
print("Original PCA Dataset:\n", df.head())
print("Original Shape:", df.shape)

X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

print("PCA Data:\n", pd.DataFrame(X2).head())
print("Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance:", round(sum(pca.explained_variance_ratio_), 3))

# 3. PCA Visualization
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y, palette='Paired')
plt.title("PCA - Dimensionality Reduction")
plt.show()

# 4. K-Means on PCA Data
km2 = KMeans(n_clusters=4, n_init=10, random_state=42)
labels2 = km2.fit_predict(X2)

sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels2, palette='viridis')
plt.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1],
            c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means on PCA-Reduced Data")
plt.legend()
plt.show()

print("PCA Silhouette Score:", round(silhouette_score(X2, labels2), 3))
