# Example code to add to your notebook for GMM tuning

# First, make sure your data is ready and scaled
X_test_scaled = scaler.transform(X_test)

# Option 1: Run the entire tuning process
from gmm_tuning import create_optimized_gmm
best_gmm = create_optimized_gmm(X_train_scaled, X_test_scaled)

# Option 2: Run specific parts of the tuning process
from gmm_tuning import tune_gmm_components, tune_gmm_covariance, tune_gmm_convergence

# Find the optimal number of components
best_n_components = tune_gmm_components(X_train_scaled, X_test_scaled)
print(f"Best number of components: {best_n_components}")

# Find the optimal covariance type
best_cov_type = tune_gmm_covariance(X_train_scaled, X_test_scaled, best_n_components)
print(f"Best covariance type: {best_cov_type}")

# Create and evaluate a model with the optimal parameters
optimized_gmm = GaussianMixture(
    n_components=best_n_components, 
    covariance_type=best_cov_type,
    random_state=42
)
optimized_gmm.fit(X_train_scaled)
optimized_predictions = optimized_gmm.predict(X_test_scaled)

# Calculate performance metrics
silhouette = silhouette_score(X_test_scaled, optimized_predictions)
davies_bouldin = davies_bouldin_score(X_test_scaled, optimized_predictions)
calinski_harabasz = calinski_harabasz_score(X_test_scaled, optimized_predictions)

print(f"Optimized GMM Performance:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

# Visualize the clusters (if your data is 2D or can be reduced to 2D)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce data to 2D for visualization
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test_scaled)

# Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=optimized_predictions, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'GMM Clustering with {best_n_components} components and {best_cov_type} covariance')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze cluster distributions
for i in range(best_n_components):
    cluster_points = X_test[optimized_predictions == i]
    if len(cluster_points) > 0:
        print(f"\nCluster {i} - Size: {len(cluster_points)} samples")
        print("Averages:")
        for col in X_test.columns:
            print(f"  {col}: {cluster_points[col].mean():.4f}") 