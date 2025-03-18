# GMM Parameter Tuning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def tune_gmm_components(X_train_scaled, X_test_scaled, n_components_range=range(2, 10)):
    """
    Find the optimal number of components for a GMM model
    """
    bic = []
    aic = []
    silhouette_scores = []

    for n_components in n_components_range:
        # Train GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X_train_scaled)
        
        # Predict clusters
        predictions = gmm.predict(X_test_scaled)
        
        # Calculate BIC and AIC
        bic.append(gmm.bic(X_train_scaled))
        aic.append(gmm.aic(X_train_scaled))
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_test_scaled, predictions)
        silhouette_scores.append(silhouette)
        
        print(f"Components: {n_components}, BIC: {bic[-1]:.2f}, AIC: {aic[-1]:.2f}, Silhouette: {silhouette:.4f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(n_components_range, bic, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('BIC by number of components')

    plt.subplot(1, 3, 2)
    plt.plot(n_components_range, aic, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('AIC')
    plt.title('AIC by number of components')

    plt.subplot(1, 3, 3)
    plt.plot(n_components_range, silhouette_scores, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by number of components')

    plt.tight_layout()
    plt.show()
    
    # Return the best number of components based on silhouette score
    return n_components_range[np.argmax(silhouette_scores)]

def tune_gmm_covariance(X_train_scaled, X_test_scaled, n_components):
    """
    Find the optimal covariance type for a GMM model
    """
    cov_types = ['full', 'tied', 'diag', 'spherical']
    results = []
    
    print(f"\nTesting covariance types with {n_components} components:")
    
    for cov_type in cov_types:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
        gmm.fit(X_train_scaled)
        predictions = gmm.predict(X_test_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_test_scaled, predictions)
        davies_bouldin = davies_bouldin_score(X_test_scaled, predictions)
        calinski_harabasz = calinski_harabasz_score(X_test_scaled, predictions)
        
        results.append({
            'covariance_type': cov_type,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'bic': gmm.bic(X_train_scaled),
            'aic': gmm.aic(X_train_scaled)
        })
        
        print(f"Covariance type: {cov_type}")
        print(f"  - Silhouette Score: {silhouette:.4f}")
        print(f"  - Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  - Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        print(f"  - BIC: {gmm.bic(X_train_scaled):.2f}")
        print(f"  - AIC: {gmm.aic(X_train_scaled):.2f}")
    
    # Find the best covariance type based on silhouette score
    best_result = max(results, key=lambda x: x['silhouette'])
    return best_result['covariance_type']

def tune_gmm_convergence(X_train_scaled, X_test_scaled, n_components, covariance_type):
    """
    Find the optimal convergence parameters for a GMM model
    """
    print("\nTesting convergence parameters:")
    
    best_silhouette = -1
    best_params = {}
    
    for n_init in [1, 5, 10]:
        for max_iter in [100, 200]:
            for tol in [1e-3, 1e-4, 1e-5]:
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_init=n_init,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=42
                    )
                    gmm.fit(X_train_scaled)
                    predictions = gmm.predict(X_test_scaled)
                    
                    silhouette = silhouette_score(X_test_scaled, predictions)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_params = {
                            'n_components': n_components,
                            'covariance_type': covariance_type,
                            'n_init': n_init,
                            'max_iter': max_iter,
                            'tol': tol
                        }
                    
                    print(f"n_init={n_init}, max_iter={max_iter}, tol={tol}, Silhouette={silhouette:.4f}")
                except Exception as e:
                    print(f"Error with n_init={n_init}, max_iter={max_iter}, tol={tol}: {str(e)}")
    
    print("\nBest GMM parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    return best_params

def create_optimized_gmm(X_train_scaled, X_test_scaled):
    """
    Create an optimized GMM model
    """
    # 1. Find optimal number of components
    best_n_components = tune_gmm_components(X_train_scaled, X_test_scaled)
    
    # 2. Find optimal covariance type
    best_cov_type = tune_gmm_covariance(X_train_scaled, X_test_scaled, best_n_components)
    
    # 3. Find optimal convergence parameters
    best_params = tune_gmm_convergence(X_train_scaled, X_test_scaled, best_n_components, best_cov_type)
    
    # 4. Create final model
    final_gmm = GaussianMixture(**best_params)
    final_gmm.fit(X_train_scaled)
    final_predictions = final_gmm.predict(X_test_scaled)
    
    print("\nFinal model performance:")
    print(f"Silhouette Score: {silhouette_score(X_test_scaled, final_predictions):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_test_scaled, final_predictions):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_test_scaled, final_predictions):.4f}")
    
    return final_gmm

# Example usage:
# X_test_scaled = scaler.transform(X_test)
# best_gmm = create_optimized_gmm(X_train_scaled, X_test_scaled) 