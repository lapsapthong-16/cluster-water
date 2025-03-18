import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy import linalg
import matplotlib.cm as cm

def plot_2d_clusters(X, labels, method='PCA', title=None, figsize=(12, 10)):
    """
    Visualize clusters in 2D using PCA or t-SNE.
    
    Parameters:
    -----------
    X : array-like
        The original feature data (scaled)
    labels : array-like
        Cluster labels from GMM prediction
    method : str, default='PCA'
        Dimensionality reduction method ('PCA' or 'TSNE')
    title : str, optional
        Plot title
    figsize : tuple, default=(12, 10)
        Figure size
    """
    n_clusters = len(np.unique(labels))
    title = title or f'GMM Clustering with {n_clusters} components ({method})'
    
    plt.figure(figsize=figsize)
    
    # Reduce dimensionality to 2D
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%} variance)'
        ylabel = f'PC2 ({explained_var[1]:.1%} variance)'
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(40, len(X)//10))
        X_2d = reducer.fit_transform(X)
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
    else:
        raise ValueError("Method must be either 'PCA' or 'TSNE'")
    
    # Plot points colored by cluster
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', 
                alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    
    return X_2d, reducer

def plot_3d_clusters(X, labels, method='PCA', figsize=(14, 12)):
    """
    Visualize clusters in 3D using PCA or t-SNE
    """
    n_clusters = len(np.unique(labels))
    
    # Reduce dimensionality to 3D
    if method.upper() == 'PCA':
        reducer = PCA(n_components=3)
        X_3d = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%})'
        ylabel = f'PC2 ({explained_var[1]:.1%})'
        zlabel = f'PC3 ({explained_var[2]:.1%})'
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(40, len(X)//10))
        X_3d = reducer.fit_transform(X)
        xlabel, ylabel, zlabel = 't-SNE 1', 't-SNE 2', 't-SNE 3'
    else:
        raise ValueError("Method must be either 'PCA' or 'TSNE'")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map
    cmap = cm.get_cmap('viridis', n_clusters)
    
    # Plot each cluster
    for cluster in range(n_clusters):
        cluster_points = X_3d[labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                  color=cmap(cluster/n_clusters), s=50, alpha=0.7, label=f'Cluster {cluster}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(f'3D Visualization of GMM Clusters ({method})', fontsize=14)
    ax.legend()
    
    return X_3d, reducer

def plot_cluster_distributions(X_df, labels, n_cols=3, figsize=(18, 15)):
    """
    Create boxplots showing feature distributions across clusters
    
    Parameters:
    -----------
    X_df : pandas DataFrame
        Original data with feature names
    labels : array-like
        Cluster labels from GMM prediction
    n_cols : int, default=3
        Number of columns in the subplot grid
    """
    df = X_df.copy()
    df['Cluster'] = labels
    
    features = X_df.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'{feature} by Cluster', fontsize=12)
        axes[i].set_xlabel('Cluster', fontsize=10)
        axes[i].set_ylabel(feature, fontsize=10)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Feature Distributions Across Clusters', fontsize=16, y=1.02)
    
    return fig

def plot_feature_correlations_by_cluster(X_df, labels, selected_features=None):
    """
    Create correlation heatmaps for each cluster
    
    Parameters:
    -----------
    X_df : pandas DataFrame
        Original data with feature names
    labels : array-like
        Cluster labels from GMM prediction
    selected_features : list, optional
        Subset of features to include
    """
    df = X_df.copy()
    df['Cluster'] = labels
    n_clusters = len(np.unique(labels))
    
    features = selected_features or X_df.columns
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]
    
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i][features]
        corr = cluster_data.corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   center=0, linewidths=.5, ax=axes[i], fmt='.2f', cbar=False)
        axes[i].set_title(f'Cluster {i} Correlations', fontsize=14)
    
    plt.tight_layout()
    plt.suptitle('Feature Correlations Within Each Cluster', fontsize=16, y=1.02)
    
    return fig

def plot_gmm_covariance_ellipses(X, gmm, feature_names=None, figsize=(12, 10)):
    """
    Visualize GMM clusters with covariance ellipses
    
    Parameters:
    -----------
    X : array-like
        The original feature data (scaled)
    gmm : GaussianMixture model
        Fitted GMM model
    feature_names : list, optional
        Names of the features to use for 2D visualization (must be exactly 2)
    """
    if feature_names and len(feature_names) == 2:
        # Use specified features
        if isinstance(X, pd.DataFrame):
            X_2d = X[feature_names].values
            xlabel, ylabel = feature_names
        else:
            raise ValueError("If feature_names is provided, X must be a DataFrame")
    else:
        # Use PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
    
    plt.figure(figsize=figsize)
    
    # Plot data points
    labels = gmm.predict(X)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.5, s=40)
    
    # Plot ellipses
    colors = cm.viridis(np.linspace(0, 1, gmm.n_components))
    
    if feature_names and len(feature_names) == 2:
        # For 2 specific features, get the covariance submatrix
        indices = [X.columns.get_loc(name) for name in feature_names]
        for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
            mean_2d = mean[indices]
            if gmm.covariance_type == 'full':
                covar_2d = covar[np.ix_(indices, indices)]
            elif gmm.covariance_type == 'tied':
                covar_2d = covar[np.ix_(indices, indices)]
            elif gmm.covariance_type == 'diag':
                covar_2d = np.diag(covar[indices])
            else:  # 'spherical'
                covar_2d = np.eye(2) * covar
            
            plot_ellipse(mean_2d, covar_2d, color, alpha=0.4, label=f'Cluster {i}')
    else:
        # For PCA, project the means and covariances
        for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
            mean_2d = pca.transform([mean])[0]
            
            # Transform covariance matrix to PCA space
            if gmm.covariance_type == 'full':
                covar_2d = pca.components_ @ covar @ pca.components_.T
                covar_2d = covar_2d[:2, :2]
            elif gmm.covariance_type == 'tied':
                covar_2d = pca.components_ @ covar @ pca.components_.T
                covar_2d = covar_2d[:2, :2]
            elif gmm.covariance_type == 'diag':
                covar_pca = pca.components_ @ np.diag(covar) @ pca.components_.T
                covar_2d = covar_pca[:2, :2]
            else:  # 'spherical'
                covar_2d = np.eye(2) * covar
            
            plot_ellipse(mean_2d, covar_2d, color, alpha=0.4, label=f'Cluster {i}')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'GMM Clustering with Covariance Ellipses ({gmm.covariance_type})', fontsize=14)
    plt.legend()
    plt.tight_layout()

def plot_ellipse(mean, covariance, color, alpha=0.5, label=None):
    """Helper function to plot an ellipse representing a covariance matrix"""
    v, w = linalg.eigh(covariance)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    
    for nsig in [1, 2, 3]:
        ell = Ellipse(mean, v[0] * nsig, v[1] * nsig,
                         180. + angle, color=color, alpha=alpha/nsig)
        ell.set_clip_box(plt.gca().bbox)
        ell.set_alpha(alpha/nsig)
        plt.gca().add_artist(ell)
        
    # Only add label for the outermost ellipse
    if label:
        plt.plot(mean[0], mean[1], 'o', markersize=8, markerfacecolor=color, 
                 markeredgecolor='k', label=label)
    else:
        plt.plot(mean[0], mean[1], 'o', markersize=8, markerfacecolor=color, 
                 markeredgecolor='k')

def plot_temporal_cluster_distribution(df, labels, figsize=(14, 8)):
    """
    Visualize how clusters are distributed over time
    
    Parameters:
    -----------
    df : pandas DataFrame
        Original data with a 'Timestamp' column
    labels : array-like
        Cluster labels from GMM prediction
    """
    if 'Timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'Timestamp' column")
    
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['Timestamp']):
        df_copy['Timestamp'] = pd.to_datetime(df_copy['Timestamp'])
    
    # Add time-based columns
    df_copy['Month'] = df_copy['Timestamp'].dt.month
    df_copy['Day'] = df_copy['Timestamp'].dt.day
    df_copy['Hour'] = df_copy['Timestamp'].dt.hour
    
    # Monthly distribution
    plt.figure(figsize=figsize)
    month_counts = pd.crosstab(df_copy['Month'], df_copy['Cluster'])
    month_counts.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Monthly Distribution of Clusters', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.figure()
    
    # Daily distribution (by hour)
    hour_counts = pd.crosstab(df_copy['Hour'], df_copy['Cluster'])
    hour_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=figsize)
    plt.title('Hourly Distribution of Clusters', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Cluster')
    plt.tight_layout()

def plot_water_specific_visualizations(df, labels, figsize=(12, 10)):
    """
    Create water-data specific visualizations for GMM clusters
    
    Parameters:
    -----------
    df : pandas DataFrame
        Water data with features like 'Temperature', 'Dissolved Oxygen', etc.
    labels : array-like
        Cluster labels from GMM prediction
    """
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    # 1. Temperature vs Dissolved Oxygen by cluster
    plt.figure(figsize=figsize)
    sns.scatterplot(x='Temperature', y='Dissolved Oxygen', 
                   hue='Cluster', data=df_copy, palette='viridis', 
                   s=60, alpha=0.7)
    plt.title('Temperature vs Dissolved Oxygen by Cluster', fontsize=14)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Dissolved Oxygen (mg/L)', fontsize=12)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.figure()
    
    # 2. Water Speed vs Direction by cluster (polar plot)
    if 'Average Water Speed' in df.columns and 'Average Water Direction' in df.columns:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        clusters = np.unique(labels)
        for cluster in clusters:
            cluster_data = df_copy[df_copy['Cluster'] == cluster]
            # Convert to radians for polar plot
            directions_rad = np.radians(cluster_data['Average Water Direction'])
            ax.scatter(directions_rad, cluster_data['Average Water Speed'], 
                      label=f'Cluster {cluster}', alpha=0.7)
        
        ax.set_theta_zero_location('N')  # 0 degrees at North
        ax.set_theta_direction(-1)  # clockwise
        ax.set_title('Water Speed and Direction by Cluster', fontsize=14)
        ax.set_ylabel('Water Speed', labelpad=20)
        ax.legend(title='Cluster')
        plt.tight_layout()
        plt.figure()
    
    # 3. pH vs Chlorophyll by cluster
    if 'pH' in df.columns and 'Chlorophyll' in df.columns:
        plt.figure(figsize=figsize)
        sns.scatterplot(x='pH', y='Chlorophyll', 
                       hue='Cluster', data=df_copy, palette='viridis', 
                       s=60, alpha=0.7)
        plt.title('pH vs Chlorophyll by Cluster', fontsize=14)
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('Chlorophyll', fontsize=12)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.figure()
    
    # 4. Salinity vs Specific Conductance
    if 'Salinity' in df.columns and 'Specific Conductance' in df.columns:
        plt.figure(figsize=figsize)
        sns.scatterplot(x='Salinity', y='Specific Conductance', 
                       hue='Cluster', data=df_copy, palette='viridis', 
                       s=60, alpha=0.7)
        plt.title('Salinity vs Specific Conductance by Cluster', fontsize=14)
        plt.xlabel('Salinity', fontsize=12)
        plt.ylabel('Specific Conductance', fontsize=12)
        plt.legend(title='Cluster')
        plt.tight_layout()

# Example usage in notebook:
# 
# # First, obtain your GMM model and predictions
# optimized_gmm = GaussianMixture(n_components=best_n_components, covariance_type=best_cov_type)
# optimized_gmm.fit(X_train_scaled)
# optimized_predictions = optimized_gmm.predict(X_test_scaled)
# 
# # Then use these visualization functions
# plot_2d_clusters(X_test_scaled, optimized_predictions, method='PCA')
# plot_3d_clusters(X_test_scaled, optimized_predictions, method='PCA')
# plot_cluster_distributions(X_test, optimized_predictions)
# plot_gmm_covariance_ellipses(X_test_scaled, optimized_gmm)
# plot_temporal_cluster_distribution(X_test.reset_index(), optimized_predictions)
# plot_water_specific_visualizations(X_test, optimized_predictions) 