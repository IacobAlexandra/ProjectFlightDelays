import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import umap

# ---------------------------------------
# DATA PREPARATION
# ---------------------------------------
def prepare_data_for_dr(df):
    "Prepares, samples and scales the data for dimensionality reduction"

    # Take a sample to avoid memory and performance issues
    df_sample = df.sample(n=10000, random_state=42).copy()

    # Create text labels for the legend
    df_sample['Flight Status'] = np.where(df_sample['ARR_DELAY'] > 15, 'Delayed (>15m)', 'On-Time')

    # Select the columns to scale
    cols_to_scale = [
        'CRS_DEP_TIME', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME', 'MONTH',
        'SCHEDULED_SPEED',
        'ORIGIN_CONGESTION',
        'DESTINATION_CONGESTION',
        'ROUTE_BOTTLENECK_INTERACTION',
        'ROUTE_FREQUENCY',
        'AIRLINE_HUB_DOMINANCE'
    ]

    # Create a copy for PCA/UMAP
    df_features = df_sample.copy()

    # Use RobustScaler to handle extreme delay outliers
    from sklearn.preprocessing import RobustScaler
    df = RobustScaler().fit_transform(df_features[cols_to_scale])
    x_scaled = RobustScaler().fit_transform(df_features[cols_to_scale])

    return x_scaled, df_features, cols_to_scale

# ---------------------------------------
# PCA
# ---------------------------------------
def run_and_plot_pca(x_scaled, df_features, cols_to_scale):
    """Runs PCA and generates the variance, projection, and loading plots."""

    # run PCA
    pca = PCA()

    # Fit the PCA model and compress the scaled data into principal components
    pca_result = pca.fit_transform(x_scaled)

    # Plot 1: Cumulative Explained Variance
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_),
             marker='o', linestyle='-', color='dodgerblue')
    plt.axhline(y=0.80, color='r', linestyle='--', label='80% Information Retained')
    plt.title('Data Compression: How Much Flight Information is Preserved by PCA?', fontsize=14, pad=15)
    plt.xlabel('Number of Principal Components (Compressed Features)')
    plt.ylabel('Percentage of Original Flight Data Retained')
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.35))
    sns.despine()
    plt.show()
    

    # Plot 2: PCA 2D Projection
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                    hue=df_features['Flight Status'].values,
                    palette={'On-Time': 'lightseagreen', 'Delayed (>15m)': 'crimson'},
                    alpha=0.5, s=15, linewidth=0)
    plt.title('Linear Dimensionality Reduction: PCA (2D Projection)', fontsize=14, pad=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Flight Outcome', loc='upper right')
    sns.despine()
    plt.show()
    

    # Plot 3: PCA Component Loadings (Heatmap)
    loadings = pd.DataFrame(
        pca.components_,
        columns=cols_to_scale,
        index=[f'PC{i + 1}' for i in range(pca.n_components_)]
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings.head(3), annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('PCA Component Loadings: Feature Contributions to Top 3 Components', fontsize=14, pad=15)
    plt.ylabel('Principal Components')
    plt.xlabel('Original Flight Features')
    plt.xticks(rotation=45, ha='right')  # Rotate x-labels to fit the feature names
    plt.tight_layout()
    plt.show()
    


# ---------------------------------------
# UMAP
# ---------------------------------------
def run_and_plot_umap(x_scaled, df_features):
    """Runs UMAP and generates the 2D projection plot."""

    # run UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    # Fit the UMAP model to build a neighborhood graph and compress the data into a non-linear 2D map
    umap_result = reducer.fit_transform(x_scaled)

    # Plot 4: UMAP 2D Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1],
                    hue=df_features['Flight Status'].values,
                    palette={'On-Time': 'lightseagreen', 'Delayed (>15m)': 'crimson'},
                    alpha=0.5, s=15, linewidth=0)
    plt.title('Non-Linear Dimensionality Reduction: UMAP', fontsize=14, pad=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Flight Outcome', loc='upper right')
    sns.despine()
    plt.show()
    


# ---------------------------------------
# DIMENSIONALITY REDUCTION PIPELINE
# ---------------------------------------
def execute_dimensionality_reduction(df):
    "Main execution function to run the entire DR pipeline"

    # 1. Data Preparation
    x_scaled, df_features, cols_to_scale = prepare_data_for_dr(df)

    # 2. PCA
    run_and_plot_pca(x_scaled, df_features, cols_to_scale)

    # 3. UMAP
    run_and_plot_umap(x_scaled, df_features)
