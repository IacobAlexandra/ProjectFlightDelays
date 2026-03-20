import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import umap

# =========================================================
# DATA PREPARATION
# =========================================================
"""
# Take a 10k sample
df_features = df.sample(n=10000, random_state=42).copy()

# Create the MONTH feature from FL_DATE (datetime format) for the sample
df_features['MONTH'] = df_features['FL_DATE'].astype(str).str[5:7].astype(int)

# Create explicit text labels for the legend
df_features['Flight Status'] = np.where(df_features['ARR_DELAY'] > 15, 'Delayed (>15m)', 'On-Time')

# Select only numeric features known BEFORE departure to prevent data leakage
cols_to_scale = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME', 'MONTH']
"""

# =========================================================
# 1. DATA PREPARATION FUNCTION
# =========================================================
def prepare_data_for_dr(df):
    #Prepares, samples, and scales the data for dimensionality reduction.

    # Take a sample to prevent memory/performance issues
    df_sample = df.sample(n=10000, random_state=42).copy()

    # Create explicit text labels for the legend
    df_sample['Flight Status'] = np.where(df_sample['ARR_DELAY'] > 15, 'Delayed (>15m)', 'On-Time')

    # Include all newly engineered features!
    cols_to_scale = [
        'CRS_DEP_TIME', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME', 'MONTH',
        'SCHEDULED_SPEED',
        'ORIGIN_CONGESTION',
        'DESTINATION_CONGESTION',
        'ROUTE_BOTTLENECK_INTERACTION',
        'ROUTE_FREQUENCY',
        'AIRLINE_HUB_DOMINANCE'
    ]

    # Keep only the rows without NaNs to avoid PCA/UMAP errors
    df_features = df_sample.dropna(subset=cols_to_scale + ['Flight Status']).copy()

    # SCALING: Using RobustScaler to handle extreme delay outliers perfectly
    from sklearn.preprocessing import RobustScaler
    x_scaled = RobustScaler().fit_transform(df_features[cols_to_scale])

    # ---> THIS IS THE LINE THAT FIXES YOUR ERROR! <---
    return x_scaled, df_features, cols_to_scale


def apply_global_scaling(df, target_col='ARR_DELAY', additional_excludes=None):
    """
    Applies Robust Scaling globally to all numerical predictor features.
    Explicitly excludes the target variable and non-numeric columns.
    """
    exclude_cols = [target_col]
    if additional_excludes:
        exclude_cols.extend(additional_excludes)

    # THE FIX: Universally select all number types
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]

    print(f"Applying scaling on: {len(cols_to_scale)}")

    scaler = RobustScaler()
    df_scaled = df.copy()

    # THE FIX: Temporarily fill NaNs with 0 just for the scaling math so it doesn't crash
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale].fillna(0))

    return df_scaled

# =========================================================
# 2. PCA FUNCTION
# =========================================================
def run_and_plot_pca(x_scaled, df_features, cols_to_scale):
    #Runs Linear PCA and generates the variance, projection, and loading plots.

    pca = PCA()
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
    plt.xticks(rotation=45, ha='right')  # Rotate x-labels to fit the new longer feature names cleanly
    plt.tight_layout()
    plt.show()


# =========================================================
# 3. UMAP FUNCTION
# =========================================================
def run_and_plot_umap(x_scaled, df_features):
    #Runs Non-Linear UMAP and generates the 2D projection plot.

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
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


# =========================================================
# 4. MAIN EXECUTION CALL
# =========================================================
def execute_dimensionality_reduction(df):
    #Main execution function to run the entire Phase 3 DR pipeline.

    # 1. Prepare and scale the data
    x_scaled, df_features, cols_to_scale = prepare_data_for_dr(df)

    # 2. Run and plot PCA
    run_and_plot_pca(x_scaled, df_features, cols_to_scale)

    # 3. Run and plot UMAP
    run_and_plot_umap(x_scaled, df_features)


file_path = '../../Project_Datasets/removed_outliers_data.csv'
df = pd.read_csv(file_path)

# Run the entire dimensionality reduction pipeline (Scaling, PCA, and UMAP)
execute_dimensionality_reduction(df)

output_filename = '../Project_Datasets/scaled_data.csv'
df.to_csv(output_filename, index=False)