library(dplyr)
library(ggplot2)
library(umap)
library(reshape2)

### ---------------------------------------
### DATA PREPARATION
### ---------------------------------------
prepare_data_for_dr <- function(df) {
    # "Prepares, samples and scales the data for dimensionality reduction"

    # Safely sample the dataset (UMAP is computationally heavy on massive datasets)
    sample_size <- min(15000, nrow(df))
    set.seed(42)
    df_sample <- df[sample(nrow(df), sample_size), ]

    # Keep only numeric columns for DR algorithms
    df_features <- df_sample %>% select(where(is.numeric))
    cols_to_scale <- colnames(df_features)

    # Custom RobustScaler implementation (matches sklearn: median removal & IQR scaling)
    robust_scale <- function(x) {
        iqr_val <- IQR(x, na.rm = TRUE)
        if(iqr_val == 0) return(x - median(x, na.rm = TRUE)) # Avoid division by zero
        return((x - median(x, na.rm = TRUE)) / iqr_val)
    }

    x_scaled <- as.data.frame(lapply(df_features, robust_scale))

    return(list(x_scaled = x_scaled, df_features = df_features, cols_to_scale = cols_to_scale))
}

### ---------------------------------------
### PCA
### ---------------------------------------
run_and_plot_pca <- function(x_scaled, df_features, cols_to_scale) {
    # """Runs PCA and generates the variance, projection, and loading plots."""

    # Run PCA (data is already scaled via robust_scale)
    pca_result <- prcomp(x_scaled, center = FALSE, scale. = FALSE)

    # 1. Cumulative Variance Plot
    explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)
    cumulative_variance <- cumsum(explained_variance)

    var_df <- data.frame(PC = 1:length(cumulative_variance), Cumulative_Variance = cumulative_variance)

    p1 <- ggplot(var_df, aes(x = PC, y = Cumulative_Variance)) +
        geom_line(color = "dodgerblue", size = 1) +
        geom_point(color = "dodgerblue", size = 2) +
        geom_hline(yintercept = 0.80, color = "red", linetype = "dashed") +
        labs(title = 'PCA - Cumulative Explained Variance',
             x = 'Number of Principal Components', y = 'Cumulative Variance Retained') +
        theme_classic()
    print(p1)

    # 2. 2D Projection Plot
    proj_df <- data.frame(PC1 = pca_result$x[, 1], PC2 = pca_result$x[, 2])

    p2 <- ggplot(proj_df, aes(x = PC1, y = PC2)) +
        geom_point(alpha = 0.5, color = "steelblue") +
        labs(title = 'Linear Dimensionality Reduction: PCA (2D Projection)',
             x = 'Principal Component 1', y = 'Principal Component 2') +
        theme_classic()
    print(p2)

    # 3. PCA Component Loadings Heatmap (Top 3 Components)
    num_components <- min(3, ncol(pca_result$rotation))
    loadings <- pca_result$rotation[, 1:num_components]
    loadings_melted <- melt(loadings)
    colnames(loadings_melted) <- c("Feature", "PC", "Loading")

    p3 <- ggplot(loadings_melted, aes(x = PC, y = Feature, fill = Loading)) +
        geom_tile(color = "white") +
        scale_fill_gradient2(low = "firebrick", high = "dodgerblue", mid = "white", midpoint = 0) +
        labs(title = 'PCA - Component Loadings (Heatmap)') +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(p3)

    return(pca_result)
}

### ---------------------------------------
### UMAP
### ---------------------------------------
run_and_plot_umap <- function(x_scaled, df_features) {
    # """Runs UMAP and generates the 2D projection plot."""

    # Run UMAP
    set.seed(42)
    umap_result <- umap(x_scaled)

    # Projection Plot (2D)
    umap_df <- data.frame(UMAP1 = umap_result$layout[, 1], UMAP2 = umap_result$layout[, 2])

    p <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2)) +
        geom_point(alpha = 0.5, color = "mediumseagreen") +
        labs(title = 'Non-Linear Dimensionality Reduction: UMAP',
             x = 'UMAP Dimension 1', y = 'UMAP Dimension 2') +
        theme_classic()
    print(p)

    return(umap_result)
}

### ---------------------------------------
### DIMENSIONALITY REDUCTION PIPELINE
### ---------------------------------------
execute_dimensionality_reduction <- function(df) {
    # "Main execution function to run the entire DR pipeline"

    cat("\n--- Starting Dimensionality Reduction Pipeline ---\n")

    # 1. Prepare data
    prep_results <- prepare_data_for_dr(df)

    # 2. PCA
    cat("Running PCA...\n")
    pca_result <- run_and_plot_pca(prep_results$x_scaled, prep_results$df_features, prep_results$cols_to_scale)

    # 3. UMAP
    cat("Running UMAP...\n")
    umap_result <- run_and_plot_umap(prep_results$x_scaled, prep_results$df_features)

    cat("--- Dimensionality Reduction Complete ---\n")
}
