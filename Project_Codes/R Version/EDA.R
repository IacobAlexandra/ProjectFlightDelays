library(dplyr)
library(ggplot2)
library(ggpubr) # for grid arrangements
library(data.table)
library(moments) # for skewness and kurtosis

# ---------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------
run_descriptive_statistics <- function(df, target_features, output_file_name) {
  # Provides a basic data overview and descriptive statistics for the continuous features
  cat("\n--- DESCRIPTIVE STATISTICS & DATA OVERVIEW ---\n")

  # 1. Basic Summary Statistics (Count, Mean, Std, Min, Max, Quantiles)
  cat("\nBasic Summary Statistics:\n")
  stats <- df %>% select(all_of(target_features)) %>% summary()
  print(round(stats, 2))

  # 2. Variance (Measures data dispersion/spread)
  cat("\nVariance:\n")
  variance <- sapply(df[target_features], var, na.rm = TRUE)
  print(round(variance, 2))

  # 3. Skewness (Measures symmetry)
  cat("\nSkewness:\n")
  skewness_vals <- sapply(df[target_features], skewness, na.rm = TRUE)
  print(round(skewness_vals, 2))

  # 4. Kurtosis (Measures the shape/tails of the distribution)
  cat("\nKurtosis:\n")
  kurtosis_vals <- sapply(df[target_features], kurtosis, na.rm = TRUE)
  print(round(kurtosis_vals, 2))

  # Save to CSV
  output_df <- as.data.frame(df[target_features])
  write.csv(output_df, output_file_name, row.names = TRUE)
}

# ---------------------------------------
# EXECUTIVE PROFILE BAR GRID
# ---------------------------------------
plot_executive_profile <- function(df, categorical_col, top_n = 15, plot_color = 'steelblue') {
  # Creates a generalized Grid Chart of Bar Plots showing the operational profile
  # based on the categorical feature passed as parameter

  # 1. Select top N categories
  top_categories <- names(sort(table(df[[categorical_col]]), decreasing = TRUE))[1:top_n]
  df_filtered <- df[df[[categorical_col]] %in% top_categories, ]

  # 2. Define features to plot
  features_to_plot <- list(
    list('ARR_DELAY', 'Target: Average Arrival Delay', 'Minutes'),
    list('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Minute'),
    list('DISTANCE', 'Fleet: Average Flight Distance', 'Miles'),
    list('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Route Bottleneck Score', 'Score'),
    list('ROUTE_FREQUENCY', 'Operations: Route Frequency', 'Flight Count'),
    list('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)')
  )

  plot_list <- list()
  for (feat in features_to_plot) {
    feature <- feat[[1]]
    title <- feat[[2]]
    ylabel <- feat[[3]]

    # Aggregate for mean ordering
    agg_order <- df_filtered %>% group_by(.data[[categorical_col]]) %>%
      summarise(mean_val = mean(.data[[feature]], na.rm = TRUE)) %>%
      arrange(desc(mean_val))

    p <- ggplot(df_filtered, aes_string(x = categorical_col, y = feature)) +
      stat_summary(fun = mean, geom = "bar", fill = plot_color, width = 0.7) +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.2) +
      labs(title = title, x = gsub("_", " ", categorical_col), y = ylabel) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

    plot_list <- c(plot_list, list(p))
  }

  # Arrange in 3 rows, 2 columns
  ggarrange(plotlist = plot_list, ncol = 2, nrow = 3)
}

# ---------------------------------------
# BOX PLOT GRID
# ---------------------------------------
plot_categorical_boxplots <- function(df) {
  # Creates a Grid Chart of Boxplots for continuous features by Flight Haul Type
  features_to_plot <- list(
    list('ARR_DELAY', 'Target: Arrival Delay Distribution', 'Minutes'),
    list('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Min'),
    list('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)'),
    list('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Bottleneck Score', 'Score'),
    list('ORIGIN_CONGESTION', 'Traffic: Origin Congestion', 'Concurrent Flights'),
    list('CRS_ELAPSED_TIME', 'Planning: Scheduled Duration', 'Minutes')
  )

  plot_list <- list()
  for (feat in features_to_plot) {
    feature <- feat[[1]]
    title <- feat[[2]]
    ylabel <- feat[[3]]

    p <- ggplot(df, aes_string(x = 'FLIGHT_HAUL_TYPE', y = feature, fill = 'FLIGHT_HAUL_TYPE')) +
      geom_boxplot(outlier.shape = 1, width = 0.4) +
      labs(title = title, x = 'Flight Haul Type', y = ylabel) +
      theme_minimal() +
      theme(legend.position = "none")

    if (feature == "ARR_DELAY") {
      p <- p + ylim(-60, 300)
    }

    plot_list <- c(plot_list, list(p))
  }

  ggarrange(plotlist = plot_list, ncol = 2, nrow = 3)
}

# ---------------------------------------
# TEMPORAL TREND
# ---------------------------------------
plot_temporal_delay_trend <- function(df, time_col, plot_title, x_label, line_color) {
  # Group by the temporal column and calculate the mean
  trend_data <- df %>%
    group_by(!!sym(time_col)) %>%
    summarize(Mean_Delay = mean(ARR_DELAY, na.rm = TRUE))

  # Plot the line chart using the modern tidy evaluation !!sym() instead of aes_string()
  ggplot(trend_data, aes(x = !!sym(time_col), y = Mean_Delay)) +
    geom_line(color = line_color, size = 1) +
    geom_point(color = line_color, size = 2) +
    theme_minimal() +
    labs(title = plot_title, x = x_label, y = "Average Arrival Delay")
}
# ---------------------------------------
# DELAY DISTRIBUTION
# ---------------------------------------
plot_delay_distribution <- function(df) {
  ggplot(df, aes(x = ARR_DELAY)) +
    geom_histogram(aes(y = ..density..), bins = 100, fill = "mediumpurple") +
    geom_density(color = "black") +
    xlim(-60, 300) +
    labs(title = "Distribution Shape: Frequency and Density of Arrival Delays",
         x = "Arrival Delay (Minutes)", y = "Density") +
    theme_minimal()
}

# ---------------------------------------
# DISTANCE VS DURATION
# ---------------------------------------
plot_distance_vs_duration <- function(df) {
  ggplot(df, aes(x = DISTANCE, y = CRS_ELAPSED_TIME)) +
    geom_point(alpha = 0.1, color = "dodgerblue") +
    labs(title = "Linear Correlation: Distance vs. Scheduled Flight Duration",
         x = "Flight Distance (Miles)", y = "Scheduled Elapsed Time (Minutes)") +
    theme_minimal()
}
plot_correlation_heatmap <- function(df) {
  library(dplyr)

  # 1. Select only numeric columns
  numeric_df <- df %>% select(where(is.numeric))

  # 2. Remove columns with zero variance or all NA
  numeric_df <- numeric_df %>%
    select(where(~ !all(is.na(.)) & sd(., na.rm = TRUE) > 0))

  # 3. Compute correlation matrix safely
  cor_matrix <- cor(numeric_df, use = "pairwise.complete.obs")

  # 4. Replace any NA correlations with 0 (optional, safer for heatmap)
  cor_matrix[is.na(cor_matrix)] <- 0

  # 5. Plot heatmap
  heatmap(
    cor_matrix,
    main = "Feature Correlation Heatmap",
    col = heat.colors(10),
    na.rm = TRUE
  )
}