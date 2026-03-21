library(dplyr)
library(isotree) # Used for Isolation Forest in R


#----------------------------
### DATA CLEANSING
#----------------------------
filter_valid_flights <- function(df) {
    # """Filters out canceled and diverted flights and ensures valid numerical parameters"""
    # Cancelled or diverted flights do not have meaningful arrival delay values
    df <- df[(df$CANCELLED == 0) & (df$DIVERTED == 0), ]

    # Keep only flights with positive distance and scheduled time
    df <- df[(df$DISTANCE > 0) & (df$CRS_ELAPSED_TIME > 0), ]

    return(df)
}

handle_inconsistency <- function(df) {
    # """Fixes formatting irregularities and ensures mathematical consistency"""
    # Convert FL_DATE from string to datetime
    df$FL_DATE <- as.Date(df$FL_DATE)

    # Standardize midnight encoding from 2400 to 0
    df$CRS_DEP_TIME[df$CRS_DEP_TIME == 2400] <- 0
    df$CRS_ARR_TIME[df$CRS_ARR_TIME == 2400] <- 0

    # Check if times are in the right format (between 0 and 2359)
    df <- df[(df$CRS_DEP_TIME >= 0) & (df$CRS_DEP_TIME <= 2359), ]
    df <- df[(df$CRS_ARR_TIME >= 0) & (df$CRS_ARR_TIME <= 2359), ]

    # Flight time consistency check: actual elapsed time equals the sum of its phases
    taxi_out <- ifelse(is.na(df$TAXI_OUT), 0, df$TAXI_OUT)
    air_time <- ifelse(is.na(df$AIR_TIME), 0, df$AIR_TIME)
    taxi_in <- ifelse(is.na(df$TAXI_IN), 0, df$TAXI_IN)

    consistent_time_mask <- (!is.na(df$ELAPSED_TIME)) & (df$ELAPSED_TIME == (taxi_out + air_time + taxi_in))
    df <- df[consistent_time_mask, ]

    # Standardize early arrivals (negative delays) to 0 minutes
    df$ARR_DELAY <- ifelse(df$ARR_DELAY < 0, 0, df$ARR_DELAY)

    return(df)
}

handle_duplicates <- function(df) {
    # """Removes duplicate flight records"""
    # Use subset to ensure accuracy of the matching
    subset_cols <- c('FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST')
    df <- df[!duplicated(df[, subset_cols]), ]

    return(df)
}

remove_irrelevant_and_leaky_features <- function(df) {
    # """Drops identifier columns, post-event variables and cancellation flags."""
    # Drop identifiers and redundant categorical representations
    cols_to_drop_identifiers <- c('AIRLINE_DOT', 'AIRLINE', 'DOT_CODE', 'FL_NUMBER')
    df <- df[, !(names(df) %in% cols_to_drop_identifiers)]

    # Prevent Data Leakage: drop post-event variables
    columns_to_drop_leaky <- c(
        'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
        'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME', 'DEP_DELAY',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    )
    df <- df[, !(names(df) %in% columns_to_drop_leaky)]

    # Drop variables related to cancellations and diversion (since we already filtered the rows)
    cols_to_drop_status <- c('CANCELLED', 'DIVERTED', 'CANCELLATION_CODE')
    df <- df[, !(names(df) %in% cols_to_drop_status)]

    return(df)
}

### ---------------------------------------
### REMOVE OUTLIERS
### ---------------------------------------
remove_outliers_isolation_forest <- function(df, features_list, contamination=0.05) {
    # """ Removes outliers using Isolation Forest, with training on a data sample"""

    cat("\n--- Removing Outliers using Isolation Forest ---\n")
    cat(sprintf("Original dataset size: %d rows\n", nrow(df)))

    # Create a copy of the dataset
    df_clean <- df

    # Safely take 200k rows, or the maximum available if the dataset is smaller!
    sample_size <- min(200000, nrow(df_clean))
    set.seed(42)
    sample_indices <- sample(seq_len(nrow(df_clean)), size = sample_size, replace = FALSE)
    sample_df <- df_clean[sample_indices, ]

    # Run Isolation Forest
    # Train model on data sample
    cat("Training model on sample...\n")
    iso_forest <- isolation.forest(
        sample_df[, features_list],
        ndim = 1,
        ntrees = 100,
        nthreads = -1,
        seed = 42
    )

    # Predict on full dataset
    predictions <- predict(iso_forest, df_clean[, features_list])

    # Keep only normal rows
    threshold <- quantile(predictions, probs = 1 - contamination)
    df_filtered <- df_clean[predictions <= threshold, ]

    # Count the number of dropped rows
    dropped_count <- nrow(df_clean) - nrow(df_filtered)

    cat(sprintf("Dropped %d anomalous rows.\n", dropped_count))
    cat(sprintf("New dataset size: %d rows\n", nrow(df_filtered)))

    return(df_filtered)
}

### ---------------------------------------
### DATA CLEANSING PIPELINE
### ---------------------------------------
run_data_cleansing_pipeline <- function(df) {
    # """ Main execution pipeline """
    df <- filter_valid_flights(df)
    df <- handle_inconsistency(df)
    df <- handle_duplicates(df)
    df <- remove_irrelevant_and_leaky_features(df)
    return(df)
}
