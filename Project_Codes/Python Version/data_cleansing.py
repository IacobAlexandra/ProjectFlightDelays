# 1. Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest

def handle_incompleteness_and_inconsistency(df):
    """Filters out canceled/diverted flights and fixes time/range inconsistencies."""

    # Filter out canceled and diverted flights
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

    # Keep only flights with positive distance and scheduled time
    df = df[(df['DISTANCE'] > 0) & (df['CRS_ELAPSED_TIME'] > 0)]

    # Convert FL_DATE from string to datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # Standardize midnight encoding from 2400 to 0
    df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].replace(2400, 0)
    df['CRS_ARR_TIME'] = df['CRS_ARR_TIME'].replace(2400, 0)

    # Check if times are in the right format (between 0 and 2359)
    df = df[(df['CRS_DEP_TIME'] >= 0) & (df['CRS_DEP_TIME'] <= 2359)]
    df = df[(df['CRS_ARR_TIME'] >= 0) & (df['CRS_ARR_TIME'] <= 2359)]

    return df


def handle_duplicates_and_logic(df):
    """Removes duplicate rows and ensures flight time phases add up logically."""

    # Use subset to ensure accuracy of the matching
    subset_cols = ['FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST']
    df = df.drop_duplicates(subset=subset_cols, keep='first')

    # Flight time consistency check: actual elapsed time equals the sum of its phases
    consistent_time_mask = df['ELAPSED_TIME'] == (
                df['TAXI_OUT'].fillna(0) + df['AIR_TIME'].fillna(0) + df['TAXI_IN'].fillna(0))
    df = df[consistent_time_mask]

    return df


def remove_irrelevant_and_leaky_features(df):
    """Drops identifier columns, post-event (leaky) variables, and cancellation flags."""

    # Drop identifiers and redundant categorical representations
    cols_to_drop_identifiers = ['AIRLINE_DOT', 'AIRLINE', 'DOT_CODE', 'FL_NUMBER']
    df = df.drop(columns=cols_to_drop_identifiers, errors='ignore')

    # Prevent Data Leakage: drop post-event variables
    columns_to_drop_leaky = [
        'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
        'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME', 'DEP_DELAY',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    df = df.drop(columns=columns_to_drop_leaky, errors='ignore')

    # Drop variables related to cancellations and diversion (since we already filtered the rows)
    cols_to_drop_status = ['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE']
    df = df.drop(columns=cols_to_drop_status, errors='ignore')

    return df


def remove_outliers_isolation_forest(df, features_list, contamination=0.05):
    # Removes outliers using Isolation Forest, trained on a fast sample.
    # contamination: expected percentage of outliers (0.05 = 5% of data will be dropped)

    print("\n--- Removing Outliers using Isolation Forest (Sampled Training) ---")
    print(f"Original dataset size: {df.shape[0]} rows")

    # Drop NaNs so the model doesn't crash
    df_clean = df.dropna(subset=features_list).copy()

    # Safely take 200k rows, or the maximum available if the dataset is smaller!
    sample_size = min(200000, len(df_clean))
    sample_df = df_clean.sample(n=sample_size, random_state=42)

    # 2. Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # 3. Fit on sample
    print("Training model on sample...")
    iso_forest.fit(sample_df[features_list])

    # 4. Predict on full dataset
    print("Predicting anomalies across the entire dataset...")
    predictions = iso_forest.predict(df_clean[features_list])

    # Keep normal rows
    df_filtered = df_clean[predictions == 1].copy()

    dropped_count = df_clean.shape[0] - df_filtered.shape[0]

    print(f"Dropped {dropped_count} anomalous rows.")
    print(f"New dataset size: {df_filtered.shape[0]} rows")

    return df_filtered


def run_data_cleansing_pipeline(df):
    """
    Main Orchestrator Function:
    Runs the entire data cleansing pipeline in procedural order.
    """
    df = handle_incompleteness_and_inconsistency(df)
    df = handle_duplicates_and_logic(df)
    df = remove_irrelevant_and_leaky_features(df)
    return df
