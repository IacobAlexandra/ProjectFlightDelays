import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

#----------------------------
# DATA CLEANSING
#----------------------------
def filter_valid_flights(df):
    """Filters out canceled and diverted flights and ensures valid numerical parameters"""
    # Cancelled or diverted flights do not have meaningful arrival delay values [4]
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

    # Keep only flights with positive distance and scheduled time
    df = df[(df['DISTANCE'] > 0) & (df['CRS_ELAPSED_TIME'] > 0)]
    return df


def handle_inconsistency(df):
    """Fixes formatting irregularities and ensures mathematical consistency"""

    # Standardize early arrivals (negative delays) to 0 minutes
    df['ARR_DELAY'] = np.where(df['ARR_DELAY'] < 0, 0, df['ARR_DELAY'])

    # Convert FL_DATE from string to datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # Standardize midnight encoding from 2400 to 0
    df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].replace(2400, 0)
    df['CRS_ARR_TIME'] = df['CRS_ARR_TIME'].replace(2400, 0)

    # Check if times are in the right format (between 0 and 2359)
    df = df[(df['CRS_DEP_TIME'] >= 0) & (df['CRS_DEP_TIME'] <= 2359)]
    df = df[(df['CRS_ARR_TIME'] >= 0) & (df['CRS_ARR_TIME'] <= 2359)]

    return df

def handle_duplicates(df):
    """Removes duplicate flight records"""

    # Use subset of features to ensure accuracy of the matching
    subset_cols = ['FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST']
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    return df



def remove_irrelevant_and_leaky_features(df):
    """Drops identifier columns, post-event variables and cancellation flags."""

    # Drop redundant identifier features
    cols_to_drop_identifiers = ['AIRLINE_DOT', 'AIRLINE', 'DOT_CODE', 'FL_NUMBER']
    df = df.drop(columns=cols_to_drop_identifiers, errors='ignore')

    # Drop post-event variables to prevent data leakage
    columns_to_drop_leaky = [
        'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
        'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME', 'DEP_DELAY',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    df = df.drop(columns=columns_to_drop_leaky, errors='ignore')

    # Drop features related to cancellations and diversion (since we already filtered the rows)
    cols_to_drop_status = ['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE']
    df = df.drop(columns=cols_to_drop_status, errors='ignore')

    return df

# ---------------------------------------
# REMOVE OUTLIERS
# ---------------------------------------
def remove_outliers_isolation_forest(df, features_list, contamination=0.05):
    """ Removes outliers using Isolation Forest, with training on a data sample"""

    print("\n--- Removing Outliers using Isolation Forest ---")
    print(f"Original dataset size: {df.shape[0]} rows")

    # Create a copy of the dataset
    df_clean = df.copy()

    # Take 200k rows or the maximum available if the dataset is smaller
    sample_size = min(200000, len(df_clean))
    sample_df = df_clean.sample(n=sample_size, random_state=42)

    # Run Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # Train model on data sample
    print("Training model on sample...")
    iso_forest.fit(sample_df[features_list])

    # Predict on full dataset
    predictions = iso_forest.predict(df_clean[features_list])

    # Keep only normal rows
    df_filtered = df_clean[predictions == 1].copy()

    # Count the number of dropped rows
    dropped_count = df_clean.shape[0] - df_filtered.shape[0]

    print(f"Dropped {dropped_count} anomalous rows.")
    print(f"New dataset size: {df_filtered.shape[0]} rows")

    return df_filtered

# ---------------------------------------
# DATA CLEANSING PIPELINE
# ---------------------------------------
def run_data_cleansing_pipeline(df):
    """ Main execution pipeline """
    df = filter_valid_flights(df)
    df = handle_inconsistency(df)
    df = handle_duplicates(df)
    df = remove_irrelevant_and_leaky_features(df)
    return df

