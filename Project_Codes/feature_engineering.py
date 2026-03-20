import numpy as np
import pandas as pd


def create_temporal_features(df):
    """Group 1: Feature Extraction & Transformation (Derived from Temporal Data)"""
    # Ensure FL_DATE is a datetime object
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # 1. MONTH
    df['MONTH'] = df['FL_DATE'].dt.month

    # 2. IS_WEEKEND (0 = Monday, 6 = Sunday)
    day_of_week = df['FL_DATE'].dt.dayofweek
    df['IS_WEEKEND'] = np.where(day_of_week >= 5, 1, 0)

    # Extract scheduled departure hour (CRS_DEP_TIME is in hhmm format)
    dep_hour = df['CRS_DEP_TIME'] // 100

    # 3 & 4. SIN_DEP_TIME and COS_DEP_TIME (Cyclic Transformation)
    df['SIN_DEP_TIME'] = np.sin(2 * np.pi * dep_hour / 24)
    df['COS_DEP_TIME'] = np.cos(2 * np.pi * dep_hour / 24)

    return df


def create_binned_features(df):
    """Group 2: Data Discretization"""

    # 5. FLIGHT_HAUL_TYPE (Binning DISTANCE)
    # Short-haul: < 700 miles, Medium-haul: 700-3000 miles, Long-haul: > 3000 miles
    df['FLIGHT_HAUL_TYPE'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 700, 3000, np.inf],
        labels=['Short-Haul', 'Medium-Haul', 'Long-Haul']
    )

    return df


def create_aggregation_features(df):
    """Group 3: Data-Driven Feature Aggregation"""

    # 6. ROUTE_FREQUENCY (Airline specific)
    df['ROUTE'] = df['ORIGIN'] + "-" + df['DEST']
    df['ROUTE_FREQUENCY'] = df.groupby(['AIRLINE_CODE', 'ROUTE'])['FL_DATE'].transform('count')

    # =========================================================================
    # 7. ORIGIN_CONGESTION (Relative Capacity Ratio - Coarser Hour Granularity)
    # =========================================================================
    # Step 1: Extract the coarser hour block
    df['DEP_HOUR'] = df['CRS_DEP_TIME'] // 100

    # Step 2: Count the absolute flights scheduled in that specific hour block
    df['ABSOLUTE_ORIGIN_FLIGHTS'] = df.groupby(['ORIGIN', 'FL_DATE', 'DEP_HOUR'])['AIRLINE_CODE'].transform('count')

    # Step 3: Find the maximum capacity (busiest hour ever) for that specific airport
    df['ORIGIN_MAX_CAPACITY'] = df.groupby('ORIGIN')['ABSOLUTE_ORIGIN_FLIGHTS'].transform('max')

    # Step 4: Cross-Feature Operation (Ratio) to normalize by airport size
    df['ORIGIN_CONGESTION'] = df['ABSOLUTE_ORIGIN_FLIGHTS'] / df['ORIGIN_MAX_CAPACITY']

    # =========================================================================
    # 8. DESTINATION_CONGESTION (Relative Capacity Ratio)
    # =========================================================================
    # Step 1: Extract the coarser hour block
    df['ARR_HOUR'] = df['CRS_ARR_TIME'] // 100

    # Step 2: Count the absolute flights scheduled in that specific hour block
    df['ABSOLUTE_DEST_FLIGHTS'] = df.groupby(['DEST', 'FL_DATE', 'ARR_HOUR'])['AIRLINE_CODE'].transform('count')

    # Step 3: Find the maximum capacity for that specific destination airport
    df['DEST_MAX_CAPACITY'] = df.groupby('DEST')['ABSOLUTE_DEST_FLIGHTS'].transform('max')

    # Step 4: Cross-Feature Operation (Ratio)
    df['DESTINATION_CONGESTION'] = df['ABSOLUTE_DEST_FLIGHTS'] / df['DEST_MAX_CAPACITY']

    # =========================================================================
    # CLEANUP
    # =========================================================================
    # Drop the temporary calculation columns to prevent the dataframe from getting bloated
    df = df.drop(columns=[
        'DEP_HOUR', 'ARR_HOUR',
        'ABSOLUTE_ORIGIN_FLIGHTS', 'ABSOLUTE_DEST_FLIGHTS',
        'ORIGIN_MAX_CAPACITY', 'DEST_MAX_CAPACITY'
    ])

    return df

def create_interaction_features(df):
    """Group 4: Cross-Feature Operations & Interactions"""

    # 9. SCHEDULED_SPEED
    df['SCHEDULED_SPEED'] = df['DISTANCE'] / df['CRS_ELAPSED_TIME']

    # 10. AIRLINE_HUB_DOMINANCE (Airline's flights at origin / Total flights at origin)
    airline_origin_count = df.groupby(['ORIGIN', 'AIRLINE_CODE'])['FL_DATE'].transform('count')
    total_origin_count = df.groupby(['ORIGIN'])['FL_DATE'].transform('count')
    df['AIRLINE_HUB_DOMINANCE'] = airline_origin_count / total_origin_count

    # 11. ROUTE_BOTTLENECK_INTERACTION
    df['ROUTE_BOTTLENECK_INTERACTION'] = df['ORIGIN_CONGESTION'] * df['DESTINATION_CONGESTION']

    return df

def run_feature_engineering_pipeline(df):
    """
    Main Orchestrator Function:
    Runs the entire feature engineering pipeline and cleans up temporary variables.
    """
    df = create_temporal_features(df)
    df = create_binned_features(df)
    df = create_aggregation_features(df)
    df = create_interaction_features(df)

    # Cleanup: Drop temporary columns used for calculations to keep the dataset clean
    columns_to_drop = ['ROUTE', 'DEP_30MIN_BIN', 'ARR_30MIN_BIN']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df