import numpy as np
import pandas as pd


def create_temporal_features(df):
    """Temporal Feature Extraction & Transformation"""

    # Ensure FL_DATE is a datetime object
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # 1. MONTH
    df['MONTH'] = df['FL_DATE'].dt.month

    # 2. IS_WEEKEND (0 = Monday, 6 = Sunday)
    day_of_week = df['FL_DATE'].dt.dayofweek
    df['IS_WEEKEND'] = np.where(day_of_week >= 5, 1, 0)

    # Extract scheduled departure and arrival hour (CRS_DEP_TIME and CRS_ARR_TIME are in hhmm format)
    df['DEP_HOUR'] = df['CRS_DEP_TIME'] // 100
    df['ARR_HOUR'] = df['CRS_ARR_TIME'] // 100

    # 3 & 4. SIN_DEP_TIME and COS_DEP_TIME (Cyclic Transformation)
    df['SIN_DEP_TIME'] = np.sin(2 * np.pi * df['DEP_HOUR'] / 24)
    df['COS_DEP_TIME'] = np.cos(2 * np.pi * df['DEP_HOUR'] / 24)

    return df


def create_binned_features(df):
    """Create binned feature based on flight distance"""

    #5. FLIGHT_HAUL_TYPE (Binning DISTANCE)
    # Short-haul: < 700 miles, Medium-haul: 700-3000 miles, Long-haul: > 3000 miles
    df['FLIGHT_HAUL_TYPE'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 700, 3000, np.inf],
        labels=['Short-Haul', 'Medium-Haul', 'Long-Haul']
    )

    return df


def create_aggregation_features(df):
    """Feature Aggregation"""

    # 6. ROUTE_FREQUENCY (by Airline)
    df['ROUTE'] = df['ORIGIN'] + "-" + df['DEST']
    df['ROUTE_FREQUENCY'] = df.groupby(['AIRLINE_CODE', 'ROUTE'])['FL_DATE'].transform('count')

    # 7. ORIGIN_CONGESTION (Relative capacity ratio)

    # step 1: Count the flights scheduled in that specific hour block
    df['ORIGIN_FLIGHTS'] = df.groupby(['ORIGIN', 'FL_DATE', 'DEP_HOUR'])['AIRLINE_CODE'].transform('count')

    # Step 2: Find the maximum capacity (busiest hour ever) for the origin airport
    df['ORIGIN_MAX_CAPACITY'] = df.groupby('ORIGIN')['ORIGIN_FLIGHTS'].transform('max')

    # Step 3: Ratio to normalize by airport size
    df['ORIGIN_CONGESTION'] = df['ORIGIN_FLIGHTS'] / df['ORIGIN_MAX_CAPACITY']


    # 8. DESTINATION_CONGESTION (Relative capacity ratio)

    # Step 1: Extract scheduled arrival hour (CRS_ARR_TIME is in hhmm format)
    df['ARR_HOUR'] = df['CRS_ARR_TIME'] // 100

    # Step 2: Count the flights scheduled in that specific hour block
    df['DEST_FLIGHTS'] = df.groupby(['DEST', 'FL_DATE', 'ARR_HOUR'])['AIRLINE_CODE'].transform('count')

    # Step 3: Find the maximum capacity for the destination airport
    df['DEST_MAX_CAPACITY'] = df.groupby('DEST')['DEST_FLIGHTS'].transform('max')

    # Step 4: Ratio to normalize by airport size
    df['DESTINATION_CONGESTION'] = df['DEST_FLIGHTS'] / df['DEST_MAX_CAPACITY']

    # Cleanup: Drop the calculation features to prevent the dataframe from getting bloated
    df = df.drop(columns=[
        'ROUTE', 'ORIGIN_FLIGHTS', 'DEST_FLIGHTS',
        'ORIGIN_MAX_CAPACITY', 'DEST_MAX_CAPACITY'
    ])

    return df


def create_interaction_features(df):
    """Feature combining and interactions"""

    # 9. SCHEDULED_SPEED
    df['SCHEDULED_SPEED'] = df['DISTANCE'] / df['CRS_ELAPSED_TIME']

    # 10. AIRLINE_HUB_DOMINANCE (Airline's flights at origin / Total flights at origin in that airport)
    airline_origin_count = df.groupby(['ORIGIN', 'AIRLINE_CODE'])['FL_DATE'].transform('count')
    total_origin_count = df.groupby(['ORIGIN'])['FL_DATE'].transform('count')
    df['AIRLINE_HUB_DOMINANCE'] = airline_origin_count / total_origin_count

    # 11. ROUTE_BOTTLENECK_INTERACTION
    df['ROUTE_BOTTLENECK_INTERACTION'] = df['ORIGIN_CONGESTION'] * df['DESTINATION_CONGESTION']

    return df


def run_feature_engineering_pipeline(df):
    """ Main execution pipeline:"""
    df = create_temporal_features(df)
    df = create_binned_features(df)
    df = create_aggregation_features(df)
    df = create_interaction_features(df)

    return df
