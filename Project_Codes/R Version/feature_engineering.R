library(dplyr)

create_temporal_features <- function(df) {
  # Temporal Feature Extraction & Transformation

  # Ensure FL_DATE is a datetime object
  df$FL_DATE <- as.Date(df$FL_DATE)

  # 1. MONTH
  df$MONTH <- as.integer(format(df$FL_DATE, "%m"))

  # 2. IS_WEEKEND (0 = Monday, 6 = Sunday)
  day_of_week <- as.POSIXlt(df$FL_DATE)$wday
  df$IS_WEEKEND <- ifelse(day_of_week >= 5, 1, 0)

  # Extract scheduled departure and arrival hour (CRS_DEP_TIME and CRS_ARR_TIME are in hhmm format)
  df$DEP_HOUR <- df$CRS_DEP_TIME %/% 100
  df$ARR_HOUR <- df$CRS_ARR_TIME %/% 100

  # 3 & 4. SIN_DEP_TIME and COS_DEP_TIME (Cyclic Transformation)
  df$SIN_DEP_TIME <- sin(2 * pi * df$DEP_HOUR / 24)
  df$COS_DEP_TIME <- cos(2 * pi * df$DEP_HOUR / 24)

  return(df)
}

create_binned_features <- function(df) {
  # Create binned feature based on flight distance

  #5. FLIGHT_HAUL_TYPE (Binning DISTANCE)
  # Short-haul: < 700 miles, Medium-haul: 700-3000 miles, Long-haul: > 3000 miles
  df$FLIGHT_HAUL_TYPE <- cut(
    df$DISTANCE,
    breaks = c(0, 700, 3000, Inf),
    labels = c("Short-Haul", "Medium-Haul", "Long-Haul"),
    include.lowest = TRUE
  )

  return(df)
}

create_aggregation_features <- function(df) {
  # Feature Aggregation

  # 6. ROUTE_FREQUENCY (by Airline)
  df <- df %>%
    mutate(ROUTE = paste0(ORIGIN, "-", DEST)) %>%
    group_by(AIRLINE_CODE, ROUTE) %>%
    mutate(ROUTE_FREQUENCY = n()) %>%
    ungroup()

  # 7. ORIGIN_CONGESTION (Relative capacity ratio)

  # step 1: Count the flights scheduled in that specific hour block
  df <- df %>%
    group_by(ORIGIN, FL_DATE, DEP_HOUR) %>%
    mutate(ORIGIN_FLIGHTS = n()) %>%
    ungroup()

  # Step 2: Find the maximum capacity (busiest hour ever) for the origin airport
  df <- df %>%
    group_by(ORIGIN) %>%
    mutate(ORIGIN_MAX_CAPACITY = max(ORIGIN_FLIGHTS, na.rm = TRUE)) %>%
    ungroup()

  # Step 3: Ratio to normalize by airport size
  df$ORIGIN_CONGESTION <- df$ORIGIN_FLIGHTS / df$ORIGIN_MAX_CAPACITY


  # 8. DESTINATION_CONGESTION (Relative capacity ratio)

  # Step 1: Extract scheduled arrival hour (CRS_ARR_TIME is in hhmm format)
  df$ARR_HOUR <- df$CRS_ARR_TIME %/% 100

  # Step 2: Count the flights scheduled in that specific hour block
  df <- df %>%
    group_by(DEST, FL_DATE, ARR_HOUR) %>%
    mutate(DEST_FLIGHTS = n()) %>%
    ungroup()

  # Step 3: Find the maximum capacity for the destination airport
  df <- df %>%
    group_by(DEST) %>%
    mutate(DEST_MAX_CAPACITY = max(DEST_FLIGHTS, na.rm = TRUE)) %>%
    ungroup()

  # Step 4: Ratio to normalize by airport size
  df$DESTINATION_CONGESTION <- df$DEST_FLIGHTS / df$DEST_MAX_CAPACITY

  # Cleanup: Drop the calculation features to prevent the dataframe from getting bloated
  df <- df %>%
    select(-ROUTE, -ORIGIN_FLIGHTS, -DEST_FLIGHTS,
           -ORIGIN_MAX_CAPACITY, -DEST_MAX_CAPACITY)

  return(df)
}

create_interaction_features <- function(df) {
  # Feature combining and interactions

  # 9. SCHEDULED_SPEED
  df$SCHEDULED_SPEED <- df$DISTANCE / df$CRS_ELAPSED_TIME

  # 10. AIRLINE_HUB_DOMINANCE (Airline's flights at origin / Total flights at origin in that airport)
  df <- df %>%
    group_by(ORIGIN, AIRLINE_CODE) %>%
    mutate(airline_origin_count = n()) %>%
    ungroup() %>%
    group_by(ORIGIN) %>%
    mutate(total_origin_count = n()) %>%
    ungroup()

  df$AIRLINE_HUB_DOMINANCE <- df$airline_origin_count / df$total_origin_count

  # 11. ROUTE_BOTTLENECK_INTERACTION
  df$ROUTE_BOTTLENECK_INTERACTION <- df$ORIGIN_CONGESTION * df$DESTINATION_CONGESTION

  return(df)
}

run_feature_engineering_pipeline <- function(df) {
  # Main execution pipeline:
  df <- create_temporal_features(df)
  df <- create_binned_features(df)
  df <- create_aggregation_features(df)
  df <- create_interaction_features(df)

  return(df)
}
