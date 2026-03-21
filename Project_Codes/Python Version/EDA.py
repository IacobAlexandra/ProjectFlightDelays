import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_descriptive_statistics(df, target_features, output_file_name):
    """ Provides a basic data overview and descriptive statistics for the continuous features"""
    print("\n--- DESCRIPTIVE STATISTICS & DATA OVERVIEW ---")

    # Force Pandas to show all columns and widen the display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 1. Basic Summary Statistics (Count, Mean, Std, Min, Max, Quantiles)
    print("\nBasic Summary Statistics:")
    stats = df[target_features].describe().round(2)
    print(stats)

    # 2. Variance (Measures data dispersion/spread)
    print("\nVariance:")
    variance = df[target_features].var().round(2)
    print(variance)
    stats.loc['variance'] = variance  # Append to main table

    # 3. Skewness (Measures symmetry)
    print("\nSkewness:")
    skewness = df[target_features].skew().round(2)
    print(skewness)
    stats.loc['skewness'] = skewness  # Append to main table

    # 4. Kurtosis (Measures the shape/tails of the distribution)
    print("\nKurtosis:")
    kurtosis = df[target_features].kurtosis().round(2)
    print(kurtosis)
    stats.loc['kurtosis'] = kurtosis  # Append to main table

    # Clean up the main table and save it to csv
    stats[''] = stats.index  # Copies the labels to the far right

    stats.to_csv(rf'C:\Users\Jhonny999\PycharmProjects\ProjectFlightDelays\Output_Files\Descriptive_Statistics\{output_file_name}')

def plot_executive_profile(df, categorical_col, top_n=15, plot_color='steelblue'):
    """Creates a generalized Grid Chart of Bar Plots showing the operational profile
     based on the categorical feature passed as parameter"""

    # 1. If the category has hundreds of unique values (like Airports), it slices the top N.
    # If it has fewer than N (like Airlines), keep all of them
    top_categories = df[categorical_col].value_counts().nlargest(top_n).index
    df_filtered = df[df[categorical_col].isin(top_categories)]

    # 2. Define the 6 features to analyze
    features_to_plot = [
        ('ARR_DELAY', 'Target: Average Arrival Delay', 'Minutes'),
        ('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Minute'),
        ('DISTANCE', 'Fleet: Average Flight Distance', 'Miles'),
        ('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Route Bottleneck Score', 'Score'),
        ('ROUTE_FREQUENCY', 'Operations: Route Frequency', 'Flight Count'),
        ('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)')
    ]

    # 3. Create the grid (3 rows, 2 columns)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    axes = axes.flatten()

    # 4. Loop through the features and plot them
    for i, (feature, title, ylabel) in enumerate(features_to_plot):
        ax = axes[i]

        # Calculate sorting order based on mean (descending) for the chosen category
        order = df_filtered.groupby(categorical_col)[feature].mean().sort_values(ascending=False).index

        # Create the bar plot with Standard Deviation error bars (to show dispersion)
        sns.barplot(
            data=df_filtered,
            x=categorical_col,
            y=feature,
            color=plot_color,
            errorbar='sd',
            capsize=0.1,
            err_kws={'linewidth': 1.5, 'color': 'black'},
            order=order,
            ax=ax
        )

        # Add value labels to the top of the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=15, color='black', size=10)

        # Formatting
        # Dynamically generate the X-axis label based on the parameter
        ax.set_title(title, fontsize=15, pad=12, weight='bold')
        ax.set_xlabel(categorical_col.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=0)

        # Remove top and right borders to reduce clutter
        sns.despine(ax=ax)

    # Adjust spacing so titles and labels don't overlap
    plt.tight_layout(pad=4.0)
    plt.show()

def plot_categorical_boxplots(df):
    """ Creates a Grid Chart of Boxplots to show the statistical
    spread of the main continuous features, grouped by Flight Haul Type"""

    # 1. Define the 6 features to analyze
    features_to_plot = [
        ('ARR_DELAY', 'Target: Arrival Delay Distribution', 'Minutes'),
        ('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Min'),
        ('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)'),
        ('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Bottleneck Score', 'Score'),
        ('ORIGIN_CONGESTION', 'Traffic: Origin Congestion', 'Concurrent Flights'),
        ('CRS_ELAPSED_TIME', 'Planning: Scheduled Duration', 'Minutes')
    ]

    # 2. Create the grid (3 rows, 2 columns)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 10))
    axes = axes.flatten()

    # 3. Take 100k rows or the maximum available if the dataset is smaller
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # 4. Loop through the features and plot them
    for i, (feature, title, ylabel) in enumerate(features_to_plot):
        ax = axes[i]

        sns.boxplot(
            data=df_sample,
            x='FLIGHT_HAUL_TYPE',
            y=feature,
            hue='FLIGHT_HAUL_TYPE',
            legend=False,
            palette='viridis',
            width=0.4,
            flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=5),
            ax=ax
        )

        # Apply zoom for ARR_DELAY to see the IQR boxes clearly
        if feature == 'ARR_DELAY':
            ax.set_ylim(-60, 300)

        # Formatting
        ax.set_title(title, fontsize=14, pad=10, weight='bold')
        ax.set_xlabel('Flight Haul Type', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        sns.despine(ax=ax)  # Clean up top/right borders

    plt.tight_layout(pad=4.0)
    plt.show()

def plot_temporal_delay_trend(df, time_col, title, xlabel, xticks_range, line_color):
    """ Plots the average arrival delay over a specific time dimension"""

    # Define plot size
    plt.figure(figsize=(10, 6))

    # Group data by the specified time column
    trend_data = df.groupby(time_col)['ARR_DELAY'].mean().reset_index()

    # Plot the continuous trend line
    sns.lineplot(data=trend_data, x=time_col, y='ARR_DELAY',
                 marker='o', markersize=8, color=line_color, linewidth=2)

    # Formatting
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel)
    plt.ylabel('Average Arrival Delay (Minutes)')
    plt.xticks(xticks_range)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plt.show()


def plot_delay_distribution(df):
    """Plots the frequency and density distribution of arrival delays"""

    # Define plot size
    plt.figure(figsize=(10, 6))

    # Increase bins to 100 for better detail in the zoomed area
    sns.histplot(data=df, x='ARR_DELAY', bins=100, kde=True, color='mediumpurple')

    # Formatting
    plt.xlim(-60, 300) # Zoom in
    plt.title('Distribution Shape: Frequency and Density of Arrival Delays', fontsize=14, pad=15)
    plt.xlabel('Arrival Delay (Minutes)')
    plt.ylabel('Frequency (Number of Flights)')
    sns.despine()
    plt.show()


def plot_distance_vs_duration(df):
    """ Plots a Scatterplot to show the linear correlation between flight distance
    and scheduled duration """

    # Define the plot size
    plt.figure(figsize=(10, 6))

    # Create the Scatterplot
    sns.scatterplot(data=df, x='DISTANCE', y='CRS_ELAPSED_TIME',
                    alpha=0.1, s=10, color='dodgerblue')

    # Formatting
    plt.title('Linear Correlation: Distance vs. Scheduled Flight Duration', fontsize=14, pad=15)
    plt.xlabel('Flight Distance (Miles)')
    plt.ylabel('Scheduled Elapsed Time (Minutes)')
    sns.despine()

    plt.show()


def plot_executive_bubble_chart(df):
    """ Plots an Executive Bubble Chart of Traffic Volume vs. Delay by Origin Airport """

    # Safely sample to avoid crashing
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Aggregate data by Airport
    airport_stats = df_sample.groupby('ORIGIN').agg(
        Flight_Volume=('ORIGIN', 'count'),
        Avg_Delay=('ARR_DELAY', 'mean'),
        Avg_Distance=('DISTANCE', 'mean')
    ).reset_index()

    plt.figure(figsize=(14, 8))

    # Keep only the Top 30 busiest airports to reduce clutter
    airport_stats = airport_stats.nlargest(30, 'Flight_Volume').reset_index(drop=True)

    sns.scatterplot(
        data=airport_stats,
        x='Flight_Volume',
        y='Avg_Delay',
        size='Avg_Distance',
        sizes=(100, 2000),
        hue='ORIGIN',
        alpha=0.6,
        edgecolor='black',
        legend=False
    )

    for i in range(airport_stats.shape[0]):
        plt.text(
                x=airport_stats.loc[i, 'Flight_Volume'],
                y=airport_stats.loc[i, 'Avg_Delay'],
                s=airport_stats.loc[i, 'ORIGIN'],
                fontsize=9,
                ha='center',
                va='center',
                weight='bold'
            )

    # Formatting
    plt.title(
        'Traffic Characteristics: Flight Volume vs Delay by Airport (Sized by Avg Distance)',
        fontsize=14,
        weight='bold',
        pad=15
    )
    plt.xlabel('Traffic Volume (Total Flights in Sample)', fontsize=12)
    plt.ylabel('Average Arrival Delay (Minutes)', fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_speed_violin_distribution(df):
    """Plots a Violin Plot showing the density and shape of scheduled
     speeds across different flight haul types """

    # Define plot size
    plt.figure(figsize=(10, 6))

    # Create the Violin Plot
    sns.violinplot(
        data=df,
        x='FLIGHT_HAUL_TYPE',
        y='SCHEDULED_SPEED',
        hue='FLIGHT_HAUL_TYPE',
        legend=False,
        palette='magma'
    )

    plt.title('Distribution Shape: Scheduled Speed by Flight Haul Type', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Flight Haul Type', fontsize=12)
    plt.ylabel('Scheduled Speed (Miles/Minute)', fontsize=12)
    sns.despine()
    plt.show()


def plot_dominance_bubble(df):
    """ Plots an Executive Bubble Chart of
    traffic congestion vs. delay by Airline, using flight volume as the bubble size """

    plt.figure(figsize=(12, 7))

    # Aggregate data by Airline
    airline_stats = df.groupby('AIRLINE_CODE').agg({
        'AIRLINE_HUB_DOMINANCE': 'mean',
        'ARR_DELAY': 'mean',
        'ROUTE_FREQUENCY': 'count'  # Using flight volume as bubble size
    }).reset_index()

    # Create the Bubble Chart
    sns.scatterplot(
        data=airline_stats,
        x='AIRLINE_HUB_DOMINANCE',
        y='ARR_DELAY',
        size='ROUTE_FREQUENCY',
        sizes=(100, 2000),
        hue='AIRLINE_CODE',
        palette='tab20',
        alpha=0.8,
        edgecolor='black'
    )

    # Add text labels inside the bubbles
    for i in range(len(airline_stats)):
        plt.text(airline_stats['AIRLINE_HUB_DOMINANCE'][i],
                 airline_stats['ARR_DELAY'][i],
                 airline_stats['AIRLINE_CODE'][i],
                 horizontalalignment='center', size='small', color='black', weight='bold')

    plt.title('Business Strategy: Average Hub Dominance vs Average Delay by Airline', fontsize=14, pad=15,
              weight='bold')
    plt.xlabel('Average Hub Dominance Ratio', fontsize=12)
    plt.ylabel('Average Arrival Delay (Minutes)', fontsize=12)
    plt.legend([], [], frameon=False)
    sns.despine()
    plt.show()


def plot_frequency_vs_delay(df):
    """ Plots a sampled Scatterplot to show the correlation
    between Route Frequency (flight volume) and Arrival Delay """

    # 1. Create a 100k sample
    df_sample = df.sample(n=100000, random_state=42)

    plt.figure(figsize=(10, 6))

    # 2. Create Scatterplot
    sns.scatterplot(
        data=df_sample,
        x='ROUTE_FREQUENCY',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='dodgerblue'
    )

    # Zoom in on the relevant delay spread
    plt.ylim(-60, 300)

    # Formatting
    plt.title('Linear Correlation: Route Frequency vs. Arrival Delay', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Route Frequency (Total Flights on this Route)', fontsize=12)
    plt.ylabel('Arrival Delay (Minutes)', fontsize=12)

    sns.despine()
    plt.show()


def plot_congestion_vs_delay_grid(df):
    """Plots a grid of Scatterplots to show the correlation
    between the Airport Origin/Destination Congestion and the Arrival Delay"""

    # Sample with replacement
    df_sample = df.sample(n=100000, replace=True, random_state=42)

    # Create a 1x2 grid
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # LEFT CHART: Origin Congestion
    sns.scatterplot(
        data=df_sample,
        x='ORIGIN_CONGESTION',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='crimson',
        ax=axes[0]
    )

    # Formatting
    axes[0].set_ylim(-60, 300)
    axes[0].set_title('Origin Congestion vs. Arrival Delay', fontsize=14, weight='bold')
    axes[0].set_xlabel('Origin Congestion (Concurrent Departures)', fontsize=12)
    axes[0].set_ylabel('Arrival Delay (Minutes)', fontsize=12)

    sns.despine(ax=axes[0])


    # RIGHT CHART: Destination Congestion
    sns.scatterplot(
        data=df_sample,
        x='DESTINATION_CONGESTION',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='mediumseagreen',
        ax=axes[1]
    )

    # Formatting
    axes[1].set_ylim(-60, 300)
    axes[1].set_title('Destination Congestion vs. Arrival Delay', fontsize=14, weight='bold')
    axes[1].set_xlabel('Destination Congestion (Concurrent Arrivals)', fontsize=12)
    axes[1].set_ylabel('')

    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.show()
