import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_descriptive_statistics(df, target_features):
    """
    Provides a basic data overview and descriptive statistics
    strictly for the continuous features specified in the parameter list.
    """
    print("\n--- DESCRIPTIVE STATISTICS & DATA OVERVIEW ---")

    # SAFETY CHECK: Ensure all requested columns actually exist in the dataset
    missing_cols = [col for col in target_features if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Cannot run stats. The following columns are missing: {missing_cols}")
        print("Did you run the Feature Engineering pipeline first?")
        return

    # 1. Basic Summary Statistics (Count, Mean, Std, Min, Max, Quantiles)
    print("\nBasic Summary Statistics:")
    # Force Pandas to show all columns and widen the display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    stats = df[target_features].describe().round(2)
    stats[''] = stats.index  # Copies the left-hand labels to the far right
    print(stats)

    # 2. Variance (Measures data dispersion/spread)
    print("\nVariance:")
    print(df[target_features].var().round(2))

    # 3. Skewness (Measures symmetry)
    print("\nSkewness:")
    print(df[target_features].skew().round(2))

    # 4. Kurtosis (Measures the shape/tails of the distribution)
    print("\nKurtosis:")
    print(df[target_features].kurtosis().round(2))

def plot_executive_profile(df, categorical_col, top_n=15, plot_color='steelblue'):
    #  Creates a generalized 'Small Multiples' Grid Chart showing the operational profile
    # based on ANY categorical feature passed as a parameter.
    # Dynamically prevents packed charts by filtering to the Top N largest categories.

    # 1. THE SMART FILTER: Avoid the "Packed Chart" penalty
    # If the category has hundreds of unique values (like Airports), it slices the top N.
    # If it has fewer than N (like Airlines), it just keeps all of them.
    top_categories = df[categorical_col].value_counts().nlargest(top_n).index
    df_filtered = df[df[categorical_col].isin(top_categories)]

    # 2. Define the 6 elite features to analyze
    features_to_plot = [
        ('ARR_DELAY', 'Target: Average Arrival Delay', 'Minutes'),
        ('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Minute'),
        ('DISTANCE', 'Fleet: Average Flight Distance', 'Miles'),
        ('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Route Bottleneck Score', 'Score'),
        ('ROUTE_FREQUENCY', 'Operations: Route Frequency', 'Flight Count'),
        ('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)')
    ]

    # 3. CREATE THE GRID: 3 rows, 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    axes = axes.flatten()

    # 4. Loop through the features and plot them on the grid
    for i, (feature, title, ylabel) in enumerate(features_to_plot):
        ax = axes[i]

        # Calculate sorting order based on mean (Descending) dynamically for the chosen category
        order = df_filtered.groupby(categorical_col)[feature].mean().sort_values(ascending=False).index

        # Create the bar plot WITH Standard Deviation error bars
        sns.barplot(
            data=df_filtered,
            x=categorical_col,
            y=feature,
            color=plot_color,
            errorbar='sd',  # Shows uncertainty/dispersion
            capsize=0.1,
            err_kws={'linewidth': 1.5, 'color': 'black'},
            order=order,
            ax=ax
        )

        # Add exact value labels to the top of the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=15, color='black', size=10)

        # Formatting: dynamically generate the X-axis label based on the parameter!
        ax.set_title(title, fontsize=15, pad=12, weight='bold')
        ax.set_xlabel(categorical_col.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=0)

        # Remove top and right borders to reduce clutter
        sns.despine(ax=ax)

    # Adjust overall spacing so titles and labels don't overlap
    plt.tight_layout(pad=4.0)
    plt.show()


def plot_categorical_boxplots(df):
    # Creates a 'Small Multiples' Grid Chart of Boxplots to show the statistical
    # spread of our elite continuous features, grouped by Flight Haul Type.

    # 1. Define the 6 continuous features to analyze
    # Format: (Column_Name, Chart_Title, Y_Axis_Label)
    features_to_plot = [
        ('ARR_DELAY', 'Target: Arrival Delay Distribution', 'Minutes'),
        ('SCHEDULED_SPEED', 'Aggressiveness: Scheduled Speed', 'Miles / Min'),
        ('AIRLINE_HUB_DOMINANCE', 'Monopoly: Hub Dominance', 'Ratio (0 to 1)'),
        ('ROUTE_BOTTLENECK_INTERACTION', 'Congestion: Bottleneck Score', 'Score'),
        ('ORIGIN_CONGESTION', 'Traffic: Origin Congestion', 'Concurrent Flights'),
        ('CRS_ELAPSED_TIME', 'Planning: Scheduled Duration', 'Minutes')
    ]

    # 2. Create the Grid: 3 rows, 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 10))
    axes = axes.flatten()

    # 3. Create a smart 100k sample for Boxplots.
    # Boxplots take a massive amount of computational power to draw 2.9 million outliers.
    df_sample = df.sample(n=100000, random_state=42)

    # 4. Loop through the features and plot them
    for i, (feature, title, ylabel) in enumerate(features_to_plot):
        ax = axes[i]

        sns.boxplot(
            data=df_sample,
            x='FLIGHT_HAUL_TYPE',
            y='ARR_DELAY',
            hue='FLIGHT_HAUL_TYPE',  # Add this (make sure it matches your x variable!)
            legend=False,  # Add this so it doesn't print a redundant legend
            palette='viridis',
            ax=ax
        )

        # Apply your zoom logic specifically for ARR_DELAY to see the IQR box clearly
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
    """
    Generalized function to plot the average arrival delay over a specific time dimension.
    """
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
    """
    Plots the frequency and density distribution of arrival delays.
    """
    plt.figure(figsize=(10, 6))

    # Increase bins to 100 for finer detail in the zoomed area
    sns.histplot(data=df, x='ARR_DELAY', bins=100, kde=True, color='mediumpurple')

    # Formatting and zoom
    plt.xlim(-60, 300)
    plt.title('Distribution Shape: Frequency and Density of Arrival Delays', fontsize=14, pad=15)
    plt.xlabel('Arrival Delay (Minutes)')
    plt.ylabel('Frequency (Number of Flights)')
    sns.despine()
    plt.show()


def plot_distance_vs_duration(df):
    """
    Plots a scatter plot to show the linear correlation between flight distance
    and scheduled duration.
    """
    plt.figure(figsize=(10, 6))

    # Scatter with reduced alpha to handle overplotting
    sns.scatterplot(data=df, x='DISTANCE', y='CRS_ELAPSED_TIME',
                    alpha=0.1, s=10, color='dodgerblue')

    # Formatting
    plt.title('Linear Correlation: Distance vs. Scheduled Flight Duration', fontsize=14, pad=15)
    plt.xlabel('Flight Distance (Miles)')
    plt.ylabel('Scheduled Elapsed Time (Minutes)')
    sns.despine()
    plt.show()


def plot_executive_bubble_chart(df):
    # Generates an Executive Bubble Chart summarizing congestion vs. delay
    # by airline, using flight volume as the bubble size.

    plt.figure(figsize=(12, 7))

    # Aggregate data by Airline to summarize the massive dataset
    airline_stats = df.groupby('AIRLINE_CODE').agg({
        'ROUTE_BOTTLENECK_INTERACTION': 'mean',
        'ARR_DELAY': 'mean',
        'ROUTE_FREQUENCY': 'count'  # Using count of flights for bubble size
    }).reset_index()

    # Create the Bubble Chart
    sns.scatterplot(
        data=airline_stats,
        x='ROUTE_BOTTLENECK_INTERACTION',
        y='ARR_DELAY',
        size='ROUTE_FREQUENCY',
        sizes=(100, 2000),  # Minimum and maximum bubble sizes
        hue='AIRLINE_CODE',
        palette='tab20',
        alpha=0.7,
        edgecolor='black'
    )

    # Add text labels inside/next to the bubbles for clarity
    for i in range(len(airline_stats)):
        plt.text(airline_stats['ROUTE_BOTTLENECK_INTERACTION'][i],
                 airline_stats['ARR_DELAY'][i],
                 airline_stats['AIRLINE_CODE'][i],
                 horizontalalignment='center', size='small', color='black', weight='bold')

    plt.title('Multivariate Summary: Congestion vs Delay by Airline Volume', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Average Route Bottleneck Score', fontsize=12)
    plt.ylabel('Average Arrival Delay (Minutes)', fontsize=12)
    plt.legend([], [], frameon=False)  # Hide the legend to reduce clutter
    sns.despine()
    plt.show()


def plot_speed_violin_distribution(df):
    # Generates a Violin Plot showing the density and shape of scheduled
    # speeds across different flight haul types.

    plt.figure(figsize=(10, 6))

    # Shows the exact probability density shape of scheduled speeds
    sns.violinplot(
        data=df,
        x='FLIGHT_HAUL_TYPE',
        y='SCHEDULED_SPEED',
        hue='FLIGHT_HAUL_TYPE',  # <-- 1. Add this (make it identical to your x variable)
        legend=False,  # <-- 2. Add this so it doesn't print a redundant legend
        palette='magma'
    )

    plt.title('Distribution Shape: Scheduled Speed by Flight Haul Type', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Flight Haul Type', fontsize=12)
    plt.ylabel('Scheduled Speed (Miles/Minute)', fontsize=12)
    sns.despine()
    plt.show()


def plot_dominance_bubble(df):
    # 2. Executive Bubble Chart
    # Summarizes the same relationship, but aggregated by Airline to
    # tell a high-level business story.

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
    # Plots a sampled scatterplot to show the mathematical correlation
    # between Route Frequency (Flight Volume) and Arrival Delay.

    # 1. Create a 100k sample to prevent PyCharm from freezing and avoid packed charts
    df_sample = df.sample(n=100000, random_state=42)

    plt.figure(figsize=(10, 6))

    # 2. Scatterplot with s=10 and alpha=0.1 to fix overplotting
    sns.scatterplot(
        data=df_sample,
        x='ROUTE_FREQUENCY',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='dodgerblue'
    )

    # Zoom in specifically on the relevant delay spread, cutting out extreme outliers
    plt.ylim(-60, 300)

    # Professional Formatting
    plt.title('Linear Correlation: Route Frequency vs. Arrival Delay', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Route Frequency (Total Flights on this Route)', fontsize=12)
    plt.ylabel('Arrival Delay (Minutes)', fontsize=12)

    sns.despine()
    plt.show()


def plot_congestion_vs_delay_grid(df):

    df_sample = df.sample(n=100000, random_state=42)

    # Create a 1x2 grid
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # -----------------------------------------------
    # LEFT CHART: Origin Congestion
    # -----------------------------------------------
    sns.scatterplot(
        data=df_sample,
        x='ORIGIN_CONGESTION',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='crimson',
        ax=axes[0]
    )

    axes[0].set_ylim(-60, 300)
    axes[0].set_title('Origin Congestion vs. Arrival Delay', fontsize=14, weight='bold')
    axes[0].set_xlabel('Origin Congestion (Concurrent Departures)', fontsize=12)
    axes[0].set_ylabel('Arrival Delay (Minutes)', fontsize=12)

    sns.despine(ax=axes[0])

    # -----------------------------------------------
    # RIGHT CHART: Destination Congestion
    # -----------------------------------------------
    sns.scatterplot(
        data=df_sample,
        x='DESTINATION_CONGESTION',
        y='ARR_DELAY',
        alpha=0.1,
        s=10,
        color='mediumseagreen',
        ax=axes[1]
    )

    axes[1].set_ylim(-60, 300)
    axes[1].set_title('Destination Congestion vs. Arrival Delay', fontsize=14, weight='bold')
    axes[1].set_xlabel('Destination Congestion (Concurrent Arrivals)', fontsize=12)
    axes[1].set_ylabel('')

    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.show()