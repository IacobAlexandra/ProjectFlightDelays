from scipy.stats import kruskal

# ---------------------------------------
# Kruskal-Wallis Test
# ---------------------------------------
def perform_kruskal_wallis(dataframe, group_col, target_col, alpha=0.05):
    """Performs the Kruskal-Wallist test for Hypothesis Testing"""

    print(f"--- Kruskal-Wallis Test: {group_col} vs {target_col} ---")

    # Group the target variable by the grouping column
    # It creates a list of arrays for each category
    groups = [group[target_col].values for name, group in dataframe.groupby(group_col)]

    # 2. Perform the test  ('*' to unpack the list of arrays)
    stat, p_value = kruskal(*groups)

    # Print formatted results
    print(f"H-statistic: {stat:.2f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the results
    if p_value < alpha:
        print(f"Conclusion: Reject H0. '{group_col}' systematically impacts '{target_col}'.\n")
    else:
        print(f"Conclusion: Fail to reject H0. No significant impact detected.\n")

    return stat, p_value
