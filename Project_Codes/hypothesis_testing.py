from scipy.stats import kruskal

# =========================================================
# PROCEDURAL FUNCTION: Kruskal-Wallis Test
# =========================================================
def perform_kruskal_wallis(dataframe, group_col, target_col, alpha=0.05):

    print(f"--- Kruskal-Wallis Test: {group_col} vs {target_col} ---")

    # 1. Group the target variable by the grouping column
    # This automatically creates a list of arrays for every unique category
    groups = [group[target_col].values for name, group in dataframe.groupby(group_col)]

    # 2. Perform the test by unpacking the list of arrays using '*'
    stat, p_value = kruskal(*groups)

    # 3. Print formatted results (2 decimals for stat, 4 for p-value)
    print(f"H-statistic: {stat:.2f}")
    print(f"P-value: {p_value:.4f}")

    # 4. Interpret the results
    if p_value < alpha:
        print(f"Conclusion: Reject H0. '{group_col}' systematically impacts '{target_col}'.\n")
    else:
        print(f"Conclusion: Fail to reject H0. No significant impact detected.\n")

    return stat, p_value
