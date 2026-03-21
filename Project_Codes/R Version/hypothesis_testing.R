library(dplyr)

# ---------------------------------------
# Kruskal-Wallis Test
# ---------------------------------------
perform_kruskal_wallis <- function(dataframe, group_col, target_col, alpha=0.05) {
  # Performs the Kruskal-Wallist test for Hypothesis Testing

  cat(paste0("--- Kruskal-Wallis Test: ", group_col, " vs ", target_col, " ---\n"))

  # Group the target variable by the grouping column
  # It creates a list of arrays for each category
  groups <- split(dataframe[[target_col]], dataframe[[group_col]])

  # 2. Perform the test  ('*' to unpack the list of arrays)
  test_result <- kruskal.test(groups)

  stat <- test_result$statistic
  p_value <- test_result$p.value

  # Print formatted results
  cat(sprintf("H-statistic: %.2f\n", stat))
  cat(sprintf("P-value: %.4f\n", p_value))

  # Interpret the results
  if (p_value < alpha) {
    cat(paste0("Conclusion: Reject H0. '", group_col, "' systematically impacts '", target_col, "'.\n\n"))
  } else {
    cat("Conclusion: Fail to reject H0. No significant impact detected.\n\n")
  }

  return(list(stat=stat, p_value=p_value))
}
