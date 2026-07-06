library(dplyr)
library(magrittr)
library(tidyr)
library(lpSolve)

calculate_binomial_prob <- function(i, j, n, q) {
  if (i >= j) {
    stop("Error: i must be less than j")
  }
  pbinom(j-1, n, q) - pbinom(i-1, n, q)
}

construct_exact_ci_intervals <- function(n, confidence_level, quantile_level) {
  # Generate all pairs (i, j) where i < j
  intervals_tmp <- expand.grid(i = 1:(n-1), j = 2:n) %>%
    filter(i < j) %>%
    mutate(interval_length = j - i) %>%
    rowwise() %>%
    mutate(interval_prob = calculate_binomial_prob(i, j, n, quantile_level))
  
  # Objective function coefficients
  objective_coeffs <- intervals_tmp$interval_length
  
  # Constraint: confidence level
  constraint_confidence_level <- intervals_tmp$interval_prob
  
  # Constraint: sum of lambda coefficients equals 1
  num_lambdas <- length(objective_coeffs)
  constraint_sum_lambda <- rep(1, num_lambdas)
  
  # Right-hand side of the constraints
  rhs <- c(confidence_level, 1)
  
  # Direction of the constraints
  direction <- c("=", "=")
  
  # Combine constraints
  constraints <- rbind(constraint_confidence_level, constraint_sum_lambda)
  
  # Solve the linear program
  solution <- lp("min", objective_coeffs, constraints, direction, rhs, compute.sens = TRUE)
  
  lp_intervals <- intervals_tmp
  lp_intervals$lambda <- solution$solution
  lp_intervals %<>% filter(lambda != 0)
  
  # Extract intervals where interval_prob >= confidence_level
  alternative_intervals <- intervals_tmp %>%
    filter(interval_prob >= confidence_level)
  
  # Find minimum interval length among alternative intervals
  if (nrow(alternative_intervals) > 0){
    min_interval_length <- min(alternative_intervals$interval_length)
    # Select intervals with minimum length and arrange by interval_prob
    alternative_intervals %<>%
      filter(interval_length == min_interval_length) %>%
      arrange(interval_prob) %>%
      head(10)
  } else {
    alternative_intervals <- NULL
  }
  
  # Store results in a list
  results <- list(
    lp_intervals = lp_intervals,
    alternative_intervals = alternative_intervals
  )
  
  return(results)
}

# Define the parameters to iterate over
# Example parameters
n_values <- c(50, 100)
quantile_levels <- c(0.05, 0.25, 0.5)
confidence_levels <- c(0.9, 0.95)

# Generate all combinations of parameters
param_combinations <- expand.grid(n = n_values, quantile_level = quantile_levels, confidence_level = confidence_levels)

# Apply the function to each combination using lapply
results_list <- lapply(1:nrow(param_combinations), function(i) {
  params <- param_combinations[i, ]
  results <- construct_exact_ci_intervals(params$n, params$confidence_level, params$quantile_level)
  
  # Add parameters to results
  results$params <- params
  
  return(results)
})

