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

CI_np_exact <- function(n, confidence_level, quantile_level, deterministic = FALSE, verbose = FALSE, seed = 42) {
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
  
  # return interval with max lambda only
  lp_intervals <- lp_intervals %>% ungroup() %>% select(-interval_length) %>% filter(lambda > 0) %>% mutate(lower = i*lambda, upper = j*lambda)

  # Check if lp_intervals is empty
  if (sum(lp_intervals$lambda) == 0) {
    # Create a data frame with NA values having the same column names
    result <- data.frame(lower = NA, upper = NA)
  } else {
    if (!deterministic) {
      if (!verbose) { # default behavior with randomization
        set.seed(seed = seed)
        random_index <- sample(1:2, size = 1, prob = lp_intervals$lambda)
        result <- lp_intervals[random_index, ] %>% 
          select(i, j) %>%
          rename(lower = i, upper = j)  
      } else { # hint: this option is needed for simulation purposes only to reduce computation time
        result <- lp_intervals %>% 
          select(i, j, lambda) %>%
          rename(lower = i, upper = j)  
      }
    } else { # give back interval with maximum lambda
      result <- lp_intervals %>%
        slice_max(order_by = lambda, n = 1) %>%
        select(i, j) %>%
        rename(lower = i, upper = j)
    }
    result <- as.data.frame(result)
  }
  return(result)
}