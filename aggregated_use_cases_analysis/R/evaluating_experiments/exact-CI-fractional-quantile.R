library(dplyr)
library(magrittr)
library(tidyr)

# implementation of exact confidence intervals approximated by fractional quantiles based on linear interpolation
# Hutson, Alan D. "Calculating nonparametric confidence intervals for quantiles using fractional order statistics." Journal of Applied Statistics 26.3 (1999): 343-353.

# helper function to calculate cdf (beta distribution) of some quantile level according to eq. (5), p. 346 (denoted by u therein)
# parameter u in this implementation of the helper function corresponds to u1, u2 (in eq. (5)) that determines the shape parameters of the beta distribution
calculate_beta_prob <- function(n, quantile_level, u){
  a <- (n+1)*u
  b <- (n+1)*(1-u)
  return(pbeta(quantile_level, a, b))
}

# helper function to numerically obtain interval limits for given confidence level according to eqn. (7), (8), p. 346
find_u <- function(n, quantile_level, target_prob){
  # Use uniroot to find u such that calculate_beta_prob(n, quantile_level, u) = target_prob
  result <- uniroot(function(u) calculate_beta_prob(n = n, quantile_level = quantile_level, u = u) - target_prob, interval = c(0, 1))
  return(result$root)
}

n <- 20
quantile_level <- 0.5
confidence_level <- 0.95
# at the end we want to have: P(u1 <= u <= u2) ≈ confidence_level

calculate_limits_exact_CI <- function(n, quantile_level, confidence_level){
  u_1 <- find_u(n = n, quantile_level = quantile_level, target_prob = 1 - (1-confidence_level)/2)
  u_2 <- find_u(n = n, quantile_level = quantile_level, target_prob = (1-confidence_level)/2)
  confidence_level_reached <- calculate_beta_prob(n = n, quantile_level = quantile_level, u = u_1) - calculate_beta_prob(n = n, quantile_level = quantile_level, u = u_2)
  return(list(lower_limit = u_1, 
              upper_limit = u_2, 
              conf_level_reached = confidence_level_reached))
}


# for testing purposes: dataframe test_settings provides results similar to Table 2, p. 348
n <- c(15, 20, 25)
quantile_level <- c(0.5, 0.75)
confidence_level <- c(0.9, 0.95, 0.99)

test_settings <- expand.grid(n = n, quantile_level = quantile_level, confidence_level = confidence_level)

test_settings %<>% rowwise() %>%
  mutate(
    limits = list(calculate_limits_exact_CI(n, quantile_level, confidence_level)),
    lower_limit = limits$lower_limit,
    upper_limit = limits$upper_limit,
    conf_level_reached = limits$conf_level_reached
  ) %>%
  select(-limits)

# helper function to estimate fractional quantile with linear interpolation according to eqn. (3), p.345
calculate_quantile_approximation_linear <- function(sample, quantile_level){
  n <- length(sample)
  orderd_sample <- sort(sample)
  fraction_index <- (n+1)*quantile_level
  fraction_index_floor <- floor(fraction_index)
  epsilon <- fraction_index-fraction_index_floor
  # Switch block based on epsilon
  if ((epsilon > 0) & (epsilon <1)){
    result <- (1 - epsilon)*orderd_sample[fraction_index_floor] + epsilon*orderd_sample[fraction_index_floor+1]
  } else if (epsilon == 0) {
    # hint: fraction_index (but not _floor) is used intentionally here to throw an error in case that something strange happens
    result <- orderd_sample[fraction_index]
  } else {
    stop("Unexpected value of epsilon")
  }
  result <- list(result = result, 
                 quantile_level = quantile_level)
  return(result)
}

calculate_approximate_non_parametric_CI <- function(sample, quantile_level, confidence_level){
  n <- length(sample)
  tmp_limits <- calculate_limits_exact_CI(n = n, quantile_level = quantile_level, confidence_level = confidence_level)
  lower_tmp <- calculate_quantile_approximation_linear(sample = sample, quantile_level = tmp_limits$lower_limit)
  upper_tmp <- calculate_quantile_approximation_linear(sample = sample, quantile_level = tmp_limits$upper_limit)
  result <- list(
    quantile_level_req = quantile_level, 
    confidence_level_req = confidence_level,
    confidence_level_reached = tmp_limits$conf_level_reached,
    lower_ci = lower_tmp$result, 
    upper_ci = upper_tmp$result, 
    quantile_level_lower_ci = lower_tmp$quantile_level,
    quantile_level_lower_ci = upper_tmp$quantile_level
  )
  return(result)
}

my_sample <- 1:10
my_confidence_level <- 0.9
my_quantile_level <- 0.5
calculate_approximate_non_parametric_CI(sample = my_sample, quantile_level = my_quantile_level, confidence_level = my_confidence_level)
