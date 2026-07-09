# implementation of semi-parametric quantile estimator, eqn. (2.2) from Hutson, A. D. (2002). A semi-parametric quantile function estimator for use in bootstrap estimation procedures. Statistics and Computing, 12(4), 331-338.
library(dplyr)
library(magrittr)
library(tidyr)
library(tibble)

# helper function for getting single quantile
get_single_extrapol_quantile <- function(ordered_sample, qlevel, positive_only = F){
  tmp_n <- length(ordered_sample)
  # only minimal error handling as function is used within wrapper function only
  n_mod <- tmp_n+1
  result <- NULL
  if (0 < qlevel & qlevel <= 1/n_mod){
    if (!positive_only)
      result <- ordered_sample[1] + (ordered_sample[2] - ordered_sample[1])*log(n_mod*qlevel) # here it is unclear from the paper which log base is used, conjecture: natural log
    else {
      epsilon <- n_mod*qlevel
      result <- epsilon*ordered_sample[1]
    }
  }
  if (1/n_mod < qlevel & qlevel < tmp_n/n_mod){
    q_index <- floor(n_mod*qlevel)
    epsilon <- n_mod*qlevel - q_index
    result <- (1-epsilon)*ordered_sample[q_index] + epsilon*ordered_sample[q_index+1]
  }
  if (tmp_n/n_mod <= qlevel & qlevel < 1){
    result <- ordered_sample[tmp_n] - (ordered_sample[tmp_n] - ordered_sample[tmp_n-1])*log(n_mod*(1-qlevel))
  }
  if (!(is.null(result))){
    return(result)  
  } else
    stop('error in extrapolation single quantile calculation occured.')
}


# wrapper function for calculating multiple quantile levels
quantile_tail_extrapolation <- function(input_sample, probs, pos_only = FALSE){
  result <- NULL
  n <- length(input_sample)
  ordered_sample <- sort(input_sample)
  if (all(probs > 0 & probs < 1)){
    result <- lapply(probs, FUN = get_single_extrapol_quantile, ordered_sample = ordered_sample, positive_only = pos_only)
    names(result) <- paste0(probs*100, '%')
  } else {
    stop(paste0('Quantile levels not 0 < u < 1: ', probs))
  }
  if (!(is.null(result))){
    return(unlist(result))  
  } else
    stop('error in extrapolation quantile calculation occured.')
}

# helper function to calculate multiple statistics 
calculate_boot_statistics <- function(x, probs, q_type) {
  quantiles <- quantile(x, probs = probs, type = q_type)
  mean_value <- mean(x)
  c(quantiles, mean = mean_value)
}

#### Docs: Semi-parametric Bootstrapping Function ####
#
# This function performs semi-parametric bootstrapping based on a given observed sample or a simulated sample.
# It calculates bootstrap statistics using the quantile tail extrapolation method described by Hutson (2002).
#
# Parameters:
#   seed (integer): Seed for random number generation.
#   num_bootsamples (integer): Number of bootstrap samples to generate.
#   q_levels (numeric vector): Quantile levels to estimate.
#   q_type (integer): Type of quantile calculation to use (default is 7).
#   observed_sample (numeric vector, optional): Observed sample values. If provided, the function performs non-parametric bootstrapping.
#   sample_size (integer): Sample size for simulation (required if observed_sample is NULL).
#   sample_distr_func (function): Distribution to simulate samples from (required if observed_sample is NULL).
#   sample_distr_params (list): Parameters for the distribution function (required if observed_sample is NULL).
#
# Returns:
#   data.frame: A data frame containing the bootstrap statistics.
#
# Examples:
# # Example 1: Using observed sample (non-parametric bootstrap)
# observed_sample <- c(4, 2, 3, 1, 5, 9, 7, 8, 6, 10)
# result_observed_sample <- semiparametric_bootstrapping(
#   seed = 123,
#   num_bootsamples = 1000,
#   q_levels = c(0.05, 0.25, 0.5, 0.75, 0.95),
#   q_type = 7,
#   observed_sample = observed_sample
# )
# print(result_observed_sample)
#
# # Example 2: Using simulation scenario
# result_simulation <- semiparametric_bootstrapping(
#   seed = 123,
#   num_bootsamples = 1000,
#   q_levels = c(0.05, 0.25, 0.5, 0.75, 0.95),
#   q_type = 7,
#   observed_sample = NULL,
#   sample_size = 100,
#   sample_distr_func = rnorm,
#   sample_distr_params = list(mean = 0, sd = 1)
# )
# print(result_simulation)
#

#### core function ####
semiparametric_bootstrapping <- function(seed, num_bootsamples, q_levels, q_type, # general config stuff
                                         observed_sample = NULL, # case of using bootstrapping concrete sample (no simulation scenario)
                                         sample_size, sample_distr_func, sample_distr_params, # simulation scenario
                                         pos_only = FALSE
) {
  
  set.seed(seed)
  
  if (!is.null(observed_sample)){
    # Some concrete sample is provided. 
    sample_size <- length(observed_sample)
    # case of non-parametric bootstrap
    initial_sample <- observed_sample
  } else { 
    # case of simulation (observed_sample == NULL)
    initial_sample <- do.call(sample_distr_func, c(list(n = sample_size), sample_distr_params))
  } 
  
  uni_samples <- replicate(n = num_bootsamples, runif(n = sample_size, min = 0, max = 1))
  
  # create bootstrap samples based on probability integral transform , single sample in column
  # dependency 'semiparametric-quant-extrapolation-est.R'
  boot_samples <- t(apply(X = uni_samples, MARGIN = 1, FUN = quantile_tail_extrapolation, input_sample = initial_sample, pos_only = pos_only))
  
  # hint for calculating further statistics, modify the helper function calculate_boot_statistics()
  boot_statistic <- apply(X = boot_samples, MARGIN = 2, FUN = calculate_boot_statistics, probs = q_levels, q_type = q_type)
  boot_statistic <- as.data.frame(t(boot_statistic))
  
  return(boot_statistic)  
}
