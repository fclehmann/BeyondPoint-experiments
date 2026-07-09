#' Nonparametric Quantile Confidence Interval (Asymptotic Normality)
#'
#' Computes a confidence interval for a quantile based on asymptotic normality
#' of order statistics.
#'
#' See Example 21.8, p.309 in Aad W. Van der Vaart. *Asymptotic Statistics*.
#' Vol. 3. Cambridge University Press, 2000. doi: 10.1017/CBO9780511802256
#'
#' @param input_sample Numeric vector. The observed sample. Ignored if
#'   `level_mode = TRUE`.
#' @param quantile_level Numeric in (0, 1). The target quantile level p.
#' @param confidence_level Numeric in (0, 1). The nominal confidence level
#'   (e.g., 0.95).
#' @param level_mode Logical. If TRUE, only the probability levels of the
#'   interval bounds are returned (no sample quantiles computed). Requires
#'   `n` to be supplied.
#' @param n Integer. Sample size. Required when `level_mode = TRUE`; otherwise
#'   inferred from `input_sample`.
#' @param negate Logical. If TRUE, the CI is computed on the sign-flipped
#'   sample using the complementary quantile level (1 - quantile_level), and
#'   the resulting bounds are negated and swapped back. This is useful for
#'   improving coverage at extreme upper quantiles, where order-statistic
#'   asymmetry can degrade the standard interval.
#'
#' @return Named numeric vector with elements `lower` and `upper`, or `NA`s
#'   if the asymptotic interval falls outside valid index bounds.
CI_np_asymptotic <- function(input_sample = NULL,
                             quantile_level,
                             confidence_level,
                             level_mode = FALSE,
                             n = NULL,
                             negate = FALSE) {
  
  # In level_mode, only 'n' is required; ignore 'input_sample' if provided
  if (level_mode) {
    if (is.null(n) || n <= 0 || n %% 1 != 0) {
      stop("In 'level_mode', a valid sample size 'n' must be provided as a positive integer.")
    }
  } else {
    if (is.null(input_sample)) {
      stop("Input sample data must be provided when 'level_mode' is FALSE.")
    }
    n <- length(input_sample)
  }
  
  # Validate inputs
  if (n <= 0 || n %% 1 != 0) stop("Sample size 'n' must be a positive integer.")
  if (quantile_level <= 0 || quantile_level >= 1) stop("Quantile level must be between 0 and 1.")
  if (confidence_level <= 0 || confidence_level >= 1) stop("Confidence level must be between 0 and 1.")
  if (!is.logical(negate) || length(negate) != 1 || is.na(negate)) {
    stop("'negate' must be a single logical value (TRUE or FALSE).")
  }
  
  # --- Sign transformation (negate sample; use complementary quantile) --------
  p_eff <- if (negate) 1 - quantile_level else quantile_level
  
  # Asymptotic interval calculation
  normality_quantile <- qnorm(p = (1 - confidence_level) / 2)
  tmp <- normality_quantile * sqrt(p_eff * (1 - p_eff) / n)
  interval_levels <- c(lower = p_eff + tmp, upper = p_eff - tmp)
  
  # --- Validity check ---------------------------------------------------------
  if (n * interval_levels['lower'] >= 1 && n * interval_levels['upper'] <= n) {
    if (level_mode) {
      result <- c(interval_levels['lower'], interval_levels['upper'])
    } else {
      sample_used <- if (negate) -input_sample else input_sample
      q_bounds    <- unname(quantile(x = sample_used, probs = interval_levels, type = 7))
      
      if (negate) {
        # Back-transform: negate and swap so that lower < upper again
        result <- c(lower = -q_bounds[2], upper = -q_bounds[1])
      } else {
        result <- c(lower = q_bounds[1], upper = q_bounds[2])
      }
    }
  } else {
    result <- c(lower = NA, upper = NA)
  }
  
  return(result)
}


#' Quantile CI Estimation Based on a Concrete Distribution
#'
#' Computes a confidence interval for a quantile assuming a known parametric
#' distribution, using the asymptotic normality of the sample quantile.
#'
#' @param data Non-empty numeric vector.
#' @param p Numeric in [0, 1]. Target quantile level.
#' @param dist_name Character. Distribution suffix used to build `q<dist_name>`
#'   and `d<dist_name>` (e.g., "norm", "exp", "gamma").
#' @param params Named list of parameters passed to the density/quantile
#'   functions.
#' @param conf_level Numeric in (0, 1). Nominal confidence level.
#'
#' @return Named numeric vector with `lower` and `upper` bounds.
quantile_ci_asympt_normality <- function(data, p, dist_name, params, conf_level = 0.95) {
  if (!is.numeric(data) || length(data) == 0) {
    stop("data must be a non-empty numeric vector.")
  }
  if (p < 0 || p > 1) {
    stop("p must be between 0 and 1.")
  }
  if (!is.list(params)) {
    stop("params must be a named list.")
  }
  
  n     <- length(data)
  q_hat <- quantile(data, probs = p)
  
  # Density at the true (theoretical) quantile
  quantile_func <- paste0("q", dist_name)
  true_quantile <- do.call(quantile_func, c(p, params))
  density_func  <- paste0("d", dist_name)
  f             <- do.call(density_func, c(true_quantile, params))
  
  se       <- sqrt((p * (1 - p)) / (n * f^2))
  z_value  <- qnorm((1 + conf_level) / 2)
  ci_lower <- q_hat - z_value * se
  ci_upper <- q_hat + z_value * se
  
  return(c(lower = ci_lower, upper = ci_upper))
}


# --- Example usage ------------------------------------------------------------
# data   <- rnorm(100)             # Sample data from a normal distribution
# p      <- 0.5                    # Median
# params <- list(mean = 0, sd = 1) # Parameters for dnorm/qnorm
#
# ci <- quantile_ci_asympt_normality(data, p, dist_name = "norm", params, conf_level = 0.95)
# print(ci)
#
# # Nonparametric CI with sign transformation for an upper-tail quantile:
# ci_np <- CI_np_asymptotic(input_sample = data,
#                           quantile_level   = 0.95,
#                           confidence_level = 0.95,
#                           negate           = TRUE)
# print(ci_np)
