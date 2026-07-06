# implementation of semi-parametric quantile estimator, eqn. (2.2) from Hutson, A. D. (2002). A semi-parametric quantile function estimator for use in bootstrap estimation procedures. Statistics and Computing, 12(4), 331-338.

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
quantile_tail_extrapolation <- function(input_sample, probs){
  result <- NULL
  n <- length(input_sample)
  ordered_sample <- sort(input_sample)
  if (all(probs > 0 & probs < 1)){
    result <- lapply(probs, FUN = get_single_extrapol_quantile, ordered_sample = ordered_sample, positive_only = F)
    names(result) <- paste0(probs*100, '%')
  } else {
    stop(paste0('Quantile levels not 0 < u < 1: ', probs))
  }
  if (!(is.null(result))){
    return(unlist(result))  
  } else
    stop('error in extrapolation quantile calculation occured.')
}

quantile_tail_extrapolation(input_sample = c(4:2, 1:10), probs = c(0.0004, 0.2, 0.999))
