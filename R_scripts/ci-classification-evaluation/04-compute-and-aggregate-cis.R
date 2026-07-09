# this script calculates and aggregates all CIs: t-distributed CI for mean, bootstrapped (only aggregation), exact non-parametric CI for quantiles, and non-parametric CIs based on asymptotic normality
# usage hint: before running the whole script check all load() and saveResult() calls for not overwriting existing stuff
library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)
library(foreach)
library(doParallel)
library(parallel)
library(purrr)
library(optparse)

##### Parallel backend #######
num_cores <- 0
# Check if SLURM_JOB_CPUS_PER_NODE is set
slurm_cpus <- Sys.getenv("SLURM_JOB_CPUS_PER_NODE", unset = NA)

if (is.na(slurm_cpus)) {
  cat("SLURM_JOB_CPUS_PER_NODE is not set. Using detectCores() to determine the number of CPU cores.\n")
  num_cores <- detectCores()
} else {
  num_cores <- as.numeric(slurm_cpus)
  if (is.na(num_cores) || num_cores <= 0) {
    stop("Invalid value for SLURM_JOB_CPUS_PER_NODE. Please check the environment variable.")
  }
  cat("SLURM_JOB_CPUS_PER_NODE is set to:", num_cores, "\n")
}


cl <- makeCluster(num_cores-2)
registerDoParallel(cl)

##### config and helper functions ####
# Define command line arguments
option_list <- list(
  make_option(c("-d", "--datafile_path"), type="character", default=NULL,
              help="Path of Rdata data file.", metavar="character")
)

# Create an option parser
opt_parser <- OptionParser(option_list=option_list)

# Parse the arguments
args <- parse_args(opt_parser)

# path stuff
data_dir     <- args$datafile_path
results_path <- file.path(data_dir, 'final_results')
true_statistics_file <- file.path(results_path, 'true_statistics.Rdata')

# lib_path is derived from this script's own location - no env vars, no config
script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
lib_path <- normalizePath(file.path(dirname(script_path), "..", "libs"))
exact_non_param_CI_source_file      <- file.path(lib_path, "lib-np-exact-CI-linopt.R")
asymptotic_normality_CI_source_file <- file.path(lib_path, "lib-np-asymptotic-CI.R")

source(exact_non_param_CI_source_file)
source(asymptotic_normality_CI_source_file)

subsample_files <- list.files(path = data_dir, pattern = '^subsamples_config=.*\\.Rdata$', full.names = TRUE)

# helper function to save intermediate results
saveResult <- function(input_object, target_path = results_path) {
  # Get the object's name as a string
  object_name <- deparse(substitute(input_object))
  if (!dir.exists(target_path)) {
    dir.create(target_path, recursive = TRUE)
  }
  full_result_path <- file.path(target_path, paste0(object_name, '.Rdata'))
  assign(object_name, input_object)  # Assign object to its original name dynamically
  save(list = object_name, file = full_result_path)
  cat('Saving object "', object_name, '" to "', full_result_path, '".\n', sep = '')
}

# helper function to get experiment meta data
get_meta_data <- function(filename) {
  file_name <- str_remove(filename[1], "\\.Rdata$")
  matches <- str_split(file_name, '-')[[1]]
  meta_data <- list(
    n = as.factor(str_match(file_name, "n(\\d+)")[, 2]),
    seed = str_match(file_name, "seed=(\\d+)")[, 2], 
    num_samples = as.factor(str_match(file_name, "(\\d+)_seed")[, 2])
  )
  return(as.data.frame(meta_data))
}

calculate_quantiles <- function(subsamples, bounds) {
  if (any(bounds < 0 | bounds > 1, na.rm = TRUE)) {
    return(rep(-999, length(bounds)))
  }
  if (all(is.na(bounds))) {
    return(rep(NA, length(bounds)))
  }
  unname(quantile(subsamples, probs = bounds, type = 7, na.rm = FALSE))
}

calculate_quantiles_mirror <- function(subsamples, bounds) {
  if (any(bounds < 0 | bounds > 1, na.rm = TRUE)) {
    return(rep(-999, length(bounds)))
  }
  if (all(is.na(bounds))) {
    return(rep(NA, length(bounds)))
  }
  
  # Apply quantile levels on the sign-transformed sample (-x)
  q_minus <- unname(quantile(-subsamples, probs = bounds, type = 7, na.rm = FALSE))
  
  # Rearrange results back onto the original (x) scale.
  # 'bounds' is structured as [lower_levels(k), upper_levels(k)] on -x.
  # Under the sign transformation X -> -X, the roles of lower/upper swap:
  #   - what was an upper bound on -x becomes the (negated) LOWER bound on x
  #   - what was a lower bound on -x becomes the (negated) UPPER bound on x
  #
  # Numerical example (k = 2, targeting p = 0.25 and p = 0.1 on x):
  #   bounds  = [lower_0.25, lower_0.1, upper_0.25, upper_0.1]
  #           = [0.5928,     0.7614,    0.9072,     0.9986]        (levels on -x)
  #   q_minus = [-4.665,    -3.147,    -1.835,     -1.012]         (quantiles of -x)
  # Rücktransformation:
  #   new_lower = -q_minus[3:4] = [ 1.835,  1.012]   ← lower on x for p=0.25, p=0.1
  #   new_upper = -q_minus[1:2] = [ 4.665,  3.147]   ← upper on x for p=0.25, p=0.1
  k         <- length(bounds) / 2
  new_lower <- -q_minus[(k + 1):(2 * k)]   # from upper-half on -x
  new_upper <- -q_minus[1:k]                # from lower-half on -x
  
  c(new_lower, new_upper)
}


##### calculate t-mean CI ####

confidence_levels <- c(0.90, 0.95)
t_mean_CI_df <- foreach(file_i = subsample_files, .combine = rbind, .packages = c("dplyr", "tidyr", 'stringr', 'magrittr', 'purrr')) %dopar% {
  env <- new.env()  # Create a new environment to avoid overwriting issues
  cat('file: ', file_i, '\n')
  load(file = file_i, envir = env)
  md <- get_meta_data(filename = file_i)
  env$subsamples_df <- cbind(env$subsamples_df, md)
  # hint: sorting is needed as non-parametric CI are calculated on indices of order statistics
  env$subsamples_df %<>% mutate(subsamples = map(subsamples, ~ sort(.)))
  env$subsamples_df <- env$subsamples_df %>%
    tidyr::crossing(confidence_level = confidence_levels)
  
  env$subsamples_df %<>% mutate(
    CI = lapply(1:length(subsamples), function(i) {
      t.test(unlist(subsamples[[i]]), 
             alternative = 'two.sided', 
             conf.level = confidence_level[i])$conf.int
    }), 
    CI_type = 't_mean', 
    statistic = 'mean'
  )
}

# extract subsamples for further calculations
subsample_data <- t_mean_CI_df %>% 
  select(experiment, n, num_samples, sample_id, subsamples) %>% distinct() %>% 
  mutate(n = as.integer(as.character(n)))

n_values <- subsample_data %>% ungroup() %>% select(n) %>% distinct() %>% pull(n) %>% as.character() %>% as.numeric()    
quantile_levels <- c(0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05)
param_combinations <- expand.grid(n = n_values, quantile_level = quantile_levels, confidence_level = confidence_levels) %>%
  mutate(sign_trafo = quantile_level < 0.5) # Sign-Trafo-Flag: for lower quantiles (p < 0.5)

#### calculate nonparametric asymptotic CI  #####
# according to CI in Example 21.8, p.309 in Aad W Van der Vaart. Asymptotic statistics. Vol. 3. Cambridge university press, 2000. doi: 10.1017/CBO9780511802256
exact_np_asympt_norm_CI_df <- lapply(1:nrow(param_combinations), function(i) {
  params <- param_combinations[i, ]
  results <- CI_np_asymptotic(n = params$n, 
                              confidence_level = params$confidence_level, 
                              quantile_level = params$quantile_level, 
                              level_mode = T,
                              negate = params$sign_trafo)
  return(data.frame(c(params, results)))
}) 
exact_np_asympt_norm_CI_df <- do.call(rbind, exact_np_asympt_norm_CI_df)
exact_np_asympt_norm_CI_df <- exact_np_asympt_norm_CI_df %>% select(quantile_level, n, confidence_level, lower, upper, sign_trafo)

# split according to sign trafo, due to different handling of quantile calculation
df_direct <- exact_np_asympt_norm_CI_df %>% filter(!sign_trafo)
df_mirror <- exact_np_asympt_norm_CI_df %>% filter( sign_trafo)

# For performance reasons, the basic strategy here is to collect all necessary quantile levels for the CI calculation for every combination of n and confidence level. Then, every subsample is processed with quantile() for all CI bounds at once. In order to do so, all bounds are collected within a vector that is used within quantile() as probs parameter. Note that calculation per quantile level in long format is far more readable, but far less performant. The difference lies in the order of hours.

collect_CI_bounds <- function(df) {
  df %>%
    select(-sign_trafo) %>%
    pivot_wider(names_from = quantile_level, values_from = c(lower, upper)) %>%
    rowwise() %>%
    mutate(bounds = list(c(c_across(starts_with("lower_")), 
                           c_across(starts_with("upper_"))))) %>%
    ungroup()
}

df_direct <- merge(subsample_data, collect_CI_bounds(df_direct), by = 'n', all.x = TRUE)
df_mirror <- merge(subsample_data, collect_CI_bounds(df_mirror), by = 'n', all.x = TRUE)

# Metadata columns to preserve throughout the pipeline
meta_cols_asymptotic <- c("experiment", "n", "num_samples", "sample_id", "confidence_level")

# final calculation of CIs based on the quantile levels
# calculation without sign trafo
CI_bounds_asymptotic_direct <- t(apply(X = df_direct, MARGIN = 1, FUN = function(row) {
  subsample <- row[["subsamples"]]
  bounds    <- row[["bounds"]]
  calculate_quantiles(subsample, bounds)
}))

# set column names wrt quantile level and bound label
bounds_colnames_direct <- colnames(df_direct)[str_detect(colnames(df_direct), "^(lower|upper)")]
colnames(CI_bounds_asymptotic_direct) <- bounds_colnames_direct

CI_bounds_asymptotic_mirror <- t(apply(X = df_mirror, MARGIN = 1, FUN = function(row) {
  subsample <- row[["subsamples"]]
  bounds    <- row[["bounds"]]
  calculate_quantiles_mirror(subsample, bounds)
}))

bounds_colnames_mirror <- colnames(df_mirror)[str_detect(colnames(df_mirror), "^(lower|upper)")]
colnames(CI_bounds_asymptotic_mirror) <- bounds_colnames_mirror

# --- Attach metadata (type-safe via cbind on the original data frames) ---
result_direct <- cbind(df_direct[, meta_cols_asymptotic], as.data.frame(CI_bounds_asymptotic_direct))
result_mirror <- cbind(df_mirror[, meta_cols_asymptotic], as.data.frame(CI_bounds_asymptotic_mirror))

# --- Merge direct and mirror results on metadata keys ---
# Merging on metadata is robust against any row-order differences between the two branches.
exact_np_asympt_norm_CI_df <- merge(result_direct, result_mirror, by = meta_cols_asymptotic)

exact_np_asympt_norm_CI_df %<>%
  pivot_longer(
    cols = starts_with("lower_") | starts_with("upper_"),
    names_to = c(".value", "quantile_level"),
    names_pattern = "^(lower|upper)_(.*)$"
  ) %>%
  rename(lower = lower, upper = upper)

exact_np_asympt_norm_CI_df %<>% 
  mutate(quantile_level = paste0(as.numeric(quantile_level)*100, '%'), 
         CI = map2(lower, upper, ~ c(.x, .y)), 
         CI_type = 'np_asymptotic') %>% 
  rename(statistic = quantile_level) %>%
  select(-lower, -upper)

cat('Finished exact_np_asympt_norm_CI_df base calculation.\n')

#### calculate nonparametric exact CI #####

# non-parametric CI do not depend on sample, but on sample size and confidence level only
# calculating CI sample indices is sufficient
exact_np_CI_df <- lapply(1:nrow(param_combinations), function(i) {
  params <- param_combinations[i, ]
  tmp_samples <- subsample_data %>% filter(n == params$n) %>% left_join(params, by = 'n')
  num_obs <- nrow(tmp_samples)
  CI_randomize <- CI_np_exact(n = params$n, confidence_level = params$confidence_level, quantile_level = params$quantile_level, verbose = T)
  if (sum(is.na(CI_randomize))>0) { # CI is NA if it does not exist caused by too small sample size
    sampled_CIs <- CI_randomize %>% slice(rep(1, num_obs)) 
    tmp_samples <- bind_cols(tmp_samples, sampled_CIs) %>% mutate(CI = NA)
  } else {
    sampled_indices <- sample(1:nrow(CI_randomize), size = num_obs, prob = CI_randomize$lambda, replace = TRUE)
    sampled_CIs <- CI_randomize[sampled_indices, c("lower", "upper")]  
    tmp_samples <- bind_cols(tmp_samples, sampled_CIs) %>% mutate(CI = pmap(list(subsamples, lower, upper), function(x, k, l) c(x[k], x[l]))) 
  }
  return(tmp_samples %>% select(-subsamples))
}) %>% bind_rows()

exact_np_CI_df %<>% 
  mutate(quantile_level = paste0(as.numeric(quantile_level)*100, '%'), 
         CI_type = 'np_exact') %>% 
  rename(statistic = quantile_level)

cat('Finished exact_np_CI_df base calculation.\n')

#### merge 'true' values of statistics of interest ####
# true_statistics file is generated within sp-bt-final-merge-3.R
load(true_statistics_file)
#true_statistics_df %<>% pivot_wider(names_from = 'statistic', values_from = 'true_value')

#### true statistics: t-mean CI ####
t_mean_CI_df %<>% ungroup() %>% select(-subsamples)
t_mean_CI_df <- left_join(t_mean_CI_df, true_statistics_df, by = c('experiment', 'statistic'))

# Constrain each CI to lie within [0, 1] by adjusting values outside the range
t_mean_CI_df %<>% mutate(invalid_values = map_dbl(CI, ~ sum(.x<0 | .x>1)), 
                         CI = map(CI, ~ pmax(pmin(.x, 1), 0)))

t_mean_CI_df %<>% mutate(lower = map_dbl(CI, first), 
                         upper = map_dbl(CI, last), 
                         is_in_CI = between(true_value, lower, upper),
                         interval_length = abs(upper - lower)) %>% select(-lower, -upper)

saveResult(t_mean_CI_df)
#load('final_results/t_mean_CI_df.Rdata')
t_mean_CI_df_aggregate <- t_mean_CI_df %>% 
  group_by(experiment, n, num_samples, statistic, confidence_level, CI_type, seed) %>% 
  summarise(emp_conf_level = mean(is_in_CI), avg_length = mean(interval_length), invalid = sum(invalid_values)/n())
cat('Finished aggregation of t_mean_CI_df.\n')
saveResult(t_mean_CI_df_aggregate)
#### true statistics: nonparametric asymptotic CI ####
exact_np_asympt_norm_CI_df %<>% ungroup()
exact_np_asympt_norm_CI_df <- left_join(exact_np_asympt_norm_CI_df, true_statistics_df, by = c('experiment', 'statistic'))

# Constrain each CI to lie within [0, 1] by adjusting values outside the range
exact_np_asympt_norm_CI_df %<>% 
  mutate(invalid_values = map_dbl(CI, ~ sum(.x < 0 | .x > 1, na.rm = TRUE)),
         CI             = map(CI, ~ pmax(pmin(.x, 1), 0)))

exact_np_asympt_norm_CI_df %<>% mutate(lower = map_dbl(CI, first), 
                                       upper = map_dbl(CI, last), 
                                       is_in_CI = between(true_value, lower, upper),
                                       interval_length = abs(upper - lower)) %>% select(-lower, -upper)

saveResult(exact_np_asympt_norm_CI_df)

exact_np_asympt_norm_CI_df_aggregate <- exact_np_asympt_norm_CI_df %>% 
  group_by(experiment, n, num_samples, statistic, confidence_level, CI_type) %>% 
  summarise(emp_conf_level = mean(is_in_CI), avg_length = mean(interval_length)) 
exact_np_asympt_norm_CI_df_aggregate$n <- factor(exact_np_asympt_norm_CI_df_aggregate$n, levels = unique(exact_np_asympt_norm_CI_df_aggregate$n))
cat('Finished aggregation of exact_np_asympt_norm_CI_df.\n')
saveResult(exact_np_asympt_norm_CI_df_aggregate)

#### true statistics: nonparametric exact CI ####
exact_np_CI_df %<>% ungroup()
exact_np_CI_df <- left_join(exact_np_CI_df, true_statistics_df, by = c('experiment', 'statistic'))

# Constrain each CI to lie within [0, 1] by adjusting values outside the range
exact_np_CI_df %<>% mutate(invalid_values = map_dbl(CI, ~ sum(.x<0 | .x>1)), 
                           CI = map(CI, ~ pmax(pmin(.x, 1), 0)))
exact_np_CI_df %<>% mutate(lower_tmp = map_dbl(CI, first), # for np exact CI there are already cols named lower and upper!!!!!
                           upper_tmp = map_dbl(CI, last), 
                           is_in_CI = between(true_value, lower_tmp, upper_tmp),
                           interval_length = abs(upper_tmp - lower_tmp)) %>% select(-lower_tmp, -upper_tmp)
saveResult(exact_np_CI_df)
#load('final_results/exact_np_CI_df.Rdata')
exact_np_CI_df_aggregate <- exact_np_CI_df %>% 
  group_by(experiment, n, num_samples, statistic, confidence_level, CI_type) %>% 
  summarise(emp_conf_level = mean(is_in_CI), avg_length = mean(interval_length), invalid = sum(invalid_values)/n())
exact_np_CI_df_aggregate$n <- factor(exact_np_CI_df_aggregate$n, levels = unique(exact_np_CI_df_aggregate$n))

cat('Finished aggregation of exact_np_CI_df.\n')
saveResult(exact_np_CI_df_aggregate)

#### rbind all single results ####
t_mean_CI_df$n <- as.integer(as.character(t_mean_CI_df$n)) # align factor and integers
single_CIs_all <- bind_rows(exact_np_asympt_norm_CI_df,
                            exact_np_CI_df,
                            t_mean_CI_df)

##### bootstrap CI #####
load(file.path(results_path, 'bootstrap_full_aggregated.Rdata'))

# Function to calculate the column names dynamically for each confidence level
apply_ci_calculation <- function(df, conf_level) {
  lower_bound <- (1 - conf_level) / 2 * 100
  upper_bound <- (1 + conf_level) / 2 * 100
  
  lower_col <- paste0("boot_perc_", lower_bound, "%")
  upper_col <- paste0("boot_perc_", upper_bound, "%")
  
  # Calculate CI and interval length
  df %>%
    mutate(
      confidence_level = conf_level,
      lower_limit = !!sym(lower_col),
      upper_limit = !!sym(upper_col)
    )
}

# Apply the function for each confidence level using lapply
results_list <- lapply(confidence_levels, function(conf_level) {
  apply_ci_calculation(bootstrap_full_aggregated, conf_level)
})

# Combine the results into a single dataframe
bootstrap_single_CI <- bind_rows(results_list) %>% select(-starts_with("boot_perc_"))
bootstrap_single_CI %<>% mutate(invalid = lower_limit < 0 | upper_limit >1,
                                lower_limit = ifelse(lower_limit < 0, 0, lower_limit),
                                upper_limit = ifelse(upper_limit > 1, 1, upper_limit), 
                                interval_length = abs(upper_limit - lower_limit),
                                is_in_CI = between(true_value, lower_limit, upper_limit))
saveResult(bootstrap_single_CI)

# Apply summarization dynamically for each confidence level
boot_CI_df_aggregate <- bootstrap_single_CI %>% 
  group_by(experiment, n, boot_statistic, num_bootsamples, num_samples, confidence_level) %>%
  summarise(emp_conf_level = mean(is_in_CI,),
            avg_length = mean(interval_length),
            invalid = sum(invalid)/n(),
            .groups = 'drop') %>%
  rename(statistic = boot_statistic)
boot_CI_df_aggregate %<>% mutate(CI_type = 'bootstrap')
saveResult(boot_CI_df_aggregate)
########## combine everything into overview dataframe ##################

# hint: the aggregation might run into trouble as there are partially slightly different cols, that arent needed for plotting
overview_signTrafo <- rbind(exact_np_asympt_norm_CI_df_aggregate, 
                            exact_np_CI_df_aggregate, 
                            t_mean_CI_df_aggregate, 
                            boot_CI_df_aggregate)

# order experiment (factor) wrt initial samples size, esp. for plotting
statistic_levels <- c("5%", "10%", "25%", "50%", "mean", "75%", "90%", "95%")
CI_type_levels = c("np_exact", "bootstrap", "np_asymptotic", "t_mean")

overview_signTrafo %<>% mutate(statistic = factor(statistic, levels = statistic_levels), 
                               CI_type = factor(CI_type, levels = CI_type_levels))


true_statistics_wide <- true_statistics_df %>% pivot_wider(names_from = statistic, values_from = true_value)
true_statistics_wide %<>% mutate(inter_decile_range = abs(`90%`- `10%`)) %>% select(experiment, inter_decile_range, counts)
overview_signTrafo <- merge(overview_signTrafo, true_statistics_wide, by = 'experiment', all.x = TRUE)
overview_signTrafo %<>% mutate(avg_length_ratio = avg_length/inter_decile_range)

saveResult(overview_signTrafo)
