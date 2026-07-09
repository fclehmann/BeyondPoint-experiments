library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)
library(foreach)
library(doParallel)
library(parallel)
library(optparse)

##### Parallel backend and helper functions #######
num_cores <- 0
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

cl <- makeCluster(num_cores - 1)
registerDoParallel(cl)


##### argument parsing stuff ####
option_list <- list(
  make_option(c("-d", "--parent_dir"), type = "character", default = NULL,
              help = "Parent directory", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
args <- parse_args(opt_parser)


####### config data #####
base_data_path <- args$parent_dir

yaml_file <- list.files(path = base_data_path, pattern = '*.yaml', full.names = TRUE)[1]
cat('reading yaml file: ', yaml_file, '\n')
config <- yaml::read_yaml(yaml_file)

Rdatafile <- file.path(base_data_path, config$global_settings$data_file)
cat('reading Rdata file: ', Rdatafile, '\n')
load(Rdatafile)

output_path <- file.path(base_data_path, 'final_results')
cat('results will be written to ', output_path, '\n')
dir.create(output_path, recursive = TRUE, showWarnings = TRUE)


#### calculation of true quantiles ######

calculate_statistics <- function(x, probs, type) {
  quantiles  <- quantile(x, probs = probs, type = q_type)
  mean_value <- mean(x)
  counts     <- length(x)
  c(quantiles, mean = mean_value, counts = as.integer(counts))
}

# getting 'true' statistics from large empirical distribution
cat('calculation of true statistics...\n')
q_levels <- config$global_settings$q_levels
q_type   <- config$global_settings$q_type
true_statistics_df <- df %>%
  group_by(experiment) %>%
  summarise(statistics = list(calculate_statistics(RMSE, probs = q_levels, type = q_type)))
true_statistics_df %<>% unnest_wider(statistics, names_sep = "_") %>%
  rename_with(~ gsub("^statistics_", "", .), starts_with("statistics_"))
true_statistics_df %<>% pivot_longer(cols = -experiment,
                                     names_to  = "statistic",
                                     values_to = "true_value")

true_file <- file.path(output_path, "true_statistics.Rdata")
cat('write true statistics to ', true_file, '\n')
save(true_statistics_df, file = true_file)


#### aggregation of bootstrap percentile dataframes ######
cat('processing aggregation....\n')
data_files_avail <- list.files(file.path(base_data_path, "boot_percentiles"),
                               pattern = '*.Rdata', full.names = TRUE)
cat('found number of files to aggregate: ', length(data_files_avail), '\n')

bootstrap_full_aggregated <- foreach(file = data_files_avail, .combine = rbind, .packages = 'base') %dopar% {
  env <- new.env()  # Create a new environment to avoid overwriting issues
  load(file, envir = env)
  env$boot_statistics_percentiles
}

colnames(true_statistics_df)[colnames(true_statistics_df) == "statistic"] <- "boot_statistic"

# Merge the data frames
bootstrap_full_aggregated <- merge(
  bootstrap_full_aggregated,
  true_statistics_df,
  by = c("experiment", "boot_statistic"),
  all.x = TRUE  # Keep all rows from bootstrap_full_aggregated
)

bootstrap_full_aggregated %<>% arrange(experiment, n, num_samples, num_bootsamples, sample_id, boot_statistic)

bootstrap_full_file <- file.path(output_path, "bootstrap_full_aggregated.Rdata")
cat('write aggregated results to ', bootstrap_full_file, '\n')
save(bootstrap_full_aggregated, file = bootstrap_full_file)
