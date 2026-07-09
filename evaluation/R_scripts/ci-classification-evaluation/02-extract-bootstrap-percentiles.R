# This script aggregates the bootstrap distributions.
# From all statistics, the percentiles defined under global_settings.boot_percentiles_levels
# in the YAML config are extracted and, together with per-experiment metadata parsed
# from the filename, written into a companion .Rdata file per input bootstrap file.

library(dplyr)
library(magrittr)
library(tidyr)
library(tidyverse)
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


#### option parser stuff ####

option_list <- list(
  make_option(c("-d", "--data_parent_dir"), type = "character", default = NULL,
              help = "Parent dir of bootstrapped statistics Rdata files.", metavar = "character"),
  make_option(c("-c", "--config_file"), type = "character", default = NULL,
              help = "Config file to use. The config file should be in the directory with all data files as it is copied after sample creation already.", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
args <- parse_args(opt_parser)

cat('args:\n  data_parent_dir: ', args$data_parent_dir, '\n  config_file:     ', args$config_file, '\n')


##### config stuff ####

parent_dir  <- args$data_parent_dir
output_path <- file.path(parent_dir, 'boot_percentiles')

# Read YAML configuration
yamlfile <- file.path(parent_dir, args$config_file)
cat('check yaml file...', yamlfile, '----', file.exists(yamlfile), '\n')
cat('trying to open yaml file: ', yamlfile, '\n')
config <- yaml::read_yaml(yamlfile)
cat('Available configs:\n', 
    paste(sapply(config$experiments, function(x) 
      paste(x$id, 
            "sample_size=",    x$sample_size, 
            "num_samples=",    x$num_samples, 
            "num_bootsamples=", x$num_bootsamples)), 
      collapse = "\n"), '\n')

# Assign parameters from YAML to variables
boot_percentiles_levels <- config$global_settings$boot_percentiles_levels

#### processing bootstrapped data ####
cat('processing aggregation....\n')
cat('getting percentile levels: ', boot_percentiles_levels, '\n')
data_dir <- list.dirs(parent_dir)

data_files <- list.files(data_dir, pattern = '*.Rdata', full.names = TRUE)
# exclude some Rdata files (subsamples, existing percentile files, final_results)
data_files <- grep("^(?!.*(subsample|boot_percent|final_results)).*$", data_files, value = TRUE, perl = TRUE)
# also exclude the raw data file (provenance copy from step 01)
data_files <- data_files[!basename(data_files) %in% config$global_settings$data_file]
cat('found number of files to aggregate: ', length(data_files), '\n')

dir.create(output_path, recursive = TRUE, showWarnings = TRUE)
cat('write files to ', output_path, '\n')

log_file <- file.path(output_path, 'debug_log.txt')
cat('write log file to ', log_file, '\n')

if (file.exists(log_file)) {
  file.remove(log_file)
}

foreach(file_i = data_files, .packages = c("dplyr", "tidyr", 'stringr', 'magrittr')) %dopar% {
  cat('loading data file: ', file_i, '\n')
  write(paste(Sys.time(), ' ##### Trying to load: ', file_i, ' #### file exists: ', file.exists(log_file), "\n ", sep = ''),
        file = log_file, append = TRUE)
  load(file = file_i)
  
  # get experiment meta data from filename
  file_name <- str_remove(file_i, "\\.Rdata$")
  matches <- str_extract_all(file_name, "(?<=\\=)[^\\-]+")[[1]]
  meta_data <- list(
    experiment      = matches[1],
    n               = matches[2],
    num_samples     = matches[3],
    num_bootsamples = matches[4],
    qlev            = matches[5],
    seed            = matches[6],
    qtype           = matches[7]
  )
  meta_data <- as.data.frame(meta_data)
  meta_data %<>% select(-qtype, -qlev, -seed)
  
  # get percentiles from bootstrap distributions
  boot_percentiles <- config$global_settings$boot_percentiles_levels
  combined_boot_samples %<>% pivot_longer(cols = -sample_id, names_to = "boot_statistic", values_to = "statistics_value")
  
  boot_statistics_percentiles <- combined_boot_samples %>%
    group_by(sample_id, boot_statistic) %>%
    summarise(boot_perc = list(quantile(statistics_value, probs = boot_percentiles)),
              boot_mean = mean(statistics_value)) %>%
    ungroup() %>%
    unnest_wider(boot_perc, names_sep = "_") %>%
    rename_with(~ paste0('boot_perc_', boot_percentiles * 100, "%"), starts_with("boot_perc"))
  
  # append metadata
  boot_statistics_percentiles <- cbind(meta_data, boot_statistics_percentiles)
  
  outfile  <- paste0('boot-percentiles-', basename(file_i))
  full_out <- file.path(output_path, outfile)
  save(boot_statistics_percentiles, file = full_out)
  
  rm(boot_statistics_percentiles)
  rm(combined_boot_samples)
  gc()
  NULL
}

stopCluster(cl)
