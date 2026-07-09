# This script aggregates the bootstrap distributions.
# from all statistics, the defined boot_percentiles are extracted (see corresponding yaml file)
# additionally, the true value of the respective statistics is calculated based on the theoretical distribution.

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


cl <- makeCluster(num_cores-1)
registerDoParallel(cl)

#### option parser stuff ####

# Define command line arguments
option_list <- list(
  make_option(c("-d", "--data_parent_dir"), type="character", default=NULL,
              help="Parent dir of bootstrapped statistics Rdata files.", metavar="character"),
  make_option(c("-c", "--config_file"), type="character", default=NULL,
              help="Config file to use. The condig file should be in the directory with all data files as it is copied after sample creation.", metavar="character")
)

# Create an option parser
opt_parser <- OptionParser(option_list=option_list)

# Parse the arguments
args <- parse_args(opt_parser)

cat('set working dir to ', args$data_parent_dir, '\n')
cat('args:\n', args$data_parent_dir, '\n', args$config_file, '\n')


##### config stuff ####

# check output directory
output_dir <- 'boot_percentiles'
parent_dir <- args$data_parent_dir
output_path <- paste0(parent_dir, '/', output_dir)

# Read YAML configuration
yamlfile <- paste0(parent_dir, '/', args$config_file)
cat('check yaml file...', yamlfile, '----', file.exists(yamlfile), '\n')
cat('trying to open yaml file: ', yamlfile)
config <- yaml::read_yaml(yamlfile)
cat('Available configs:\n', paste(sapply(config$distributions, function(x) paste(x$name)), collapse = "\n"), '\n')

# Assign parameters from YAML to variables
boot_percentiles_levels <- config$global_settings$boot_percentiles_levels

#### processing bootstrapped data ####
cat('processing aggregation....\n')
cat('getting percentils levels: ', boot_percentiles_levels, '\n')
data_dir <- list.dirs(parent_dir)

data_files <- list.files(data_dir, pattern = '*.Rdata', full.names = T)
data_files <- grep("^(?!.*(subsample|boot_percent|final)).*$", data_files, value = TRUE, perl = TRUE)

cat('found number of files to aggregate: ', length(data_files), '\n')

# Create the directory path
dir.create(output_path, recursive = TRUE, showWarnings = T)
cat('write files to ', output_path, '\n')

log_file <- paste0(output_path, '/debug_log.txt')
cat('write log file to ', log_file, '\n')

if (file.exists(log_file)) {
  file.remove(log_file)
}

foreach(file_i = data_files, .packages = c("dplyr", "tidyr", 'stringr', 'magrittr')) %dopar% {
  #file_i = data_files[1]
  cat('loading data file: ', file_i, '\n')
  write(paste(Sys.time(), ' ##### Trying to load: ', file_i, ' #### file exists: ', file.exists(log_file), "\n ", sep = ''), file = log_file, append = TRUE)
  load(file = file_i)
  
  # get experiment meta data
  file_name <- str_remove(basename(file_i), "\\.Rdata$")
  matches <- str_extract_all(file_name, "(?<=\\=)[^\\-]+")[[1]]
  meta_data <- list(
    experiment = matches[1],
    n = matches[2],
    num_samples = matches[3],
    num_bootsamples = matches[4]
    # qlev = matches[5],
    # seed = matches[6],
    # qtype = matches[7]
  )
  meta_data <- as.data.frame(meta_data) 
  #meta_data %<>% select(-qtype, -qlev, -seed)
  
  
  # get percentiles from bootstrap distributions
  boot_percentiles <- config$global_settings$boot_percentiles_levels
  combined_boot_samples %<>% pivot_longer(cols = -c(sample_id), names_to = "boot_statistic", values_to = "statistics_value")
  
  boot_statistics_percentiles <- combined_boot_samples %>%
    group_by(sample_id, boot_statistic) %>%
    summarise(boot_perc = list(quantile(statistics_value, probs = boot_percentiles))) %>% 
    ungroup() %>%
    unnest_wider(boot_perc, names_sep = "_") %>%
    rename_with(~ paste0('boot_perc_', boot_percentiles * 100, "%"), starts_with("boot_perc"))
  
  # append metadata
  boot_statistics_percentiles <- cbind(meta_data, boot_statistics_percentiles)
  
  outfile <- paste0('boot-percentiles-', basename(file_i))
  full_out <- paste0(output_path, '/', outfile)
  save(boot_statistics_percentiles, file = full_out)
  # free up memory
  rm(boot_statistics_percentiles)
  rm(combined_boot_samples)
  gc()
  NULL  # Return NULL to avoid accumulating results in memory
}

# Stop the cluster
stopCluster(cl)

# Remove all objects from the global environment
rm(list = ls())
