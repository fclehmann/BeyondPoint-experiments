# implementation of bootstrapped semi-parametric quantile estimator, eqn. (2.2) from Hutson, A. D. (2002). A semi-parametric quantile function estimator for use in bootstrap estimation procedures. Statistics and Computing, 12(4), 331-338.
# The basic idea is to modify the creation of bootstrap samples. 
# Here: probability integral transform is used in connection with a semi-parametric quantile estimator.
# This semi-parametric estimator uses extrapolation esp. for extreme quantiles and therefore reflects more about the tails of a distribution. The sample quantiles are no longer restricted by min and max.
# This script realizes as simulation of semi-parametric bootstrap for different known distributions.

library(jsonlite)
library(boot)
library(dplyr)
library(magrittr)
library(tidyr)
library(tibble)
library(stringr)
library(KScorrect) # used for normal mixture distribution
library(foreach)
library(doParallel)
library(parallel)
library(optparse)
library(evd)

#### option parser stuff ####

# Define command line arguments
option_list <- list(
  make_option(c("-o", "--output_path"), type="character", default=NULL,
              help="Working directory", metavar="character"),
  make_option(c("-c", "--config_file"), type="character", default=NULL,
              help="Config file to use", metavar="character"),
  make_option(c("-i", "--configuration_index"), type="integer", default=NULL,
              help="The index of the configuration in the yaml file to run", metavar="integer"),
  make_option(c("-d", "--datafile_path"), type="character", default=NULL,
              help="Path of json data file.", metavar="character")
)

# Create an option parser
opt_parser <- OptionParser(option_list=option_list)

# Parse the arguments
args <- parse_args(opt_parser)

flush.console()
# Read YAML configuration
config <- yaml::read_yaml(args$config_file)
# lib_path is derived from this script's own location
script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
lib_path <- normalizePath(file.path(dirname(script_path), "..", "libs"))

cat('lib path: ', lib_path, '\n')

# Build lookup: experiment name -> positive_support flag (from YAML)
# Inline version of create_distribution_name() to remove function dependency.
make_exp_name <- function(dname, params) {
  param_str <- paste(names(params), params, sep = "_", collapse = "_")
  gsub("\\.", "_", paste0(dname, "_", param_str))
}

positive_support_lookup <- setNames(
  sapply(config$distributions, function(d) isTRUE(d$positive_support)),
  sapply(config$distributions, function(d) make_exp_name(d$name, d$params))
)

cat('Experiment -> positive_support mapping:\n')
print(positive_support_lookup)

# include semi-parametric quantile estimation
source(file.path(lib_path, "lib-semiparam-quantile-est.R"))

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

# helper function replacing dots with underscores in names and concatenate
format_params <- function(some_list) {
  # Replace dots with underscores in names and concatenate
  formatted_params <- paste(
    sapply(names(some_list), function(name) {
      # Replace dots with underscores
      name <- gsub("\\.", "_", name)
      # Concatenate name and value with an equal sign
      paste(name, some_list[[name]], sep = "=")
    }),
    collapse = "-"
  )
  
  return(formatted_params)
}

# helper function for file copying
copy_file_if_not_exists <- function(source_file, dest_file) {
  if (file.exists(dest_file)) {
    cat("The file already exists:\n", dest_file, "\n")
  } else {
    cat("The file does not exist. Copying the file now.\n")
    cat("Source file:\n", source_file, "\n")
    cat("Destination file:\n", dest_file, "\n")
    file.copy(source_file, dest_file)
  }
}

##### config stuff ####

# Read YAML configuration
config_index <- args$configuration_index
cat('Available configs:\n', paste(sapply(config$variations, function(x) paste(x$id, x$name)), collapse = "\n"), '\n')
config_id_exp <- config$variations[[which(sapply(config$variations, function(e) e$id == config_index))]]
cat('run config: ', config$id, '\n')
print(config)

# Assign parameters from YAML to variables
distribution_config <- config$distributions
id <- config_id_exp$id
num_bootsamples <- config_id_exp$num_bootsamples
num_samples <- config_id_exp$num_samples
sample_size <- config_id_exp$sample_size
q_levels <- config$global_settings$q_levels
q_type <- config$global_settings$q_type
seed <- config$global_settings$seeds

config_name <- paste0('semiparam-bt-simulation-', 
                      "n=", sample_size, 
                      "-num_samples=", num_samples,
                      "-num_bootsamples=", num_bootsamples, 
                      '-seed=', seed)

# check output directory
parent_dir <- tools::file_path_sans_ext(basename(args$config_file))
output_base <- paste0(args$output_path, '/', parent_dir, '/')
output_path <- paste0(output_base, '/', config_name, '/')

# Create the directory path
dir.create(output_path, recursive = TRUE, showWarnings = T)

# Function to generate samples based on distribution configuration
generate_samples <- function(dist_name, params, sample_size) {
  # Construct the function name for random sampling
  rfunc <- paste0("r", dist_name)
  
  # Generate a single sample
  sample <- do.call(
    rfunc,
    c(list(n = sample_size), params)
  )
  
  return(as.vector(sample))
}

# Function to create a description for the distribution
create_distribution_name <- function(dist_name, params) {
  param_str <- paste(
    names(params),
    params,
    sep = "_",
    collapse = "_"
  )
  
  # Combine the distribution name and parameter string
  dist_description <- paste0(dist_name, "_", param_str)
  
  # Replace all '.' with '_'
  dist_description <- gsub("\\.", "_", dist_description)
  
  return(dist_description)
}


# Initialize an empty list to hold the samples
samples_list <- list()

# Loop through distributions and generate samples
for (dist in config$distributions) {
  dist_type <- dist$type
  dist_name <- dist$name
  params <- dist$params
  
  for (i in 1:num_samples) {
    # Generate sample
    sample_data <- generate_samples(dist_type, params, sample_size)
    
    # Create a data frame entry for this sample, ensuring 'actual_sample' is treated as a single vector
    sample_entry <- data.frame(
      experiment = create_distribution_name(dist_name, params),
      sample_id = i,
      subsamples = I(list(as.vector(sample_data)))  # Ensure 'sample_data' is stored as a vector
    )
    
    # Add to the list
    samples_list[[length(samples_list) + 1]] <- sample_entry
  }
}

# Combine all sample entries into a single data frame
subsamples_df <- do.call(rbind, samples_list)
######################################

# save subsamples
subsample_name_string <- paste0('subsamples_config=', parent_dir, '-', config_name, '.Rdata')
save_path_subsamples <- paste0(output_base, '/', subsample_name_string)
cat('save subsample data to ', save_path_subsamples, '\n')
save(subsamples_df, file = save_path_subsamples)

# foreach is iterating over different experiment settings
all_experiments <- subsamples_df$experiment %>% unique()

cat('Starting calculation of semiparametric bootstrap distributions...\nCheck output folder: ', output_path, ' for results.')
cat('Number of different settings to calculate: ', length(all_experiments), '.\nNumber of cores used for parallelization: ', length(cl), '.\n')

config_name <- paste0("n=", sample_size, 
                      "-num_samples=", num_samples,
                      "-num_bootsamples=", num_bootsamples, 
                      '-seed=', seed)

foreach(experiment_i = all_experiments, 
        .packages = c("dplyr", "tidyr", 'magrittr'),
        .export   = c("positive_support_lookup")) %dopar% {
  tmp <- subsamples_df %>% filter(experiment == experiment_i) 
  
  # Look up flag by experiment name (defaults to FALSE if missing)
  pos_only_flag <- isTRUE(positive_support_lookup[[experiment_i]])
  
  # generating bootstrap samples for single subsamples from large empirical distribution
  boot_samples <- lapply(X = seq.int(nrow(tmp)), FUN = function(index) {
    sample_data <- semiparametric_bootstrapping(seed, num_bootsamples = num_bootsamples, q_levels = q_levels, q_type = q_type,
                                                observed_sample = tmp$subsamples[[index]], pos_only = pos_only_flag                 )
    sample_data$sample_id <- tmp$sample_id[index]
    return(sample_data)
  })
  combined_boot_samples <- bind_rows(boot_samples)
  
  out_file <- paste0(output_path, 'semiparametric_bt-exp=', experiment_i, '-', config_name, '.Rdata')
  save(combined_boot_samples, file = out_file)
  cat('results saved into: ', out_file, '\n')
  
  # free up memory
  rm(sample_data)
  gc()
  NULL  # Return NULL to avoid accumulating results in memory
}


# Stop the cluster
stopCluster(cl)

# copying yaml and data file into output location
source_config <- args$config_file
dest_config <- paste0(output_base, '/', basename(args$config_file))
copy_file_if_not_exists(source_file = source_config, dest_file = dest_config)

# Remove all objects from the global environment
rm(list = ls())
