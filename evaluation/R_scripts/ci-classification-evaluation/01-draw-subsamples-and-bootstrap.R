# implementation of bootstrapped semi-parametric quantile estimator, eqn. (2.2) from Hutson, A. D. (2002). A semi-parametric quantile function estimator for use in bootstrap estimation procedures. Statistics and Computing, 12(4), 331-338.
# The basic idea is to modify the creation of bootstrap samples. 
# Here: probability integral transform is used in connection with a semi-parametric quantile estimator.
# This semi-parametric estimator uses extrapolation esp. for extreme quantiles and therefore reflects more about the tails of a distribution. The sample quantiles are no longer restricted by min and max.
# This script applies semi-parametric bootstrap to empirical data.

library(boot)
library(dplyr)
library(magrittr)
library(tidyr)
library(tibble)
library(stringr)
library(foreach)
library(doParallel)
library(parallel)
library(optparse)

# example call for this script:
# Rscript semi-parametric-bootstrap-CI-application.R --output_path /path/to/output --config_file /path/to/config_file.yaml --configuration_index 1


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
              help="Path of Rdata data file.", metavar="character")
)

# Create an option parser
opt_parser <- OptionParser(option_list=option_list)

# Parse the arguments
args <- parse_args(opt_parser)

# Locate ../libs/ relative to this script's own location.
script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
lib_path <- normalizePath(file.path(dirname(script_path), "..", "libs"))
flush.console()

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

create_filename <- function(config, sub_experiment, folder, file_extension = '.pdf') {
  file.path(folder, paste0(
    'semiparametric-BT-exp=', sub_experiment, '-',
    "n=", config$sample_size, "-",
    "num_samples=", config$num_samples, '-',
    "num_boot=", config$num_bootsamples, "-",
    "qlev=", paste(gsub("\\.", "_", config$q_levels), collapse = "~"), '-',
    "seed=", config$seed, "-",
    "qtype=", config$q_type,
    file_extension
  ))
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
config <- yaml::read_yaml(args$config_file)
config_index <- args$configuration_index

cat('Available configs:\n', 
    paste(sapply(config$experiments, function(x) 
      paste(x$id, 
            "sample_size=",    x$sample_size, 
            "num_samples=",    x$num_samples, 
            "num_bootsamples=", x$num_bootsamples)), 
      collapse = "\n"), '\n')

config_id_exp <- config$experiments[[which(sapply(config$experiments, function(e) e$id == config_index))]]
config <- c(config$global_settings, config_id_exp)
dataset_label <- tools::file_path_sans_ext(basename(config$data_file))
base_name     <- paste0("semiparametric-", dataset_label,
                        "-n", config$sample_size,
                        "-", config$num_samples)
config_name   <- paste0(base_name, "-", config$num_bootsamples)


cat('run config: ', config$id, '\n')
print(config)

# check output directory
parent_dir <- tools::file_path_sans_ext(basename(args$config_file))
output_base          <- file.path(args$output_path, parent_dir)
output_path          <- file.path(output_base, config_name)

# Create the directory path
dir.create(output_path, recursive = TRUE, showWarnings = T)

# Assign parameters from YAML to variables
Rdata_file_path      <- file.path(args$datafile_path, config$data_file)
num_bootsamples <- config$num_bootsamples
num_samples <- config$num_samples
sample_size <- config$sample_size
q_levels <- config$q_levels
q_type <- config$q_type
seed <- eval(parse(text = config$seeds))
boot_percentiles_levels <- config$boot_percentiles_levels

##### reading data ####
cat('check Rdata file...', file.exists(Rdata_file_path), '\n')
cat('trying to open Rdata file: ', Rdata_file_path, '\n')
load(Rdata_file_path)

#### quantile calculation, subsampling, bootstrapping ####

# Function to draw num_samples sub samples of size sample_size from a vector
draw_subsamples <- function(data, num_samples, sample_size, seed) {
  set.seed(seed)
  replicate(num_samples, sample(data, sample_size, replace = TRUE), simplify = FALSE)
}

# Draw groupwise subsamples
subsamples_df <- df %>%
  group_by(experiment) %>%
  summarise(subsamples = list(draw_subsamples(accuracy, num_samples, sample_size, seed = seed))) %>%
  unnest(subsamples) %>%
  group_by(experiment) %>%
  mutate(sample_id = row_number())

# save subsamples
subsample_name_string <- paste0('subsamples_config=', 
                                sub("-[^-]*$", "", config_name),
                                '_seed=', seed, '.Rdata')
save_path_subsamples <- file.path(output_base, subsample_name_string)
cat('save subsample data to ', save_path_subsamples, '\n')
save(subsamples_df, file = save_path_subsamples)

# foreach is iterating over different experiment settings
all_experiments <- subsamples_df$experiment %>% unique()

# # calculation of memory consumption per experiment
# size_in_bytes <- length(q_levels)*num_samples*num_bootsamples*4/3*8
# paste0('Memory comsumption per CPU: ', formatC(size_in_bytes / 1024^2, format = "e", digits = 2), " MB")
# paste0('Trying to use ', num_cores, ' CPUs. Total memory consumption: ', formatC(num_cores*size_in_bytes / 1024^2, format = "e", digits = 2), " MB")
 
cat('Starting calculation of semiparametric bootstrap distributions...\nCheck output folder: ', output_path, ' for results.')
cat('Number of different settings to calculate: ', length(all_experiments), '.\nNumber of cores used for parallelization: ', length(cl), '.\n')
foreach(experiment_i = all_experiments, .packages = c("dplyr", "tidyr")) %dopar% {
  tmp <- subsamples_df %>% filter(experiment == experiment_i) 
  
  # generating bootstrap samples for single subsamples from large empirical distribution
  boot_samples <- lapply(X = seq.int(nrow(tmp)), FUN = function(index) {
    sample_data <- semiparametric_bootstrapping(seed, num_bootsamples = num_bootsamples, q_levels = q_levels, q_type = q_type,
                                                observed_sample = tmp$subsamples[[index]])
    sample_data$sample_id <- tmp$sample_id[index]
    return(sample_data)
  })
  combined_boot_samples <- bind_rows(boot_samples)
  
  out_file <- create_filename(config = config, sub_experiment = experiment_i, folder = output_path, file_extension = '.Rdata')
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
dest_config          <- file.path(output_base, basename(args$config_file))
copy_file_if_not_exists(source_file = source_config, dest_file = dest_config)

source_data <- Rdata_file_path
dest_data            <- file.path(output_base, basename(config$data_file))
copy_file_if_not_exists(source_file = source_data, dest_file = dest_data)
