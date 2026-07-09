library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)
library(foreach)
library(doParallel)
library(parallel)
library(optparse)
library(KScorrect) # needed for normal mixture distribution
library(evd) # needed for gumbel dist.

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


##### argument parsing stuff ####
option_list <- list(
  make_option(c("-d", "--parent_dir"), type="character", default=NULL,
              help="Parent directory", metavar="character")
)

# Create an option parser
opt_parser <- OptionParser(option_list=option_list)

# Parse the arguments
args <- parse_args(opt_parser)

####### config data #####
base_data_path <- args$parent_dir


yaml_file <- list.files(path = base_data_path, pattern = '*.yaml', full.names = T)[1]
cat('reading yaml file: ', yaml_file, '\n')
config <- yaml::read_yaml(yaml_file)

output_path <- paste0(base_data_path, '/final_results/')

# Create the directory path
cat('results will be written to ', output_path, '\n')
dir.create(output_path, recursive = TRUE, showWarnings = T)

#### calculation of true quantiles ######
# Function to generate samples based on distribution configuration
calculate_quantile <- function(dist_name, params, probs) {
  # Construct the function name for random sampling
  qfunc <- paste0("q", dist_name)
  
  # Generate a single sample
  quantile_tmp <- do.call(
    qfunc,
    c(list(p = probs), params)
  )
  
  return(quantile_tmp)
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
  
  return(as.character(dist_description))
}

true_statistics_df <- data.frame()
for (dist in config$distributions) {
  dist_type <- dist$type
  dist_name <- dist$name
  params <- dist$params
  probs <- config$global_settings$q_levels
  
  # Calculate the true quantiles
  tmp_quantiles <- calculate_quantile(dist_name = dist_type, params = params, probs = probs)
  names(tmp_quantiles) <- paste0(probs * 100, '%')
  tmp_quantiles <- as.data.frame(t(tmp_quantiles))
  tmp_quantiles$mean <- dist$theo_mean
  # Add experiment name to the quantiles
  tmp_quantiles$experiment <- create_distribution_name(dist_name, params)

  # Bind the result to the final data frame
  true_statistics_df <- rbind(true_statistics_df, tmp_quantiles)
}

true_statistics_df %<>% mutate(mean = unname(sapply(mean, function(x) eval(parse(text = x)))))

true_statistics_df %<>% pivot_longer(cols = -experiment, names_to = 'statistic', values_to = 'true_value')

true_file <- paste0(output_path, '/true_statistics.Rdata')
cat('write true statistics to ', true_file, '\n')
save(true_statistics_df, file = true_file)

#### aggregation of bootstrap percentile dataframes ######
cat('processing aggregation....\n')
data_files_dir <- '/boot_percentiles'
data_files_avail <- list.files(paste0(base_data_path, data_files_dir), pattern = '*.Rdata', full.names = T)
cat('found number of files to aggregate: ', length(data_files_avail), '\n')

bootstrap_full_aggregated <- foreach(file = data_files_avail, .combine = rbind, .packages = 'base') %dopar% {
  env <- new.env()  # Create a new environment to avoid overwriting issues
  load(file, envir = env)
  env$boot_statistics_percentiles
}

# Merge the data frames
bootstrap_full_aggregated <- merge(
  bootstrap_full_aggregated,
  true_statistics_df,
  by.x = c("experiment", "boot_statistic"),  
  by.y = c("experiment", "statistic"), 
  all.x = TRUE  # Keep all rows from bootstrap_full_aggregated
)

bootstrap_full_aggregated %<>% arrange(experiment, n, num_samples, num_bootsamples, sample_id, boot_statistic)
# overview <- bootstrap_full_aggregated %>% group_by(experiment, n, num_samples, num_bootsamples) %>% summarise(counts = n(), boot_statistics = list(unique(boot_statistic))) 

bootstrap_full_file <- paste0(output_path, '/bootstrap_full_aggregated.Rdata')
cat('write aggregated results to ', bootstrap_full_file, '\n')
save(bootstrap_full_aggregated, file = bootstrap_full_file)
