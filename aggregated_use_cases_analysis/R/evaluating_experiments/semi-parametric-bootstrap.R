# implementation of bootstrapped semi-parametric quantile estimator, eqn. (2.2) from Hutson, A. D. (2002). A semi-parametric quantile function estimator for use in bootstrap estimation procedures. Statistics and Computing, 12(4), 331-338.

library(boot)
library(ggplot2)
library(dplyr)
library(magrittr)
library(tidyr)
library(tibble)
library(foreach)
library(doParallel)

# Set up parallel backend
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

my_seed <- 123
set.seed(my_seed)
# include semi-parametric quantile estimation
source(file = 'semiparametric-quant-extrapolation-est.R')

#### helper function to run bootstrap seed controlled ######
run_bootstrapping <- function(seed){

  set.seed(seed)
  orig_sample <- rnorm(n = sample_size, mean = 100, sd = 10)
  ###### apply bootstrap based on probability integral transform #########
  uni_samples <- replicate(n = num_bootsamples, runif(n = sample_size, min = 0, max = 1))
  
  # create bootstrap samples, single sample in column
  boot_samples <- t(apply(X = uni_samples, MARGIN = 1, FUN = quantile_tail_extrapolation, input_sample = orig_sample))
  
  boot_statistic <- apply(X = boot_samples, MARGIN = 2, FUN = quantile, probs = q_levels)
  boot_statistic <- as.data.frame(t(boot_statistic))
  
    boot_statistic_long <- boot_statistic %>% pivot_longer(cols = everything(), names_to = 'q_level', values_to = 'boot_statistic') %>% mutate(seed = seed)
  
    
  return(boot_statistic_long)
}

########## parameter setting for bootstrapping ############

# Define parameters
sample_size <- 10

q_levels <- c(0.95, 0.9, 0.5)
true_quantiles <- qnorm(p = q_levels, mean = 100, sd = 10)
# Create a data frame from the true quantiles for horizontal lines in final plot
true_quantiles_df <- data.frame(
  q_level = paste0(as.character(q_levels*100), '%'),
  true_quantile = true_quantiles
)



num_bootsamples <- 1500
my_seeds <- 1:500

start_time <- Sys.time()  # Start time

#seed_runs <- do.call(rbind, lapply(my_seeds, run_bootstrapping))
seed_runs <- foreach(seed = my_seeds, .combine = rbind, .packages = c("dplyr", "tidyr")) %dopar% {
  run_bootstrapping(seed)
}

end_time <- Sys.time()  # End time
elapsed_time <- end_time - start_time  # Elapsed time
print(elapsed_time)

# plot only with bootstrapped distribution percentiles
boot_percentiles_levels <- c(0.025, .975)
boot_conf_level <- boot_percentiles_levels[2] - boot_percentiles_levels[1]
boot_percentiles <- seed_runs %>%
  group_by(q_level, seed) %>%
  summarise(
    ymin = quantile(boot_statistic, probs = boot_percentiles_levels[1]),
    ymax = quantile(boot_statistic, probs = boot_percentiles_levels[2]),
    .groups = 'drop'
  )

# Merge true_quantiles_df with quantile_ranges
combined_df <- boot_percentiles %>%
  left_join(true_quantiles_df, by = "q_level")

# Check if true_quantile falls within the range of ymin and ymax
combined_df <- combined_df %>%
  mutate(is_within_range = ymin <= true_quantile & true_quantile <= ymax)

# empirical confidence level
emp_conf_level <- combined_df %>% group_by(q_level) %>% summarise(emp_conf_level = sum(is_within_range)/n())
# Generate custom labels for facets
facet_labels <- emp_conf_level %>%
  mutate(title = paste0(q_level, " (emp. c-level: ", 
                        scales::percent(emp_conf_level), ', theo. level: ', boot_conf_level*100, '%)')) %>%
  select(q_level, title) %>%
  deframe()

bootstrap_title_string <- paste0('Percentiles of bootstrap distributions (',
                                 boot_percentiles_levels[1]*100,
                                 '%, ',
                                 boot_percentiles_levels[2]*100,
                                 '%) for multiple runs (seed controlled) of semi-parametric quantiles,\nboot sample size: ', num_bootsamples, ', number of runs: ', length(my_seeds), ', initial sample size: n=', sample_size)
#bootstrap_title_string <- NULL

boot_percentiles %>% ggplot(aes(x = seed)) + 
  #  geom_boxplot() +
  geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.1) +
  geom_hline(data = true_quantiles_df, 
             aes(yintercept = true_quantile, 
                 color = q_level), 
             linetype = "dashed") +
  facet_wrap(q_level ~ ., scales = 'free_y', nrow = 3, labeller = as_labeller(facet_labels)) + 
  theme_light() +
  scale_color_manual(values = c("red", 'blue', 'orange'), name = "True Quantiles") +
  labs(title = bootstrap_title_string,
       x = "seed run",
       y = "Bootstrapped Statistic")




# plot with boxplots
seed_runs %>%
  ggplot(aes(y = boot_statistic, group = seed)) + 
  geom_boxplot() +
  geom_hline(data = true_quantiles_df, 
             aes(yintercept = true_quantile, 
                 color = q_level), 
             linetype = "dashed") +
  facet_wrap(q_level ~ ., scales = 'free_y', nrow = 3) + 
  theme_light() +
  scale_color_manual(values = c("red", 'blue', 'orange'), name = "True Quantiles") +
  labs(title = paste0('Bootstrap distributions for multiple runs (seed controlled) of semi-parametric quantiles,\nboot sample size: ', num_bootsamples, ', number of runs: ', length(my_seeds), ', initial sample size: n=', sample_size),
       x = "seed run",
       y = "Bootstrapped Statistic")


# Stop the cluster
stopCluster(cl)
