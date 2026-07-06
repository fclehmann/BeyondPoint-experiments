# uncertainty estimation for ML applications
# Basic ideas: get condidence intervals based on quantile estimation, advantage here: with median, some mean concept is covered as well
# 1) CI based on normal approximation
# 2) bootstrap CI

#### lib stuff #####

library(jsonlite)
library(ggplot2)
library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(scales)
library(ggthemes)
library(nortest)
library(purrr)
library(colourvalues)
library(purrr)

#### data loading stuff ####
json_file_path <- 'data_03june2024.json'

# Read JSON data from file
data <- fromJSON(json_file_path)

save_plot <- FALSE
# Convert the list of records to a data frame
df <- as.data.frame(data) %>% 
  mutate(experiment = as.factor(experiment))
df$experiment %>% unique()

df_reduced <- df %>% 
  select(experiment, metric_values) %>% 
  filter(str_detect(experiment, "simpsons")) %>% 
  #filter(str_detect(experiment, "^(21|16)")) %>% 
  #unnest(cols = c(seeds, metric_values)) %>% 
  mutate(accuracy = map(metric_values, ~ 1 - .x), 
         exp_category = case_when(
           str_detect(experiment, "hopt") ~ "HOPT+OPTIMIZER",
           str_detect(experiment, "split") ~ "SPLIT",
           str_detect(experiment, "dropout") ~ "DROPOUT",
           str_detect(experiment, "augment") ~ "AUGMENT",
           str_detect(experiment, "weights") ~ "WEIGHTS",
           TRUE ~ "OTHER"
         ) %>%
           factor(levels = c("HOPT+OPTIMIZER", "SPLIT", "DROPOUT",'AUGMENT', 'WEIGHTS', "OTHER"))
         )

df_reduced %<>% mutate(initial_quantiles = map(accuracy, ~ quantile(.x, probs = quantile_probs <- c(0.025, 0.5, 0.975)))) %>% select(-metric_values)

df_long <- df_reduced %>% unnest(cols = accuracy)

str(df_long)

#### creating subsamples ########

# Define the number of samples and vector of number of observations per sample
num_samples <- 5
size_per_sample <- c(5, 10, 20)

# Function to create subsamples
create_subsamples <- function(df, num_samples, size_per_sample) {
  df %>%
    group_by(experiment) %>%
    summarize(subsamples = list(
      map(size_per_sample, function(size) {
        map(seq_len(num_samples), function(i) {
          slice_sample(across(c(accuracy)), n = size) %>%
            mutate(sample_number = i, sample_size = size)
        })
      }) %>%
        flatten()
    ), .groups = 'drop') %>%
    unnest(subsamples)
}

df_subsamples <- create_subsamples(df_long, num_samples, size_per_sample)

#### calculate CI based on normal approximation of accuracy rate ####

quantile_probs <- c(0.025, 0.5, 0.975)
df_subsamples %<>% mutate(empiricial_quantiles = map(subsamples, ~ quantile(.x$accuracy, probs = quantile_probs)), 
                          pval_lillie = map(subsamples, ~ round(lillie.test(unlist(.x$accuracy))$p.value, digits = 3)),
                          pval_shapiro = map(subsamples, ~ round(shapiro.test(unlist(.x$accuracy))$p.value, digits = 3)),
                          )

#### calculate bootstrap CI #####
library(boot)

Mboot = boot(Data$Likert,
             function(x,i) median(x[i]),
             R=10000)

boot.ci(Mboot,
        conf = 0.95,
        type = c("norm", "basic" ,"perc", "bca")
)