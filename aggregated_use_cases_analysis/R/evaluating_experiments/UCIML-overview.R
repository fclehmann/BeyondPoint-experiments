# Install and load the jsonlite package
#install.packages("jsonlite")
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
# Specify the path to your JSON file
json_file_path <- 'data_03june2024.json'

# Read JSON data from file
data <- fromJSON(json_file_path)

save_plot <- FALSE
# Convert the list of records to a data frame
df <- as.data.frame(data) %>% 
  mutate(experiment = as.factor(experiment))

df$experiment %>% unique() 

p_vals <- df %>% 
  group_by(experiment) %>% 
  summarise(ad = round(ad.test(unlist(metric_values))$p.value, digits = 3), 
            lf = round(lillie.test(unlist(metric_values))$p.value, digits = 3)) 

#### simpsons plot #########

plot_data <- df %>% 
  select(experiment, seeds, metric_values) %>% 
  filter(str_detect(experiment, "simpsons")) %>% 
  #filter(str_detect(experiment, "^(21|16)")) %>% 
  unnest(cols = c(seeds, metric_values)) %>% 
  mutate(accuracy = 1- metric_values, 
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

plot_data %>% ggplot(aes(x = accuracy)) + geom_density(adjust = 0.7, bounds = c(0,1)) + 
  theme_light()

overview <- df %>% 
  select(experiment, metric_values) %>% group_by(experiment) %>% 
  summarise(count = unlist(metric_values) %>% length()) %>% 
  arrange(experiment, count)
  
  
# Data frame with captions for each facet (count of runs)
caption_df <- plot_data %>%
  group_by(experiment) %>%
  summarise(caption = paste("# runs:", n()), 
            AD_pvalue = paste0('pval_AD:', round(ad.test(unlist(accuracy))$p.value, digits = 3)), 
            lillie_pvalue = paste0('pval_lillie:', round(lillie.test(unlist(accuracy))$p.value, digits = 3))
  )


plot <- 
  plot_data %>%
  ggplot(aes(x = accuracy, fill = experiment)) +
  geom_density(adjust = 0.7, bounds = c(0, 1)) +
  #geom_histogram() +
  #scale_color_colorblind() +
  facet_wrap(.~exp_category, scales = "free_y", ncol = 1) +
  # geom_text(
  #   data = caption_df,
  #   aes(
  #     x = 0.812,
  #     y = -Inf,
  #     label = paste(caption, ' '),#, AD_pvalue, ' ', lillie_pvalue),
  #     hjust = -.25,
  #     vjust = -1
  #   )
  # ) +
  labs(title = 'empirical distributions of accuracy rate for image classification after of training of neural network (CNN), \nfor different experiment settings (variation of optimiziers, train-test split etc., approx. 1000 runs per setting)', 
       x = 'accuracy rate', y = "kernel density estimate") +
  paletteer::scale_fill_paletteer_d("Polychrome::glasbey") +
  theme_light() +
  theme(
    plot.caption = element_text(margin = margin(t = 10)),
    strip.text = element_text(size = 10)#, 
    #legend.position = 'none'
  ) +
  scale_x_continuous(breaks = seq(from = 0.3, to = 1, by = 0.025)) 
save_plot = TRUE
if (save_plot == TRUE){
  ggsave(plot = plot, 
       path = '.', 
       filename = 'UCIML-exp-categorized-simpsons.pdf', 
       dpi = 300, 
       scale = .6, 
       width = 16, 
       height = 9, 
       limitsize = FALSE)
} else {
  plot
}


#### simpsons 3D plot (rgl approach) ##########
library(rgl)

plot_data <- df %>% 
  select(experiment, seeds, metric_values) %>% 
  filter(str_detect(experiment, "simpsons")) %>% 
  #filter(str_detect(experiment, "^(10|11|13)")) %>% 
  unnest(cols = c(seeds, metric_values)) %>% 
  mutate(accuracy = 1- metric_values)

# Create a grid of x values for density calculation
x_grid <- seq(min(plot_data$accuracy), max(plot_data$accuracy), length.out = 5000)

# Update factor levels based on filtered data
exp_levels <- unique(plot_data$experiment)

df_kde <- plot_data %>%
  group_by(experiment) %>%
  do({
    dens <- density(.$accuracy, from = min(x_grid), to = max(x_grid), n = length(x_grid))
    data.frame(
      accuracy = dens$x,
      kde = dens$y,
      mean_acc = mean(.$accuracy)
    )
  }) %>%
  ungroup()

df_kde <- df_kde %>% arrange(mean_acc) %>% mutate(experiment = factor(experiment, levels = exp_levels))

df_kde$color <- colour_values(df_kde$mean_acc, palette = "plasma")


# Plot
plot3d( 
  x=df_kde$accuracy, y=df_kde$experiment, z=df_kde$kde, 
  col = df_kde$color, 
  type = 'p', 
  radius = .1,
  xlab="accuracy", ylab="experiment", zlab="kde")

# Set axis ticks
grid3d(c("y+", 'z-'))

rglwidget()






#### simpsons 3D plot ##########
plot_data <- df %>% 
  select(experiment, seeds, metric_values) %>% 
  filter(str_detect(experiment, "simpsons")) %>% 
  #filter(str_detect(experiment, "^(10|11|13)")) %>% 
  unnest(cols = c(seeds, metric_values)) %>% 
  mutate(accuracy = 1- metric_values)

# Create a grid of x values for density calculation
x_grid <- seq(min(plot_data$accuracy), max(plot_data$accuracy), length.out = 1000)

# Update factor levels based on filtered data
exp_levels <- unique(plot_data$experiment)

df_kde <- plot_data %>%
  group_by(experiment) %>%
  do({
    dens <- density(.$accuracy, from = min(x_grid), to = max(x_grid), n = length(x_grid))
    data.frame(
      accuracy = dens$x,
      kde = dens$y,
      mean_acc = mean(.$accuracy)
    )
  }) %>%
  ungroup()

df_kde <- df_kde %>% arrange(mean_acc) %>% mutate(experiment = factor(experiment, levels = exp_levels))

# Assign colors to df_kde based on the experiment variable
df_kde$color <- colour_values(df_kde$experiment, palette = "plasma")

# Plot using plotly
fig <- plot_ly(df_kde, x = ~accuracy, y = ~experiment, z = ~kde,
               color =  'blue',
               alpha = 1,
               type = 'scatter3d',  
               mode = 'lines', size = I(5))

fig <- fig %>%
  layout(scene = list(
    xaxis = list(title = 'accuracy'),
    yaxis = list(title = ''),  # Use ordered levels
    zaxis = list(title = 'kernel density estimate')
  )) %>%
  layout(showlegend = FALSE)

fig

df_kde %>% select(experiment, mean_acc) %>% unique()

df_kde

NULL

#### low resource language experiment plot #########
plot_data <- df %>% 
  select(experiment, seeds, metric_values) %>% 
  filter(str_detect(experiment, "lrl")) %>% 
  #filter(str_detect(experiment, "^(10|11|13|16|21)")) %>% 
  unnest(cols = c(seeds, metric_values)) 

overview <- df %>% 
  select(experiment, metric_values) %>% group_by(experiment) %>% 
  summarise(count = unlist(metric_values) %>% length()) %>% 
  arrange(experiment, count)


# Data frame with captions for each facet (count of runs)
caption_df <- plot_data %>%
  group_by(experiment) %>%
  summarise(caption = paste("# runs:", n()), 
            AD_pvalue = paste0('pval_AD:', round(ad.test(unlist(metric_values))$p.value, digits = 3)), 
            lillie_pvalue = paste0('pval_lillie:', round(lillie.test(unlist(metric_values))$p.value, digits = 3))
  )


# Create a ggplot with geom_text for different captions per facet
plot <- 
  plot_data %>%
  ggplot(aes(x = metric_values, fill = experiment)) +
  #geom_density(adjust = 0.7, bounds = c(0, 1)) +
  geom_histogram() +
  #scale_color_colorblind() +
facet_wrap(.~experiment, scales = "free_y", ncol = 1) +
geom_text(
  data = caption_df,
  aes(
    x = 0.5,
    y = -Inf,
    label = paste(caption, ' ', AD_pvalue, ' ', lillie_pvalue),
    hjust = -.25,
    vjust = -1
  )
) +
labs(x = 'F1 score (macro)', y = "kernel density estimate") +
  theme_light() +
  theme(
    plot.caption = element_text(margin = margin(t = 10)),
    strip.text = element_text(size = 10)
  ) +
scale_x_continuous(breaks = seq(from = 0.0, to = 1, by = 0.05))
plot

if (save_plot == TRUE){
  ggsave(plot = plot, 
         path = '.', 
         filename = 'UCIML-overview-lrl.pdf', 
         dpi = 900, 
         scale = 1.0, 
         width = 16, 
         height = 9, 
         limitsize = FALSE)
} else {
  plot
}
