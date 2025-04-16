###### packages and config ######

library(lme4)
library(tidyverse) # data wrangling and visualization
library(readr)
library(tidyr)
library(dplyr)
library(mice)
library(corrplot)  # For visualization
library(caret)     # For correlation-related functions (optional)
library(car)
library(broom.mixed)
#library(rsample)
library(boot)
library(sjPlot)
library(flextable)
library(broom.mixed)
library(performance)
library(stringr)
library(pROC)
library(Metrics)   # for logLoss()

options(scipen = 999)

###### load data ######

#change to appropriate working directory
setwd("C:/Users/BootMR/Documents/data_processed/CHECKIFYOUREALLYNEEDTHISFOLDER")
df <- read_csv("ratingsFeatures_baselcorr_17-3_C1WPCSWSFBCNSR_merged.csv")

###### weighted sampling and rating mapping ######

# Remove rows with rating 0
df <- df[df$rating != 0, ]

# Map rating -1 to 0
df$rating[df$rating == -1] <- 0

# Compute rating weight
df$rating_weight <- 1 / ave(df$p_id, df$p_id, FUN=length)

# Set desired sample size (adjust as needed)
sample_size <- 600  # Example: selecting 500 samples

# Perform weighted random sampling
set.seed(42)
df_sampled <- df[sample(1:nrow(df), size=sample_size, prob=df$rating_weight, replace=FALSE), ]

###### impute ######

# Function to impute missing values with the median for all continuous variables
impute_missing_values_median <- function(df) {
  continuous_vars <- c("interval_id", "warnings_slowdown_count", "H10_hr_mean", "H10_hrv_mean", 
                       "H10_HRV_MeanNN", "E4_eda_tonic_mean", "E4_eda_phasic_mean", "E4_eda_tonic_peaks", 
                       "E4_eda_phasic_peaks", "cadence_avg", "cadence_pvr", "velocity_avg", "velocity_slope", 
                       "wind_speed", "temperature", "BMI", "pleasantnessdisposition_mapped", 
                       "FB_hinder_mapping", "FB_wegtype_mapping", "FB_wegkwal_mapping")
  
  # Loop through each continuous variable and impute missing values with the median
  for (var in continuous_vars) {
    df[[var]][is.na(df[[var]])] <- median(df[[var]], na.rm = TRUE)
  }
  
  return(df)
}

# Apply the median imputation function to your dataset
df_imputed_median <- impute_missing_values_median(df_sampled)


###### run single model ######


# gives error p_id invalid grouping factor: sociodem_fitness 
# drops random effect to 0: context_perceivedinfluence
# drops random effect to very low: weather_wind_speed +  weather_temperature +


# Define your predictors as a character vector
predictors <- c(
  "rating_totalperpid", "H10_hr_mean", "H10_hrv_mean", "H10_HRV_RMSSD", "H10_HRV_MeanNN", "H10_HRV_HF", 
  "H10_HRV_SD1", "H10_HRV_SD2", "E4_eda_phasic_mean", "E4_eda_phasic_peaks", "E4_eda_phasic_max", 
  "E4_eda_phasic_n_above_mean", "E4_eda_tonic_mean", "E4_eda_tonic_peaks", "E4_eda_tonic_n_above_mean", 
  "cadence_avg", "velocity_avg", "velocity_avg_change", "warnings_slowdown_count", "warnings_tactile_warning", 
  "warnings_audio_warning", "warnings_warning_value", "sociodem_income", "sociodem_education", 
  "sociodem_cycling_experience", "sociodem_BMI", "context_surface_type", "context_road_quality", 
  "context_scenic_beauty", "context_hindrance", "context_road_type", "sociodem_mood", 
  "sociodem_pleasantness_disposition", "cycling_perceivedintensity"
)

# Paste predictors into a formula string, Add outcome and random effect to create full formula, Convert to formula object
model_formula <- as.formula(paste("rating ~", paste(predictors, collapse = " + "), "+ (1 | p_id)"))

# Fit the model
model <- glmer(  formula = model_formula, data = df_imputed_median, family = binomial, control = glmerControl(optimizer = "bobyqa"))

# Display summary
summary(model)

###### run model 10 fold CV, without p_id RE ######

# Create 10 group-level folds using caret's groupKFold
folds <- groupKFold(group = df_imputed_median$p_id, k = 10)

# Prepare lists to collect results
coef_list <- list()

for (i in seq_along(folds)) {
  cat("Running fold", i, "...\n")
  
  train_idx <- folds[[i]]
  train_data <- df_imputed_median[train_idx, ]
  
  # Fit model on training data
  model <- tryCatch({
    glmer(model_formula, data = train_data, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  }, error = function(e) NULL)
  
  if (!is.null(model)) {
    tidy_model <- broom.mixed::tidy(model, effects = "fixed")
    coef_list[[i]] <- tidy_model
  }
}

# Combine all coefficient results
coef_df <- bind_rows(coef_list, .id = "fold")

# Summarize: mean estimate and p-value across folds
summary_df <- coef_df %>%
  group_by(term) %>%
  summarise(
    mean_estimate = mean(estimate, na.rm = TRUE),
    mean_p_value = mean(p.value, na.rm = TRUE)
  ) %>%
  arrange(mean_p_value)

# Show summary
print(summary_df)


###### run model 10 fold CV, without p_id RE ######

# Create 10 group-level folds using caret's groupKFold
folds <- groupKFold(group = df_imputed_median$p_id, k = 10)

# Prepare lists to collect results
coef_list <- list()
re_var_list <- numeric(length(folds))  # To store random effect variance

for (i in seq_along(folds)) {
  cat("Running fold", i, "...\n")
  
  train_idx <- folds[[i]]
  train_data <- df_imputed_median[train_idx, ]
  
  # Fit model on training data
  model <- tryCatch({
    glmer(model_formula, data = train_data, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  }, error = function(e) NULL)
  
  if (!is.null(model)) {
    # Fixed effects
    tidy_model <- broom.mixed::tidy(model, effects = "fixed")
    coef_list[[i]] <- tidy_model
    
    # Extract variance of random effect for p_id
    var_components <- as.data.frame(VarCorr(model))
    p_id_var <- var_components %>% filter(grp == "p_id") %>% pull(vcov)
    re_var_list[i] <- ifelse(length(p_id_var) == 1, p_id_var, NA)
  } else {
    re_var_list[i] <- NA
  }
}

# Combine all coefficient results
coef_df <- bind_rows(coef_list, .id = "fold")

# Summarize fixed effects: mean estimate and p-value across folds
summary_df <- coef_df %>%
  group_by(term) %>%
  summarise(
    mean_estimate = mean(estimate, na.rm = TRUE),
    mean_p_value = mean(p.value, na.rm = TRUE)
  ) %>%
  arrange(mean_p_value)

# Compute mean and standard deviation of p_id random effect variance
mean_re_variance <- mean(re_var_list, na.rm = TRUE)
sd_re_variance <- sd(re_var_list, na.rm = TRUE)

# Show summary
print(summary_df)
cat("\nRandom intercept variance for p_id across folds:\n")
cat("  Mean: ", round(mean_re_variance, 4), "\n")
cat("  SD:   ", round(sd_re_variance, 4), "\n")


###### corrplot ######

# Subset the data
df_corr <- df_imputed_median %>% select(all_of(predictors))

# Compute correlation matrix
cor_matrix <- cor(df_corr, use = "complete.obs")

# Plot using corrplot
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 0.6,
         diag = FALSE,
         order = "hclust")  # Optional: cluster similar variables



