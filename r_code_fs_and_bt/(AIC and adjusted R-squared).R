# List of all filenames

setwd("C:/Users/enthe/Documents/projectX/PliableLasso/")
source("C:/Users/enthe/Documents/Internship/iform/iform_3.R")
filenames <- c(
  "microRNA_MCI_dataset_50_20_0.1.csv",
  "microRNA_MCI_dataset_50_20_0.01.csv",
  "microRNA_MCI_dataset_50_20_1.0.csv",
  "microRNA_MCI_dataset_500_20_0.1.csv",
  "microRNA_MCI_dataset_500_20_0.01.csv",
  "microRNA_MCI_dataset_500_20_1.0.csv",
  "microRNA_MCI_dataset_500_300_0.1.csv",
  "microRNA_MCI_dataset_500_300_0.01.csv",
  "microRNA_MCI_dataset_500_300_1.0.csv",
  "microRNA_MCI_dataset_50_300_0.1.csv",
)





# Initialize empty lists to store results
results <- data.frame(filename = character(), iForm_AIC = numeric(), iForm_adj_R2 = numeric(),
                      forward_AIC = numeric(), forward_adj_R2 = numeric(), stringsAsFactors = FALSE)

# Loop through each file
for (filename in filenames) {
  # Load the dataset
  sim_dataset <- read.table(filename, header = TRUE, sep = ",", quote = "\"")
  
  # Remove columns with only one unique value
  sim_dataset <- sim_dataset[, sapply(sim_dataset, function(x) length(unique(x)) > 1)]
  
  # Handle NAs (if any)
  sim_dataset <- na.omit(sim_dataset)
  
  # Get predictor column names
  predictor_columns <- colnames(sim_dataset)[colnames(sim_dataset) != "MCI"]
  
  # Create formula string for the model
  predictor_str <- paste(predictor_columns, collapse = " + ")
  formula_str <- paste("MCI ~", predictor_str)
  formula_obj <- as.formula(formula_str)
  
  # Build iForm model
  iForm_fit <- iForm(formula_str, sim_dataset, heredity = "strong", higher_order = TRUE)
  
  # Get iForm performance metrics
  #iForm_AIC <- AIC(iForm_fit)
  iForm_adj_R2 <- summary(iForm_fit)$adj.r.squared
  rss_iForm_model <- sum(residuals(iForm_fit)^2)
  # Change the last column name to "y"
  colnames(sim_dataset)[dim(sim_dataset)[2]] <- "y"
  #formula_str_2 <- paste("y ~", predictor_str)
  formula_obj_2 <- as.formula(paste("y~ (", paste(predictor_str, collapse = " + "), ")^2"))
  print(formula_obj_2)
  # Build forward selection model
  full_model <- lm(formula_obj_2, data = sim_dataset)
  null_model <- lm(y ~ 1, data = sim_dataset)
  forward_model <- step(null_model, scope = formula(full_model), direction = "forward", trace = FALSE)
  
  # Get forward selection performance metrics
  #forward_AIC <- AIC(forward_model)
  forward_adj_R2 <- summary(forward_model)$adj.r.squared
  rss_forward_model <- sum(residuals(forward_model)^2)
  # Append the results
  results <- rbind(results, data.frame(
    filename = filename,
    #iForm_AIC = iForm_AIC,
    rss_iForm_model = rss_iForm_model,
    #iForm_adj_R2 = iForm_adj_R2,
    #forward_AIC = forward_AIC,
    #forward_adj_R2 = forward_adj_R2
    rss_forward_model = rss_forward_model
  ))
}


# Required library
library(ggplot2)

# Melt data frame for ggplot
library(reshape2)
results_melt <- melt(results, id.vars = "filename", variable.name = "model_metric", value.name = "value")

# Separate the metrics for better visualization (AIC and adjusted R-squared)
ggplot(results_melt, aes(x = filename, y = value, fill = model_metric)) + 
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Comparison of iForm and Forward Models Across Files", 
       x = "Filename", y = "Value")



