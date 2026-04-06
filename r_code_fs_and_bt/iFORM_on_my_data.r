# nolint: line_length_linter.
source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM.R")
#load filename

#filename = "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.0.csv" # nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.5.csv" #nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.5.csv" #nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.5.csv" #nolint
###filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.8.csv" #nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.0.csv" #nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.5.csv" #nolint
##filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.05_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.2_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_500_0.5_500_0.8.csv" #nolint
#ilename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.2_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_500_0.5.csv" #nolint
filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.5_500_0.8.csv" #nolint

# Load the dataset
sim_dataset <- read.table(filename, header = TRUE, sep = ",", quote = "\"")
  
# Remove columns with only one unique value
sim_dataset <- sim_dataset[, sapply(sim_dataset, function(x) length(unique(x)) > 1)] # nolint: line_length_linter.
  
# Handle NAs (if any)
sim_dataset <- na.omit(sim_dataset)
  
# Get predictor column names
predictor_columns <- colnames(sim_dataset)[!colnames(sim_dataset) %in% c("y", "p_i")] #nolint
  
# Create formula string for the model
predictor_str <- paste(predictor_columns, collapse = " + ")
formula_str <- paste("y ~", predictor_str)
formula_obj <- as.formula(formula_str)
  
# Build iForm model
iForm_fit <- iForm(formula_str, sim_dataset, heredity = "strong", higher_order = FALSE) #nolint
  
# Get iForm performance metrics
#iForm_AIC <- AIC(iForm_fit)
iForm_adj_R2 <- summary(iForm_fit)$adj.r.squared
rss_iForm_model <- sum(residuals(iForm_fit)^2)
  # Change the last column name to "y"
colnames(sim_dataset)[dim(sim_dataset)[2]] <- "y"
  #formula_str_2 <- paste("y ~", predictor_str)
formula_obj_2 <- as.formula(paste("y~ (", paste(predictor_str, collapse = " + "), ")^2")) #nolint
print(formula_obj_2)
  # Build forward selection model
#full_model <- lm(formula_obj_2, data = sim_dataset)
#null_model <- lm(y ~ 1, data = sim_dataset)
#forward_model <- step(null_model, scope = formula(full_model), direction = "forward", trace = FALSE)
  
  # Get forward selection performance metrics
  #forward_AIC <- AIC(forward_model)
#forward_adj_R2 <- summary(forward_model)$adj.r.squared
#rss_forward_model <- sum(residuals(forward_model)^2)
  # Append the results
rss_iForm_model = rss_iForm_model
    #iForm_adj_R2 = iForm_adj_R2,
    #forward_AIC = forward_AIC,
    #forward_adj_R2 = forward_adj_R2
#rss_forward_model = rss_forward_model
print(rss_iForm_model)# ...existing code...

# # --- Forward selection predictions and metrics ---
# forward_pred <- predict(forward_model, sim_dataset)
# forward_pred_label <- as.integer(forward_pred > 0.5)
# forward_mse <- mean((sim_dataset$y - forward_pred)^2)
# forward_accuracy <- mean(forward_pred_label == sim_dataset$y)

# cat("Forward selection MSE:", forward_mse, "\n")
# cat("Forward selection Accuracy:", forward_accuracy, "\n")

# # Extract selected features from forward selection
# forward_selected <- names(coef(forward_model))[-1]  # remove intercept
# cat("Forward selection selected features:\n")
# print(forward_selected)

library(jsonlite)

# ...existing code...

# Extract parameters from filename
filename_base <- basename(filename)
truth_base <- gsub("simulated_dataset_", "truth_simulated_dataset_", filename_base)
truth_base <- gsub(".csv$", ".json", truth_base)
truth_path <- file.path("C:/Users/enthe/Desktop/Thesis/data/simulated_data", truth_base)

# Load truth file
truth <- fromJSON(truth_path)

# ...existing code...

# --- Load truth file ---
#truth_path <- "C:/Users/enthe/Desktop/Thesis/data/simulated_data/truth_simulated_dataset_250_0.05_20_0.0.json"
#truth <- fromJSON(truth_path)

# --- True main and interaction sets (0-indexed) ---
main_idx <- as.integer(truth$main_idx)
print("True main effects (0-indexed):")
print(main_idx)
#wrong interaction_pairs <- lapply(truth$interaction_pairs, function(x) as.integer(x)) # nolint: line_length_linter.
# Extract 0-indexed pairs
# If truth$interaction_pairs is a matrix:
interaction_pairs <- split(truth$interaction_pairs, seq(nrow(truth$interaction_pairs)))
interaction_pairs <- lapply(interaction_pairs, as.integer)
print("1st")
print(interaction_pairs)

# --- Use iForm selected indices for main effects ---
coef_names <- names(iForm_fit$coefficients)
print("coef names")
print(coef_names)
main_names <- coef_names[!grepl(":", coef_names)]
print("main names")
print(main_names)
main_indices <- as.integer(gsub("X", "", main_names)) - 1
print(main_indices)

interaction_names <- coef_names[grepl(":", coef_names)]
interaction_indices <- lapply(strsplit(gsub("X", "", interaction_names), ":"), function(x) as.integer(x) - 1)


print("iForm selected interactions (0-indexed):")
print(interaction_indices)

iform_selected_main <- main_indices  # already 0-indexed
print(iform_selected_main)
print(iForm_fit)

# --- Extract selected interactions from iForm (still by name) ---
extract_inter_pairs <- function(selected_names) {
  pairs <- selected_names[grepl(":", selected_names)]
  lapply(strsplit(gsub("X", "", pairs), ":"), function(x) sort(as.integer(x) - 1))
}

iform_selected_inter <- extract_inter_pairs(coef_names)

# --- F1 calculation ---
f1_score_main <- function(true_set, pred_set) {
  tp <- length(intersect(true_set, pred_set))
  precision <- if (length(pred_set) == 0) 0 else tp / length(pred_set)
  recall <- if (length(true_set) == 0) 0 else tp / length(true_set)
  f1 <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  list(f1 = f1, precision = precision, recall = recall)
}

pair_to_str <- function(pair) paste(sort(pair), collapse = "_")
f1_score_inter <- function(true_pairs, pred_pairs) {
  true_str <- vapply(true_pairs, pair_to_str, character(1))
  pred_str <- vapply(pred_pairs, pair_to_str, character(1))
  tp <- length(intersect(true_str, pred_str))
  precision <- if (length(pred_str) == 0) 0 else tp / length(pred_str)
  recall <- if (length(true_str) == 0) 0 else tp / length(true_str)
  f1 <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  list(f1 = f1, precision = precision, recall = recall)
}

# Calculate F1 for main effects
main_f1 <- f1_score_main(main_idx, iform_selected_main)
cat(sprintf("iForm Main effects: F1=%.3f, Precision=%.3f, Recall=%.3f\n", main_f1$f1, main_f1$precision, main_f1$recall))

# Calculate F1 for interactions
true_inter_pairs <- lapply(interaction_pairs, sort)
iform_selected_inter_pairs <- lapply(iform_selected_inter, sort)
inter_f1 <- f1_score_inter(true_inter_pairs, iform_selected_inter_pairs)
cat(sprintf("iForm Interactions: F1=%.3f, Precision=%.3f, Recall=%.3f\n", inter_f1$f1, inter_f1$precision, inter_f1$recall))