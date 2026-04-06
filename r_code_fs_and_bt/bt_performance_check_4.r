# Load necessary libraries
library(LassoBacktracking)
library(jsonlite)
library(Matrix)

setwd("C:/Users/enthe/Documents/projectX/PliableLasso")
# Function to process each dataset

#filename = "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.0.csv" # nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.05_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_20_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.2_500_0.8.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.0.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.5.csv" #nolint
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_250_0.5_20_0.8.csv" #nolint
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
#filename =  "C:/Users/enthe/Desktop/Thesis/data/simulated_data/simulated_dataset_1000_0.05_20_0.0.csv" #nolint
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





# Build truth file path from filename
filename_base <- basename(filename)
truth_base <- gsub("simulated_dataset_", "truth_simulated_dataset_", filename_base)
truth_base <- gsub(".csv$", ".json", truth_base)
truth_path <- file.path("C:/Users/enthe/Desktop/Thesis/data/simulated_data", truth_base)

# Read truth JSON
truth <- fromJSON(truth_path)
true_main <- as.integer(truth$main_idx) + 1
#true_interactions <- lapply(truth$interaction_pairs, function(pair) sort(as.integer(pair) + 1))
#true_interactions <- apply(truth$interaction_pairs, 1, function(pair) sort(as.integer(pair) + 1))
true_interactions <- lapply(seq_len(nrow(truth$interaction_pairs)), function(i) sort(as.integer(truth$interaction_pairs[i, ]) + 1))
print("Raw interaction pairs from truth file:")
print(truth$interaction_pairs)

# Read simulated dataset
data <- read.csv(filename)
X <- as.matrix(data[, grep("^X", names(data))])
y <- data$y


# print(any(is.na(X)))
# print(any(is.na(y)))
# print(dim(X))
# print(apply(X, 2, function(col) length(unique(col))))
# X <- as.matrix(data[, grep("^X", names(data))])
# X <- X[, apply(X, 2, function(col) length(unique(col)) > 1), drop = FALSE]
# if (ncol(X) == 0) stop("No non-constant columns in X!")
# print(class(X))
# print(table(y))
# print(colnames(X))
# ...rest of your code: modeling, evaluation, etc...


process_dataset <- function(filename) {
  #set.seed(42)
  #RNGkind("L'Ecuyer-CMRG")
  
  data <- read.csv(filename)
  cat("here:")
  #print(typeof(data))
  #print(data)
  #X <- as.matrix(data[, -which(names(data) == "y")])
  X <- as.matrix(data[, grep("^X", names(data))])
  storage.mode(X) <- "double"

  y <- data$y
  #print(table(y))
  
  nlambda <- 100
  iter_max <- 100
  lambda_min_ratio <- ifelse(nrow(X) < ncol(X), 0.01, 1e-8)
  thresh <- 1e-8
  verbose <- TRUE
  
  lasso_model <- LassoBT(
    x = X,
    y = y,
    nlambda = nlambda,
    iter_max = iter_max,
    lambda.min.ratio = lambda_min_ratio,
    thresh = thresh,
    verbose = verbose
  )
  
  cv_model <- cvLassoBT(
    x = X,
    y = y,
    nlambda = nlambda,
    lambda.min.ratio = lambda_min_ratio,
    nfolds = 5,
    nperms = 1,
    mc.cores = 1
  )
  
  n_main_vars <- lasso_model$nvars
  opt_lambda_idx <- cv_model$cv_opt[1]
  last_iter <- length(lasso_model$beta)
  print(last_iter)
  beta_at_lambda <- lasso_model$beta[[last_iter]][, opt_lambda_idx]
  #print(beta_at_lambda)
  beta_at_lambda <- lasso_model$beta[[last_iter]][, opt_lambda_idx]
  summary(beta_at_lambda)
  
  main_coeffs <- beta_at_lambda[1:n_main_vars]
  print(main_coeffs)
  
  selected_main_1 <- which(main_coeffs >= 1e-4)
  names(selected_main_1) <- NULL
  # Extract the optimal lambda value
  opt_lambda <- cv_model$lambda[cv_model$cv_opt[1]]
  #opt_lambda = opt_lambda - 1
  print(opt_lambda)
  predictions <- predict(lasso_model, newx = X, s = opt_lambda)
  
  binary_predictions <- ifelse(predictions > 0.5, 1, 0)
  accuracy <- mean(binary_predictions == y)
  mse <- mean((y - predictions)^2)
  
  tp <- sum(binary_predictions == 1 & y == 1)
  fp <- sum(binary_predictions == 1 & y == 0)
  fn <- sum(binary_predictions == 0 & y == 1)
  
  precision <- ifelse((tp + fp) == 0, NA, tp / (tp + fp))
  recall <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
  f1_score <- ifelse((precision + recall) == 0, NA, 2 * precision * recall / (precision + recall))
  
  # Extract selected main effects
  #selected_main <- sort(unique(lasso_model$active_vars)) + 1  # adjust to 1-based indexing
  print(selected_main_1)
  # Extract selected interaction matrix
  interactions_matrix <- if (!is.null(lasso_model$interactions)) {
    lasso_model$interactions + 1  # adjust to 1-based indexing
  } else {
    matrix(nrow = 2, ncol = 0)
  }
  
  return(list(
    filename = filename,
    accuracy = accuracy,
    mse = mse,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    cv_model = cv_model,
    lasso_model = lasso_model,
    interactions_matrix = interactions_matrix,
    selected_main = selected_main_1
  ))
}

evaluate_main_effects <- function(true_main, selected_main) {
  tp <- sum(selected_main %in% true_main)
  fp <- sum(!(selected_main %in% true_main))
  fn <- sum(!(true_main %in% selected_main))
  
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  f1 <- ifelse(is.na(precision) || is.na(recall) || (precision + recall == 0),
               NA,
               2 * precision * recall / (precision + recall))
  
  return(list(
    TP = tp, FP = fp, FN = fn,
    precision = precision,
    recall = recall,
    f1 = f1
  ))
}
cat("True interaction pairs:\n")
print(true_interactions)
cat("Selected interaction pairs:\n")
if (!is.null(predicted$interactions_matrix) && ncol(predicted$interactions_matrix) > 0) {
  print(apply(predicted$interactions_matrix, 2, function(x) sort(as.integer(x))))
} else {
  print("None selected")
}
cat("Number of true pairs:", length(true_interactions), "\n")
cat("Number of selected pairs:", ifelse(is.null(predicted$interactions_matrix), 0, ncol(predicted$interactions_matrix)), "\n")

evaluate_interactions <- function(true_list, selected_matrix) {
  true_keys <- unique(sapply(true_list, function(x) paste(sort(x), collapse = "-")))
  if (is.null(selected_matrix) || ncol(selected_matrix) == 0) {
    selected_keys <- character(0)
  } else {
    selected_keys <- unique(apply(selected_matrix, 2, function(x) paste(sort(as.integer(x)), collapse = "-")))
  }
  tp <- sum(selected_keys %in% true_keys)
  fp <- sum(!(selected_keys %in% true_keys))
  fn <- sum(!(true_keys %in% selected_keys))
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  f1 <- ifelse(is.na(precision) || is.na(recall) || (precision + recall == 0),
               NA,
               2 * precision * recall / (precision + recall))
  return(list(
    TP = tp, FP = fp, FN = fn,
    precision = precision,
    recall = recall,
    f1 = f1
  ))
}

# List of dataset filenames
datasets <- list.files(pattern = "microRNA_MCI_dataset_.*\\.csv")
true_var_datasets <- list.files(pattern = "chosen_indices_and_coefficients_.*\\.csv")

# get_true_support <- function(filename) {
#   df <- read.csv(filename, stringsAsFactors = FALSE)
  
#   # Extract main effect indices (fixing 0-based indexing)
#   true_main <- as.integer(df$Main.Effect.Indices)
#   true_main <- true_main + 1
#   print(true_main)
#   # Parse interaction indices, e.g. "(9, 17)" → c(9,17)
#   interaction_raw <- gsub("[()]", "", df$Interaction.Effect.Indices)
#   interaction_pairs <- strsplit(interaction_raw, ",\\s*")
#   true_interactions <- lapply(interaction_pairs, function(p) {
#     sorted_1_based <- sort(as.integer(p) + 1)  # Add 1 here
#     return(sorted_1_based)
#   })
  
  
#   return(list(main = sort(unique(true_main)), interactions = unique(true_interactions)))
# }

# Evaluate both main and interaction effects for one dataset
#true_support <- get_true_support(true_var_datasets[6])
predicted <- process_dataset(filename)

#predicted_TGS <- process_dataset(TGS_data)

#main_eval <- evaluate_main_effects(true_support$main, predicted$selected_main)
main_eval <- evaluate_main_effects(true_main, predicted$selected_main)

print(predicted$selected_main)
#interaction_eval <- evaluate_interactions(true_support$interactions, predicted$interactions_matrix)
interaction_eval <- evaluate_interactions(true_interactions, predicted$interactions_matrix)


cat("Main Effects:\n")
print(main_eval)

cat("\nInteraction Effects:\n")
print(interaction_eval)



