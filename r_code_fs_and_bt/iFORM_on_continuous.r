source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM.R")
library(jsonlite)
library(tools)

data_dir <- "C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous"
out_dir  <- data_dir

dataset_files <- list.files(
  data_dir,
  pattern = "^simulated_dataset_.*\\.csv$",
  full.names = TRUE
)

## ---------- helpers (global) ----------

evaluate_main_effects <- function(true_main, selected_main) {
  tp <- sum(selected_main %in% true_main)
  fp <- sum(!(selected_main %in% true_main))
  fn <- sum(!(true_main %in% selected_main))
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  recall    <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  f1 <- ifelse(
    is.na(precision) || is.na(recall) || (precision + recall == 0),
    NA,
    2 * precision * recall / (precision + recall)
  )
  list(
    TP = tp, FP = fp, FN = fn,
    precision = precision,
    recall = recall,
    f1 = f1
  )
}

evaluate_interactions <- function(true_list, selected_matrix) {
  true_keys <- unique(
    sapply(true_list, function(x) paste(sort(x), collapse = "-"))
  )

  if (is.null(selected_matrix) || length(selected_matrix) == 0) {
    selected_keys <- character(0)

  } else if (is.matrix(selected_matrix)) {
    selected_keys <- unique(
      apply(selected_matrix, 2, function(x)
        paste(sort(as.integer(x)), collapse = "-"))
    )

  } else if (is.list(selected_matrix)) {
    selected_keys <- unique(
      sapply(selected_matrix, function(x)
        paste(sort(as.integer(x)), collapse = "-"))
    )

  } else {
    stop("selected_matrix must be NULL, a matrix, or a list")
  }

  tp <- sum(selected_keys %in% true_keys)
  fp <- sum(!(selected_keys %in% true_keys))
  fn <- sum(!(true_keys %in% selected_keys))
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  recall    <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  f1 <- ifelse(
    is.na(precision) || is.na(recall) || (precision + recall == 0),
    NA,
    2 * precision * recall / (precision + recall)
  )
  list(
    TP = tp, FP = fp, FN = fn,
    precision = precision,
    recall = recall,
    f1 = f1
  )
}

## ---------- outer CV settings ----------

outer_k <- 5   # number of outer folds for unbiased MSE

## ---------- main loop over datasets ----------

for (filename in dataset_files) {
  cat("\nProcessing:", filename, "\n")

  # Build truth file path
  filename_base <- basename(filename)
  truth_base <- gsub("simulated_dataset_", "truth_simulated_dataset_", filename_base)
  truth_base <- gsub(".csv$", ".json", truth_base)
  truth_path <- file.path(data_dir, truth_base)

  if (!file.exists(truth_path)) {
    cat("Truth file not found for", filename, "\n")
    next
  }

  # Read truth (0-based indices)
  truth <- fromJSON(truth_path)
  true_main <- as.integer(truth$main_idx)               # 0-based
  true_interactions <- lapply(
    seq_len(nrow(truth$interaction_pairs)),
    function(i) sort(as.integer(truth$interaction_pairs[i, ]))  # still 0-based
  )

  # Read data
  data <- read.csv(filename)

  # Remove columns with only one unique value
  data <- data[, sapply(data, function(x) length(unique(x)) > 1)]

  # Remove rows with NA
  data <- na.omit(data)

  # Identify X columns and response
  X_cols <- colnames(data)[grepl("^X", colnames(data))]
  y_col  <- "y"

  # Build formula
  predictor_str <- paste(X_cols, collapse = " + ")
  formula_str   <- paste("y ~", predictor_str)

  n <- nrow(data)

  ## ===== Outer CV for unbiased MSE =====

  set.seed(123)  # reproducible folds per dataset
  fold_ids <- sample(rep(1:outer_k, length.out = n))
  outer_mse <- numeric(outer_k)

  for (k in seq_len(outer_k)) {
    test_idx  <- which(fold_ids == k)
    train_idx <- setdiff(seq_len(n), test_idx)

    data_train <- data[train_idx, , drop = FALSE]
    data_test  <- data[test_idx,  , drop = FALSE]

    # Fit iFORM on training fold only
    iForm_cv <- iForm(
      formula_str,
      data_train,
      heredity     = "strong",
      higher_order = FALSE
    )

    # Predict on test fold
    y_pred_test <- as.numeric(predict(iForm_cv, data_test))
    outer_mse[k] <- mean((data_test$y - y_pred_test)^2)
  }

  mse_outer_mean <- mean(outer_mse)
  mse_outer_sd   <- sd(outer_mse)

  ## ===== Final iFORM fit on full data (for selection) =====

  iForm_fit <- iForm(
    formula_str,
    data,
    heredity     = "strong",
    higher_order = FALSE
  )

  # Extract coefficients
  coef_names <- names(iForm_fit$coefficients)
  coef_vals  <- as.numeric(iForm_fit$coefficients)

  # Main effects: names without ":" (including possible intercept)
  main_mask   <- !grepl(":", coef_names)
  main_names  <- coef_names[main_mask]
  main_coefs  <- coef_vals[main_mask]

  # Remove intercept if present
  intercept_idx <- which(main_names == "(Intercept)")
  if (length(intercept_idx) > 0) {
    main_names <- main_names[-intercept_idx]
    main_coefs <- main_coefs[-intercept_idx]
  }

  # Convert "Xj" -> j-1 (0-based)
  main_indices <- as.integer(gsub("X", "", main_names)) - 1L

  # Interactions: names containing ":"
  inter_mask  <- grepl(":", coef_names)
  inter_names <- coef_names[inter_mask]
  inter_coefs <- coef_vals[inter_mask]

  # Each interaction is "Xi:Xj" -> c(i-1, j-1)
  inter_indices <- lapply(
    strsplit(gsub("X", "", inter_names), ":"),
    function(x) as.integer(x) - 1L
  )

  # Training predictions and MSE on full data
  y_pred_full <- as.numeric(predict(iForm_fit, data))
  mse_train_full <- mean((data$y - y_pred_full)^2)

  ## ===== Build JSON outputs =====

  # Selected mains (0-based)
  selected_mains <- as.integer(main_indices)

  # Selected interactions (0-based pairs)
  selected_interactions <- lapply(inter_indices, function(pair) as.integer(pair))

  # Metrics + F1
  metrics <- list(
    mse_outer_mean = mse_outer_mean,
    mse_outer_sd   = mse_outer_sd,
    mse_train_full = mse_train_full,
    n_total        = length(data$y)
  )

  main_eval  <- evaluate_main_effects(true_main, main_indices)
  inter_eval <- evaluate_interactions(true_interactions, inter_indices)

  metrics$f1_mains               <- main_eval$f1
  metrics$precision_mains        <- main_eval$precision
  metrics$recall_mains           <- main_eval$recall
  metrics$f1_interactions        <- inter_eval$f1
  metrics$precision_interactions <- inter_eval$precision
  metrics$recall_interactions    <- inter_eval$recall

  chosen <- list(
    selected_mains        = selected_mains,
    selected_interactions = selected_interactions,
    selected_hyperparams  = list(
      heredity     = "strong",
      higher_order = FALSE,
      outer_k      = outer_k
    ),
    metrics = metrics
  )

  chosen_json <- file.path(
    out_dir,
    paste0("chosen_continuous_iform_", file_path_sans_ext(filename_base), ".json")
  )

  write(toJSON(chosen, pretty = TRUE, auto_unbox = TRUE), chosen_json)
  cat("Saved:", chosen_json, "\n")
}
