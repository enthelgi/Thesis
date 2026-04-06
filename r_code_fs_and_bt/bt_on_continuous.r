library(LassoBacktracking)
library(jsonlite)
library(Matrix)
library(tools)

# Set your data directory
data_dir <- "C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous"
out_dir  <- data_dir  # or set another output directory

# List all simulated dataset files
dataset_files <- list.files(
  data_dir,
  pattern = "^simulated_dataset_.*\\.csv$",
  full.names = TRUE
)

## ---------- helpers ----------

# Evaluate main and interaction effects
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

  if (is.null(selected_matrix) || ncol(selected_matrix) == 0) {
    selected_keys <- character(0)
  } else {
    selected_keys <- unique(
      apply(selected_matrix, 2, function(x)
        paste(sort(as.integer(x)), collapse = "-"))
    )
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

## ---------- tuning parameters ----------

outer_k          <- 5   # outer CV folds (for unbiased MSE)
inner_nfolds     <- 5   # inner CV folds (for lambda/iter selection)
inner_nperms     <- 1   # permutations for cvLassoBT
nlambda          <- 100
iter_max         <- 70
thresh           <- 1e-8
verbose          <- FALSE

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

  # Read truth
  truth <- fromJSON(truth_path)
  true_main <- as.integer(truth$main_idx) + 1
  true_interactions <- lapply(
    seq_len(nrow(truth$interaction_pairs)),
    function(i) sort(as.integer(truth$interaction_pairs[i, ]) + 1)
  )

  # Read data
  data <- read.csv(filename)
  X <- as.matrix(data[, grep("^X", names(data))])
  y <- data$y
  n <- nrow(X)
  p <- ncol(X)

  # lambda_min_ratio (same logic you used before)
  lambda_min_ratio <- ifelse(n < p, 0.01, 1e-8)

  ## ===== Outer K-fold CV for unbiased performance =====

  set.seed(123)  # reproducible folds per dataset
  fold_ids <- sample(rep(1:outer_k, length.out = n))
  outer_mse <- numeric(outer_k)

  for (k in seq_len(outer_k)) {
    test_idx  <- which(fold_ids == k)
    train_idx <- setdiff(seq_len(n), test_idx)

    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test  <- X[test_idx, , drop = FALSE]
    y_test  <- y[test_idx]

    # Inner CV ONLY on training data
    cv_inner <- cvLassoBT(
      x = X_train,
      y = y_train,
      nlambda = nlambda,
      lambda.min.ratio = lambda_min_ratio,
      nfolds = inner_nfolds,
      nperms = inner_nperms,
      iter_max = iter_max,
      thresh = thresh,
      verbose = verbose,
      mc.cores = 1L
    )

    opt_lambda_idx <- cv_inner$cv_opt[1]
    opt_iter       <- cv_inner$cv_opt[2]
    opt_lambda     <- cv_inner$lambda[opt_lambda_idx]

    # Predict on outer test fold
    preds_test <- predict(
      cv_inner$BT_fit,
      newx = X_test,
      s    = opt_lambda,
      iter = opt_iter,
      type = "response"
    )

    outer_mse[k] <- mean((y_test - preds_test)^2)
  }

  mse_outer_mean <- mean(outer_mse)
  mse_outer_sd   <- sd(outer_mse)

  ## ===== Final model on full data (for variable selection etc.) =====

  cv_full <- cvLassoBT(
    x = X,
    y = y,
    nlambda = nlambda,
    lambda.min.ratio = lambda_min_ratio,
    nfolds = inner_nfolds,
    nperms = inner_nperms,
    iter_max = iter_max,
    thresh = thresh,
    verbose = verbose,
    mc.cores = 1L
  )

  lambda_seq          <- cv_full$lambda
  opt_lambda_idx_full <- cv_full$cv_opt[1]
  opt_iter_full       <- cv_full$cv_opt[2]
  opt_lambda_full     <- lambda_seq[opt_lambda_idx_full]

  bt_full <- cv_full$BT_fit   # "BT" object fit on the full data
  n_main_vars <- bt_full$nvars

  # Coefficients at (lambda*, iter*) using predict(..., type="coefficients")
  coef_vec <- as.numeric(
    predict(
      bt_full,
      newx = X,                        # ignored for type="coefficients"
      s    = opt_lambda_full,
      iter = opt_iter_full,
      type = "coefficients"
    )
  )

  # Split into intercept, main, and interaction coefficients
  intercept <- coef_vec[1]
  beta_main <- coef_vec[2:(n_main_vars + 1)]

  if (length(coef_vec) > (n_main_vars + 1)) {
    beta_interact <- coef_vec[(n_main_vars + 2):length(coef_vec)]
  } else {
    beta_interact <- numeric(0)
  }

  # Selected main effects (1-based indices)
  selected_main <- which(abs(beta_main) >= 1e-4)
  names(selected_main) <- NULL

  # Interaction structure from bt_full
  interactions_matrix <- if (!is.null(bt_full$interactions) &&
                             ncol(bt_full$interactions) > 0) {
    bt_full$interactions  # 0-based indices internally
  } else {
    matrix(nrow = 2, ncol = 0)
  }

  # Which interactions are active?
  if (length(beta_interact) > 0) {
    selected_inter_idx <- which(abs(beta_interact) >= 1e-4)
  } else {
    selected_inter_idx <- integer(0)
  }

  # 1-based interaction indices matrix for evaluation
  selected_interactions_matrix <- if (length(selected_inter_idx) > 0) {
    interactions_matrix[, selected_inter_idx, drop = FALSE] + 1L
  } else {
    matrix(nrow = 2, ncol = 0)
  }

  # Training predictions on full data (mainly for curiosity)
  preds_full <- predict(
    bt_full,
    newx = X,
    s    = opt_lambda_full,
    iter = opt_iter_full,
    type = "response"
  )
  mse_train_full <- mean((y - preds_full)^2)

  ## ===== Build JSON pieces =====

  # Selected mains (0-based for JSON)
  selected_mains <- as.integer(selected_main - 1L)

  # Selected interactions (0-based pairs) for JSON
  selected_interactions <- list()
  if (length(selected_inter_idx) > 0) {
    for (k in seq_along(selected_inter_idx)) {
      pair <- interactions_matrix[, selected_inter_idx[k]]
      selected_interactions[[k]] <- as.integer(pair - 1L)
    }
  }

  # F1 / precision / recall
  main_eval  <- evaluate_main_effects(true_main, selected_main)
  inter_eval <- evaluate_interactions(true_interactions,
                                      selected_interactions_matrix)

  metrics <- list(
    mse_outer_mean          = mse_outer_mean,
    mse_outer_sd            = mse_outer_sd,
    mse_train_full          = mse_train_full,
    n_total                 = length(y),
    f1_mains                = main_eval$f1,
    precision_mains         = main_eval$precision,
    recall_mains            = main_eval$recall,
    f1_interactions         = inter_eval$f1,
    precision_interactions  = inter_eval$precision,
    recall_interactions     = inter_eval$recall
  )

  chosen <- list(
    selected_mains        = selected_mains,
    selected_interactions = selected_interactions,
    selected_hyperparams  = list(
      nlambda         = nlambda,
      iter_max        = iter_max,
      lambda_min_ratio = lambda_min_ratio,
      inner_nfolds     = inner_nfolds,
      inner_nperms     = inner_nperms,
      outer_k          = outer_k
    ),
    metrics = metrics
  )

  # Write JSON
  chosen_json <- file.path(
    out_dir,
    paste0("chosen_continuous_bt_", file_path_sans_ext(filename_base), ".json")
  )
  write(toJSON(chosen, pretty = TRUE, auto_unbox = TRUE), chosen_json)
  cat("Saved:", chosen_json, "\n")
}


