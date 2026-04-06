library(glmnet)
source("Mplasso.R")
library(pROC)

cat("=== Starting TGS mpliable run ===\n")

# --- CONFIG FLAGS ---
USE_1SE_PLIABLE <- TRUE   # toggle 1SE rule for pliable lasso
USE_1SE_LASSO   <- TRUE   # toggle 1SE rule for logistic L1

# --- Helper: 1SE rule index for a lambda path (robust, no nfolds needed) ---
lambda_1se_index <- function(cv_vals) {
  # cv_vals: vector of CV (or deviance) values per lambda
  cv_vals <- as.numeric(cv_vals)

  ok <- is.finite(cv_vals)
  if (!any(ok)) {
    stop("lambda_1se_index: all cv_vals are NA/NaN/Inf")
  }

  cv_use  <- cv_vals[ok]
  idx_map <- which(ok)

  # Use standard deviation as a scale for "1 SE"
  se <- stats::sd(cv_use)
  if (!is.finite(se) || length(se) == 0) {
    se <- 0
  }

  min_idx_local <- which.min(cv_use)
  thr <- cv_use[min_idx_local] + se
  cand_local <- which(cv_use <= thr)

  if (length(cand_local) == 0) {
    chosen_local <- min_idx_local
  } else {
    chosen_local <- max(cand_local)
  }

  # Map back to original index
  idx_map[chosen_local]
}

# --- Helper: stratified K-folds so each fold has both classes ---
make_stratified_folds <- function(y, K) {
  y <- as.factor(y)
  folds <- integer(length(y))

  for (cls in levels(y)) {
    idx_cls <- which(y == cls)
    n_cls <- length(idx_cls)
    # random order within class
    idx_cls <- sample(idx_cls, n_cls)
    # assign approximately equal number from this class to each fold
    fold_seq <- rep(1:K, length.out = n_cls)
    folds[idx_cls] <- fold_seq
  }

  folds
}

# --- Load Data ---
cat("Loading data...\n")
tgs <- read.csv("../data/tgs_data/tgs_dataset_normalized_balanced.csv")
X <- as.matrix(tgs[, setdiff(names(tgs), "MCI")])
dim_X <- dim(X)
cat("X dim:", dim_X[1], "x", dim_X[2], "\n")

y <- as.integer(tgs$MCI)
y <- y + 1 # classes 1/2 for binomial
N <- nrow(X)
p <- ncol(X)
cat("N =", N, " p =", p, "\n")

# --- (Optional) Lasso for Modifier Selection (kept for reference) ---
cat("Running cv.glmnet for modifier selection...\n"); flush.console()
set.seed(123)
lasso_cv <- cv.glmnet(X, y, family = "binomial", alpha = 1)
cat("cv.glmnet done.\n"); flush.console()

lambda_loose <- lasso_cv$lambda.min * 0.7
cat("lambda.min:", lasso_cv$lambda.min, " lambda_loose:", lambda_loose, "\n")

lasso_coef <- coef(lasso_cv, s = lambda_loose)[-1]
selected_mod_idx <- which(abs(lasso_coef) > 1e-8)
cat("Selected modifier indices (", length(selected_mod_idx), "):",
    paste(selected_mod_idx, collapse = ", "), "\n")

# --- Define modifier sets for model comparison ---
# Model A: all modifiers
Z_all <- X
cat("Using Z_all = X; dim(Z_all):", nrow(Z_all), "x", ncol(Z_all), "\n")

# Model B: stable modifiers 4, 8, 13
Z_stable <- X[, c(4, 8, 13), drop = FALSE]
cat("Using Z_stable = X[, c(4, 8, 13)]; dim(Z_stable):",
    nrow(Z_stable), "x", ncol(Z_stable), "\n")

# --- Helper: build full design for logistic L1 with all interactions (Model C) ---
make_full_interaction_matrix <- function(X) {
  p <- ncol(X)
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", seq_len(p))
  }
  int_list <- list()
  int_names <- character()

  idx <- 1
  for (j in 1:(p - 1)) {
    for (k in (j + 1):p) {
      int_list[[idx]] <- X[, j] * X[, k]
      int_names[idx] <- paste0(colnames(X)[j], ":", colnames(X)[k])
      idx <- idx + 1
    }
  }

  if (length(int_list) > 0) {
    X_int <- do.call(cbind, int_list)
    colnames(X_int) <- int_names
    X_ext <- cbind(X, X_int)
  } else {
    X_ext <- X
  }
  X_ext
}

cat("Building full interaction design matrix for logistic L1...\n"); flush.console()
X_lasso_full <- make_full_interaction_matrix(X)
dim_X_lasso <- dim(X_lasso_full)
cat("X_lasso_full dim (main + all pairwise interactions):",
    dim_X_lasso[1], "x", dim_X_lasso[2], "\n")

# --- fit_and_select function (fits pliable lasso on a subset) ---
fit_and_select <- function(sub_idx, X, y, Z,
                           lasso_lambda = NULL,
                           mpliable_lambda = NULL) {
  cat("fit_and_select(): starting with |sub_idx| =",
      length(sub_idx), "\n"); flush.console()

  Xb <- X[sub_idx, , drop = FALSE]
  yb <- y[sub_idx]
  Zb <- Z[sub_idx, , drop = FALSE]
  p <- ncol(Xb)
  p_mod <- ncol(Zb)
  cat("  Subsample dims: Xb =", nrow(Xb), "x", p,
      " Zb =", nrow(Zb), "x", p_mod, "\n"); flush.console()

  # Fit mpliable lasso
  cat("  Calling plasso_fit1()...\n"); flush.console()
  t0 <- Sys.time()
  fit <- plasso_fit1(
    y = yb, X = Xb, Z = Zb, nlambda = 50,
    alpha = .5, new_t = 1, my_mbeta = .09, intercept = 0.01,
    step = .05, number = 10, maxgrid = 50, tol = 1e-3, run = 2,
    lambda_min = .001, for_v = 10, sv = 0, fq = 50, st = 50,
    mv = 20, ms = 50, cv_run = 0
  )
  t1 <- Sys.time()
  cat("  plasso_fit1() finished in",
      round(as.numeric(difftime(t1, t0, units = "secs")), 2),
      "seconds\n"); flush.console()

  # Lambda selection: 1SE rule or min deviance
  lambda_seq <- as.numeric(fit$Lambdas)
  dev_vec <- as.numeric(fit$path$DEV)

  if (is.null(mpliable_lambda)) {
    if (USE_1SE_PLIABLE) {
      lambda_idx <- lambda_1se_index(dev_vec)
    } else {
      lambda_idx <- which.min(dev_vec)
    }
  } else {
    # if a specific lambda is provided, use the closest
    lambda_idx <- which.min(abs(lambda_seq - mpliable_lambda))
  }

  if (!is.finite(lambda_idx) || lambda_idx < 1 || lambda_idx > length(lambda_seq)) {
    stop("Invalid lambda_idx selected in fit_and_select: ", lambda_idx)
  }

  cat("  Selected lambda_idx:", lambda_idx,
      " lambda:", lambda_seq[lambda_idx],
      " DEV:", dev_vec[lambda_idx], "\n"); flush.console()

  # Extract coefficients at selected lambda
  beta <- fit$beta[[2]][, lambda_idx]
  theta <- fit$theta[[2]][, , lambda_idx]

  cat("  beta nonzero:", sum(abs(beta) > 1e-8),
      " /", length(beta), "\n")
  cat("  theta nonzero:",
      sum(abs(theta) > 1e-8),
      " /", length(theta), "\n"); flush.console()

  # Main effects: nonzero beta
  mains_sel <- abs(beta) > 1e-8
  # Interactions: nonzero theta (p x p_mod logical matrix)
  inter_sel <- abs(theta) > 1e-8

  cat("fit_and_select(): done.\n"); flush.console()

  return(list(
    mains_sel = mains_sel,
    inter_sel = inter_sel,
    beta = beta,          # at selected lambda
    theta = theta,        # at selected lambda
    dev_vec = dev_vec     # full path deviance
  ))
}

# --- Helper: compute metrics for pliable models ---
compute_metrics <- function(X, Z, y, beta, theta, idx) {
  Xb <- X[idx, , drop = FALSE]
  Zb <- Z[idx, , drop = FALSE]
  yb <- y[idx]

  # Linear predictor: main + interaction
  eta <- as.numeric(Xb %*% beta)
  if (!is.null(theta)) {
    eta <- eta + rowSums((Xb %*% theta) * Zb)
  }

  prob <- 1 / (1 + exp(-eta))
  y_pred <- ifelse(prob > 0.5, 2, 1)
  acc <- mean(y_pred == yb)

  # AUC (handle case with only one class)
  if (length(unique(yb)) < 2) {
    auc_score <- NA_real_
  } else {
    roc_obj <- roc(yb, prob)
    auc_score <- as.numeric(auc(roc_obj))
  }

  list(acc = acc, auc = auc_score)
}

# --- Repeated K-fold CV for pliable models (Models A & B) ---
run_repeated_cv_pliable <- function(X, y, Z, model_name,
                                    K = 5, n_repeats = 3) {
  N <- nrow(X)
  results <- data.frame(
    rep_id = integer(0),
    fold = integer(0),
    model = character(0),
    train_acc = numeric(0),
    train_auc = numeric(0),
    test_acc = numeric(0),
    test_auc = numeric(0),
    stringsAsFactors = FALSE
  )

  for (r in seq_len(n_repeats)) {
    cat("\n=== ", model_name, ": Repeat", r, "of", n_repeats, "===\n")
    set.seed(1000 + r)  # different splits per repeat

    # Stratified K-fold assignment
    fold_ids <- make_stratified_folds(y, K)

    for (k in 1:K) {
      cat("\n--", model_name, ": Repeat", r, " Fold", k, "of", K, "--\n")
      test_idx <- which(fold_ids == k)
      train_idx <- which(fold_ids != k)

      # Fit on training data
      sel <- fit_and_select(train_idx, X, y, Z)

      beta <- sel$beta
      theta <- sel$theta

      # Metrics on training set
      train_metrics <- compute_metrics(X, Z, y, beta, theta, train_idx)
      # Metrics on test set
      test_metrics  <- compute_metrics(X, Z, y, beta, theta, test_idx)

      results <- rbind(
        results,
        data.frame(
          rep_id = r,
          fold = k,
          model = model_name,
          train_acc = train_metrics$acc,
          train_auc = train_metrics$auc,
          test_acc = test_metrics$acc,
          test_auc = test_metrics$auc,
          stringsAsFactors = FALSE
        )
      )

      cat(sprintf("%s (repeat %d, fold %d): train_acc = %.3f, train_auc = %.3f, test_acc = %.3f, test_auc = %.3f\n",
                  model_name, r, k,
                  train_metrics$acc, train_metrics$auc,
                  test_metrics$acc, test_metrics$auc))
      flush.console()
    }
  }

  return(results)
}

# --- Repeated K-fold CV for logistic L1 with main + all interactions (Model C) ---
run_repeated_cv_lasso_full <- function(X_ext, y,
                                       model_name = "logistic_L1_main_int",
                                       K = 5, n_repeats = 3) {
  N <- nrow(X_ext)
  results <- data.frame(
    rep_id = integer(0),
    fold = integer(0),
    model = character(0),
    train_acc = numeric(0),
    train_auc = numeric(0),
    test_acc = numeric(0),
    test_auc = numeric(0),
    stringsAsFactors = FALSE
  )

  for (r in seq_len(n_repeats)) {
    cat("\n=== ", model_name, ": Repeat", r, "of", n_repeats, "===\n")
    set.seed(2000 + r)  # different splits per repeat

    # Stratified K-fold assignment
    fold_ids <- make_stratified_folds(y, K)

    for (k in 1:K) {
      cat("\n--", model_name, ": Repeat", r, " Fold", k, "of", K, "--\n")
      test_idx <- which(fold_ids == k)
      train_idx <- which(fold_ids != k)

      X_train <- X_ext[train_idx, , drop = FALSE]
      X_test  <- X_ext[test_idx, , drop = FALSE]
      y_train <- y[train_idx]
      y_test  <- y[test_idx]

      # glmnet expects 0/1 or factor for binomial
      y_train01 <- y_train - 1  # y in {1,2} -> {0,1}

      # Fit logistic L1 with CV on training data only
      cvfit <- cv.glmnet(
        X_train, y_train01,
        family = "binomial",
        alpha = 1
      )

      # Lambda selection: 1SE rule or lambda.min
      if (USE_1SE_LASSO) {
        cv_cvm <- cvfit$cvm
        Lambda <- cvfit$lambda
        lambda_idx <- lambda_1se_index(cv_cvm)
        if (!is.finite(lambda_idx) || lambda_idx < 1 || lambda_idx > length(Lambda)) {
          stop("Invalid lambda_idx selected in logistic CV: ", lambda_idx)
        }
        lambda_opt <- Lambda[lambda_idx]
      } else {
        lambda_opt <- cvfit$lambda.min
      }

      cat("  glmnet lambda used:", lambda_opt, "\n")

      # Probabilities: P(Y=1 in {0,1} coding) => P(y == 2 in original coding)
      prob_train <- as.numeric(predict(cvfit, newx = X_train,
                                       s = lambda_opt, type = "response"))
      prob_test  <- as.numeric(predict(cvfit, newx = X_test,
                                       s = lambda_opt, type = "response"))

      # Predicted labels in {1,2}, using 0.5 threshold on prob of class '2'
      y_pred_train <- ifelse(prob_train > 0.5, 2, 1)
      y_pred_test  <- ifelse(prob_test > 0.5, 2, 1)

      train_acc <- mean(y_pred_train == y_train)
      test_acc  <- mean(y_pred_test == y_test)

      # AUC (handle degenerate case)
      if (length(unique(y_train)) < 2) {
        train_auc <- NA_real_
      } else {
        roc_train <- roc(y_train, prob_train)
        train_auc <- as.numeric(auc(roc_train))
      }

      if (length(unique(y_test)) < 2) {
        test_auc <- NA_real_
      } else {
        roc_test <- roc(y_test, prob_test)
        test_auc <- as.numeric(auc(roc_test))
      }

      results <- rbind(
        results,
        data.frame(
          rep_id = r,
          fold = k,
          model = model_name,
          train_acc = train_acc,
          train_auc = train_auc,
          test_acc = test_acc,
          test_auc = test_auc,
          stringsAsFactors = FALSE
        )
      )

      cat(sprintf("%s (repeat %d, fold %d): train_acc = %.3f, train_auc = %.3f, test_acc = %.3f, test_auc = %.3f\n",
                  model_name, r, k,
                  train_acc, train_auc,
                  test_acc, test_auc))
      flush.console()
    }
  }

  return(results)
}

# --- Run comparison: Model A, Model B, and Model C ---
K <- 5
n_repeats <- 5  # smaller for speed

cat("\n=== Running repeated K-fold CV for Model A: all modifiers (pliable) ===\n")
results_all <- run_repeated_cv_pliable(X, y, Z_all,
                                       model_name = "pliable_all_modifiers",
                                       K = K, n_repeats = n_repeats)

cat("\n=== Running repeated K-fold CV for Model B: stable modifiers 4,8,13 (pliable) ===\n")
results_stable <- run_repeated_cv_pliable(X, y, Z_stable,
                                          model_name = "pliable_stable_4_8_13",
                                          K = K, n_repeats = n_repeats)

cat("\n=== Running repeated K-fold CV for Model C: logistic L1 with main + all interactions ===\n")
results_lasso_full <- run_repeated_cv_lasso_full(X_lasso_full, y,
                                                 model_name = "logistic_L1_main_int",
                                                 K = K, n_repeats = n_repeats)

# Combine all results
all_results <- rbind(results_all, results_stable, results_lasso_full)

# --- Summarise performance across repeats/folds (out-of-sample) ---
summary_list <- lapply(split(all_results, all_results$model), function(df) {
  data.frame(
    model = unique(df$model),
    test_acc_mean = mean(df$test_acc, na.rm = TRUE),
    test_acc_sd   = sd(df$test_acc, na.rm = TRUE),
    test_auc_mean = mean(df$test_auc, na.rm = TRUE),
    test_auc_sd   = sd(df$test_auc, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
})

summary_table <- do.call(rbind, summary_list)

cat("\n=== Cross-validated test performance summary ===\n")
print(summary_table)

cat("\n=== Done ===\n"); flush.console()
