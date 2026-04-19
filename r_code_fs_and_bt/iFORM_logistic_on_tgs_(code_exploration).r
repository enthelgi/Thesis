library(pROC)
library(PRROC)
library(ggplot2)

#source("iFORM_logistic.r")
source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM_logistic_(code_exploration).r")
# --- Helper to load and split data (unchanged) ---
load_and_split <- function(csv_path, test_frac = 0.3, seed = 42) {
  dat <- read.csv(csv_path, stringsAsFactors = FALSE)
  y <- as.numeric(dat$MCI)
  X <- as.matrix(dat[, setdiff(colnames(dat), "MCI")])
  set.seed(seed)
  idx <- sample(seq_len(nrow(X)))
  n_test <- round(test_frac * nrow(X))
  test_idx <- idx[1:n_test]
  train_idx <- idx[-(1:n_test)]
  list(
    X = X[train_idx, , drop = FALSE],
    y = y[train_idx],
    X_test = X[test_idx, , drop = FALSE],
    y_test = y[test_idx]
  )
}

make_stratified_folds <- function(y, k) {
  y <- as.factor(y)
  folds <- integer(length(y))
  for (cls in levels(y)) {
    idx <- which(y == cls)
    # assign class indices to folds as evenly as possible
    folds[idx] <- sample(rep(1:k, length.out = length(idx)))
  }
  folds
}

# --- Metric helper to avoid repeating code ---
compute_metrics <- function(y_true, pred_probs) {
  # Numerical stability for log
  pred_probs <- pmin(pmax(pred_probs, 1e-15), 1 - 1e-15)
  
  dev <- -2 * sum(
    y_true * log(pred_probs) +
      (1 - y_true) * log(1 - pred_probs)
  )
  
  roc_obj <- roc(y_true, pred_probs, quiet = TRUE)
  auc_score <- as.numeric(auc(roc_obj))
  
  pr_obj <- pr.curve(
    scores.class0 = pred_probs[y_true == 1],
    scores.class1 = pred_probs[y_true == 0],
    curve = FALSE
  )
  pr_auc <- pr_obj$auc.integral
  
  list(
    dev = dev,
    auc = auc_score,
    pr_auc = pr_auc,
    roc_obj = roc_obj,
    pr_obj = pr_obj
  )
}

# --- Paths to datasets ---
imbal_path <- "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv"
bal_path   <- "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv"

datasets <- list(
  imbalanced = load_and_split(imbal_path),
  balanced   = load_and_split(bal_path)
)

# --- Nested CV parameters ---
k_outer <- 5
k_inner <- 1          # set this to 1 to skip inner CV
set.seed(42)          # for reproducible folds

if (k_inner < 1) {
  stop("k_inner must be at least 1.")
}

# --- Fit and evaluate iFORM_logistic for each dataset ---
for (tag in names(datasets)) {
  cat("\n==============================\n")
  cat("==== Dataset:", tag, "====\n")
  cat("==============================\n")
  
  dat <- datasets[[tag]]
  
  ## ------------------------------------------------------------------
  ## 1) ORIGINAL TRAIN/TEST EVALUATION (single split)
  ## ------------------------------------------------------------------
  df_train <- as.data.frame(dat$X)
  df_train$y <- dat$y
  df_test <- as.data.frame(dat$X_test)
  df_test$y <- dat$y_test
  
  # Fit iFORM_logistic on all features (main effects only)
  formula_str <- paste("y ~ 0 +", paste(colnames(dat$X), collapse = "+"))
  formula <- as.formula(formula_str)
  model <- iForm_logistic(formula, data = df_train,
                          heredity = "strong", higher_order = FALSE)
  
  # Predict on test set
  pred_probs <- predict(model, newdata = df_test, type = "response")
  
  # Compute metrics on this single test split
  m_test <- compute_metrics(dat$y_test, pred_probs)
  
  cat(sprintf("Single-split Test ROC AUC: %.4f, PR AUC: %.4f, Deviance: %.4f\n",
              m_test$auc, m_test$pr_auc, m_test$dev))
  
  # Plots for this single split
  windows()
  par(mfrow = c(1, 2))
  # ROC
  plot(1 - m_test$roc_obj$specificities, m_test$roc_obj$sensitivities,
       type = "l", col = "blue",
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       main = paste("ROC\nAUC =", round(m_test$auc, 3)))
  abline(0, 1, lty = 3)
  # PR
  pr_curve <- pr.curve(
    scores.class0 = pred_probs[dat$y_test == 1],
    scores.class1 = pred_probs[dat$y_test == 0],
    curve = TRUE
  )
  curve_mat <- as.matrix(pr_curve$curve)
  plot(curve_mat[, 1], curve_mat[, 2],
       type = "l", col = "blue",
       xlab = "Recall", ylab = "Precision",
       main = paste("PR\nAUC =", round(pr_curve$auc.integral, 3)))
  
  ## ------------------------------------------------------------------
  ## 2) NESTED CROSS-VALIDATION FOR GENERALIZATION PERFORMANCE
  ## ------------------------------------------------------------------
  cat("\n--- Nested cross-validation ---\n")
  
  # Reconstruct full dataset (train + test) for nested CV
  X_all <- rbind(dat$X, dat$X_test)
  y_all <- c(dat$y, dat$y_test)
  n_all <- length(y_all)
  
  # Create outer folds (stratified)
  outer_fold_ids <- make_stratified_folds(y_all, k_outer)
  
  outer_aucs    <- numeric(k_outer)
  outer_pr_aucs <- numeric(k_outer)
  outer_devs    <- numeric(k_outer)
  
  for (k in 1:k_outer) {
    cat(sprintf("\nOuter fold %d/%d\n", k, k_outer))
    
    # Split into outer train/test
    test_idx_outer  <- which(outer_fold_ids == k)
    train_idx_outer <- setdiff(seq_len(n_all), test_idx_outer)
    
    X_outer_train <- X_all[train_idx_outer, , drop = FALSE]
    y_outer_train <- y_all[train_idx_outer]
    X_outer_test  <- X_all[test_idx_outer, , drop = FALSE]
    y_outer_test  <- y_all[test_idx_outer]
    
    ## -------------------------
    ## Inner CV (on outer train)
    ## -------------------------
    if (k_inner > 1) {
      n_outer_train <- length(y_outer_train)
      inner_fold_ids <- make_stratified_folds(y_outer_train, k_inner)
      
      inner_aucs    <- numeric(k_inner)
      inner_pr_aucs <- numeric(k_inner)
      inner_devs    <- numeric(k_inner)
      
      for (j in 1:k_inner) {
        val_idx_inner   <- which(inner_fold_ids == j)
        train_idx_inner <- setdiff(seq_len(n_outer_train), val_idx_inner)
        
        X_inner_train <- X_outer_train[train_idx_inner, , drop = FALSE]
        y_inner_train <- y_outer_train[train_idx_inner]
        X_inner_val   <- X_outer_train[val_idx_inner, , drop = FALSE]
        y_inner_val   <- y_outer_train[val_idx_inner]
        
        df_inner_train <- as.data.frame(X_inner_train)
        df_inner_train$y <- y_inner_train
        df_inner_val <- as.data.frame(X_inner_val)
        df_inner_val$y <- y_inner_val
        
        formula_str_inner <- paste("y ~ 0 +", paste(colnames(X_inner_train), collapse = "+"))
        formula_inner <- as.formula(formula_str_inner)
        
        model_inner <- iForm_logistic(formula_inner, data = df_inner_train,
                                      heredity = "strong", higher_order = FALSE)
        pred_probs_inner <- predict(model_inner, newdata = df_inner_val, type = "response")
        
        m_inner <- compute_metrics(y_inner_val, pred_probs_inner)
        
        inner_aucs[j]    <- m_inner$auc
        inner_pr_aucs[j] <- m_inner$pr_auc
        inner_devs[j]    <- m_inner$dev
      }
      
      cat(sprintf("  Inner CV mean ROC AUC: %.4f (sd=%.4f)\n",
                  mean(inner_aucs), sd(inner_aucs)))
      cat(sprintf("  Inner CV mean PR AUC:  %.4f (sd=%.4f)\n",
                  mean(inner_pr_aucs), sd(inner_pr_aucs)))
      cat(sprintf("  Inner CV mean Dev:      %.4f (sd=%.4f)\n",
                  mean(inner_devs), sd(inner_devs)))
      
    } else {
      cat("  (Skipping inner CV because k_inner = 1)\n")
    }
    
    ## -------------------------------------------------
    ## Refit on full outer training set and test outside
    ## -------------------------------------------------
    df_outer_train <- as.data.frame(X_outer_train)
    df_outer_train$y <- y_outer_train
    df_outer_test <- as.data.frame(X_outer_test)
    df_outer_test$y <- y_outer_test
    
    formula_str_outer <- paste("y ~ 0 +", paste(colnames(X_outer_train), collapse = "+"))
    formula_outer <- as.formula(formula_str_outer)
    
    model_outer <- iForm_logistic(formula_outer, data = df_outer_train,
                                  heredity = "strong", higher_order = FALSE)
    
    pred_probs_outer <- predict(model_outer, newdata = df_outer_test, type = "response")
    m_outer <- compute_metrics(y_outer_test, pred_probs_outer)
    
    outer_aucs[k]    <- m_outer$auc
    outer_pr_aucs[k] <- m_outer$pr_auc
    outer_devs[k]    <- m_outer$dev
    
    cat(sprintf("  Outer test ROC AUC: %.4f, PR AUC: %.4f, Dev: %.4f\n",
                m_outer$auc, m_outer$pr_auc, m_outer$dev))
  }
  
  ## Summary of outer (test) performance: nested-CV estimate
  cat("\nNested CV (outer folds) summary for", tag, ":\n")
  cat(sprintf("  ROC AUC: mean = %.4f, sd = %.4f\n",
              mean(outer_aucs), sd(outer_aucs)))
  cat(sprintf("  PR AUC:  mean = %.4f, sd = %.4f\n",
              mean(outer_pr_aucs), sd(outer_pr_aucs)))
  cat(sprintf("  Dev:     mean = %.4f, sd = %.4f\n",
              mean(outer_devs), sd(outer_devs)))
}
