############################################################
# mpl (pliable lasso) + L1 + iFORM logistic with
# stability-selection files
#
# UPDATED (as requested):
#   - Design matrices (column sets) are defined ONCE per dataset
#     and then reused across all CV splits (splits pick rows only).
#
#   - mpl_fixed: X = stable mains, Z = fixed modifiers (4,8,13)
#   - mpl_stab : X = stable mains, Z = vars in stable interactions
#                (interaction file i/j are 0-indexed -> +1 into all_mains)
#   - mpl_full : X = full, Z = X
#
# Everything else remains the same.
############################################################

library(glmnet)
library(pROC)
library(PRROC)
library(ggplot2)

# ---- iFORM implementation ----
source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM_logistic.r")

# ---- mplasso implementation ----
setwd("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt")
source("Mplasso.R")
source("cv_mplasso.R")  # not strictly needed, but harmless

# --- Global parameters ---

tau            <- 0.6   # stability threshold
n_lambda       <- 100  # glmnet nlambda
n_folds_outer  <- 5     # outer CV folds
n_permutations <- 1     # outer CV repetitions
n_folds_inner  <- 5     # inner CV folds
base_seed_cv   <- 123

# mplasso hyper-parameters (mirroring example script)
mpl_nlambda    <- 100
mpl_alpha      <- 0.5
mpl_for_v      <- 10
mpl_sv         <- 0
mpl_fq         <- 100
mpl_st         <- 100
mpl_mv         <- 20
mpl_ms         <- 100
mpl_tol        <- 1e-4
mpl_lambda_min <- 0.001
mpl_step       <- 0.05
mpl_number     <- 100
mpl_maxgrid    <- 100

############################################################
# --- Read stability selection results ---
############################################################

read_stable_features <- function(mains_csv, inter_csv, tau) {
  mains <- read.csv(
    mains_csv,
    row.names = 1,
    stringsAsFactors = FALSE
  )
  mains$freq <- as.numeric(mains$freq)
  stable_mains <- rownames(mains)[mains$freq > tau]

  inter <- read.csv(
    inter_csv,
    stringsAsFactors = FALSE
  )
  if (!("freq" %in% names(inter))) {
    stop("Interaction CSV must have a 'freq' column.")
  }
  inter$freq <- as.numeric(inter$freq)
  stable_inter <- subset(inter, freq > tau)

  list(
    mains      = stable_mains,
    inter      = stable_inter,  # data.frame with i,j,freq
    all_mains  = rownames(mains)
  )
}

############################################################
# Stratified splitting helpers
############################################################

make_stratified_folds <- function(y, k) {
  y <- as.factor(y)
  folds <- vector("list", k)
  for (cls in levels(y)) {
    idx_cls <- which(y == cls)
    idx_cls <- sample(idx_cls)
    fold_ids_cls <- cut(seq_along(idx_cls), breaks = k, labels = FALSE)
    for (fold in seq_len(k)) {
      folds[[fold]] <- c(folds[[fold]], idx_cls[fold_ids_cls == fold])
    }
  }
  folds
}

make_stratified_foldid <- function(y, k) {
  folds <- make_stratified_folds(y, k)
  foldid <- integer(length(y))
  for (i in seq_along(folds)) {
    foldid[folds[[i]]] <- i
  }
  foldid
}

############################################################
# Helper: inner CV for mplasso (binary y in {0,1}) with deviance as inner layer metric
############################################################

mplasso_cv <- function(X, Z, y, foldid, tag = "") {
  # plasso_fit1 expects N, p, and K defined in its environment (global in their code)
  assign("N", nrow(X), envir = .GlobalEnv)
  assign("p", ncol(X), envir = .GlobalEnv)
  assign("K", 2L,       envir = .GlobalEnv)  # binary case: classes = {1,2}

  # map y in {0,1} to {1,2} as required by mplasso
  y_pl <- ifelse(y == 1, 2L, 1L)
  if (length(unique(y_pl)) < 2L) stop("mplasso_cv: y must have at least two classes.")

  cat("\n[", tag, "] mplasso_cv: fitting full path on training data...\n")

  # full-data path to define a common Lambda grid
  result_full <- plasso_fit1(
    y = y_pl, X = X, Z = Z, nlambda = mpl_nlambda,
    alpha = mpl_alpha, new_t = 1, my_mbeta = .09, intercept = 0.01,
    step = mpl_step, number = mpl_number, maxgrid = mpl_maxgrid,
    tol = mpl_tol, run = 2, lambda_min = mpl_lambda_min,
    for_v = mpl_for_v, sv = mpl_sv, fq = mpl_fq, st = mpl_st,
    mv = mpl_mv, ms = mpl_ms, cv_run = 0
  )

  Lambda <- as.numeric(result_full$Lambdas)

  cv_cvm <- numeric(length(Lambda))
  cv_cnt <- numeric(length(Lambda))

  folds <- sort(unique(foldid))

  for (ii in folds) {
    tr <- foldid != ii
    te <- foldid == ii

    if (sum(tr) < 2 || sum(te) < 1) next

    cat("[", tag, "]  inner fold", ii, ": fitting plasso_fit1...\n")

    # update N, p, K for this fold as well
    assign("N", sum(tr),  envir = .GlobalEnv)
    assign("p", ncol(X),  envir = .GlobalEnv)
    assign("K", 2L,       envir = .GlobalEnv)

    fit_i <- plasso_fit1(
      y = y_pl[tr],
      X = X[tr, , drop = FALSE],
      Z = Z[tr, , drop = FALSE],
      nlambda = mpl_nlambda,
      alpha = mpl_alpha, new_t = 1, my_mbeta = .09, intercept = 0.01,
      step = mpl_step, number = mpl_number, maxgrid = mpl_maxgrid,
      tol = mpl_tol, run = 2, lambda_min = mpl_lambda_min,
      for_v = mpl_for_v, sv = mpl_sv, fq = mpl_fq, st = mpl_st,
      mv = mpl_mv, ms = mpl_ms, cv_run = 0
    )

    lam_i   <- as.numeric(fit_i$Lambdas)
    idx_map <- sapply(Lambda, function(L) which.min(abs(lam_i - L)))

    for (jj in seq_along(Lambda)) {
      k <- idx_map[jj]
      pr <- predict_lasso(
        fit_i,
        X = X[te, , drop = FALSE],
        Z = Z[te, , drop = FALSE],
        y = y_pl[te],
        lambda = lam_i[k]
      )
      cv_cvm[jj] <- cv_cvm[jj] + pr$deviance
      cv_cnt[jj] <- cv_cnt[jj] + length(y_pl[te])
    }
  }

  cv_cvm <- (2 * cv_cvm) / cv_cnt  # average deviance per obs × 2
  lambda_min <- Lambda[which.min(cv_cvm)]

  cat("[", tag, "] mplasso_cv: chosen lambda_min =", signif(lambda_min, 4), "\n")

  list(
    lambda_min = lambda_min,
    Lambda     = Lambda,
    cv_cvm     = cv_cvm,
    fit_full   = result_full
  )
}

############################################################
# NEW helper: precompute MPL column sets ONCE (per dataset)
############################################################

precompute_mpl_cols <- function(coln, stable_mpl, tau, fixed_mod_idx = c(4, 8, 13)) {
  p <- length(coln)

  # X mains for mpl_fixed and mpl_stab: stable mains
  cols_X_mains <- integer(0)
  if (!is.null(stable_mpl$mains) && length(stable_mpl$mains) > 0) {
    cols_X_mains <- match(stable_mpl$mains, coln)
    cols_X_mains <- cols_X_mains[!is.na(cols_X_mains)]
  }

  # Z fixed for mpl_fixed: indices 4/8/13 in all_mains space
  idx_fixed <- intersect(fixed_mod_idx, seq_along(stable_mpl$all_mains))
  nm_fixed  <- stable_mpl$all_mains[idx_fixed]
  cols_Z_fixed <- match(nm_fixed, coln)
  cols_Z_fixed <- cols_Z_fixed[!is.na(cols_Z_fixed)]

  # Z stab for mpl_stab: vars involved in stable interactions (i/j 0-indexed -> +1 into all_mains)
  cols_Z_stab <- integer(0)
  if (!is.null(stable_mpl$inter) && nrow(stable_mpl$inter) > 0) {
    valid <- which(
      stable_mpl$inter$freq >= tau &
        stable_mpl$inter$i >= 0 & stable_mpl$inter$j >= 0 &
        stable_mpl$inter$i < length(stable_mpl$all_mains) &
        stable_mpl$inter$j < length(stable_mpl$all_mains)
    )
    if (length(valid) > 0) {
      nm_i <- stable_mpl$all_mains[stable_mpl$inter$i[valid] + 1]
      nm_j <- stable_mpl$all_mains[stable_mpl$inter$j[valid] + 1]
      nm_stab <- unique(c(nm_i, nm_j))
      cols_Z_stab <- match(nm_stab, coln)
      cols_Z_stab <- cols_Z_stab[!is.na(cols_Z_stab)]
    }
  }

  list(
    cols_X_mains = cols_X_mains,
    cols_Z_fixed = cols_Z_fixed,
    cols_Z_stab  = cols_Z_stab,
    cols_full    = seq_len(p)
  )
}

############################################################
# Read stability files for each method + dataset
############################################################

base_path <- "c:/Users/enthe/Desktop/Thesis/results/tgs_results"

# L1 logistic stability
stable_imbal_l1 <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_l1logistic_imbalanced.csv"),
  file.path(base_path, "stability_interactions_tgs_l1logistic_imbalanced.csv"),
  tau
)
stable_bal_l1 <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_l1logistic_balanced.csv"),
  file.path(base_path, "stability_interactions_tgs_l1logistic_balanced.csv"),
  tau
)

# mpl stability
stable_imbal_mpl <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_imbalanced.csv"),
  file.path(base_path, "stability_interactions_tgs_imbalanced.csv"),
  tau
)
stable_bal_mpl <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_balanced.csv"),
  file.path(base_path, "stability_interactions_tgs_balanced.csv"),
  tau
)

# iFORM stability (loaded but not used directly here)
stable_imbal_iform <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_iform_imbalanced.csv"),
  file.path(base_path, "stability_interactions_tgs_iform_imbalanced.csv"),
  tau
)
stable_bal_iform <- read_stable_features(
  file.path(base_path, "stability_mains_tgs_iform_balanced.csv"),
  file.path(base_path, "stability_interactions_tgs_iform_balanced.csv"),
  tau
)

############################################################
# Load real TGS datasets (full, for CV)
############################################################

load_tgs_full <- function(csv_path) {
  dat <- read.csv(csv_path, stringsAsFactors = FALSE)
  y <- as.numeric(dat$MCI)
  feature_cols <- setdiff(colnames(dat), "MCI")
  X <- as.matrix(dat[, feature_cols])

  list(
    X = X,
    y = y
  )
}

full_imbal <- load_tgs_full(
  "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv"
)
full_bal <- load_tgs_full(
  "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv"
)

cat("\nFull TGS datasets loaded for nested CV:\n")
cat("Imbalanced: ", nrow(full_imbal$X), "samples,",
    ncol(full_imbal$X), "features\n")
cat("Balanced:   ", nrow(full_bal$X), "samples,",
    ncol(full_bal$X), "features\n")

cat("\nBalanced outcome distribution (y):\n")
print(table(full_bal$y))

############################################################
# PRECOMPUTE MPL column sets ONCE per dataset (GLOBAL across splits)
############################################################

mpl_cols_imbal <- precompute_mpl_cols(colnames(full_imbal$X), stable_imbal_mpl, tau)
mpl_cols_bal   <- precompute_mpl_cols(colnames(full_bal$X),   stable_bal_mpl,   tau)

############################################################
# Fit + evaluate for one outer train/test split
############################################################

fit_and_eval_nested <- function(X_train, y_train,
                                X_test,  y_test,
                                stable_l1,
                                stable_mpl,
                                mpl_cols,         
                                tag,
								perm = NA_integer_,   
                                fold = NA_integer_,   
                                n_folds_inner = 5) {

  p    <- ncol(X_train)
  coln <- colnames(X_train)
  pred_list <- list()   # NEW: store outer-test predictions here
  ## ---------- L1 feature sets (unchanged) ----------

  ## L1 STAB: all mains above tau + all interactions above tau

  main_idx_l1_stab  <- integer(0)
  inter_idx_l1_stab <- integer(0)

  ## ----- mains above tau -----
  if (!is.null(stable_l1$mains) && length(stable_l1$mains) > 0) {
    main_idx_l1_stab <- match(stable_l1$mains, coln)
    main_idx_l1_stab <- main_idx_l1_stab[!is.na(main_idx_l1_stab)]
  }

  ## ----- interactions above tau -----
  if (!is.null(stable_l1$inter) && length(stable_l1$inter) > 0) {
    inter_rows <- which(
      stable_l1$inter$freq >= tau &
        stable_l1$inter$i    >= 0 &
        stable_l1$inter$j    >= 0 &
        stable_l1$inter$i    <  length(stable_l1$all_mains) &
        stable_l1$inter$j    <  length(stable_l1$all_mains)
    )

    if (length(inter_rows) > 0) {
      i_0 <- stable_l1$inter$i[inter_rows] # 0-based
      j_0 <- stable_l1$inter$j[inter_rows]

      names_i_l1 <- stable_l1$all_mains[i_0 + 1] # to 1-based
      names_j_l1 <- stable_l1$all_mains[j_0 + 1]

      ## assumes interaction columns in coln are like "A:B"
      inter_names_l1 <- paste(names_i_l1, names_j_l1, sep = ":")

      inter_idx_l1_stab <- match(inter_names_l1, coln)
      inter_idx_l1_stab <- inter_idx_l1_stab[!is.na(inter_idx_l1_stab)]
    }
  }

  ## final: mains + interactions (both above tau), nothing else
  feat_idx_l1_stab <- sort(unique(c(main_idx_l1_stab, inter_idx_l1_stab)))

  ## L1 FULL-X: all features
  feat_idx_l1_full <- seq_len(p)

  ## ---------- Diagnostics: MPL global column sets ----------
  cat("\n[", tag, "] MPL global column sets (fixed across splits)\n", sep = "")
  cat("[", tag, "] mpl_fixed: X=stable mains (n=",
      length(mpl_cols$cols_X_mains), "), Z=fixed (n=",
      length(mpl_cols$cols_Z_fixed), ")\n", sep = "")
  cat("[", tag, "] mpl_stab : X=stable mains (n=",
      length(mpl_cols$cols_X_mains), "), Z=interaction vars (n=",
      length(mpl_cols$cols_Z_stab), ")\n", sep = "")
  cat("[", tag, "] mpl_full : X=full (n=", p, "), Z=X\n", sep = "")

  ## ---------- Design matrices (UPDATED as requested) ----------

  # mpl_fixed: X = stable mains, Z = fixed (4,8,13)
  X_train_mpl_fixed <- if (length(mpl_cols$cols_X_mains) > 0)
    X_train[, mpl_cols$cols_X_mains, drop = FALSE] else NULL
  X_test_mpl_fixed  <- if (!is.null(X_train_mpl_fixed))
    X_test[, mpl_cols$cols_X_mains, drop = FALSE] else NULL

  Z_train_mpl_fixed <- if (length(mpl_cols$cols_Z_fixed) > 0)
    X_train[, mpl_cols$cols_Z_fixed, drop = FALSE] else NULL
  Z_test_mpl_fixed  <- if (!is.null(Z_train_mpl_fixed))
    X_test[, mpl_cols$cols_Z_fixed, drop = FALSE] else NULL

  # mpl_stab: X = stable mains, Z = vars involved in stable interactions (0-indexed i/j -> +1 already handled in mpl_cols)
  X_train_mpl_stab <- if (length(mpl_cols$cols_X_mains) > 0)
    X_train[, mpl_cols$cols_X_mains, drop = FALSE] else NULL
  X_test_mpl_stab  <- if (!is.null(X_train_mpl_stab))
    X_test[, mpl_cols$cols_X_mains, drop = FALSE] else NULL

  Z_train_mpl_stab <- if (length(mpl_cols$cols_Z_stab) > 0)
    X_train[, mpl_cols$cols_Z_stab, drop = FALSE] else NULL
  Z_test_mpl_stab  <- if (!is.null(Z_train_mpl_stab))
    X_test[, mpl_cols$cols_Z_stab, drop = FALSE] else NULL

  # mpl_full: X = full, Z = X
  X_train_mpl_full <- X_train
  X_test_mpl_full  <- X_test
  Z_train_mpl_full <- X_train
  Z_test_mpl_full  <- X_test

  # L1 sets
  X_train_l1_stab <- if (length(feat_idx_l1_stab) >= 2)
    X_train[, feat_idx_l1_stab, drop = FALSE] else NULL
  X_test_l1_stab  <- if (!is.null(X_train_l1_stab))
    X_test[, feat_idx_l1_stab, drop = FALSE] else NULL

  X_train_l1_full <- if (length(feat_idx_l1_full) >= 2)
    X_train[, feat_idx_l1_full, drop = FALSE] else NULL
  X_test_l1_full  <- if (!is.null(X_train_l1_full))
    X_test[, feat_idx_l1_full, drop = FALSE] else NULL

  ## ---------- Dev function (binary) ----------
  dev_fun <- function(p) {
    p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
    -2 * sum(y_test * log(p) + (1 - y_test) * log(1 - p))
  }

  ## ---------- Initialize result containers ----------
  best_auc_mpl_fixed    <- pr_auc_mpl_fixed    <- dev_mpl_fixed    <- NA_real_
  best_lambda_mpl_fixed <- NA_real_

  best_auc_mpl_stab     <- pr_auc_mpl_stab     <- dev_mpl_stab     <- NA_real_
  best_lambda_mpl_stab  <- NA_real_

  best_auc_mpl_full     <- pr_auc_mpl_full     <- dev_mpl_full     <- NA_real_
  best_lambda_mpl_full  <- NA_real_

  best_auc_l1_stab      <- pr_auc_l1_stab      <- dev_l1_stab      <- NA_real_
  best_lambda_l1_stab   <- NA_real_

  best_auc_l1_full      <- pr_auc_l1_full      <- dev_l1_full      <- NA_real_
  best_lambda_l1_full   <- NA_real_

  auc_iform  <- pr_auc_iform <- dev_iform <- NA_real_

  # Stratified inner folds
  inner_foldid <- make_stratified_foldid(y_train, n_folds_inner)

  ## ---------- mpl FIXED (pliable lasso) ----------
  if (!is.null(X_train_mpl_fixed) &&
      !is.null(Z_train_mpl_fixed) &&
      ncol(Z_train_mpl_fixed) > 0 &&
      length(unique(y_train)) > 1) {

    mpl_cv_fixed <- mplasso_cv(
      X = X_train_mpl_fixed,
      Z = Z_train_mpl_fixed,
      y = y_train,
      foldid = inner_foldid,
      tag = paste(tag, "mpl_fixed")
    )

    best_lambda_mpl_fixed <- mpl_cv_fixed$lambda_min

    y_test_pl <- ifelse(y_test == 1, 2L, 1L)

    pr_fixed <- predict_lasso(
      mpl_cv_fixed$fit_full,
      X = X_test_mpl_fixed,
      Z = Z_test_mpl_fixed,
      y = y_test_pl,
      lambda = best_lambda_mpl_fixed
    )

    P_fixed <- as.data.frame(pr_fixed$y_hat)
    if (ncol(P_fixed) >= 2) {
      p_mpl_fixed <- as.numeric(P_fixed[, 2])
    } else {
      p_mpl_fixed <- as.numeric(P_fixed[, 1])
    }

	pred_list[["mpl_fixed"]] <- data.frame(
      perm  = perm,
      fold  = fold,
      Model = "mpl_fixed",
      y     = y_test,
      p     = p_mpl_fixed
    )
    pred_list[["mpl_fixed"]] <- rbind(pred_list[["mpl_fixed"]], pred_list[["mpl_fixed"]])

    if (length(unique(p_mpl_fixed)) > 1 && length(unique(y_test)) > 1) {
      roc_mpl_fixed <- roc(y_test, p_mpl_fixed, quiet = TRUE)
      best_auc_mpl_fixed <- as.numeric(auc(roc_mpl_fixed))
      dev_mpl_fixed      <- dev_fun(p_mpl_fixed)

      pr_mpl_fixed <- pr.curve(
        scores.class0 = p_mpl_fixed[y_test == 1],
        scores.class1 = p_mpl_fixed[y_test == 0],
        curve = TRUE
      )
      pr_auc_mpl_fixed <- pr_mpl_fixed$auc.integral

      cat("\n[", tag, "] mpl (FIXED) nested ROC AUC:",
          round(best_auc_mpl_fixed, 4),
          " (lambda =", signif(best_lambda_mpl_fixed, 4), ")\n")
    } else {
      cat("\n[", tag, "] mpl (FIXED) predictions constant in outer test.\n")
    }
  } else {
    cat("\n[", tag, "] mpl (FIXED): insufficient features/modifiers or class variety.\n")
  }

  ## ---------- mpl STAB (pliable) ----------
  if (!is.null(X_train_mpl_stab) &&
      !is.null(Z_train_mpl_stab) &&
      ncol(Z_train_mpl_stab) > 0 &&
      length(unique(y_train)) > 1) {

    mpl_cv_stab <- mplasso_cv(
      X = X_train_mpl_stab,
      Z = Z_train_mpl_stab,
      y = y_train,
      foldid = inner_foldid,
      tag = paste(tag, "mpl_stab")
    )

    best_lambda_mpl_stab <- mpl_cv_stab$lambda_min
    y_test_pl <- ifelse(y_test == 1, 2L, 1L)

    pr_stab <- predict_lasso(
      mpl_cv_stab$fit_full,
      X = X_test_mpl_stab,
      Z = Z_test_mpl_stab,
      y = y_test_pl,
      lambda = best_lambda_mpl_stab
    )

    P_stab <- as.data.frame(pr_stab$y_hat)
    p_mpl_stab <- if (ncol(P_stab) >= 2) {
      as.numeric(P_stab[, 2])
    } else {
      as.numeric(P_stab[, 1])
    }

    if (length(unique(p_mpl_stab)) > 1 && length(unique(y_test)) > 1) {
      roc_mpl_stab <- roc(y_test, p_mpl_stab, quiet = TRUE)
      best_auc_mpl_stab <- as.numeric(auc(roc_mpl_stab))
      dev_mpl_stab      <- dev_fun(p_mpl_stab)

      pr_mpl_stab <- pr.curve(
        scores.class0 = p_mpl_stab[y_test == 1],
        scores.class1 = p_mpl_stab[y_test == 0],
        curve         = TRUE
      )
      pr_auc_mpl_stab <- pr_mpl_stab$auc.integral

      cat("[", tag, "] mpl (STAB)  nested ROC AUC:",
          round(best_auc_mpl_stab, 4),
          " (lambda =", signif(best_lambda_mpl_stab, 4), ")\n")
    } else {
      cat("[", tag, "] mpl (STAB) predictions constant in outer test.\n")
    }
  } else {
    cat("[", tag, "] mpl (STAB): insufficient features/modifiers or class variety.\n")
  }

  ## ---------- mpl FULL-X (pliable) ----------
  if (!is.null(X_train_mpl_full) &&
      !is.null(Z_train_mpl_full) &&
      ncol(Z_train_mpl_full) > 0 &&
      length(unique(y_train)) > 1) {

    mpl_cv_full <- mplasso_cv(
      X = X_train_mpl_full,
      Z = Z_train_mpl_full,
      y = y_train,
      foldid = inner_foldid,
      tag = paste(tag, "mpl_full")
    )

    best_lambda_mpl_full <- mpl_cv_full$lambda_min
    y_test_pl <- ifelse(y_test == 1, 2L, 1L)

    pr_full <- predict_lasso(
      mpl_cv_full$fit_full,
      X = X_test_mpl_full,
      Z = Z_test_mpl_full,
      y = y_test_pl,
      lambda = best_lambda_mpl_full
    )

    P_full <- as.data.frame(pr_full$y_hat)
    p_mpl_full <- if (ncol(P_full) >= 2) {
      as.numeric(P_full[, 2])
    } else {
      as.numeric(P_full[, 1])
    }

    if (length(unique(p_mpl_full)) > 1 && length(unique(y_test)) > 1) {
      roc_mpl_full <- roc(y_test, p_mpl_full, quiet = TRUE)
      best_auc_mpl_full <- as.numeric(auc(roc_mpl_full))
      dev_mpl_full      <- dev_fun(p_mpl_full)

      pr_mpl_full <- pr.curve(
        scores.class0 = p_mpl_full[y_test == 1],
        scores.class1 = p_mpl_full[y_test == 0],
        curve         = TRUE
      )
      pr_auc_mpl_full <- pr_mpl_full$auc.integral

      cat("[", tag, "] mpl (FULL)  nested ROC AUC:",
          round(best_auc_mpl_full, 4),
          " (lambda =", signif(best_lambda_mpl_full, 4), ")\n")
    } else {
      cat("[", tag, "] mpl (FULL) predictions constant in outer test.\n")
    }
  } else {
    cat("[", tag, "] mpl (FULL): insufficient features/modifiers or class variety.\n")
  }

  ## ---------- L1 STAB (glmnet) ----------
  if (!is.null(X_train_l1_stab) && length(unique(y_train)) > 1) {
    cv_l1_stab <- cv.glmnet(
      X_train_l1_stab, y_train,
      family       = "binomial",
      alpha        = 1,
      nlambda      = n_lambda,
      foldid       = inner_foldid,
      type.measure = "auc"
    )
    best_lambda_l1_stab <- cv_l1_stab$lambda.min

    fit_l1_stab <- glmnet(
      X_train_l1_stab, y_train,
      family = "binomial",
      alpha  = 1,
      lambda = best_lambda_l1_stab
    )

    p_l1_stab <- as.numeric(predict(fit_l1_stab, X_test_l1_stab, type = "response"))
	    pred_list[["L1_stab"]] <- data.frame(
      perm  = perm,
      fold  = fold,
      Model = "L1_stab",
      y     = y_test,
      p     = p_l1_stab
    )

    if (length(unique(p_l1_stab)) > 1 && length(unique(y_test)) > 1) {
      roc_l1_stab <- roc(y_test, p_l1_stab, quiet = TRUE)
      best_auc_l1_stab <- as.numeric(auc(roc_l1_stab))
      dev_l1_stab      <- dev_fun(p_l1_stab)

      pr_l1_stab <- pr.curve(
        scores.class0 = p_l1_stab[y_test == 1],
        scores.class1 = p_l1_stab[y_test == 0],
        curve         = TRUE
      )
      pr_auc_l1_stab <- pr_l1_stab$auc.integral

      cat("[", tag, "] L1  (STAB) nested ROC AUC:",
          round(best_auc_l1_stab, 4),
          " (lambda =", signif(best_lambda_l1_stab, 4), ")\n")
    } else {
      cat("[", tag, "] L1  (STAB) predictions constant in outer test.\n")
    }
  } else {
    cat("[", tag, "] L1  (STAB): not enough class variety or features.\n")
  }

  ## ---------- L1 FULL (glmnet) ----------
  if (!is.null(X_train_l1_full) && length(unique(y_train)) > 1) {
    cv_l1_full <- cv.glmnet(
      X_train_l1_full, y_train,
      family       = "binomial",
      alpha        = 1,
      nlambda      = n_lambda,
      foldid       = inner_foldid,
      type.measure = "auc"
    )
    best_lambda_l1_full <- cv_l1_full$lambda.min

    fit_l1_full <- glmnet(
      X_train_l1_full, y_train,
      family = "binomial",
      alpha  = 1,
      lambda = best_lambda_l1_full
    )

    p_l1_full <- as.numeric(predict(fit_l1_full, X_test_l1_full, type = "response"))

    if (length(unique(p_l1_full)) > 1 && length(unique(y_test)) > 1) {
      roc_l1_full <- roc(y_test, p_l1_full, quiet = TRUE)
      best_auc_l1_full <- as.numeric(auc(roc_l1_full))
      dev_l1_full      <- dev_fun(p_l1_full)

      pr_l1_full <- pr.curve(
        scores.class0 = p_l1_full[y_test == 1],
        scores.class1 = p_l1_full[y_test == 0],
        curve         = TRUE
      )
      pr_auc_l1_full <- pr_l1_full$auc.integral

      cat("[", tag, "] L1  (FULL) nested ROC AUC:",
          round(best_auc_l1_full, 4),
          " (lambda =", signif(best_lambda_l1_full, 4), ")\n")
    } else {
      cat("[", tag, "] L1  (FULL) predictions constant in outer test.\n")
    }
  } else {
    cat("[", tag, "] L1  (FULL): not enough class variety or features.\n")
  }

  ## ---------- iFORM model (outer only) ----------
  if (length(unique(y_train)) > 1) {
    df_outer_train <- as.data.frame(X_train)
    df_outer_train$y <- y_train
    df_outer_test <- as.data.frame(X_test)
    df_outer_test$y <- y_test

    formula_str_outer <- paste("y ~ 0 +", paste(colnames(X_train), collapse = "+"))
    formula_outer <- as.formula(formula_str_outer)

    model_iform <- iForm_logistic(
      formula_outer,
      data         = df_outer_train,
      heredity     = "strong",
      higher_order = FALSE
    )

    p_iform <- as.numeric(predict(model_iform, newdata = df_outer_test, type = "response"))

    if (length(unique(p_iform)) > 1 && length(unique(y_test)) > 1) {
      roc_iform <- roc(y_test, p_iform, quiet = TRUE)
      auc_iform <- as.numeric(auc(roc_iform))
      dev_iform <- dev_fun(p_iform)

      pr_iform <- pr.curve(
        scores.class0 = p_iform[y_test == 1],
        scores.class1 = p_iform[y_test == 0],
        curve         = TRUE
      )
      pr_auc_iform <- pr_iform$auc.integral

      cat("[", tag, "] iFORM outer ROC AUC:",
          round(auc_iform, 4), "\n")
    } else {
      cat("[", tag, "] iFORM: predictions constant in outer test.\n")
    }
  } else {
    cat("[", tag, "] iFORM: not enough class variety in training fold.\n")
  }
  preds <- if (length(pred_list) > 0) do.call(rbind, pred_list) else data.frame()
  ## ---------- Return results ----------
  list(
    roc_auc_mpl_fixed   = best_auc_mpl_fixed,
    pr_auc_mpl_fixed    = pr_auc_mpl_fixed,
    dev_mpl_fixed       = dev_mpl_fixed,
    lambda_mpl_fixed    = best_lambda_mpl_fixed,

    roc_auc_mpl_stab    = best_auc_mpl_stab,
    pr_auc_mpl_stab     = pr_auc_mpl_stab,
    dev_mpl_stab        = dev_mpl_stab,
    lambda_mpl_stab     = best_lambda_mpl_stab,

    roc_auc_mpl_full    = best_auc_mpl_full,
    pr_auc_mpl_full     = pr_auc_mpl_full,
    dev_mpl_full        = dev_mpl_full,
    lambda_mpl_full     = best_lambda_mpl_full,

    roc_auc_l1_stab     = best_auc_l1_stab,
    pr_auc_l1_stab      = pr_auc_l1_stab,
    dev_l1_stab         = dev_l1_stab,
    lambda_l1_stab      = best_lambda_l1_stab,

    roc_auc_l1_full     = best_auc_l1_full,
    pr_auc_l1_full      = pr_auc_l1_full,
    dev_l1_full         = dev_l1_full,
    lambda_l1_full      = best_lambda_l1_full,

    roc_auc_iform       = auc_iform,
    pr_auc_iform        = pr_auc_iform,
    dev_iform           = dev_iform,

	preds = preds # NEW: return outer-test predictions
  )
}

############################################################
# Repeated nested K-fold CV (outer K-fold, permutations)
############################################################

run_nested_cv <- function(X, y,
                          stable_l1,
                          stable_mpl,
                          mpl_cols,         # NEW: global MPL col sets
                          tag,
                          n_folds_outer  = 5,
                          n_permutations = 1,
                          n_folds_inner  = 5,
                          base_seed      = 123) {

  n <- nrow(X)
  
  preds_all <- data.frame()

  results <- data.frame(
    perm                 = integer(),
    fold                 = integer(),

    roc_auc_mpl_fixed    = numeric(),
    pr_auc_mpl_fixed     = numeric(),
    dev_mpl_fixed        = numeric(),
    lambda_mpl_fixed     = numeric(),

    roc_auc_mpl_stab     = numeric(),
    pr_auc_mpl_stab      = numeric(),
    dev_mpl_stab         = numeric(),
    lambda_mpl_stab      = numeric(),

    roc_auc_mpl_full     = numeric(),
    pr_auc_mpl_full      = numeric(),
    dev_mpl_full         = numeric(),
    lambda_mpl_full      = numeric(),

    roc_auc_l1_stab      = numeric(),
    pr_auc_l1_stab       = numeric(),
    dev_l1_stab          = numeric(),
    lambda_l1_stab       = numeric(),

    roc_auc_l1_full      = numeric(),
    pr_auc_l1_full       = numeric(),
    dev_l1_full          = numeric(),
    lambda_l1_full       = numeric(),

    roc_auc_iform        = numeric(),
    pr_auc_iform         = numeric(),
    dev_iform            = numeric(),
    stringsAsFactors     = FALSE
  )

  for (perm in seq_len(n_permutations)) {
    set.seed(base_seed + perm)
    folds <- make_stratified_folds(y, n_folds_outer)

    for (fold in seq_len(n_folds_outer)) {
      test_idx  <- folds[[fold]]
      train_idx <- setdiff(seq_len(n), test_idx)

      res <- fit_and_eval_nested(
        X_train       = X[train_idx, , drop = FALSE],
        y_train       = y[train_idx],
        X_test        = X[test_idx,  , drop = FALSE],
        y_test        = y[test_idx],
        stable_l1     = stable_l1,
        stable_mpl    = stable_mpl,
        mpl_cols      = mpl_cols,   # NEW
        tag           = paste(tag, "perm", perm, "fold", fold),
		perm          = perm,   # NEW
        fold          = fold,   # NEW
        n_folds_inner = n_folds_inner
      )

      if (is.null(res)) {
        this_row <- data.frame(
          perm                 = perm,
          fold                 = fold,
          roc_auc_mpl_fixed    = NA_real_,
          pr_auc_mpl_fixed     = NA_real_,
          dev_mpl_fixed        = NA_real_,
          lambda_mpl_fixed     = NA_real_,
          roc_auc_mpl_stab     = NA_real_,
          pr_auc_mpl_stab      = NA_real_,
          dev_mpl_stab         = NA_real_,
          lambda_mpl_stab      = NA_real_,
          roc_auc_mpl_full     = NA_real_,
          pr_auc_mpl_full      = NA_real_,
          dev_mpl_full         = NA_real_,
          lambda_mpl_full      = NA_real_,
          roc_auc_l1_stab      = NA_real_,
          pr_auc_l1_stab       = NA_real_,
          dev_l1_stab          = NA_real_,
          lambda_l1_stab       = NA_real_,
          roc_auc_l1_full      = NA_real_,
          pr_auc_l1_full       = NA_real_,
          dev_l1_full          = NA_real_,
          lambda_l1_full       = NA_real_,
          roc_auc_iform        = NA_real_,
          pr_auc_iform         = NA_real_,
          dev_iform            = NA_real_
        )
      } else {
        this_row <- data.frame(
          perm                 = perm,
          fold                 = fold,
          roc_auc_mpl_fixed    = res$roc_auc_mpl_fixed,
          pr_auc_mpl_fixed     = res$pr_auc_mpl_fixed,
          dev_mpl_fixed        = res$dev_mpl_fixed,
          lambda_mpl_fixed     = res$lambda_mpl_fixed,

          roc_auc_mpl_stab     = res$roc_auc_mpl_stab,
          pr_auc_mpl_stab      = res$pr_auc_mpl_stab,
          dev_mpl_stab         = res$dev_mpl_stab,
          lambda_mpl_stab      = res$lambda_mpl_stab,

          roc_auc_mpl_full     = res$roc_auc_mpl_full,
          pr_auc_mpl_full      = res$pr_auc_mpl_full,
          dev_mpl_full         = res$dev_mpl_full,
          lambda_mpl_full      = res$lambda_mpl_full,

          roc_auc_l1_stab      = res$roc_auc_l1_stab,
          pr_auc_l1_stab       = res$pr_auc_l1_stab,
          dev_l1_stab          = res$dev_l1_stab,
          lambda_l1_stab       = res$lambda_l1_stab,

          roc_auc_l1_full      = res$roc_auc_l1_full,
          pr_auc_l1_full       = res$pr_auc_l1_full,
          dev_l1_full          = res$dev_l1_full,
          lambda_l1_full       = res$lambda_l1_full,

          roc_auc_iform        = res$roc_auc_iform,
          pr_auc_iform         = res$pr_auc_iform,
          dev_iform            = res$dev_iform
        )
      }

      results <- rbind(results, this_row)
	  if (!is.null(res$preds) && nrow(res$preds) > 0) {
        preds_all <- rbind(preds_all, res$preds)
      }

    }
  }


  summary <- list(
    per_outer_fold            = results,
	preds_outer    = preds_all, 

    mean_roc_auc_mpl_fixed    = mean(results$roc_auc_mpl_fixed, na.rm = TRUE),
    sd_roc_auc_mpl_fixed      = sd(  results$roc_auc_mpl_fixed, na.rm = TRUE),
    mean_pr_auc_mpl_fixed     = mean(results$pr_auc_mpl_fixed,  na.rm = TRUE),
    sd_pr_auc_mpl_fixed       = sd(  results$pr_auc_mpl_fixed,  na.rm = TRUE),

    mean_roc_auc_mpl_stab     = mean(results$roc_auc_mpl_stab,  na.rm = TRUE),
    sd_roc_auc_mpl_stab       = sd(  results$roc_auc_mpl_stab,  na.rm = TRUE),
    mean_pr_auc_mpl_stab      = mean(results$pr_auc_mpl_stab,   na.rm = TRUE),
    sd_pr_auc_mpl_stab        = sd(  results$pr_auc_mpl_stab,   na.rm = TRUE),

    mean_roc_auc_mpl_full     = mean(results$roc_auc_mpl_full,  na.rm = TRUE),
    sd_roc_auc_mpl_full       = sd(  results$roc_auc_mpl_full,  na.rm = TRUE),
    mean_pr_auc_mpl_full      = mean(results$pr_auc_mpl_full,   na.rm = TRUE),
    sd_pr_auc_mpl_full        = sd(  results$pr_auc_mpl_full,   na.rm = TRUE),

    mean_roc_auc_l1_stab      = mean(results$roc_auc_l1_stab,   na.rm = TRUE),
    sd_roc_auc_l1_stab        = sd(  results$roc_auc_l1_stab,   na.rm = TRUE),
    mean_pr_auc_l1_stab       = mean(results$pr_auc_l1_stab,    na.rm = TRUE),
    sd_pr_auc_l1_stab         = sd(  results$pr_auc_l1_stab,    na.rm = TRUE),

    mean_roc_auc_l1_full      = mean(results$roc_auc_l1_full,   na.rm = TRUE),
    sd_roc_auc_l1_full        = sd(  results$roc_auc_l1_full,   na.rm = TRUE),
    mean_pr_auc_l1_full       = mean(results$pr_auc_l1_full,    na.rm = TRUE),
    sd_pr_auc_l1_full         = sd(  results$pr_auc_l1_full,    na.rm = TRUE),

    mean_roc_auc_iform        = mean(results$roc_auc_iform,     na.rm = TRUE),
    sd_roc_auc_iform          = sd(  results$roc_auc_iform,     na.rm = TRUE),
    mean_pr_auc_iform         = mean(results$pr_auc_iform,      na.rm = TRUE),
    sd_pr_auc_iform           = sd(  results$pr_auc_iform,      na.rm = TRUE)
  )

  cat("\n==== Nested CV summary for", tag, "====\n")
  cat("mpl (FIXED)    ROC AUC: mean", round(summary$mean_roc_auc_mpl_fixed, 4),
      "sd", round(summary$sd_roc_auc_mpl_fixed, 4), "\n")
  cat("mpl (FIXED)    PR  AUC: mean", round(summary$mean_pr_auc_mpl_fixed, 4),
      "sd", round(summary$sd_pr_auc_mpl_fixed, 4), "\n")

  cat("mpl (STAB)     ROC AUC: mean", round(summary$mean_roc_auc_mpl_stab, 4),
      "sd", round(summary$sd_roc_auc_mpl_stab, 4), "\n")
  cat("mpl (STAB)     PR  AUC: mean", round(summary$mean_pr_auc_mpl_stab, 4),
      "sd", round(summary$sd_pr_auc_mpl_stab, 4), "\n")

  cat("mpl (FULL)     ROC AUC: mean", round(summary$mean_roc_auc_mpl_full, 4),
      "sd", round(summary$sd_roc_auc_mpl_full, 4), "\n")
  cat("mpl (FULL)     PR  AUC: mean", round(summary$mean_pr_auc_mpl_full, 4),
      "sd", round(summary$sd_pr_auc_mpl_full, 4), "\n")

  cat("L1  (STAB)     ROC AUC: mean", round(summary$mean_roc_auc_l1_stab, 4),
      "sd", round(summary$sd_roc_auc_l1_stab, 4), "\n")
  cat("L1  (STAB)     PR  AUC: mean", round(summary$mean_pr_auc_l1_stab, 4),
      "sd", round(summary$sd_pr_auc_l1_stab, 4), "\n")

  cat("L1  (FULL)     ROC AUC: mean", round(summary$mean_roc_auc_l1_full, 4),
      "sd", round(summary$sd_roc_auc_l1_full, 4), "\n")
  cat("L1  (FULL)     PR  AUC: mean", round(summary$mean_pr_auc_l1_full, 4),
      "sd", round(summary$sd_pr_auc_l1_full, 4), "\n")

  cat("iFORM          ROC AUC: mean", round(summary$mean_roc_auc_iform, 4),
      "sd", round(summary$sd_roc_auc_iform, 4), "\n")
  cat("iFORM          PR  AUC: mean", round(summary$mean_pr_auc_iform, 4),
      "sd", round(summary$sd_pr_auc_iform, 4), "\n")

  summary
}

############################################################
# Run nested CV for both datasets
############################################################

# cv_imbal <- run_nested_cv(
#   full_imbal$X, full_imbal$y,
#   stable_l1      = stable_imbal_l1,
#   stable_mpl     = stable_imbal_mpl,
#   mpl_cols       = mpl_cols_imbal,   # NEW
#   tag            = "imbalanced",
#   n_folds_outer  = n_folds_outer,
#   n_permutations = n_permutations,
#   n_folds_inner  = n_folds_inner,
#   base_seed      = base_seed_cv
# )
cv_imbal <- NULL


cv_bal <- run_nested_cv(
  full_bal$X, full_bal$y,
  stable_l1      = stable_bal_l1,
  stable_mpl     = stable_bal_mpl,
  mpl_cols       = mpl_cols_bal,     # NEW
  tag            = "balanced",
  n_folds_outer  = n_folds_outer,
  n_permutations = n_permutations,
  n_folds_inner  = n_folds_inner,
  base_seed      = base_seed_cv
)
make_calibration_bins <- function(pred_df, n_bins = 10,
                                  models = c("mpl_fixed", "L1_stab"),
                                  breaks = c("equal", "quantile")) {
  breaks <- match.arg(breaks)
  df <- subset(pred_df, Model %in% models)
  df <- df[is.finite(df$p) & !is.na(df$y), ]
  df$p <- pmin(pmax(df$p, 1e-15), 1 - 1e-15)

  # Define bins
  if (breaks == "equal") {
    br <- seq(0, 1, length.out = n_bins + 1)
    df$bin <- cut(df$p, breaks = br, include.lowest = TRUE)
  } else {
    # quantile bins computed per model (safer if distributions differ a lot)
    df$bin <- NA
    for (m in unique(df$Model)) {
      pm <- df$p[df$Model == m]
      br <- quantile(pm, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
      br[1] <- 0; br[length(br)] <- 1
      br <- unique(br)
      if (length(br) < 3) next
      df$bin[df$Model == m] <- cut(pm, breaks = br, include.lowest = TRUE)
    }
  }

  # Summarize
  out <- do.call(rbind, lapply(split(df, df$Model), function(d) {
    ##agg <- aggregate(cbind(p, y) ~ bin, data = d, FUN = mean)
	agg <- aggregate(p ~ bin, data = d, FUN = median)
	agg_y <- aggregate(y ~ bin, data = d, FUN = mean)
	n   <- aggregate(y ~ bin, data = d, FUN = length)
	names(agg) <- c("bin", "median_pred")
	names(agg_y) <- c("bin", "obs_rate")
	agg <- merge(agg, agg_y, by = "bin")
    
    ##names(agg) <- c("bin", "mean_pred", "obs_rate")
    agg$n <- n$y
    agg$Model <- unique(d$Model)

    # 95% normal approx CI for observed rate (optional)
    se <- sqrt(pmax(agg$obs_rate * (1 - agg$obs_rate), 0) / pmax(agg$n, 1))
    agg$lo <- pmax(0, agg$obs_rate - 1.96 * se)
    agg$hi <- pmin(1, agg$obs_rate + 1.96 * se)
    agg
  }))

  out
}

cal_df <- make_calibration_bins(
  cv_bal$preds_outer,
  n_bins = 10,
  models = c("mpl_fixed", "L1_stab"),
  breaks = "quantile"   # try "quantile" too if you prefer
)

print(cal_df[, c("Model", "bin", "n")])

cal_df$Model <- factor(cal_df$Model, levels = c("mpl_fixed", "L1_stab"))

windows()
print(
  ggplot(cal_df, aes(x = median_pred, y = obs_rate, color = Model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_line() +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.01, alpha = 0.6) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = "Calibration (outer-fold predictions): mpl_fixed vs L1_stab",
      x = "Mean predicted probability (bin)",
      y = "Observed event rate (bin)"
    ) +
    theme_bw()
)

cat("\nDone nested CV.\n")

############################################################
# PR AUC, ROC AUC & DEV BOXPLOTS – BALANCED ONLY
############################################################

make_pr_long <- function(res_df, dataset_name) {
  rbind(
    data.frame(Dataset = dataset_name,
               Model   = "mpl_fixed",
               PR_AUC  = res_df$pr_auc_mpl_fixed),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_stab",
               PR_AUC  = res_df$pr_auc_mpl_stab),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_full",
               PR_AUC  = res_df$pr_auc_mpl_full),
    data.frame(Dataset = dataset_name,
               Model   = "L1_stab",
               PR_AUC  = res_df$pr_auc_l1_stab),
    data.frame(Dataset = dataset_name,
               Model   = "L1_full",
               PR_AUC  = res_df$pr_auc_l1_full),
    data.frame(Dataset = dataset_name,
               Model   = "iFORM",
               PR_AUC  = res_df$pr_auc_iform)
  )
}

make_dev_long <- function(res_df, dataset_name) {
  rbind(
    data.frame(Dataset = dataset_name,
               Model   = "mpl_fixed",
               Deviance = res_df$dev_mpl_fixed),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_stab",
               Deviance = res_df$dev_mpl_stab),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_full",
               Deviance = res_df$dev_mpl_full),
    data.frame(Dataset = dataset_name,
               Model   = "L1_stab",
               Deviance = res_df$dev_l1_stab),
    data.frame(Dataset = dataset_name,
               Model   = "L1_full",
               Deviance = res_df$dev_l1_full),
    data.frame(Dataset = dataset_name,
               Model   = "iFORM",
               Deviance = res_df$dev_iform)
  )
}

make_roc_long <- function(res_df, dataset_name) {
  rbind(
    data.frame(Dataset = dataset_name,
               Model   = "mpl_fixed",
               ROC_AUC = res_df$roc_auc_mpl_fixed),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_stab",
               ROC_AUC = res_df$roc_auc_mpl_stab),
    data.frame(Dataset = dataset_name,
               Model   = "mpl_full",
               ROC_AUC = res_df$roc_auc_mpl_full),
    data.frame(Dataset = dataset_name,
               Model   = "L1_stab",
               ROC_AUC = res_df$roc_auc_l1_stab),
    data.frame(Dataset = dataset_name,
               Model   = "L1_full",
               ROC_AUC = res_df$roc_auc_l1_full),
    data.frame(Dataset = dataset_name,
               Model   = "iFORM",
               ROC_AUC = res_df$roc_auc_iform)
  )
}

pr_long_bal   <- make_pr_long(cv_bal$per_outer_fold,   "Balanced")
pr_long <- pr_long_bal[!is.na(pr_long_bal$PR_AUC), ]
pr_long$Model <- factor(
  pr_long$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)

dev_long_bal   <- make_dev_long(cv_bal$per_outer_fold,   "Balanced")
dev_long <- dev_long_bal[!is.na(dev_long_bal$Deviance), ]
dev_long$Model <- factor(
  dev_long$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)

roc_long_bal   <- make_roc_long(cv_bal$per_outer_fold,   "Balanced")
roc_long <- roc_long_bal[!is.na(roc_long_bal$ROC_AUC), ]
roc_long$Model <- factor(
  roc_long$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)

## ---- Reality check: PR_AUC distribution per model ----
cat("\n=== PR_AUC summary per model (Balanced) ===\n")
print(t(sapply(split(pr_long$PR_AUC, pr_long$Model), summary)))

cat("\n=== PR_AUC means used in boxplot vs nested-CV summary ===\n")
model_means_box <- tapply(pr_long$PR_AUC, pr_long$Model, mean)
print(model_means_box)

cat("\nFrom nested-CV summary object:\n")
cat("mpl_fixed mean PR AUC =", cv_bal$mean_pr_auc_mpl_fixed, "\n")
cat("mpl_stab  mean PR AUC =", cv_bal$mean_pr_auc_mpl_stab,  "\n")
cat("mpl_full  mean PR AUC =", cv_bal$mean_pr_auc_mpl_full,  "\n")
cat("L1_stab   mean PR AUC =", cv_bal$mean_pr_auc_l1_stab,   "\n")
cat("L1_full   mean PR AUC =", cv_bal$mean_pr_auc_l1_full,   "\n")
cat("iFORM     mean PR AUC =", cv_bal$mean_pr_auc_iform,     "\n")

## ---- PR AUC boxplot ----
windows()
print(
  ggplot(pr_long, aes(x = Model, y = PR_AUC, fill = Model)) +
    geom_boxplot(alpha = 0.8) +
    stat_summary(
      fun = mean,
      geom = "point",
      shape = 23,
      size  = 3.5,
      fill  = "yellow",
      color = "black"
    ) +
    labs(
      title = "PR AUC across outer folds (Balanced dataset)",
      x = "Model",
      y = "PR AUC"
    ) +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )+
    coord_cartesian(ylim = c(0, 1)) 
)
## ---- ROC AUC boxplot (separate figure) ----
windows()
print(
  ggplot(roc_long, aes(x = Model, y = ROC_AUC, fill = Model)) +
    geom_boxplot(alpha = 0.8) +
    stat_summary(
      fun = mean,
      geom = "point",
      shape = 23,
      size  = 3.5,
      fill  = "yellow",
      color = "black"
    ) +
    labs(
      title = "ROC AUC across outer folds (Balanced dataset)",
      x = "Model",
      y = "ROC AUC"
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )+
    coord_cartesian(ylim = c(0, 1))
)


## ---- Deviance boxplot ----
windows()
print(
  ggplot(dev_long, aes(x = Model, y = Deviance, fill = Model)) +
    geom_boxplot(alpha = 0.8) +
    stat_summary(
      fun = mean,
      geom = "point",
      shape = 23,
      size  = 3.5,
      fill  = "yellow",
      color = "black"
    ) +
    labs(
      title = "Deviance across outer folds (Balanced dataset)",
      x = "Model",
      y = "Deviance (-2 log-likelihood)"
    ) +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
)

############################################################
# Deviance vs log(lambda) – BALANCED (glmnet L1 models only)
############################################################

# get_feature_sets_l1 <- function(X, stable_l1) {
#   coln <- colnames(X)
#   p <- ncol(X)

#   main_idx_l1 <- match(stable_l1$mains, coln)
#   main_idx_l1 <- main_idx_l1[!is.na(main_idx_l1)]
#   mod_names_l1 <- character(0)
#   if (!is.null(stable_l1$inter) && nrow(stable_l1$inter) > 0) {
#     valid_i_l1 <- stable_l1$inter$i[
#       stable_l1$inter$i >= 1 & stable_l1$inter$i <= length(stable_l1$all_mains)
#     ]
#     valid_j_l1 <- stable_l1$inter$j[
#       stable_l1$inter$j >= 1 & stable_l1$inter$j <= length(stable_l1$all_mains)
#     ]
#     names_i_l1 <- stable_l1$all_mains[valid_i_l1]
#     names_j_l1 <- stable_l1$all_mains[valid_j_l1]
#     mod_names_l1 <- unique(c(names_i_l1, names_j_l1))
#   }
#   mod_idx_l1 <- match(mod_names_l1, coln)
#   mod_idx_l1 <- mod_idx_l1[!is.na(mod_idx_l1)]
#   feat_idx_l1 <- sort(unique(c(main_idx_l1, mod_idx_l1)))

#   list(
#     L1_stab = feat_idx_l1,
#     L1_full = seq_len(p)
#   )
# }


get_feature_sets_l1 <- function(X, stable_l1) {
  coln <- colnames(X)
  p <- ncol(X)

  # Mains above tau
  main_idx_l1 <- match(stable_l1$mains, coln)
  main_idx_l1 <- main_idx_l1[!is.na(main_idx_l1)]

  # Interactions above tau (as "A:B" columns)
  inter_idx_l1 <- integer(0)
  if (!is.null(stable_l1$inter) && nrow(stable_l1$inter) > 0) {
    inter_rows <- which(
      stable_l1$inter$freq >= tau &
        stable_l1$inter$i    >= 0 &
        stable_l1$inter$j    >= 0 &
        stable_l1$inter$i    <  length(stable_l1$all_mains) &
        stable_l1$inter$j    <  length(stable_l1$all_mains)
    )
    if (length(inter_rows) > 0) {
      i_0 <- stable_l1$inter$i[inter_rows]
      j_0 <- stable_l1$inter$j[inter_rows]
      names_i_l1 <- stable_l1$all_mains[i_0 + 1]
      names_j_l1 <- stable_l1$all_mains[j_0 + 1]
      inter_names_l1 <- paste(names_i_l1, names_j_l1, sep = ":")
      inter_idx_l1 <- match(inter_names_l1, coln)
      inter_idx_l1 <- inter_idx_l1[!is.na(inter_idx_l1)]
    }
  }

  feat_idx_l1 <- sort(unique(c(main_idx_l1, inter_idx_l1)))

  list(
    L1_stab = feat_idx_l1,
    L1_full = seq_len(p)
  )
}


make_deviance_curves_l1 <- function(X, y, feature_sets, dataset_name) {
  out <- list()
  foldid_dev <- make_stratified_foldid(y, 5)

  idx <- feature_sets$L1_stab
  if (length(idx) >= 2 && length(unique(y)) > 1) {
    cv <- cv.glmnet(
      X[, idx, drop = FALSE], y,
      family      = "binomial",
      alpha       = 1,
      nlambda     = n_lambda,
      type.measure = "deviance",
      foldid      = foldid_dev
    )
    out[["L1_stab"]] <- data.frame(
      Dataset    = dataset_name,
      Model      = "L1_stab",
      log_lambda = log(cv$lambda),
      Deviance   = cv$cvm
    )
  }

  idx <- feature_sets$L1_full
  if (length(idx) >= 2 && length(unique(y)) > 1) {
    cv <- cv.glmnet(
      X[, idx, drop = FALSE], y,
      family      = "binomial",
      alpha       = 1,
      nlambda     = n_lambda,
      type.measure = "deviance",
      foldid      = foldid_dev
    )
    out[["L1_full"]] <- data.frame(
      Dataset    = dataset_name,
      Model      = "L1_full",
      log_lambda = log(cv$lambda),
      Deviance   = cv$cvm
    )
  }

  if (length(out) == 0) return(data.frame())
  do.call(rbind, out)
}

fs_bal_l1        <- get_feature_sets_l1(full_bal$X, stable_bal_l1)
dev_curve_bal_l1 <- make_deviance_curves_l1(full_bal$X, full_bal$y, fs_bal_l1, "Balanced")

dev_curve_bal_l1$Model <- factor(
  dev_curve_bal_l1$Model,
  levels = c("L1_stab", "L1_full")
)

## ============================================================
## Add MPL deviance-vs-lambda curves (UPDATED to global MPL sets)
## ============================================================

X_all <- full_bal$X
y_all <- full_bal$y

foldid_dev_mpl <- make_stratified_foldid(y_all, 5)

dev_curve_bal_mpl <- data.frame()

if (length(unique(y_all)) < 2) {
  cat("MPL dev-curves skipped: y has <2 classes.\n")
} else {

  # mpl_fixed: X = stable mains, Z = fixed
  if (length(mpl_cols_bal$cols_X_mains) == 0 || length(mpl_cols_bal$cols_Z_fixed) == 0) {
    cat("mpl_fixed dev-curve skipped: empty X mains or Z fixed.\n")
  } else {
    X_fixed <- X_all[, mpl_cols_bal$cols_X_mains, drop = FALSE]
    Z_fixed <- X_all[, mpl_cols_bal$cols_Z_fixed, drop = FALSE]

    mpl_cv_fixed <- mplasso_cv(
      X = X_fixed, Z = Z_fixed, y = y_all,
      foldid = foldid_dev_mpl,
      tag = "Balanced mpl_fixed devcurve"
    )
    dev_curve_bal_mpl <- rbind(dev_curve_bal_mpl,
      data.frame(Dataset="Balanced", Model="mpl_fixed",
                 log_lambda=log(mpl_cv_fixed$Lambda), Deviance=mpl_cv_fixed$cv_cvm)
    )
  }

  # mpl_stab: X = stable mains, Z = interaction vars
  if (length(mpl_cols_bal$cols_X_mains) == 0 || length(mpl_cols_bal$cols_Z_stab) == 0) {
    cat("mpl_stab dev-curve skipped: empty X mains or Z stab.\n")
  } else {
    X_stab <- X_all[, mpl_cols_bal$cols_X_mains, drop = FALSE]
    Z_stab <- X_all[, mpl_cols_bal$cols_Z_stab, drop = FALSE]

    mpl_cv_stab <- mplasso_cv(
      X = X_stab, Z = Z_stab, y = y_all,
      foldid = foldid_dev_mpl,
      tag = "Balanced mpl_stab devcurve"
    )
    dev_curve_bal_mpl <- rbind(dev_curve_bal_mpl,
      data.frame(Dataset="Balanced", Model="mpl_stab",
                 log_lambda=log(mpl_cv_stab$Lambda), Deviance=mpl_cv_stab$cv_cvm)
    )
  }

  # mpl_full: X = full, Z = X
  mpl_cv_full <- mplasso_cv(
    X = X_all, Z = X_all, y = y_all,
    foldid = foldid_dev_mpl,
    tag = "Balanced mpl_full devcurve"
  )
  dev_curve_bal_mpl <- rbind(dev_curve_bal_mpl,
    data.frame(Dataset="Balanced", Model="mpl_full",
               log_lambda=log(mpl_cv_full$Lambda), Deviance=mpl_cv_full$cv_cvm)
  )
}

## combine with your existing L1 curves and plot
dev_curve_bal_all <- rbind(
  dev_curve_bal_l1[, c("Dataset","Model","log_lambda","Deviance")],
  dev_curve_bal_mpl
)

dev_curve_bal_all$Model <- factor(
  dev_curve_bal_all$Model,
  levels = c("mpl_fixed","mpl_stab","mpl_full","L1_stab","L1_full")
)

windows()
print(
  ggplot(dev_curve_bal_all, aes(x = log_lambda, y = Deviance, colour = Model)) +
    geom_line() +
    geom_point(size = 0.8) +
    labs(
      title = "Balanced dataset: CV deviance vs log(lambda) (mpl + L1)",
      x = "log(lambda)",
      y = "CV Deviance"
    ) +
    theme_bw()
)

############################################################
# Combined PR + ROC computation – Balanced only
############################################################

make_pr_roc_all_models_bal <- function(X, y, stable_l1, stable_mpl, mpl_cols, dataset_name) {

  coln <- colnames(X)
  p    <- ncol(X)

  # ---- MPL matrices (UPDATED to global sets) ----
  X_mpl_fixed <- if (length(mpl_cols$cols_X_mains) > 0)
    X[, mpl_cols$cols_X_mains, drop = FALSE] else NULL
  Z_mpl_fixed <- if (length(mpl_cols$cols_Z_fixed) > 0)
    X[, mpl_cols$cols_Z_fixed, drop = FALSE] else NULL

  X_mpl_stab  <- if (length(mpl_cols$cols_X_mains) > 0)
    X[, mpl_cols$cols_X_mains, drop = FALSE] else NULL
  Z_mpl_stab  <- if (length(mpl_cols$cols_Z_stab) > 0)
    X[, mpl_cols$cols_Z_stab, drop = FALSE] else NULL

  X_mpl_full  <- X
  Z_mpl_full  <- X

  # ---- L1 feature sets (unchanged) ----
  main_idx_l1 <- match(stable_l1$mains, coln)
  main_idx_l1 <- main_idx_l1[!is.na(main_idx_l1)]
  mod_names_l1 <- character(0)
  if (!is.null(stable_l1$inter) && nrow(stable_l1$inter) > 0) {
    valid_i_l1 <- stable_l1$inter$i[
      stable_l1$inter$i >= 1 & stable_l1$inter$i <= length(stable_l1$all_mains)
    ]
    valid_j_l1 <- stable_l1$inter$j[
      stable_l1$inter$j >= 1 & stable_l1$inter$j <= length(stable_l1$all_mains)
    ]
    names_i_l1 <- stable_l1$all_mains[valid_i_l1]
    names_j_l1 <- stable_l1$all_mains[valid_j_l1]
    mod_names_l1 <- unique(c(names_i_l1, names_j_l1))
  }
  mod_idx_l1 <- match(mod_names_l1, coln)
  mod_idx_l1 <- mod_idx_l1[!is.na(mod_idx_l1)]
  feat_idx_l1_stab <- sort(unique(c(main_idx_l1, mod_idx_l1)))

  feat_idx_l1_full <- seq_len(p)

  X_l1_stab   <- if (length(feat_idx_l1_stab) >= 2)
    X[, feat_idx_l1_stab, drop = FALSE] else NULL
  X_l1_full   <- if (length(feat_idx_l1_full) >= 2)
    X[, feat_idx_l1_full, drop = FALSE] else NULL

  models <- c("mpl_fixed", "mpl_stab", "mpl_full",
              "L1_stab", "L1_full", "iFORM")

  pr_curve_list  <- list()
  pr_best_list   <- list()
  roc_curve_list <- list()
  roc_best_list  <- list()
  score_list     <- list()

  foldid_shared <- make_stratified_foldid(y, 5)

  for (m in models) {
    cat("\n[PR/ROC] Fitting model:", m, "on", dataset_name, "\n")

    if (m == "mpl_fixed") {
      if (is.null(X_mpl_fixed) || is.null(Z_mpl_fixed) ||
          ncol(Z_mpl_fixed) == 0 || length(unique(y)) < 2) next
      mpl_cv <- mplasso_cv(
        X = X_mpl_fixed, Z = Z_mpl_fixed,
        y = y, foldid = foldid_shared,
        tag = paste(dataset_name, "mpl_fixed_full")
      )
      lambda_best <- mpl_cv$lambda_min
      y_pl <- ifelse(y == 1, 2L, 1L)
      pr_m <- predict_lasso(
        mpl_cv$fit_full,
        X = X_mpl_fixed,
        Z = Z_mpl_fixed,
        y = y_pl,
        lambda = lambda_best
      )
      P_hat <- as.data.frame(pr_m$y_hat)
      p <- if (ncol(P_hat) >= 2) as.numeric(P_hat[, 2]) else as.numeric(P_hat[, 1])

    } else if (m == "mpl_stab") {
      if (is.null(X_mpl_stab) || is.null(Z_mpl_stab) ||
          ncol(Z_mpl_stab) == 0 || length(unique(y)) < 2) next
      mpl_cv <- mplasso_cv(
        X = X_mpl_stab, Z = Z_mpl_stab,
        y = y, foldid = foldid_shared,
        tag = paste(dataset_name, "mpl_stab_full")
      )
      lambda_best <- mpl_cv$lambda_min
      y_pl <- ifelse(y == 1, 2L, 1L)
      pr_m <- predict_lasso(
        mpl_cv$fit_full,
        X = X_mpl_stab,
        Z = Z_mpl_stab,
        y = y_pl,
        lambda = lambda_best
      )
      P_hat <- as.data.frame(pr_m$y_hat)
      p <- if (ncol(P_hat) >= 2) as.numeric(P_hat[, 2]) else as.numeric(P_hat[, 1])

    } else if (m == "mpl_full") {
      if (is.null(X_mpl_full) || is.null(Z_mpl_full) ||
          ncol(Z_mpl_full) == 0 || length(unique(y)) < 2) next
      mpl_cv <- mplasso_cv(
        X = X_mpl_full, Z = Z_mpl_full,
        y = y, foldid = foldid_shared,
        tag = paste(dataset_name, "mpl_full_full")
      )
      lambda_best <- mpl_cv$lambda_min
      y_pl <- ifelse(y == 1, 2L, 1L)
      pr_m <- predict_lasso(
        mpl_cv$fit_full,
        X = X_mpl_full,
        Z = Z_mpl_full,
        y = y_pl,
        lambda = lambda_best
      )
      P_hat <- as.data.frame(pr_m$y_hat)
      p <- if (ncol(P_hat) >= 2) as.numeric(P_hat[, 2]) else as.numeric(P_hat[, 1])

    } else if (m == "L1_stab") {
      if (is.null(X_l1_stab) || length(unique(y)) < 2) next
      cv <- cv.glmnet(
        X_l1_stab, y,
        family      = "binomial",
        alpha       = 1,
        nlambda     = n_lambda,
        type.measure = "auc",
        foldid      = foldid_shared
      )
      lambda_best <- cv$lambda.min
      fit <- glmnet(
        X_l1_stab, y,
        family = "binomial",
        alpha  = 1,
        lambda = lambda_best
      )
      p <- as.numeric(predict(fit, X_l1_stab, type = "response"))

    } else if (m == "L1_full") {
      if (is.null(X_l1_full) || length(unique(y)) < 2) next
      cv <- cv.glmnet(
        X_l1_full, y,
        family      = "binomial",
        alpha       = 1,
        nlambda     = n_lambda,
        type.measure = "auc",
        foldid      = foldid_shared
      )
      lambda_best <- cv$lambda.min
      fit <- glmnet(
        X_l1_full, y,
        family = "binomial",
        alpha  = 1,
        lambda = lambda_best
      )
      p <- as.numeric(predict(fit, X_l1_full, type = "response"))

    } else if (m == "iFORM") {
      if (length(unique(y)) < 2) next
      df_all <- as.data.frame(X)
      df_all$y <- y
      formula_str <- paste("y ~ 0 +", paste(colnames(X), collapse = "+"))
      formula <- as.formula(formula_str)
      model_if <- iForm_logistic(
        formula,
        data         = df_all,
        heredity     = "strong",
        higher_order = FALSE
      )
      p <- as.numeric(predict(model_if, newdata = df_all, type = "response"))
    } else {
      next
    }

    score_list[[m]] <- p

    ## ---- PR curve ----
    pr <- pr.curve(
      scores.class0 = p[y == 1],
      scores.class1 = p[y == 0],
      curve = TRUE
    )

    df_pr <- data.frame(
      Recall    = pr$curve[, 1],
      Precision = pr$curve[, 2],
      Threshold = pr$curve[, 3],
      Model     = m,
      Dataset   = dataset_name
    )

    f1 <- 2 * df_pr$Precision * df_pr$Recall /
      (df_pr$Precision + df_pr$Recall + 1e-15)
    best_idx_pr <- which.max(f1)
    pr_auc_val  <- pr$auc.integral

    pr_best_list[[m]] <- data.frame(
      Recall    = df_pr$Recall[best_idx_pr],
      Precision = df_pr$Precision[best_idx_pr],
      Threshold = df_pr$Threshold[best_idx_pr],
      F1        = f1[best_idx_pr],
      PR_AUC    = pr_auc_val,
      Model     = m,
      Dataset   = dataset_name
    )

    pr_curve_list[[m]] <- df_pr

    ## ---- ROC curve ----
    roc_obj <- roc(y, p, quiet = TRUE)

    df_roc <- data.frame(
      FPR     = 1 - roc_obj$specificities,
      TPR     = roc_obj$sensitivities,
      Model   = m,
      Dataset = dataset_name
    )

    coords_best <- coords(
      roc_obj, "best",
      best.method = "youden",
      ret = c("threshold", "sensitivity", "specificity")
    )

    thresh_vec <- as.numeric(coords_best[["threshold"]])
    sens_vec   <- as.numeric(coords_best[["sensitivity"]])
    spec_vec   <- as.numeric(coords_best[["specificity"]])

    thresh <- thresh_vec[1]
    sens   <- sens_vec[1]
    spec   <- spec_vec[1]

    roc_auc_val <- as.numeric(auc(roc_obj))

    roc_best_list[[m]] <- data.frame(
      FPR       = 1 - spec,
      TPR       = sens,
      Threshold = thresh,
      ROC_AUC   = roc_auc_val,
      Model     = m,
      Dataset   = dataset_name
    )

    roc_curve_list[[m]] <- df_roc
  }

  # check correlation between mpl_stab and L1_stab scores (debug)
  if (all(c("mpl_stab", "L1_stab") %in% names(score_list))) {
    cor_p  <- cor(score_list[["mpl_stab"]], score_list[["L1_stab"]])
    max_dp <- max(abs(score_list[["mpl_stab"]] - score_list[["L1_stab"]]))
    cat(sprintf("[SCORE-DEBUG] cor(p_mpl_stab, p_L1_stab) = %.6f, max |Δp| = %.6f\n",
                cor_p, max_dp))
  }

  pr_curves  <- if (length(pr_curve_list)  > 0) do.call(rbind, pr_curve_list)  else data.frame()
  pr_best    <- if (length(pr_best_list)   > 0) do.call(rbind, pr_best_list)   else data.frame()
  roc_curves <- if (length(roc_curve_list) > 0) do.call(rbind, roc_curve_list) else data.frame()
  roc_best   <- if (length(roc_best_list)  > 0) do.call(rbind, roc_best_list)  else data.frame()

  list(
    pr_curves  = pr_curves,
    pr_best    = pr_best,
    roc_curves = roc_curves,
    roc_best   = roc_best
  )
}

pr_roc_all_balanced <- make_pr_roc_all_models_bal(
  full_bal$X, full_bal$y,
  stable_bal_l1, stable_bal_mpl,
  mpl_cols_bal,          # NEW
  "Balanced"
)

pr_curves_bal  <- pr_roc_all_balanced$pr_curves
pr_best_bal    <- pr_roc_all_balanced$pr_best
roc_curves_bal <- pr_roc_all_balanced$roc_curves
roc_best_bal   <- pr_roc_all_balanced$roc_best

############################################################
# PR plot – Balanced (all models incl. mplasso + iFORM)
############################################################

pr_curves_bal$Model <- factor(
  pr_curves_bal$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)
pr_best_bal$Model   <- factor(
  pr_best_bal$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)

pr_text <- data.frame()
if (nrow(pr_best_bal) > 0) {
  pr_text <- data.frame(
    x = 0.02,
    y = seq(0.05, by = 0.05, length.out = nrow(pr_best_bal)),
    label = sprintf(
      #"%s: thr=%.3f, PR-AUC=%.3f, F1=%.3f",
	  "%s: thr=%.3f, PR-AUC=%.3f",
      pr_best_bal$Model,
      pr_best_bal$Threshold,
      pr_best_bal$PR_AUC
      #pr_best_bal$F1
    )
  )
}

windows()
print(
  ggplot(
    pr_curves_bal,
    aes(x = Recall, y = Precision, colour = Model)
  ) +
    geom_line(size = 1) +
    geom_point(
      data = pr_best_bal,
      aes(x = Recall, y = Precision, colour = Model),
      size = 3, shape = 4, stroke = 1.5, show.legend = FALSE
    ) +
    geom_text(
      data = pr_text,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      hjust = 0, size = 3
    ) +
    labs(
      title    = "PR curves – Balanced dataset (all models)",
      subtitle = "Crosses mark best-F1 threshold for each model",
      x = "Recall",
      y = "Precision"
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    theme_bw()
)

############################################################
# ROC plot – Balanced (all models incl. mplasso + iFORM)
############################################################

roc_curves_bal$Model <- factor(
  roc_curves_bal$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)
roc_best_bal$Model   <- factor(
  roc_best_bal$Model,
  levels = c("mpl_fixed", "mpl_stab", "mpl_full", "L1_stab", "L1_full", "iFORM")
)

roc_text <- data.frame()
if (nrow(roc_best_bal) > 0) {
  roc_text <- data.frame(
    x = 0.02,
    y = seq(0.05, by = 0.05, length.out = nrow(roc_best_bal)),
    label = sprintf(
      "%s: thr=%.3f, ROC-AUC=%.3f",
      roc_best_bal$Model,
      roc_best_bal$Threshold,
      roc_best_bal$ROC_AUC
    )
  )
}

windows()
print(
  ggplot(
    roc_curves_bal,
    aes(x = FPR, y = TPR, colour = Model)
  ) +
    geom_line(size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_point(
      data = roc_best_bal,
      aes(x = FPR, y = TPR, colour = Model),
      size = 3, shape = 4, stroke = 1.5, show.legend = FALSE
    ) +
    geom_text(
      data = roc_text,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      hjust = 0, size = 3
    ) +
    labs(
      title    = "ROC curves – Balanced dataset (all models)",
      subtitle = "Crosses mark best threshold for each model",
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    coord_cartesian(ylim = c(0, 1), xlim = c(0, 1)) +
    theme_bw()
)

cv_bal <- run_nested_cv(
  full_bal$X, full_bal$y,
  stable_l1      = stable_bal_l1,
  stable_mpl     = stable_bal_mpl,
  mpl_cols       = mpl_cols_bal,
  tag            = "balanced",
  n_folds_outer  = n_folds_outer,
  n_permutations = n_permutations,
  n_folds_inner  = n_folds_inner,
  base_seed      = base_seed_cv
)

print(cal_df[, c("Model", "bin", "n")])
summary(cv_bal$preds_outer$p)
hist(cv_bal$preds_outer$p, breaks = 20)

