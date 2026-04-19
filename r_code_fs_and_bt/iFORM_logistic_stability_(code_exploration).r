library(jsonlite)

source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM_logistic_(code_exploration).r")

# --- CONFIG ---
DATA_PATHS <- list(
  imbalanced = "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv",
  balanced   = "c:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv"
)
RESULTS_DIR <- "c:/Users/enthe/Desktop/Thesis/results/tgs_results"
B <- 50
tau_list <- c(0.5, 0.55, 0.6, 0.65, 0.7)
set.seed(123)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# --- Helper functions ---
read_tgs_csv <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  y <- as.numeric(df$MCI)
  x_cols <- setdiff(colnames(df), "MCI")
  X <- as.matrix(df[, x_cols, drop = FALSE])
  list(X = X, y = y, x_cols = x_cols)
}

get_selected <- function(model) {
  # Extract selected features (main effects and interactions) from iForm_logistic model
  terms <- attr(model$terms, "term.labels")
  mains <- terms[!grepl(":", terms)]
  inters <- terms[grepl(":", terms)]
  list(mains = mains, inters = inters)
}

# Build empirical pmf on {0, 1/B_eff, ..., 1}
build_pmf <- function(values, B_eff) {
  if (length(values) == 0 || all(is.na(values))) {
    return(rep(0, B_eff + 1))
  }
  idx <- round(values * B_eff)
  idx[idx < 0] <- 0
  idx[idx > B_eff] <- B_eff
  # tabulate expects 1-based
  tab <- tabulate(idx + 1, nbins = B_eff + 1)
  pmf <- tab / sum(tab)
  pmf
}

# Check unimodality of a discrete pmf
is_unimodal <- function(pmf, tol = 1e-10) {
  if (all(is.na(pmf))) return(NA)
  pmf[is.na(pmf)] <- 0
  idx <- which(pmf > tol)
  if (length(idx) <= 2L) return(TRUE)  # trivial cases
  pmf_trim <- pmf[min(idx):max(idx)]
  k_max <- which.max(pmf_trim)
  left  <- pmf_trim[1:k_max]
  right <- pmf_trim[k_max:length(pmf_trim)]
  is_nondec <- all(diff(left)  >= -tol)
  is_noninc <- all(diff(right) <=  tol)
  is_nondec && is_noninc
}

# Check r-concavity: f^r should be convex for r < 0 (discrete second differences >= 0)
is_r_concave <- function(pmf, r, tol = 1e-8) {
  if (all(is.na(pmf))) return(NA)
  pmf[is.na(pmf)] <- 0
  idx <- which(pmf > 0)
  if (length(idx) <= 2L) return(TRUE)
  f <- pmf[idx]^r
  d2 <- diff(f, differences = 2)
  all(d2 >= -tol)
}

# --- CPSS stability selection ---
cpss_stability_selection_tgs <- function(X, y, B_pairs = 50, seed = 123) {
  set.seed(seed)
  n <- nrow(X)
  p <- ncol(X)

  mains_pair_count <- numeric(p)      # for Π̃_B (simultaneous selection)
  inter_pair_count <- matrix(0, p, p)
  mains_half_count <- numeric(p)      # for Π̂_B (CPSS selection, mains only)

  K_eff <- 0                          # effective number of valid complementary pairs

  K_mains_halves <- c()               # #mains per half-sample
  K_inter_halves <- c()               # #interactions per half-sample
  K_model_size_halves <- c()          # total model size per half (mains + inters)

  x_cols <- colnames(X)

  for (b in seq_len(B_pairs)) {
    # Stratified split by class
    classes <- unique(y)
    idx_by_class <- lapply(classes, function(c) which(y == c))
    half_A <- integer(0)
    half_B <- integer(0)
    for (idx_c in idx_by_class) {
      perm <- sample(idx_c)
      n_c <- length(perm)
      kA <- ceiling(n_c / 2)
      half_A <- c(half_A, perm[seq_len(kA)])
      half_B <- c(half_B, perm[(kA + 1):n_c])
    }
    # Shuffle halves
    half_A <- sample(half_A)
    half_B <- sample(half_B)

    # Skip if either half has only one class
    if (length(unique(y[half_A])) < 2 || length(unique(y[half_B])) < 2) {
      cat(sprintf("[CPSS pair %d] Skipping: one half has only one class.\n", b))
      next
    }

    # Fit iFORM_logistic on each half
    df_A <- as.data.frame(X[half_A, , drop = FALSE])
    df_A$y <- y[half_A]
    df_B <- as.data.frame(X[half_B, , drop = FALSE])
    df_B$y <- y[half_B]

    formula_str <- paste("y ~ 0 +", paste(x_cols, collapse = "+"))
    formula <- as.formula(formula_str)

    model_A <- iForm_logistic(formula, data = df_A, heredity = "strong", higher_order = FALSE)
    model_B <- iForm_logistic(formula, data = df_B, heredity = "strong", higher_order = FALSE)
    print(attr(model_A$terms, "term.labels"))
    print(attr(model_B$terms, "term.labels"))

    sel_A <- get_selected(model_A)
    sel_B <- get_selected(model_B)

    # --- Main effects ---
    mains_A <- sel_A$mains
    mains_B <- sel_B$mains

    # count simultaneous selection in a pair (for Π̃_B)
    mains_pair <- intersect(mains_A, mains_B)
    mains_idx <- match(mains_pair, x_cols)
    mains_idx <- mains_idx[!is.na(mains_idx)]
    mains_pair_count[mains_idx] <- mains_pair_count[mains_idx] + 1

    # count per-half selections (for Π̂_B)
    idx_A <- match(mains_A, x_cols)
    idx_A <- idx_A[!is.na(idx_A)]
    mains_half_count[idx_A] <- mains_half_count[idx_A] + 1

    idx_B <- match(mains_B, x_cols)
    idx_B <- idx_B[!is.na(idx_B)]
    mains_half_count[idx_B] <- mains_half_count[idx_B] + 1

    K_mains_halves <- c(K_mains_halves, length(mains_A), length(mains_B))

    # --- Interactions ---
    inters_A <- sel_A$inters
    inters_B <- sel_B$inters
    inters_pair <- intersect(inters_A, inters_B)
    for (inter in inters_pair) {
      vars <- unlist(strsplit(inter, ":"))
      idx1 <- match(vars[1], x_cols)
      idx2 <- match(vars[2], x_cols)
      if (!is.na(idx1) && !is.na(idx2)) {
        inter_pair_count[idx1, idx2] <- inter_pair_count[idx1, idx2] + 1
        inter_pair_count[idx2, idx1] <- inter_pair_count[idx2, idx1] + 1
      }
    }
    K_inter_halves <- c(K_inter_halves, length(inters_A), length(inters_B))

    # total model size per half (mains + interactions)
    size_A <- length(mains_A) + length(inters_A)
    size_B <- length(mains_B) + length(inters_B)
    K_model_size_halves <- c(K_model_size_halves, size_A, size_B)

    K_eff <- K_eff + 1
  }

  if (K_eff == 0) stop("All CPSS pairs failed; no valid splits.")

  # --- Frequencies used for CPSS ---
  mains_freq <- mains_pair_count / K_eff
  inter_freq <- inter_pair_count / K_eff

  # --- q estimates ---
  q_mains  <- mean(K_mains_halves)          # expected #mains per half
  q_inter  <- mean(K_inter_halves)          # expected #interactions per half
  q_total  <- mean(K_model_size_halves)     # total model size per half

  # --- Π̂_B(k) and Π̃_B(k) (mains only) ---
  Pi_hat   <- mains_half_count / (2 * K_eff)  # CPSS proportion per variable
  Pi_tilde <- mains_pair_count / K_eff        # simultaneous selection proportion

  # sanity: q_mains should equal sum_k Π̂_B(k)
  q_mains_sum <- sum(Pi_hat)

  # --- θ and L_θ ---
  p <- length(Pi_hat)
  theta_hat <- q_mains / p   # natural θ ≈ q/p
  delta_theta <- 0.05        # small window around θ to define "low prob" vars

  L_idx <- which(Pi_hat <= theta_hat + delta_theta)
  Pi_tilde_L <- Pi_tilde[L_idx]
  Pi_hat_L   <- Pi_hat[L_idx]

  # --- Shape checks for bounds ---
  unimodal_tilde   <- NA
  rconc_tilde_m12  <- NA
  rconc_hat_m14    <- NA
  bound_regime     <- "worst_case"

  if (length(L_idx) >= 3) {
    # effective B for Π̃_B is K_eff, for Π̂_B is 2*K_eff
    pmf_tilde_L <- build_pmf(Pi_tilde_L, B_eff = K_eff)
    pmf_hat_L   <- build_pmf(Pi_hat_L,   B_eff = 2 * K_eff)

    unimodal_tilde  <- is_unimodal(pmf_tilde_L)
    rconc_tilde_m12 <- is_r_concave(pmf_tilde_L, r = -0.5)
    rconc_hat_m14   <- is_r_concave(pmf_hat_L,   r = -0.25)

    # determine which bound regime is supported
    bound_regime <- "worst_case"
    if (isTRUE(unimodal_tilde)) {
      bound_regime <- "unimodal"
    }
    if (isTRUE(rconc_tilde_m12) && isTRUE(rconc_hat_m14)) {
      bound_regime <- "r_concave"
    }
  } else {
    bound_regime <- "worst_case (few L_theta vars)"
  }

  list(
    mains_freq          = mains_freq,
    inter_freq          = inter_freq,
    K_mains_halves      = K_mains_halves,
    K_inter_halves      = K_inter_halves,
    K_model_size_halves = K_model_size_halves,
    q_mains             = q_mains,
    q_inter             = q_inter,
    q_total             = q_total,
    q_mains_sum         = q_mains_sum,
    Pi_hat              = Pi_hat,
    Pi_tilde            = Pi_tilde,
    theta_hat           = theta_hat,
    L_idx               = L_idx,
    unimodal_tilde      = unimodal_tilde,
    rconc_tilde_m12     = rconc_tilde_m12,
    rconc_hat_m14       = rconc_hat_m14,
    bound_regime        = bound_regime,
    K_eff               = K_eff,
    x_cols              = x_cols
  )
}

# --- Main loop over datasets ---
for (balance_tag in names(DATA_PATHS)) {
  cat("\n==== Processing:", balance_tag, "====\n")
  dat <- read_tgs_csv(DATA_PATHS[[balance_tag]])
  X <- dat$X
  y <- dat$y
  x_cols <- dat$x_cols

  res <- cpss_stability_selection_tgs(X, y, B_pairs = B, seed = 123)

  # --- Print q and bound checks ---
  cat(sprintf("Estimated q_mains (E|S_{n/2}|, mains only) for %s: %.3f\n",
              balance_tag, res$q_mains))
  cat(sprintf("Estimated q_total (E|S_{n/2}|, mains + interactions) for %s: %.3f\n",
              balance_tag, res$q_total))
  cat(sprintf("Check: sum_k Pi_hat(k) = %.3f (should be ~ q_mains)\n",
              res$q_mains_sum))
  cat(sprintf("theta_hat = q_mains / p = %.4f\n", res$theta_hat))

  cat("Bound regime suggested by shape checks:", res$bound_regime, "\n")
  cat("  Unimodal Π̃_B on L_theta: ",
      if (isTRUE(res$unimodal_tilde)) "TRUE" else if (identical(res$unimodal_tilde, FALSE)) "FALSE" else "NA",
      "\n", sep = "")
  cat("  r-concave Π̃_B (r = -1/2) on L_theta: ",
      if (isTRUE(res$rconc_tilde_m12)) "TRUE" else if (identical(res$rconc_tilde_m12, FALSE)) "FALSE" else "NA",
      "\n", sep = "")
  cat("  r-concave Π̂_B (r = -1/4) on L_theta: ",
      if (isTRUE(res$rconc_hat_m14)) "TRUE" else if (identical(res$rconc_hat_m14, FALSE)) "FALSE" else "NA",
      "\n", sep = "")

  # --- Save mains frequencies ---
  mains_freq_path <- file.path(RESULTS_DIR, sprintf("stability_mains_tgs_iform_%s.csv", balance_tag))
  mains_freq_df <- data.frame(freq = res$mains_freq, row.names = x_cols)
  write.csv(mains_freq_df, mains_freq_path)
  cat("Saved mains CPSS frequencies →", mains_freq_path, "\n")

  # --- Save interaction frequencies ---
  inter_freq_path <- file.path(RESULTS_DIR, sprintf("stability_interactions_tgs_iform_%s.csv", balance_tag))
  rows <- list()
  p <- length(x_cols)
  for (i in seq_len(p)) {
    for (j in seq(i, p)) {
      rows[[length(rows) + 1]] <- list(i = i - 1, j = j - 1, freq = res$inter_freq[i, j])
    }
  }
  inter_freq_df <- do.call(rbind, rows)
  write.csv(inter_freq_df, inter_freq_path, row.names = FALSE)
  cat("Saved interaction CPSS frequencies →", inter_freq_path, "\n")

  # --- Save selected mains/interactions for each tau threshold ---
  for (freq_threshold in tau_list) {
    selected_mains <- x_cols[which(res$mains_freq > freq_threshold)]
    selected_interactions <- list()
    for (row in rows) {
      if (row$freq > freq_threshold && row$i < row$j) {
        selected_interactions[[length(selected_interactions) + 1]] <-
          list(x_cols[row$i + 1], x_cols[row$j + 1])
      }
    }
    thresh_json <- list(
      selected_mains = selected_mains,
      selected_interactions = selected_interactions
    )
    thresh_json_path <- file.path(
      RESULTS_DIR,
      sprintf("stability_selected_tgs_iform_thresh_%.2f_%s.json", freq_threshold, balance_tag)
    )
    write(toJSON(thresh_json, pretty = TRUE, auto_unbox = TRUE), file = thresh_json_path)
    cat("Saved mains/interactions above threshold JSON →", thresh_json_path, "\n")
  }

  # --- Save top 10 mains/interactions as JSON ---
  top_n <- 10
  top_mains_idx <- order(res$mains_freq, decreasing = TRUE)[seq_len(top_n)]
  selected_mains <- x_cols[top_mains_idx]

  inter_freq_flat <- data.frame(
    i = rep(seq_len(p), p),
    j = rep(seq_len(p), each = p),
    freq = as.vector(res$inter_freq)
  )
  inter_freq_flat <- inter_freq_flat[inter_freq_flat$i < inter_freq_flat$j, ]
  top_inter <- head(inter_freq_flat[order(inter_freq_flat$freq, decreasing = TRUE), ], top_n)
  selected_interactions <- mapply(
    function(i, j) list(x_cols[i], x_cols[j]),
    top_inter$i, top_inter$j, SIMPLIFY = FALSE
  )

  top_json <- list(
    selected_mains = selected_mains,
    selected_interactions = selected_interactions
  )
  top_json_path <- file.path(RESULTS_DIR, sprintf("top10_stability_selected_tgs_iform_%s.json", balance_tag))
  write(toJSON(top_json, pretty = TRUE, auto_unbox = TRUE), file = top_json_path)
  cat("Saved top 10 mains/interactions JSON →", top_json_path, "\n")
}
