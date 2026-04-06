source("c:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt/iFORM.R")
library(jsonlite)
library(tools)
library(leaps)
library(ggplot2)
library(dplyr)

data_dir <- "C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous"
dataset_files <- list.files(data_dir, pattern = "^simulated_dataset_.*\\.csv$", full.names = TRUE)

results <- data.frame(
  dataset = character(),
  n = integer(),
  p = integer(),
  model = character(),
  RSS = numeric(),
  stringsAsFactors = FALSE
)

for (filename in dataset_files) {
  cat("\nProcessing:", filename, "\n")
  # Read data
  data <- read.csv(filename)
  data <- data[, sapply(data, function(x) length(unique(x)) > 1)]
  data <- na.omit(data)
  X_cols <- colnames(data)[grepl("^X", colnames(data))]
  y_col <- "y"
  n <- nrow(data)
  p <- length(X_cols)
  if (n == 0 || p == 0) next

  # --- iFORM ---
  predictor_str <- paste(X_cols, collapse = " + ")
  formula_str <- paste("y ~", predictor_str)
  iForm_fit <- iForm(formula_str, data, heredity = "strong", higher_order = FALSE)
  y_pred_iform <- as.numeric(predict(iForm_fit, data))
  rss_iform <- sum((data$y - y_pred_iform)^2)
  results <- rbind(results, data.frame(
    dataset = basename(filename),
    n = n,
    p = p,
    model = "iFORM",
    RSS = rss_iform
  ))

  max_p_fs <- 100  # set your threshold
  # if (p <= max_p_fs) {
  #   # --- Classic Forward Selection with Interactions ---
  #   Xmat <- data[, X_cols, drop = FALSE]
  #   inter_terms <- combn(X_cols, 2, simplify = TRUE)
  #   for (k in 1:ncol(inter_terms)) {
  #     colname <- paste0(inter_terms[1, k], ":", inter_terms[2, k])
  #     Xmat[[colname]] <- Xmat[[inter_terms[1, k]]] * Xmat[[inter_terms[2, k]]]
  #   }
  #   regfit <- regsubsets(
  #     x = as.matrix(Xmat),
  #     y = data$y,
  #     nvmax = min(n - 1, ncol(Xmat)),
  #     method = "forward"
  #   )
  #   rss_vec <- summary(regfit)$rss
  #   best_idx <- which.min(rss_vec)
  #   coef_best <- coef(regfit, best_idx)
  #   selected_vars <- names(coef_best)[-1]
  #   X_selected <- cbind(1, as.matrix(Xmat[, selected_vars, drop = FALSE]))
  #   y_pred_fs <- X_selected %*% coef_best
  #   rss_fs <- sum((data$y - y_pred_fs)^2)
  #   results <- rbind(results, data.frame(
  #     dataset = basename(filename),
  #     n = n,
  #     p = p,
  #     model = "ForwardSelection",
  #     RSS = rss_fs
  #   ))
  # } else {
  #   cat("Skipping Forward Selection for", filename, "due to large p =", p, "\n")
  # }
  # Remove or comment out the if (p <= max_p_fs) check
  # Always run the forward selection code
  tryCatch({
    # --- Classic Forward Selection with Interactions ---
    Xmat <- data[, X_cols, drop = FALSE]
    inter_terms <- combn(X_cols, 2, simplify = TRUE)
    for (k in 1:ncol(inter_terms)) {
      colname <- paste0(inter_terms[1, k], ":", inter_terms[2, k])
      Xmat[[colname]] <- Xmat[[inter_terms[1, k]]] * Xmat[[inter_terms[2, k]]]
    }
    regfit <- regsubsets(
      x = as.matrix(Xmat),
      y = data$y,
      nvmax = min(n - 1, ncol(Xmat)),
      method = "forward"
    )
    rss_vec <- summary(regfit)$rss
    best_idx <- which.min(rss_vec)
    coef_best <- coef(regfit, best_idx)
    selected_vars <- names(coef_best)[-1]
    X_selected <- cbind(1, as.matrix(Xmat[, selected_vars, drop = FALSE]))
    y_pred_fs <- X_selected %*% coef_best
    rss_fs <- sum((data$y - y_pred_fs)^2)
    results <- rbind(results, data.frame(
      dataset = basename(filename),
      n = n,
      p = p,
      model = "ForwardSelection",
      RSS = rss_fs
    ))
  }, error = function(e) {
    cat("Error in Forward Selection for", filename, ":", conditionMessage(e), "\n")
  })  
}

# --- Add group column for n < p and p < n ---
results$group <- ifelse(results$n < results$p, "n < p", ifelse(results$p < results$n, "p < n", NA))
results <- results[!is.na(results$group), ]

# --- Set consistent y-axis limits and colors ---
ylim_all <- range(results$RSS, na.rm = TRUE)
model_colors <- c("iFORM" = "#1f77b4", "ForwardSelection" = "#ff7f0e") # blue for iFORM, orange for FS

# n < p
ggplot(subset(results, group == "n < p"), aes(x = model, y = RSS, fill = model)) +
  geom_boxplot() +
  ggtitle("RSS for n < p (across datasets)") +
  theme_bw() +
  scale_fill_manual(values = model_colors) +
  ylim(ylim_all)
ggsave("boxplot_rss_n_lt_p.png", width = 5, height = 5, dpi = 200)

# p < n
ggplot(subset(results, group == "p < n"), aes(x = model, y = RSS, fill = model)) +
  geom_boxplot() +
  ggtitle("RSS for p < n (across datasets)") +
  theme_bw() +
  scale_fill_manual(values = model_colors) +
  ylim(ylim_all)
ggsave("boxplot_rss_p_lt_n.png", width = 5, height = 5, dpi = 200)

cat("Done. Boxplots saved as boxplot_rss_n_lt_p.png and boxplot_rss_p_lt_n.png\n")