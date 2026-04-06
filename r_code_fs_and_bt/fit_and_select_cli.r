# fit_and_select_cli.r  (FINAL VERSION USING MPLIABLE LASSO)
# Usage:
# Rscript fit_and_select_cli.r X.csv y.csv Z.csv mains_out.csv inter_out.csv

library(glmnet)

# --- Set working directory to the script's directory so source() works ---
args_full <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_full[grepl("^--file=", args_full)])
if (length(script_path)) {
  script_dir <- dirname(normalizePath(script_path))
  setwd(script_dir)
}

cat("Working directory:", getwd(), "\n")
source("Mplasso.R")  # Mplasso.R should be in the same folder as this script

# --- Parse arguments from Python ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Expected 5 arguments: X_path y_path Z_path mains_out_path inter_out_path")
}

X_path         <- args[1]
y_path         <- args[2]
Z_path         <- args[3]
mains_out_path <- args[4]
inter_out_path <- args[5]

cat("PLASSO SCRIPT: starting fit_and_select_cli.r\n")
cat("X_path:", X_path, "\n")
cat("y_path:", y_path, "\n")
cat("Z_path:", Z_path, "\n")

# --- Load data from Python ---
Xb <- as.matrix(read.csv(X_path, header = FALSE))
yb_raw <- as.numeric(read.csv(y_path, header = FALSE)[, 1])
yb <- as.integer(yb_raw)
yb <- yb - min(yb) + 1  # map labels to {1, 2} for binomial
Zb <- as.matrix(read.csv(Z_path, header = FALSE))

cat("dim(Xb):", paste(dim(Xb), collapse = " x "), "\n")
cat("length(yb):", length(yb), "\n")
cat("dim(Zb):", paste(dim(Zb), collapse = " x "), "\n")
cat("unique(yb):", paste(sort(unique(yb)), collapse = ", "), "\n")

p     <- ncol(Xb)
p_mod <- ncol(Zb)
K     <- ncol(Zb)
# These globals are needed by Mplasso's objective()
N <- nrow(Xb)
assign("N", N, envir = .GlobalEnv)  # make sure objective() sees it

# (optional but often used too)
assign("p", p, envir = .GlobalEnv)
assign("p_mod", p_mod, envir = .GlobalEnv)


# --- Fit mpliable lasso (as in your original R code) ---
cat("Calling plasso_fit1...\n")

fit <- plasso_fit1(
  y = yb, X = Xb, Z = Zb, nlambda = 50,
  alpha = .5, new_t = 1, my_mbeta = .09, intercept = 0.01,
  step = .05, number = 10, maxgrid = 50, tol = 1e-3, run = 2,
  lambda_min = .001, for_v = 10, sv = 0, fq = 50, st = 50, mv = 20, ms = 50,
  cv_run = 0
)

cat("plasso_fit1 completed.\n")

lambda_seq <- as.numeric(fit$Lambdas)
lambda_idx <- which.min(fit$path$DEV)
cat("Selected lambda index:", lambda_idx, "\n")

# For binomial, use class 2 (same as your fit_and_select example)
beta  <- fit$beta[[2]][, lambda_idx]        # length p
theta <- fit$theta[[2]][, , lambda_idx]     # p x p_mod

mains_sel <- abs(beta)  > 1e-8
inter_sel <- abs(theta) > 1e-8

cat("Number of selected mains:", sum(mains_sel), "\n")
cat("Number of selected interactions:", sum(inter_sel), "\n")

# --- Write results back for Python ---
write.table(
  as.integer(mains_sel),
  file = mains_out_path,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)

write.table(
  as.integer(inter_sel),
  file = inter_out_path,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)

cat("PLASSO SCRIPT: wrote mains/inter selections and exiting normally.\n")
