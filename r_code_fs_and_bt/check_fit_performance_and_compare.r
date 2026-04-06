# ===== Setup =================================================================
setwd("C:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt")
source("Mplasso.R")
source("cv_mplasso.R")
library(glmnet)

# ===== Load TGS Data =========================================================
tgs <- read.csv("../data/tgs_data/tgs_dataset_normalized_balanced.csv")

X <- as.matrix(tgs[, setdiff(names(tgs), "MCI")])

# 0/1 labels for glm / glmnet
y_glm <- as.integer(tgs$MCI)

# 1/2 labels for mplasso
y_mpl <- y_glm + 1

N <- nrow(X)
p <- ncol(X)
K <- length(unique(y_glm))  # Should be 2 for binary

# ===== Modifiers Z ===========================================================
# If you have modifiers Z, define here.

# Example 1: intercept-only modifiers
# Z <- matrix(1, N, 1)

# Example 2: specific features as modifiers (your current choice)
Z <- X[, c(4, 8, 13), drop = FALSE]

Z = X
# ===== Fit mplasso PATH ======================================================
nlambda <- 50
for_v <- 10; sv <- 0; fq <- 50; st <- 50; mv <- 20; ms <- 50; tol <- 1e-3

result <- plasso_fit1(
  y = y_mpl,               # <-- 1/2 labels for mplasso
  X = X,
  Z = Z,
  nlambda    = nlambda,
  alpha      = 0.5,
  new_t      = 1,
  my_mbeta   = 0.09,
  intercept  = 0.01,
  step       = 0.05,
  number     = 10,
  maxgrid    = 50,
  tol        = tol,
  run        = 2,
  lambda_min = 0.001,
  for_v      = for_v,
  sv         = sv,
  fq         = fq,
  st         = st,
  mv         = mv,
  ms         = ms,
  cv_run     = 0
)

path   <- result$path
Lambda <- as.numeric(result$Lambdas)

# ===== Baseline glmnet over [X, Z] ===========================================
X_Z <- cbind(X, Z)

fit3 <- glmnet(
  x       = X_Z,
  y       = y_glm,         # <-- 0/1 labels for glmnet lasso
  alpha   = 1.0,
  family  = "binomial",
  nlambda = nlambda
)

fit4 <- glmnet(
  x       = X_Z,
  y       = y_glm,         # <-- 0/1 labels for glmnet elastic net
  alpha   = 0.5,
  family  = "binomial",
  nlambda = nlambda
)

fitdiv1 <- deviance(fit3) / length(y_glm)
fitdiv2 <- deviance(fit4) / length(y_glm)

windows()
plot(
  log(fit3$lambda), fitdiv1,
  type = "o", col = "red",
  xlab = "log(lambda)", ylab = "Train Deviance",
  ylim = range(c(path$DEV, fitdiv1, fitdiv2)),
  xlim = range(c(log(Lambda), log(fit3$lambda), log(fit4$lambda)))
)
lines(log(Lambda),     path$DEV,  col = "blue",  type = "o")
lines(log(fit4$lambda), fitdiv2,  col = "green", type = "o")
legend(
  "topright",
  legend = c("lasso", "plasso", "elastic net"),
  col    = c("red",   "blue",   "green"),
  lty    = 1:2,
  cex    = 0.8
)

# ===== K-fold CV for mplasso =================================================
set.seed(3321)
nfolds <- 5
foldid <- sample(rep(1:nfolds, length.out = N))
Z_for_cv <- Z

cv_cvm <- numeric(length(Lambda))
cv_cnt <- numeric(length(Lambda))

for (ii in 1:nfolds) {
  tr <- foldid != ii
  te <- foldid == ii

  fit_i <- plasso_fit1(
    y = y_mpl[tr],                         # <-- 1/2 labels in training
    X = X[tr, , drop = FALSE],
    Z = Z_for_cv[tr, , drop = FALSE],
    nlambda    = nlambda,
    alpha      = 0.5,
    new_t      = 1,
    my_mbeta   = 0.09,
    intercept  = 0.01,
    step       = 0.05,
    number     = 10,
    maxgrid    = 50,
    tol        = tol,
    run        = 2,
    lambda_min = 0.001,
    for_v      = for_v,
    sv         = sv,
    fq         = fq,
    st         = st,
    mv         = mv,
    ms         = ms,
    cv_run     = 0
  )

  lam_i   <- as.numeric(fit_i$Lambdas)
  idx_map <- sapply(Lambda, function(L) which.min(abs(lam_i - L)))

  for (jj in seq_along(Lambda)) {
    k <- idx_map[jj]
    pr <- predict_lasso(
      fit_i,
      X      = X[te, , drop = FALSE],
      Z      = Z_for_cv[te, , drop = FALSE],
      y      = y_mpl[te],                # <-- 1/2 labels for deviance
      lambda = lam_i[k]
    )
    cv_cvm[jj] <- cv_cvm[jj] + pr$deviance
    cv_cnt[jj] <- cv_cnt[jj] + length(y_mpl[te])
  }
}

# Average deviance per observation (×2)
cv_cvm <- (2 * cv_cvm) / cv_cnt

lambda_min <- Lambda[which.min(cv_cvm)]
lambda_1se <- {
  se      <- sqrt(stats::var(cv_cvm) / nfolds)
  min_idx <- which.min(cv_cvm)
  thr     <- cv_cvm[min_idx] + se
  cand    <- which(cv_cvm <= thr)
  Lambda[max(cand)]  # largest lambda within 1SE
}

windows()
plot(
  log(Lambda), cv_cvm,
  type = "o", col = "red",
  xlab = "log(lambda)", ylab = "CV Deviance"
)
abline(v = log(lambda_min), lty = 3)
abline(v = log(lambda_1se), lty = 3)

# ===== Accuracy at lambda_min ================================================
# --- pliable at lambda_min (using 1/2 labels internally) ---
pr_pl_min <- predict_lasso(
  result,
  X      = X,
  Z      = Z,
  y      = y_mpl,           # <-- 1/2 labels for deviance
  lambda = lambda_min
)

# pr_pl_min$y_hat is a list; for single lambda, take [[1]]:
P_pl <- as.data.frame(pr_pl_min$y_hat[[1]])  # N x 2 matrix: probs for classes 1 and 2

# --- glmnet lasso / elastic net at lambda_min (using 0/1 labels) ---
P_la <- data.frame(matrix(
  predict(
    fit3, X_Z,
    type = "response",
    s    = fit3$lambda[which.min(abs(fit3$lambda - lambda_min))]
  ),
  N, 1
))

P_en <- data.frame(matrix(
  predict(
    fit4, X_Z,
    type = "response",
    s    = fit4$lambda[which.min(abs(fit4$lambda - lambda_min))]
  ),
  N, 1
))

# --------------- with the less complex model lambda_1se ----------------------
# Recompute predictions at lambda_1se (overwriting the above)
pr_pl_min <- predict_lasso(
  result,
  X      = X,
  Z      = Z,
  y      = y_mpl,          # <-- 1/2 labels for deviance
  lambda = lambda_1se
)
P_pl <- as.data.frame(pr_pl_min$y_hat[[1]])  # N x 2 matrix again

P_la <- data.frame(matrix(
  predict(
    fit3, X_Z,
    type = "response",
    s    = fit3$lambda[which.min(abs(fit3$lambda - lambda_1se))]
  ),
  N, 1
))

P_en <- data.frame(matrix(
  predict(
    fit4, X_Z,
    type = "response",
    s    = fit4$lambda[which.min(abs(fit4$lambda - lambda_1se))]
  ),
  N, 1
))

# ===== Convert all predictions to 0/1 and compute accuracy ===================
# pliable: pick class with max probability (1 or 2), then convert to 0/1
class_pl_12 <- max.col(as.matrix(P_pl))       # values in {1,2}
class_pl_01 <- class_pl_12 - 1                # convert to {0,1}

# glmnet: threshold probabilities at 0.5
class_la_01 <- as.integer(P_la[, 1] >= 0.5)
class_en_01 <- as.integer(P_en[, 1] >= 0.5)

acc <- c(
  pliable = mean(class_pl_01 == y_glm),
  lasso   = mean(class_la_01 == y_glm),
  elastic = mean(class_en_01 == y_glm)
)
print(acc)
