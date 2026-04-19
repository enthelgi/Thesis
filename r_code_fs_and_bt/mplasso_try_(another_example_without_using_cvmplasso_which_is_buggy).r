# ===== Setup =================================================================
setwd("C:/Users/enthe/Desktop/Thesis/r_code_fs_and_bt")
source("Mplasso.R")

library(glmnet)

# helpers
error.bars <- function(x, upper, lower, width = 0.02, ...) {
  segments(x, lower, x, upper, ...); segments(x - width, upper, x + width, upper, ...)
  segments(x - width, lower, x + width, lower, ...)
}
log_term <- function(y, pr) y*log(pr) + (1 - y)*log(1 - pr)

# ===== Simulate TRAIN data ====================================================
set.seed(1)
N  <- 100
p  <- 10
nz <- 3               # modifiers
K  <- 3               # number of classes; keep K == nz for this generator


X <- matrix(rnorm(N*p), N, p)
X <- scale(X, colMeans(X), sqrt(apply(X, 2, var)))
windows();plot(colMeans(X), rep(0, p)); abline(h = 0)
X <- matrix(as.numeric(X), N, p)

Z <- matrix(rnorm(N*nz), N, nz)
Z <- scale(Z, colMeans(Z), sqrt(apply(Z, 2, var)))
Z <- matrix(as.numeric(Z), N, nz)

e <- matrix(1, N, 1)

beta_1 <- beta_2 <- beta_3 <- rep(0, p)
beta_1[1:5] <- c( 2,  2, 2, 2, 2)
beta_2[1:5] <- c(-2, -2,-2,-2,-2)
beta_3[1:5] <- c( 4, -2, 1, 3, 2)

coeffs1 <- cbind(beta_1[1] + 5*Z[,1], beta_1[2], beta_1[3] + 3*Z[,2],
                 beta_1[4]*(e - 2*Z[,3]), beta_1[5]*(e - 2*Z[,nz])) + 0.5*rnorm(N)
coeffs2 <- cbind(beta_2[1] + 5*Z[,1], beta_2[2], beta_2[3] + 3*Z[,2],
                 beta_2[4]*(e - 2*Z[,3]), beta_2[5]*(e - 2*Z[,nz])) + 0.5*rnorm(N)
coeffs3 <- cbind(beta_3[1] + 5*Z[,1], beta_3[2], beta_3[3] + 3*Z[,2],
                 beta_3[4]*(e - 2*Z[,3]), beta_3[5]*(e - 2*Z[,nz])) + 0.5*rnorm(N)

vProb <- cbind(
  exp(diag(X[, 1:5] %*% t(coeffs1))),
  exp(diag(X[, 1:5] %*% t(coeffs2))),
  exp(diag(X[, 1:5] %*% t(coeffs3)))
)
mChoices <- t(apply(vProb, 1, rmultinom, n = 1, size = 1))
y <- apply(mChoices, 1, function(x) which(x == 1))
table(y)

# keep K consistent with data
K <- length(unique(y)); stopifnot(ncol(Z) == K)

# ===== Fit mplasso PATH =======================================================
nlambda <- 50
for_v <- 10; sv <- 0; fq <- 50; st <- 50; mv <- 20; ms <- 50; tol <- 1e-3

system.time(
  result <- plasso_fit1(
    y = y, X = X, Z = Z, nlambda = nlambda,
    alpha = .5, new_t = 1, my_mbeta = .09, intercept = 0.01,
    step = .05, number = 10, maxgrid = 50, tol = tol, run = 2,
    lambda_min = .001, for_v = for_v, sv = sv, fq = fq, st = st, mv = mv, ms = ms,
    cv_run = 0
  )
)

path   <- result$path
Lambda <- as.numeric(result$Lambdas)

# ===== Baseline glmnet over [X, Z] ============================================
dat <- data.frame(X, Z)
colnames(dat) <- c(paste0("X", 1:ncol(X)), paste0("Z", 1:ncol(Z)))
X_Z <- as.matrix(dat)

fit3 <- glmnet(x = X_Z, y = y, alpha = 1.0, family = "multinomial", nlambda = nlambda)
fit4 <- glmnet(x = X_Z, y = y, alpha = 0.5, family = "multinomial", nlambda = nlambda)

fitdiv1 <- deviance(fit3) / length(y)
fitdiv2 <- deviance(fit4) / length(y)



take50 <- function(v) v[seq_len(min(50, length(v)))]



windows();plot(log(fit3$lambda), fitdiv1, type = "o", col = "red",
     xlab = "log(lambda)", ylab = "Train Deviance",
     ylim = range(c(path$DEV, fitdiv1, fitdiv2)),
     xlim = range(c(log(Lambda), log(fit3$lambda), log(fit4$lambda))));lines(log(take50(Lambda)), take50(path$DEV), col = "blue", type = "o");lines(log(take50(fit4$lambda)), take50(fitdiv2), type = "o", col = "green");legend(-10, 2, legend = c("lasso", "plasso", "elastic net"),
       col = c("red", "blue", "green"), lty = 1:2, cex = 0.8)

M_pliable_50 <- take50(path$DEV)
lasso_50     <- take50(fitdiv1)
enet_50      <- take50(fitdiv2)

common_len <- min(length(M_pliable_50), length(lasso_50), length(enet_50))

boxplot(data.frame(
  M_pliable   = M_pliable_50[1:common_len],
  lasso       = lasso_50[1:common_len],
  elastic_net = enet_50[1:common_len]
),
main = "N=100, p=10, K=3", ylab = "Train Deviance")


# ===== Simple, robust K-fold CV for mplasso (replaces cv_mpliable) ===========
set.seed(3321)
nfolds <- 5
foldid <- sample(rep(1:nfolds, length.out = N))

# (We can still compute zhat, but we'll use TRUE Z for CV per the DGP)
zhat <- matrix(NA_real_, N, ncol(Z))
for (ii in 1:nfolds) {
  zfit <- cv.glmnet(X[foldid != ii, , drop = FALSE],
                    Z[foldid != ii, , drop = FALSE], family = "mgaussian")
  zhat[foldid == ii, ] <- predict(zfit, X[foldid == ii, , drop = FALSE],
                                  s = zfit$lambda.min)
}
Z_for_cv <- Z   # <-- use TRUE Z in CV for a fair comparison

# Cross-validate deviance along the SAME Lambda grid learned on full data
cv_cvm <- numeric(length(Lambda))
cv_cnt <- numeric(length(Lambda))

for (ii in 1:nfolds) {
  tr <- foldid != ii
  te <- foldid == ii

  fit_i <- plasso_fit1(
    y = y[tr], X = X[tr, , drop = FALSE], Z = Z_for_cv[tr, , drop = FALSE],
    nlambda = nlambda,
    alpha = .5, new_t = 1, my_mbeta = .09, intercept = 0.01,
    step = .05, number = 10, maxgrid = 50, tol = tol, run = 2,
    lambda_min = .001, for_v = for_v, sv = sv, fq = fq, st = st, mv = mv, ms = ms,
    cv_run = 0
  )

  lam_i   <- as.numeric(fit_i$Lambdas)
  idx_map <- sapply(Lambda, function(L) which.min(abs(lam_i - L)))

  for (jj in seq_along(Lambda)) {
    k <- idx_map[jj]
    pr <- predict_lasso(fit_i,
                        X = X[te, , drop = FALSE],
                        Z = Z_for_cv[te, , drop = FALSE],
                        y = y[te],
                        lambda = lam_i[k])
    cv_cvm[jj] <- cv_cvm[jj] + pr$deviance
    cv_cnt[jj] <- cv_cnt[jj] + length(y[te])
  }
}

cv_cvm <- (2 * cv_cvm) / cv_cnt   # average deviance per obs (×2)

lambda_min <- Lambda[which.min(cv_cvm)]
lambda_1se <- {
  se <- sqrt(stats::var(cv_cvm) / nfolds)
  min_idx <- which.min(cv_cvm)
  thr <- cv_cvm[min_idx] + se
  cand <- which(cv_cvm <= thr)
  Lambda[max(cand)]  # largest lambda within 1SE
}

windows();plot(log(Lambda), cv_cvm, type = "o", col = "red",
     xlab = "log(lambda)", ylab = "CV Deviance");abline(v = log(lambda_min), lty = 3);abline(v = log(lambda_1se), lty = 3)

# ===== Simulate TEST data =====================================================
set.seed(2)
N_test <- 500
X_test <- matrix(rnorm(N_test*p), N_test, p)
X_test <- scale(X_test, colMeans(X), sqrt(apply(X, 2, var)))  # scale like train

# Use TRUE Z_test drawn like train (NOT predicted)
Z_test <- matrix(rnorm(N_test * ncol(Z)), N_test, ncol(Z))
Z_test <- scale(Z_test, colMeans(Z), sqrt(apply(Z, 2, var)))

e_test <- matrix(1, N_test, 1)

coeffs1_t <- cbind(beta_1[1] + 5*Z_test[,1], beta_1[2], beta_1[3] + 3*Z_test[,2],
                   beta_1[4]*(e_test - 2*Z_test[,3]), beta_1[5]*(e_test - 2*Z_test[,nz])) + 0.5*rnorm(N_test)
coeffs2_t <- cbind(beta_2[1] + 5*Z_test[,1], beta_2[2], beta_2[3] + 3*Z_test[,2],
                   beta_2[4]*(e_test - 2*Z_test[,3]), beta_2[5]*(e_test - 2*Z_test[,nz])) + 0.5*rnorm(N_test)
coeffs3_t <- cbind(beta_3[1] + 5*Z_test[,1], beta_3[2], beta_3[3] + 3*Z_test[,2],
                   beta_3[4]*(e_test - 2*Z_test[,3]), beta_3[5]*(e_test - 2*Z_test[,nz])) + 0.5*rnorm(N_test)

vProb_t <- cbind(
  exp(diag(X_test[, 1:5] %*% t(coeffs1_t))),
  exp(diag(X_test[, 1:5] %*% t(coeffs2_t))),
  exp(diag(X_test[, 1:5] %*% t(coeffs3_t)))
)
mChoices_t <- t(apply(vProb_t, 1, rmultinom, n = 1, size = 1))
y_test <- apply(mChoices_t, 1, function(x) which(x == 1))
table(y_test)

# ===== Compare test deviance across methods ==================================
# fit glmnet baselines on TRAIN (CV models)
cvfit1 <- cv.glmnet(x = X_Z, y = y, alpha = 1.0, family = "multinomial", nfolds = 5, nlambda = nlambda)
cvfit2 <- cv.glmnet(x = X_Z, y = y, alpha = 0.5, family = "multinomial", nfolds = 5, nlambda = nlambda)

dat_test <- data.frame(X_test, Z_test)
colnames(dat_test) <- c(paste0("X", 1:ncol(X_test)), paste0("Z", 1:ncol(Z_test)))
X_Z_test <- as.matrix(dat_test)

to_run <- min(50, length(Lambda), length(cvfit1$lambda), length(cvfit2$lambda))
dev_P <- dev_L <- dev_E <- numeric(to_run)

for (n in 1:to_run) {
  pr_pl <- predict_lasso(result, X = X_test, Z = Z_test, y = y_test, lambda = Lambda[n])
  dev_P[n] <- pr_pl$deviance

  glmnet_pred1 <- predict(cvfit1$glmnet.fit, newx = X_Z_test, type = "response", s = cvfit1$lambda[n])
  glmnet_pred2 <- predict(cvfit2$glmnet.fit, newx = X_Z_test, type = "response", s = cvfit2$lambda[n])
  pr_l <- data.frame(glmnet_pred1)
  pr_e <- data.frame(glmnet_pred2)

  Dev_l <- Dev_e <- numeric(length(y_test))
  for (l in seq_along(Dev_l)) {
    ll_l <- ll_e <- numeric(K)
    for (d in 1:K) {
      y_d <- as.integer(y_test[l] == d)
      prob_d_l <- matrix(unlist(pr_l[, d]), N_test)
      prob_d_e <- matrix(unlist(pr_e[, d]), N_test)
      ll_l[d] <- log_term(y = y_d, pr = prob_d_l[l])
      ll_e[d] <- log_term(y = y_d, pr = prob_d_e[l])
    }
    Dev_l[l] <- sum(ll_l); Dev_e[l] <- sum(ll_e)
  }
  dev_L[n] <- (2 * sum(Dev_l)) / length(y_test)
  dev_E[n] <- (2 * sum(Dev_e)) / length(y_test)
}

windows();boxplot(data.frame(plasso = dev_P, lasso = dev_L, E_net = dev_E), ylab = "Test deviance")

windows();plot(log(Lambda[1:to_run]), dev_P, type = "o", col = "red",
     ylim = range(c(dev_P, dev_L, dev_E)),
     xlim = range(c(log(Lambda[1:to_run]), log(cvfit1$lambda[1:to_run]), log(cvfit2$lambda[1:to_run]))),
     xlab = "log(lambda)", ylab = "Test Deviance", main = "N=100, p=10, K=3");lines(log(cvfit1$lambda[1:to_run]), dev_L, type = "o", col = "blue");lines(log(cvfit2$lambda[1:to_run]), dev_E, type = "o", col = "green");legend(-10, 2.5, legend = c("plasso", "lasso", "elastic net"),
       col = c("red", "blue", "green"), lty = 1:2, cex = 0.8)

# ===== Accuracy at lambda_min ================================================
# use the mplasso lambda picked by our manual CV
pr_pl_min  <- predict_lasso(result, X = X_test, Z = Z_test, y = y_test, lambda = lambda_min)
P_pl       <- data.frame(pr_pl_min$y_hat)

P_la <- data.frame(matrix(
  predict(cvfit1$glmnet.fit, X_Z_test, type = "response", s = cvfit1$lambda.min),
  N_test, K
))
P_en <- data.frame(matrix(
  predict(cvfit2$glmnet.fit, X_Z_test, type = "response", s = cvfit2$lambda.min),
  N_test, K
))

argmax_idx <- function(M) apply(M, 1, which.max)
acc <- c(
  pliable = mean(argmax_idx(P_pl) == y_test),
  lasso   = mean(argmax_idx(P_la) == y_test),
  elastic = mean(argmax_idx(P_en) == y_test)
)
print(acc)
