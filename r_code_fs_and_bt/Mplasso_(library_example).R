############################################################
##  Full mpliable library with theta-dimension fixes
############################################################

library(class)
library(pracma)

############################################################
##  Basic utilities
############################################################

quad_solution<-function(u, v, w){
  temp = ((v^2) - (4 * u * w))^0.5
  root1 = (-v + temp) / (2 * u)
  root2 = (-v - temp) / (2 * u)
  roots<-list(root1, root2)
  return (roots)
}

S_func <- function(x, a) {  # Soft Thresholding Operator
  pmax(abs(x) - a, 0) * sign(x)
}

Log <- function(y, pr) {
  if (isTRUE(y == 0)) {
    a <- 0
  } else {
    a <- y * log((y / pr))
  }
  return(a)
}

errfun.binomial <- function(y, yhat, w = rep(1, length(y))) {
  prob_min = 1e-05
  prob_max = 1 - prob_min
  predmat = pmin(pmax(yhat, prob_min), prob_max)
  -w * (y * log(predmat) + (1 - y) * log(1 - predmat))
}

reg <- function(r, Z) {
  K = ncol(Z)
  my_one <- matrix(1, nrow(Z))
  my_w = data.frame(Z, my_one)
  my_w <- as.matrix(my_w)
  my_inv <- pinv(t(my_w) %*% my_w)
  my_res <- my_inv %*% (t(my_w) %*% r)
  beta0 <- matrix(my_res[(K + 1)])
  theta0 <- matrix(my_res[c(1:K)])
  return(list(beta0, theta0))
}

error.bars <- function(x, upper, lower, width = 0.02, ...) {
  xlim <- range(x)
  barw <- diff(xlim) * width
  segments(x, upper, x, lower, ...)
  segments(x - barw, upper, x + barw, upper, ...)
  segments(x - barw, lower, x + barw, lower, ...)
}

twonorm <- function(x) {
  sqrt(sum(x * x))
}

############################################################
##  Core objective / gradient helpers (UPDATED)
############################################################

objective <- function(r, beta, theta, alpha, lambda) {
  # Ensure theta is p x K matrix
  if (is.null(dim(theta))) {
    p <- length(beta)
    K <- length(theta) / p
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  p <- length(beta)
  norm_1 <- lapply(
    seq_len(p),
    function(g) {
      lambda * (1 - alpha) * (
        norm(matrix(c(beta[g], theta[g, ]), ncol = 1), type = "F") +
          norm(matrix(theta[g, ], ncol = 1), type = "F")
      ) +
        lambda * alpha * sum(abs(theta[g, ]))
    }
  )
  
  # Uses global N like your original code
  objective_1 <- sum(r^2) / (2 * N) + sum(unlist(norm_1))
  return(objective_1)
}

gradient_j <- function(beta, theta, U, U2, U3, y, X, W, r,
                       alpha, lambda, K, j, N) {
  # Robustify theta shape: p x K
  if (is.null(dim(theta))) {
    p <- length(beta)
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  theta_j <- matrix(as.numeric(theta[j, ]), nrow = 1)
  dJ_dtheta_j <- matrix(0, nrow = 1, ncol = K)
  R <- c()
  
  if (isTRUE(any(c(beta[j], theta_j) != 0))) {
    u <- as.numeric(beta[j]) / norm(matrix(c(beta[j], theta_j)), type = "F")
    U <- c(U, u)
  } else {
    if (length(U) >= 1) {
      for (z in 1:length(U)) {
        if (norm(matrix(U[z]), type = "F") <= 1) {
          R <- c(R, U[z])
          next(z)
        } else {
          next(z)
        }
      }
    } else {
      R <- c(R, 0)
    }
    u <- sample(c(R), size = 1)
    U <- c(U, u)
  }
  
  dJ_dbeta_j <- -(t(matrix(X[, j])) %*% matrix(matrix(r))) / N +
    (1 - alpha) * lambda * u
  
  R <- c()
  if (isTRUE(any(c(beta[j], theta_j) != 0))) {
    u2 <- as.numeric(theta_j) /
      norm(matrix(c(as.numeric(beta[j]), theta_j)), type = "F")
    U2 <- c(U2, u2)
  } else {
    if (length(U2) >= 1) {
      for (z in 1:length(U2)) {
        if (norm(matrix(U2[z]), type = "F") <= 1) {
          R <- c(R, U2[z])
          next(z)
        } else {
          next(z)
        }
      }
    } else {
      R <- c(R, 0)
    }
    u2 <- sample(c(R), size = 1)
    U2 <- c(U2, u2)
  }
  
  d <- c()
  if (isTRUE(any(c(theta_j) != 0))) {
    u3 <- as.numeric(theta_j) / norm(matrix(c(theta_j)), type = "F")
    U3 <- c(U3, u3)
  } else {
    if (length(U3) >= 1) {
      for (z in 1:length(U3)) {
        if (norm(matrix(U3[z]), type = "F") <= 1) {
          d <- c(d, U3[z])
          next(z)
        } else {
          next(z)
        }
      }
    } else {
      d <- c(d, 0)
    }
    u3 <- sample(c(d), size = 1)
    U3 <- c(U3, u3)
  }
  
  v <- sign(as.numeric(theta_j))
  
  dJ_dtheta_j <- -((t(data.frame(W[j])) %*% matrix(matrix(r)))) / N +
    (1 - alpha) * lambda * (u2 + u3) +
    alpha * lambda * v
  
  L1 <- dJ_dbeta_j
  L2 <- dJ_dtheta_j
  
  return(list(L1, L2, U, U2, U3))
}

compute_pliable <- function(X, Z, theta) {
  N <- nrow(X)
  p <- ncol(X)
  K <- ncol(Z)
  
  # Ensure theta is p x K
  if (is.null(dim(theta))) {
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  xz_theta <- lapply(
    seq_len(p),
    function(j) {
      (matrix(X[, j], nrow = N, ncol = K) * Z) %*% t(theta)[, j]
    }
  )
  xz_term <- Reduce(f = "+", x = xz_theta)
  return(xz_term)
}

concat_beta_theta <- function(beta, theta) {
  p <- length(beta)
  if (is.null(dim(theta))) {
    K <- length(theta) / p
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  K <- ncol(theta)
  my_matrix <- matrix(0, p, (K + 1))
  my_matrix[, c(1:K)] <- theta
  my_matrix[, (K + 1)] <- beta
  return(my_matrix)
}

model <- function(beta0, theta0, beta, theta, X, Z) {
  N <- nrow(X)
  p <- ncol(X)
  K <- ncol(Z)
  
  # Make sure theta is p x K
  if (is.null(dim(theta))) {
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  intercepts <- as.numeric(beta0) + Z %*% (matrix(theta0))
  shared_model <- X %*% matrix(beta)
  pliable <- compute_pliable(X, Z, theta)
  return(intercepts + shared_model + pliable)
}

model_min_j <- function(beta0, theta0, beta, theta, X, Z, j, W) {
  p <- length(beta)
  K <- ncol(Z)
  if (is.null(dim(theta))) {
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  beta[j] <- 0.0
  theta[j, ] <- 0.0
  return(model(beta0, theta0, beta, theta, X, Z))
}

model_j <- function(beta_j, theta_j, x_j, W, j, Z) {
  # r_j ~ beta_j * X_j + W_j @ theta_j
  w_j <- as.matrix(data.frame(W[j]))
  zz <- w_j %*% theta_j
  return(beta_j * x_j + zz)
}

penalties_min_j <- function(beta_0, theta_0, beta, theta, X, Z, y, W,
                            j, E, n_i, b) {
  N <- nrow(X)
  p <- ncol(X)
  K <- ncol(Z)
  
  if (is.null(dim(theta))) {
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  n_i_b <- model_min_j(beta_0, theta_0, beta, theta, X, Z, j, W)
  pr_b <- matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N)
  
  B <- pr_b * (1 - pr_b)
  B[B == 0] <- as.numeric(10e-9)
  
  y1 <- 1 * (y == b)
  M <- (y1 - pr_b)
  A <- n_i_b + M / B
  rv <- B * (A - n_i_b)
  
  mse <- (1 / (2 * N)) / sum((rv)^2)
  
  coef_matrix <- concat_beta_theta(beta, theta)
  
  # Ignore the jth modifier from the model
  coef_matrix[j, ] <- 0.0
  theta[j, ] <- 0.0
  
  penalty_1 <- 0
  penalty_2 <- 0
  for (jj in 1:p) {
    penalty_1 <- penalty_1 + twonorm(matrix(coef_matrix[jj, ]))
    penalty_2 <- penalty_2 + twonorm(matrix(theta[jj, ]))
  }
  penalty_3 <- sum(abs(theta))
  
  return(list(mse, penalty_1, penalty_2, penalty_3))
}

objective_j <- function(beta0, theta0, beta, theta, X, Z, y, W,
                        alpha, lambda, p, K, N, j, r_min_j, b, n_i, E, B) {
  # Ensure matrix shape for theta first
  if (is.null(dim(theta))) {
    p <- length(beta)
    K <- ncol(Z)
    theta <- matrix(theta, nrow = p, ncol = K)
  }
  
  beta_j <- beta[j]
  theta_j <- theta[j, ]
  theta_j <- as.vector(theta_j)
  K <- length(theta_j)
  
  precomputed <- penalties_min_j(
    beta0, theta0, beta, theta, X, Z, y, W, j, E, n_i, b
  )
  
  mse       <- as.numeric(unlist(precomputed[1]))
  penalty_1 <- as.numeric(unlist(precomputed[2]))
  penalty_2 <- as.numeric(unlist(precomputed[3]))
  penalty_3 <- as.numeric(unlist(precomputed[4]))
  
  r_hat <- model_j(beta[j], theta_j, X[, j], W, j, Z)
  mse_1 <- (1 / (2 * N)) * (sum((r_min_j - B * r_hat)^2)) + mse
  
  coef_vector <- matrix(0, (K + 1))
  coef_vector[c(1:K)] <- as.numeric(theta_j)
  coef_vector[(K + 1)] <- beta_j
  penalty_1 <- penalty_1 + twonorm(matrix(coef_vector))
  
  penalty_2 <- penalty_2 + twonorm(matrix(theta_j))
  penalty_3 <- penalty_3 + sum(abs(theta_j))
  
  objective_l <- mse_1 +
    (1 - alpha) * lambda * (penalty_1 + penalty_2) +
    alpha * lambda * penalty_3
  
  return(objective_l)
}

############################################################
##  IMPORTANT: you must paste your original quadratic()
##  function here. It should call the helpers above.
############################################################

# quadratic <- function(beta, theta, alpha, lambda,
#                       beta0, theta0, j, b,
#                       W, X, Z, y, N, n_i, xbar, zbar,
#                       t = NULL, r_min_j) {
#   ## <<< paste your original implementation >>>
# }
quadratic = function(beta, theta, alpha, lambda, beta0, theta0, j, b, W, X, Z, y, N, n_i, xbar, zbar,
                     big_delta_1 = NULL, big_delta_2 = NULL, b_1 = NULL, b_2 = NULL, t = NULL, r_min_j) {
  if (isTRUE(is.null(big_delta_1)) == TRUE) { big_delta_1 = 0; lambda_d = 0 } else { big_delta_1 = big_delta_1; lambda_d = lambda }
  if (isTRUE(is.null(big_delta_2)) == TRUE) { big_delta_2 = 0 } else { big_delta_2 = big_delta_2 }

  if (isTRUE(is.null(b_1)) == TRUE) {
    b_1 = matrix(rep(0, p))
  } else {
    b_1 = matrix(b_1)
  }

  if (isTRUE(is.null(b_2)) == TRUE) {
    b_2 = matrix(0, p, 1)
  } else {
    b_2 = b_2
  }

  U <- matrix(0); U2 <- matrix(0); U3 <- matrix(0)

  big = 10e9; eps = 1e-5

  t = 1; nes = 1; beta_u = beta; theta_u = theta; beta_new = beta; theta_new = theta; v_beta = beta; v_theta = theta; n_l = n_i; n_r = n_i
  j = j
  okay = 0

  while (okay < 1) {

    theta_transpose_l <- t(theta_u); theta_transpose_r <- t(theta_new)

    n_r[b] <- list(model(beta0, theta0, beta_new, theta_new, X, Z))

    E_r <- matrix(0, N, max(y))
    for (x in 1:max(y)) {
      E_r[, x] <- exp(unlist(n_r[x]))
    }

    n_r_b <- matrix(unlist(n_r[b]), N)
    y_hat_r = n_r_b

    pr_r <- (matrix(as.numeric(exp(n_r_b) / ((rowSums(E_r)))), N))

    B_r <- pr_r * (1 - pr_r)
    B_r[B_r == 0] = as.numeric(10e-9)

    y1 = 1 * (y == b)

    M_r <- (y1 - pr_r)

    A_r <- y_hat_r + M_r / B_r

    r_r <- B_r * (A_r - y_hat_r)

    grad_j <- gradient_j(beta = beta_new, theta = theta_new, U, U2, U3, y, X, W, r = r_r, alpha, lambda, K, j, N)

    L1 = matrix(unlist(grad_j[1])); L2 <- matrix(unlist(grad_j[2]))

    U = c(unlist(grad_j[3])); U2 = c(unlist(grad_j[4])); U3 = c(unlist(grad_j[5]))

    beta_j = as.numeric(beta_new[j]); theta_j = as.numeric(theta_new[j, ])

    a <- as.numeric(norm(matrix(beta_j), type = "F")); bb <- as.numeric(norm(matrix(theta_j), type = "F")); rho_2 <- as.numeric(sqrt(a^2 + bb^2))

    c <- as.numeric(t * (1 - alpha) * lambda); g_1 <- as.numeric(abs(beta_j - t * as.numeric(L1)))
    v <- matrix(0, nrow = 1, ncol = K)

    v <- S_func(theta_j - t * L2, t * alpha * lambda)

    g_2 <- as.numeric(norm(matrix(v), type = "F"))

    root = quad_solution(1, 2 * c, 2 * c * g_2 - g_1^2 - g_2^2)
    root1 <- unlist(root[1])
    root2 <- unlist(root[2])

    a = c(
      g_1 * root1 / (c + root1),
      g_1 * root2 / (c + root2),
      g_1 * root1 / (c + root2),
      g_1 * root2 / (c + root1)
    )

    bb = c(
      g_1 * root1 * (c - g_2) / (c + root1),
      g_1 * root2 * (c - g_2) / (c + root2),
      g_1 * root1 * (c - g_2) / (c + root2),
      g_1 * root2 * (c - g_2) / (c + root1)
    )

    x_min = big

    j_hat = 0; k_hat = 0
    for (jjj in 1:4) {
      for (kkk in 1:4) {
        denominator = (a[jjj]^2 + bb[kkk]^2)^0.5
        if (isTRUE(denominator > 0) == TRUE) {
          val1 = (1 + (c / denominator)) * a[jjj] - g_1
          val2 = (1 + c * (1 / bb[kkk] + 1 / denominator)) * bb[kkk] - g_2

          temp = abs(val1) + abs(val2)
          if (isTRUE(temp < x_min) == TRUE) {
            j_hat = jjj; k_hat = kkk
            x_min = temp
          }
        }
      }
    }

    xnorm = (a[j_hat]^2 + bb[k_hat]^2)^0.5

    new_v_beta <- ((beta_j - t * L1)) / (1 + c / xnorm)

    new_v_theta <- (matrix(S_func(theta_j - t * L2, t * alpha * lambda))) /
      (1 + c * ((1 / xnorm) + (1 / abs(bb[k_hat]))))

    beta_u[j] <- new_v_beta; theta_u[j, ] <- new_v_theta

    theta_transpose_l <- t(theta_u); theta_transpose_r <- t(theta_new)

    n_l[b] <- list(model(beta0, theta0, beta_u, theta_u, X, Z))

    n_r[b] <- list(model(beta0, theta0, beta_new, theta_new, X, Z))

    E_l <- matrix(0, N, max(y))
    E_r <- matrix(0, N, max(y))
    for (x in 1:max(y)) {
      E_l[, x] <- exp(unlist(n_l[x]))
      E_r[, x] <- exp(unlist(n_r[x]))
    }
    n_l_b <- matrix(unlist(n_l[b]), N)
    n_r_b <- matrix(unlist(n_r[b]), N)
    y_hat_l = n_l_b; y_hat_r = n_r_b

    pr_l <- (matrix(as.numeric(exp(n_l_b) / ((rowSums(E_l)))), N)); pr_r <- (matrix(as.numeric(exp(n_r_b) / ((rowSums(E_r)))), N))

    B_l <- pr_l * (1 - pr_l); B_r <- pr_r * (1 - pr_r)

    y1 = 1 * (y == b)

    M_l <- (y1 - pr_l); M_r <- (y1 - pr_r)

    A_l <- y_hat_l + M_l / B_l; A_r <- y_hat_r + M_r / B_r

    r_l <- B_l * (A_l - y_hat_l); r_r <- B_r * (A_r - y_hat_r)

    objective_l <- objective_j(beta0, theta0, beta, theta, X, Z, y, W, alpha, lambda, p, K, N, j, r_min_j, b, n_l, E_l, B = B_l)
    objective_r <- objective_j(beta0, theta0, beta_new, theta_new, X, Z, y, W, alpha, lambda, p, K, N, j, r_min_j, b, n_r, E_r, B = B_r)

    rhs <- objective_r +
      matrix(c(L1, L2), nrow = 1) %*% matrix((c(beta_u[j], theta_u[j, ]) - c(beta_new[j], theta_new[j, ])), ncol = 1) +
      (1 / (2 * t)) * (norm(matrix((c(beta_u[j], theta_u[j, ]) - c(beta_new[j], theta_new[j, ])), ncol = 1), "F"))^2

    if (isTRUE(objective_l <= rhs) == TRUE) {
      okay = 1
    } else {

      old_beta = v_beta; old_theta = v_theta
      v_beta = beta_u
      v_theta = theta_u
      beta_new = v_beta + (nes / (nes + 3)) * (v_beta - old_beta)
      theta_new = v_theta + (nes / (nes + 3)) * (v_theta - old_theta)

      nes = nes + 1
      t = .8 * t
    }

    if (isTRUE(nes > 200) == TRUE) {
      okay = 1
    }

  } # while

  beta1 <- beta_u[j]; theta1 <- theta_u[j, ]; t = t

  return(list(beta1, theta1, t))
}


############################################################
##  Main training function: plasso_fit1 (original logic,
##  with xbar/zbar added)
############################################################

for_v = 10; sv = 0; fq = 50; st = 50; mv = 20; ms = 50; tol = 1e-3

plasso_fit1 <- function(y, X, Z, nlambda, alpha, new_t, my_mbeta,
                        number, intercept, step, maxgrid, tol,
                        run, lambda_min, my_lambda = NULL, tt = NULL,
                        for_v, sv, fq, st, mv, ms, cv_run,
                        max_iter = 100) {
  
  # xbar / zbar used in quadratic(...)
  xbar <- colMeans(X)
  zbar <- colMeans(Z)
  
  orig.X <- X
  N <- length(y)
  p <- ncol(X)
  K <- ncol(Z)
  
  rat <- lambda_min
  tolerance <- tol
  
  W <- lapply(seq_len(p),
              function(j) (matrix(X[, j], nrow = N, ncol = K) * Z))
  
  Y_hat <- matrix(0, nrow = (nlambda))
  Lambda <- matrix(0, nrow = (nlambda))
  non_zero <- matrix(0, nrow = (nlambda), ncol = max(y))
  non_zero_theta <- matrix(0, nrow = (nlambda), ncol = max(y))
  DEV <- matrix(0, nrow = (nlambda), ncol = 1)
  DEV1 <- matrix(0, nrow = (nlambda), ncol = 1)
  
  BETA0 <- lapply(seq_len(max(y)),
                  function(j) (matrix(0, nrow = (nlambda))))
  
  BETA <- lapply(seq_len(max(y)),
                 function(j) (matrix(0, nrow = p, ncol = nlambda)))
  
  THETA0 <- lapply(seq_len(max(y)),
                   function(j) (matrix(0, nrow = K, ncol = nlambda)))
  
  THETA <- lapply(seq_len(max(y)),
                  function(j) (array(0, c(p, K, nlambda))))
  
  BETA01 <- lapply(seq_len(max(y)),
                   function(j) (0))
  old_BETA01 <- lapply(seq_len(max(y)),
                       function(j) (0))
  BETA1 <- lapply(seq_len(max(y)),
                  function(j) (matrix(0, nrow = 1, ncol = p)))
  old_BETA1 <- lapply(seq_len(max(y)),
                      function(j) (matrix(0, nrow = 1, ncol = p)))
  
  THETA01 <- lapply(seq_len(max(y)),
                    function(j) (matrix(0, nrow = 1, ncol = K)))
  old_THETA01 <- lapply(seq_len(max(y)),
                        function(j) (matrix(0, nrow = 1, ncol = K)))
  THETA1 <- lapply(seq_len(max(y)),
                   function(j) (matrix(0, nrow = p, ncol = K)))
  old_THETA1 <- lapply(seq_len(max(y)),
                       function(j) (matrix(0, nrow = p, ncol = K)))
  Y_hat1 <- lapply(seq_len(max(y)),
                   function(j) (matrix(0, nrow = N)))
  
  XZ_term <- lapply(seq_len(max(y)),
                    function(j) (matrix(0, nrow = N)))
  n_i <- lapply(seq_len(max(y)),
                function(j) (matrix(0, nrow = N)))
  pr <- lapply(seq_len(max(y)),
               function(j) (matrix(0, nrow = N)))
  n_il <- lapply(seq_len(max(y)),
                 function(j) (matrix(0, nrow = N)))
  n_ir <- lapply(seq_len(max(y)),
                 function(j) (matrix(0, nrow = N)))
  prl <- lapply(seq_len(max(y)),
                function(j) (matrix(0, nrow = N)))
  prr <- lapply(seq_len(max(y)),
                function(j) (matrix(0, nrow = N)))
  
  for (x in 1:max(y)) {
    my_X <- X
    my_Z <- Z
    
    beta0 <- BETA0[[x]][1]
    beta <- BETA[[x]][, 1]
    theta <- as.matrix(THETA[[x]][,, 1])
    theta_transpose <- t(theta)
    theta0 <- THETA0[[x]][, 1]
    
    n_i[x] <- list(model(beta0, theta0, beta, theta, X, Z))
  }
  
  E <- matrix(0, N, max(y))
  for (x in 1:max(y)) {
    E[, x] <- exp(unlist(n_i[x]))
  }
  
  my_pr <- matrix(0, N, (max(y) - 1))
  for (x in 1:(max(y))) {
    n_i_b <- matrix(unlist(n_i[x]), N)
    pr[x] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
  }
  
  ## Lambda sequence
  if (is.null(my_lambda)) {
    
    O_1 <- matrix(0, max(y), 1)
    
    for (b in 1:max(y)) {
      n_i_b <- matrix(unlist(n_i[b]), N)
      pr_b <- matrix(unlist(pr[b]), N)
      
      B <- pr_b * (1 - pr_b)
      B[B == 0] <- as.numeric(10e-9)
      
      y1 <- 1 * (y == b)
      M <- (y1 - pr_b)
      A <- n_i_b + M / B
      rv <- B * (A - n_i_b)
      
      O_1[b] <- max(abs(t(X) %*% rv) / length(rv)) / (1 - alpha)
    }
    lambda_max_to_select <- max(O_1)
    lambda <- max(lambda_max_to_select)
    big_lambda <- lambda
    Lambda_min <- rat * big_lambda
    
    lambda_i <- exp(seq(log(big_lambda), log(big_lambda * rat), length = maxgrid))
    lambda_i[1] <- big_lambda
    lambda_i[maxgrid] <- Lambda_min
    
  } else {
    lambda_i <- my_lambda
  }
  
  big_t <- matrix(0, nlambda, max(y))
  Mbeta <- matrix(0, nlambda)
  mbeta <- my_mbeta
  
  dev_percent <- matrix(0, nlambda)
  sec_active1 <- matrix(0, p, max(y))
  sec_active2 <- matrix(0, p, max(y))
  
  ACTIVE <- matrix(0, nlambda, max(y))
  ACTIVE1 <- matrix(0, nlambda, max(y))
  my_v <- sv
  my_V <- for_v
  
  active_set1 <- matrix(0, p, max(y))
  active_set2 <- matrix(0, p, max(y))
  strong_set <- matrix(0, p, max(y))
  strong_set1 <- matrix(0, p, max(y))
  
  my_q <- 1
  my_ok <- 0
  q <- 1
  
  U <- matrix(0); U2 <- matrix(0); U3 <- matrix(0)
  
  while (isTRUE(my_ok == 0) | isTRUE(q <= nlambda)) {

    my_v <- my_v
    my_V <- my_V
    
    if (q <= 1) {
      ########################################################
      ## First lambda
      ########################################################
      i <- q
      lambda <- lambda_i[i]
      
      for (b in 1:max(y)) {
        
        beta0 <- BETA0[[b]][q]
        beta  <- BETA[[b]][, q]
        v_beta <- beta
        theta <- as.matrix(THETA[[b]][,, q])
        theta_transpose <- t(theta)
        v_theta <- theta
        theta_transpose1 <- t(v_theta)
        theta0 <- THETA0[[b]][, q]
        
        beta01 <- beta0
        beta1 <- beta
        theta1 <- theta
        theta01 <- theta
        norm1 <- matrix(0, p, 1)
        norm3 <- matrix(0, p, 1)
        
        for (iii in 1:max_iter) {
          n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
          n_i_b <- matrix(unlist(n_i[b]), N)
          
          E[, b] <- exp(unlist(n_i[b]))
          pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
          pr_b <- matrix(unlist(pr[b]), N)
          
          B <- pr_b * (1 - pr_b)
          B[B == 0] <- as.numeric(10e-9)
          
          y1 <- 1 * (y == b)
          M <- (y1 - pr_b)
          A <- n_i_b + M / B
          rv <- B * (A - n_i_b)
          
          iter_prev_score <- objective(rv, beta, theta, alpha, lambda)
          
          v <- reg(rv, Z)
          
          if (b == max(y)) {
            my_beta0 <- matrix(0, (max(y) - 1), ncol = 1)
            for (l in 1:(max(y) - 1)) {
              my_beta0[l] <- BETA0[[l]][i]
            }
            beta0 <- 1 - sum(my_beta0)
            theta0 <- matrix(unlist(v[2]))
          } else {
            beta0 <- matrix(unlist(v[1]))
            theta0 <- matrix(unlist(v[2]))
          }
          
          for (j in 1:p) {
            if (is.null(tt)) {
              t <- NULL
            } else {
              t <- tt[q]
            }
            
            if (i <= 1) {
              checkk <- 2 * lambda - lambda_i[1]
            } else {
              checkk <- 2 * lambda - lambda_i[i - 1]
            }
            
            if (as.numeric(abs((t(matrix((X[, j]))) %*% matrix(((rv)))) ) / (N * alpha)) < checkk) {
              next(j)
            } else {
              
              res_j <- rv + B * model_j(beta[j], theta[j, ], X[, j], W, j, Z)
              
              cond1 <- as.numeric(abs((t(matrix(X[, j])) %*% matrix(res_j)) / N))
              cond2 <- as.numeric(
                norm(
                  matrix(
                    S_func((t(data.frame(W[j])) %*% (res_j)) / N,
                           alpha * (lambda)
                    ),
                    ncol = 1
                  ),
                  type = "F"
                )
              )
              
              if (cond1 <= (1 - alpha) * (lambda)) {
                strong_set[j, b] <- 0
              } else {
                strong_set[j, b] <- 1
              }
              
              if (cond2 <= 2 * (1 - alpha) * (lambda)) {
                strong_set1[j, b] <- 0
              } else {
                strong_set1[j, b] <- 1
              }
              
              if (strong_set[j, b] == 0 & strong_set1[j, b] == 0) {
                next(j)
              } else {
                
                beta1_j <- (N / sum(B * (matrix(X[, j]) * matrix(X[, j])))) *
                  S_func((t(matrix(X[, j])) %*% matrix(res_j)) / N,
                         (1 - alpha) * lambda)
                
                cond3 <- norm(
                  matrix(
                    S_func(
                      t(data.frame(W[j])) %*% ((res_j) -
                                                 (B) * matrix(X[, j]) *
                                                 as.numeric(beta1_j)) / N,
                      alpha * lambda
                    ),
                    ncol = 1
                  ),
                  type = "F"
                )
                
                if (cond3 <= 2 * (1 - alpha) * lambda) {
                  beta[j] <- as.numeric(beta1_j)
                  active_set1[j, b] <- 1
                  next(j)
                } else {
                  value1 <- quadratic(
                    beta, theta, alpha, lambda, beta0, theta0, j, b,
                    W, X, Z, y, N, n_i, xbar, zbar,
                    t = t, r_min_j = res_j
                  )
                  beta[j] <- unlist(value1[[1]])
                  theta[j, ] <- unlist(value1[[2]])
                  t <- unlist(value1[[3]])
                  active_set1[j, b] <- 1
                  active_set2[j, b] <- 1
                  next(j)
                }
              }
            }
          } # j-loop
          
          n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
          n_i_b <- matrix(unlist(n_i[b]), N)
          
          E[, b] <- exp(unlist(n_i[b]))
          pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
          pr_b <- matrix(unlist(pr[b]), N)
          
          B <- pr_b * (1 - pr_b)
          B[B == 0] <- as.numeric(10e-9)
          
          y1 <- 1 * (y == b)
          M <- (y1 - pr_b)
          A <- n_i_b + M / B
          rv <- B * (A - n_i_b)
          
          if (i > new_t) {
            iter_current_score <- objective(rv, beta, theta, alpha, lambda)
            if (abs(iter_prev_score - iter_current_score) < tolerance) {
              break
            } else {
              next(iii)
            }
          } else {
            break
          }
        } # max_iter
        
        BETA01[b] <- list(beta0)
        BETA1[b] <- list(beta)
        THETA01[b] <- list(matrix(theta0, 1, K))
        THETA1[b] <- list(as.matrix(theta, p, K))
        
        BETA0[[b]][q] <- beta0
        THETA0[[b]][, q] <- theta0
        BETA[[b]][, q] <- beta
        THETA[[b]][,, q] <- theta
        
        next(b)
      } # b-loop
      
      m_for_beta <- matrix(0, p, max(y))
      for (l in 1:max(y)) {
        m_for_beta[, l] <- BETA[[l]][, q]
      }
      for (l in 1:nrow(m_for_beta)) {
        med <- mean(m_for_beta[l, ])
        m_for_beta[l, ] <- m_for_beta[l, ] - med
      }
      for (l in 1:max(y)) {
        BETA[[l]][, q] <- m_for_beta[, l]
      }
      
      for (x in 1:max(y)) {
        beta0 <- BETA0[[x]][q]
        beta  <- BETA[[x]][, q]
        theta <- as.matrix(THETA[[x]][,, q])
        theta0 <- THETA0[[x]][, q]
        n_i[x] <- list(model(beta0, theta0, beta, theta, X, Z))
      }
      
      E <- matrix(1, N, max(y))
      for (x in 1:max(y)) {
        E[, x] <- exp(unlist(n_i[x]))
      }
      
      for (x in 1:(max(y))) {
        n_i_b <- matrix(unlist(n_i[x]), N)
        pr[x] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
      }
      
      Dev <- matrix(0, nrow = length(y))
      Dev1 <- matrix(0, nrow = length(y))
      for (l in 1:length(Dev)) {
        deviance_y1 <- matrix(0, 1, max(y))
        deviance_y2 <- matrix(0, 1, max(y))
        for (d in 1:max(y)) {
          y_d <- 1 * (y[l] == d)
          prob_d <- matrix(unlist(pr[d]), N)
          deviance_y1[d] <- Log(y = y_d, pr = prob_d[l])
          deviance_y2[d] <- Log(y = y_d, pr = mean(1 * (y == d)))
        }
        Dev[l]  <- sum(deviance_y1)
        Dev1[l] <- sum(deviance_y2)
      }
      
      DEV1[i] <- ((2) * sum(Dev1)) / length(y)
      DEV[i]  <- ((2) * sum(Dev))  / length(y)
      if (is.null(t)) {
        t <- 0
      }
      big_t[i] <- t
      Mbeta[i] <- mbeta
      
      Y_hat[i] <- list(matrix(c(y, unlist(pr)), N, (max(y) + 1)))
      
      for (v in 1:max(y)) {
        nzero_beta <- BETA[[v]][, i]
        nzero_theta <- unlist(THETA[[v]][,, i])
        non_zero[i, v]        <- sum(nzero_beta != 0)
        non_zero_theta[i, v]  <- sum(nzero_theta != 0)
      }
      
      Lambda[i] <- lambda
      print(c(q, max(non_zero[i, ]), big_t[i], DEV[i]))
      
      my_q <- my_q + 1
      q <- q + 1
      
    } else if (q > 1 & q <= fq) {
      ########################################################
      ## Warm-start for next lambdas up to fq
      ########################################################
      i <- q
      lambda <- lambda_i[i]
      
      for (b in 1:max(y)) {
        lambda <- lambda_i[q]
        
        beta0 <- BETA0[[b]][q - 1]
        beta  <- BETA[[b]][, q - 1]
        v_beta <- BETA[[b]][, q - 2]
        theta <- as.matrix(THETA[[b]][,, q - 1])
        theta_transpose <- t(theta)
        v_theta <- as.matrix(THETA[[b]][,, q - 2])
        theta_transpose1 <- t(v_theta)
        theta0 <- THETA0[[b]][, q - 1]
        
        beta01 <- beta0
        beta1  <- beta
        theta1 <- theta
        theta01 <- theta
        
        norm1 <- matrix(0, p, 1)
        norm3 <- matrix(0, p, 1)
        
        for (iii in 1:max_iter) {
          n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
          n_i_b <- matrix(unlist(n_i[b]), N)
          E[, b] <- exp(unlist(n_i[b]))
          pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
          pr_b <- matrix(unlist(pr[b]), N)
          
          B <- pr_b * (1 - pr_b)
          B[B == 0] <- as.numeric(10e-9)
          
          y1 <- 1 * (y == b)
          M  <- (y1 - pr_b)
          A  <- n_i_b + M / B
          rv <- B * (A - n_i_b)
          iter_prev_score <- objective(rv, beta, theta, alpha, lambda)
          
          v <- reg(rv, Z)
          if (b == max(y)) {
            my_beta0 <- matrix(0, (max(y) - 1))
            for (l in 1:(max(y) - 1)) {
              my_beta0[l] <- BETA0[[l]][q - 1]
            }
            beta0 <- 1 - sum(my_beta0)
            theta0 <- matrix(unlist(v[2]))
          } else {
            beta0 <- matrix(unlist(v[1]))
            theta0 <- matrix(unlist(v[2]))
          }
          
          DJ_BETA <- matrix(0, 1)
          DJ_THETA <- matrix(0, 1, K)
          
          for (j in 1:p) {
            if (is.null(tt)) {
              t <- NULL
            } else {
              t <- tt[q]
            }
            
            if (i <= 1) {
              checkk <- 2 * lambda - lambda_i[1]
            } else {
              checkk <- 2 * lambda - lambda_i[i - 1]
            }
            
            if (as.numeric(abs((t(matrix((X[, j]))) %*% matrix(((rv)))) ) / (N * alpha)) < checkk) {
              next(j)
            } else {
              res_j <- rv + B * model_j(beta[j], theta[j, ], X[, j], W, j, Z)
              
              cond1 <- as.numeric(abs((t(matrix(X[, j])) %*% matrix(res_j)) / N))
              cond2 <- as.numeric(
                norm(
                  matrix(
                    S_func((t(data.frame(W[j])) %*% (res_j)) / N,
                           alpha * (lambda)
                    ),
                    ncol = 1
                  ),
                  type = "F"
                )
              )
              
              if (cond1 <= (1 - alpha) * (lambda)) {
                strong_set[j, b] <- 0
              } else {
                strong_set[j, b] <- 1
              }
              
              if (cond2 <= 2 * (1 - alpha) * (lambda)) {
                strong_set1[j, b] <- 0
              } else {
                strong_set1[j, b] <- 1
              }
              
              if (strong_set[j, b] == 0 & strong_set1[j, b] == 0) {
                next(j)
              } else {
                beta1_j <- (N / sum(B * (matrix(X[, j]) * matrix(X[, j])))) *
                  S_func((t(matrix(X[, j])) %*% matrix(res_j)) / N,
                         (1 - alpha) * lambda)
                
                cond3 <- norm(
                  matrix(
                    S_func(
                      t(data.frame(W[j])) %*% ((res_j) -
                                                 (B) * matrix(X[, j]) *
                                                 as.numeric(beta1_j)) / N,
                      alpha * lambda
                    ),
                    ncol = 1
                  ),
                  type = "F"
                )
                
                if (cond3 <= 2 * (1 - alpha) * lambda) {
                  beta[j] <- as.numeric(beta1_j)
                  active_set1[j, b] <- 1
                  next(j)
                } else {
                  value1 <- quadratic(
                    beta, theta, alpha, lambda, beta0, theta0, j, b,
                    W, X, Z, y, N, n_i, xbar, zbar,
                    t = t, r_min_j = res_j
                  )
                  beta[j] <- unlist(value1[[1]])
                  theta[j, ] <- unlist(value1[[2]])
                  t <- unlist(value1[[3]])
                  active_set1[j, b] <- 1
                  active_set2[j, b] <- 1
                  next(j)
                }
              }
            }
          } # j-loop
          
          n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
          n_i_b <- matrix(unlist(n_i[b]), N)
          E[, b] <- exp(unlist(n_i[b]))
          pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
          pr_b <- matrix(unlist(pr[b]), N)
          
          B <- pr_b * (1 - pr_b)
          B[B == 0] <- as.numeric(10e-9)
          
          y1 <- 1 * (y == b)
          M  <- (y1 - pr_b)
          A  <- n_i_b + M / B
          rv <- B * (A - n_i_b)
          
          if (i > new_t) {
            iter_current_score <- objective(rv, beta, theta, alpha, lambda)
            if (abs(iter_prev_score - iter_current_score) < tolerance) {
              break
            } else {
              next(iii)
            }
          } else {
            break
          }
        } # max_iter
        
        BETA01[b] <- list(beta0)
        BETA1[b]  <- list(beta)
        THETA01[b] <- list(matrix(theta0, 1, K))
        THETA1[b]  <- list(as.matrix(theta, p, K))
        
        BETA0[[b]][q] <- beta0
        THETA0[[b]][, q] <- theta0
        BETA[[b]][, q] <- beta
        THETA[[b]][,, q] <- theta
        
        next(b)
      } # b-loop
      
      m_for_beta <- matrix(0, p, max(y))
      for (l in 1:max(y)) {
        m_for_beta[, l] <- BETA[[l]][, q]
      }
      for (l in 1:nrow(m_for_beta)) {
        med <- mean(m_for_beta[l, ])
        m_for_beta[l, ] <- m_for_beta[l, ] - med
      }
      for (l in 1:max(y)) {
        BETA[[l]][, q] <- m_for_beta[, l]
      }
      
      for (x in 1:max(y)) {
        beta0 <- BETA0[[x]][q]
        beta  <- BETA[[x]][, q]
        theta <- as.matrix(THETA[[x]][,, q])
        theta0 <- THETA0[[x]][, q]
        n_i[x] <- list(model(beta0, theta0, beta, theta, X, Z))
      }
      
      E <- matrix(1, N, max(y))
      for (x in 1:max(y)) {
        E[, x] <- exp(unlist(n_i[x]))
      }
      
      for (x in 1:(max(y))) {
        n_i_b <- matrix(unlist(n_i[x]), N)
        E_1 <- E
        E_1[, x] <- 0
        pr[x] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
      }
      
      Dev <- matrix(0, nrow = length(y))
      Dev1 <- matrix(0, nrow = length(y))
      for (l in 1:length(Dev)) {
        deviance_y1 <- matrix(0, 1, max(y))
        deviance_y2 <- matrix(0, 1, max(y))
        for (d in 1:max(y)) {
          y_d <- 1 * (y[l] == d)
          prob_d <- matrix(unlist(pr[d]), N)
          deviance_y1[d] <- Log(y = y_d, pr = prob_d[l])
          deviance_y2[d] <- Log(y = y_d, pr = mean(1 * (y == d)))
        }
        Dev[l]  <- sum(deviance_y1)
        Dev1[l] <- sum(deviance_y2)
      }
      
      DEV1[i] <- ((2) * sum(Dev1)) / length(y)
      DEV[i]  <- ((2) * sum(Dev))  / length(y)
      if (is.null(t)) {
        t <- 0
      }
      big_t[q] <- t
      Mbeta[q] <- mbeta
      
      Y_hat[q] <- list(matrix(c(y, unlist(pr)), N, (max(y) + 1)))
      
      for (v in 1:max(y)) {
        nzero_beta <- BETA[[v]][, q]
        nzero_theta <- unlist(THETA[[v]][,, i])
        non_zero[i, v]       <- sum(nzero_beta != 0)
        non_zero_theta[i, v] <- sum(nzero_theta != 0)
      }
      
      Lambda[q] <- lambda
      print(c(q, max(non_zero[q, ]), big_t[q], DEV[q], DEV[q - 1]))
      
      my_q <- my_q + 1
      q <- q + 1
      
    } else if (q > fq) {
      ########################################################
      ## Later lambdas with active-set logic
      ########################################################
      i <- q
      lambda <- lambda_i[q]
      
      for (b in 1:max(y)) {
        lambda <- lambda_i[q]
        
        beta0 <- BETA0[[b]][q - 1]
        beta  <- BETA[[b]][, q - 1]
        v_beta <- BETA[[b]][, q - 2]
        theta <- as.matrix(THETA[[b]][,, q - 1])
        theta_transpose <- t(theta)
        v_theta <- as.matrix(THETA[[b]][,, q - 2])
        theta_transpose1 <- t(v_theta)
        theta0 <- THETA0[[b]][, q - 1]
        
        theta01 <- theta0
        beta01  <- beta0
        beta1   <- beta
        theta1  <- theta
        
        n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
        n_i_b <- matrix(unlist(n_i[b]), N)
        E[, b] <- exp(unlist(n_i[b]))
        pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
        pr_b <- matrix(unlist(pr[b]), N)
        
        B <- pr_b * (1 - pr_b)
        B[B == 0] <- as.numeric(10e-9)
        
        y1 <- 1 * (y == b)
        M  <- (y1 - pr_b)
        A  <- n_i_b + M / B
        rv <- B * (A - n_i_b)
        r  <- (A - n_i_b)
        
        v <- reg(rv, Z)
        if (b == max(y)) {
          my_beta0 <- matrix(0, (max(y) - 1))
          for (l in 1:(max(y) - 1)) {
            my_beta0[l] <- BETA0[[l]][i - 1]
          }
          beta0 <- 1 - sum(my_beta0)
          theta0 <- matrix(unlist(v[2]))
        } else {
          beta0 <- matrix(unlist(v[1]))
          theta0 <- matrix(unlist(v[2]))
        }
        
        for (v2 in 1:p) {
          if (i <= 1) {
            checkk <- 2 * lambda - lambda_i[1]
          } else {
            checkk <- 2 * lambda - lambda_i[i - 1]
          }
          
          if (as.numeric(abs((t(matrix((X[, v2]))) %*% matrix(((rv)))) ) / (N * alpha)) < checkk) {
            next(v2)
          } else if (active_set1[v2, b] == 1) {
            strong_set[v2, b]  <- 1
            strong_set1[v2, b] <- 1
            next(v2)
          } else {
            res_j <- rv + B * model_j(beta[v2], theta[v2, ], X[, v2], W, v2, Z)
            
            cond1 <- as.numeric(abs((t(matrix(X[, v2])) %*% matrix(res_j)) / N))
            cond2 <- as.numeric(
              norm(
                matrix(
                  S_func((t(data.frame(W[v2])) %*% (res_j)) / N,
                         alpha * (lambda)
                  ),
                  ncol = 1
                ),
                type = "F"
              )
            )
            
            if (cond1 <= (1 - alpha) * (lambda)) {
              strong_set[v2, b] <- 0
            } else {
              strong_set[v2, b] <- 1
            }
            
            if (cond2 <= 2 * (1 - alpha) * (lambda)) {
              strong_set1[v2, b] <- 0
            } else {
              strong_set1[v2, b] <- 1
            }
          }
        } # v2
      } # b
      
      while (my_v <= my_V) {
        if (q > mv) {
          my_V <- ms
        } else {
          my_V <- my_V
        }
        print(c(my_v, my_V))
        
        i <- q
        lambda <- lambda_i[q]
        
        for (b in 1:max(y)) {
          lambda <- lambda_i[q]
          
          beta0 <- BETA0[[b]][q - 1]
          beta  <- BETA[[b]][, q - 1]
          v_beta <- beta
          theta <- as.matrix(THETA[[b]][,, q - 1])
          theta_transpose <- t(theta)
          v_theta <- theta
          theta_transpose1 <- t(v_theta)
          theta0 <- THETA0[[b]][, q - 1]
          
          beta01 <- beta0
          beta1  <- beta
          theta1 <- theta
          theta01 <- theta
          
          for (iii in 1:max_iter) {
            n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
            n_i_b <- matrix(unlist(n_i[b]), N)
            E[, b] <- exp(unlist(n_i[b]))
            pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
            pr_b <- matrix(unlist(pr[b]), N)
            
            B <- pr_b * (1 - pr_b)
            B[B == 0] <- as.numeric(10e-9)
            
            y1 <- 1 * (y == b)
            M  <- (y1 - pr_b)
            A  <- n_i_b + M / B
            rv <- B * (A - n_i_b)
            iter_prev_score <- objective(rv, beta, theta, alpha, lambda)
            
            v <- reg(rv, Z)
            if (b == max(y)) {
              my_beta0 <- matrix(0, (max(y) - 1))
              for (l in 1:(max(y) - 1)) {
                my_beta0[l] <- BETA0[[l]][q - 1]
              }
              beta0 <- 1 - sum(my_beta0)
              theta0 <- matrix(unlist(v[2]))
            } else {
              beta0 <- matrix(unlist(v[1]))
              theta0 <- matrix(unlist(v[2]))
            }
            
            DJ_BETA <- matrix(0, 1)
            DJ_THETA <- matrix(0, 1, K)
            
            for (j in 1:p) {
              if (is.null(tt)) {
                t <- NULL
              } else {
                t <- tt[q]
              }
              
              if (active_set1[j, b] == 1 & active_set2[j, b] == 1) {
                value1 <- quadratic(
                  beta, theta, alpha, lambda, beta0, theta0, j, b,
                  W, X, Z, y, N, n_i, xbar, zbar,
                  t = t, r_min_j = res_j
                )
                beta[j]   <- unlist(value1[[1]])
                theta[j, ] <- unlist(value1[[2]])
                t         <- unlist(value1[[3]])
                
                active_set1[j, b] <- 1
                active_set2[j, b] <- 1
                next(j)
              } else if (as.numeric(abs((t(matrix(X[, j])) %*% matrix(rv))) ) < 2 * lambda - lambda_i[(i - 1)]) {
                next(j)
              } else if (strong_set[j, b] == 0 & strong_set1[j, b] == 0) {
                next(j)
              } else {
                res_j <- rv + B * model_j(beta[j], theta[j, ], X[, j], W, j, Z)
                
                beta1_j <- (N / sum(B * (matrix(X[, j]) * matrix(X[, j])))) *
                  S_func((t(matrix(X[, j])) %*% matrix(res_j)) / N,
                         (1 - alpha) * lambda)
                
                cond3 <- norm(
                  matrix(
                    S_func(
                      t(data.frame(W[j])) %*% ((res_j) -
                                                 (B) * matrix(X[, j]) *
                                                 as.numeric(beta1_j)) / N,
                      alpha * lambda
                    ),
                    ncol = 1
                  ),
                  type = "F"
                )
                
                if (cond3 <= (1 - alpha) * lambda) {
                  beta[j] <- as.numeric(beta1_j)
                  active_set1[j, b] <- 1
                  next(j)
                } else {
                  value1 <- quadratic(
                    beta, theta, alpha, lambda, beta0, theta0, j, b,
                    W, X, Z, y, N, n_i, xbar, zbar,
                    t = t, r_min_j = res_j
                  )
                  beta[j]   <- unlist(value1[[1]])
                  theta[j, ] <- unlist(value1[[2]])
                  t         <- unlist(value1[[3]])
                  
                  active_set1[j, b] <- 1
                  active_set2[j, b] <- 1
                  next(j)
                }
              }
            } # j-loop
            
            n_i[b] <- list(model(beta0, theta0, beta, theta, X, Z))
            n_i_b <- matrix(unlist(n_i[b]), N)
            E[, b] <- exp(unlist(n_i[b]))
            pr[b] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
            pr_b <- matrix(unlist(pr[b]), N)
            
            B <- pr_b * (1 - pr_b)
            B[B == 0] <- as.numeric(10e-9)
            
            y1 <- 1 * (y == b)
            M  <- (y1 - pr_b)
            A  <- n_i_b + M / B
            rv <- B * (A - n_i_b)
            
            if (i > new_t) {
              iter_current_score <- objective(rv, beta, theta, alpha, lambda)
              if (abs(iter_prev_score - iter_current_score) < tolerance) {
                break
              } else {
                next(iii)
              }
            } else {
              break
            }
          } # max_iter
          
          BETA01[b] <- list(beta0)
          BETA1[b]  <- list(beta)
          THETA01[b] <- list(matrix(theta0, 1, K))
          THETA1[b]  <- list(as.matrix(theta, p, K))
          
          BETA0[[b]][q] <- beta0
          THETA0[[b]][, q] <- theta0
          BETA[[b]][, q] <- beta
          THETA[[b]][,, q] <- theta
          
          next(b)
        } # b-loop
        
        m_for_beta <- matrix(0, p, max(y))
        for (l in 1:max(y)) {
          m_for_beta[, l] <- BETA[[l]][, q]
        }
        for (l in 1:nrow(m_for_beta)) {
          med <- mean(m_for_beta[l, ])
          m_for_beta[l, ] <- m_for_beta[l, ] - med
        }
        for (l in 1:max(y)) {
          BETA[[l]][, q] <- m_for_beta[, l]
        }
        
        for (x in 1:max(y)) {
          beta0 <- BETA0[[x]][q]
          beta  <- BETA[[x]][, q]
          theta <- as.matrix(THETA[[x]][,, q])
          theta0 <- THETA0[[x]][, q]
          n_i[x] <- list(model(beta0, theta0, beta, theta, X, Z))
        }
        
        E <- matrix(1, N, max(y))
        for (x in 1:max(y)) {
          E[, x] <- exp(unlist(n_i[x]))
        }
        
        for (x in 1:(max(y))) {
          n_i_b <- matrix(unlist(n_i[x]), N)
          E_1 <- E
          E_1[, x] <- 0
          pr[x] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
        }
        
        Dev <- matrix(0, nrow = length(y))
        Dev1 <- matrix(0, nrow = length(y))
        for (l in 1:length(Dev)) {
          deviance_y1 <- matrix(0, 1, max(y))
          deviance_y2 <- matrix(0, 1, max(y))
          for (d in 1:max(y)) {
            y_d <- 1 * (y[l] == d)
            prob_d <- matrix(unlist(pr[d]), N)
            deviance_y1[d] <- Log(y = y_d, pr = prob_d[l])
            deviance_y2[d] <- Log(y = y_d, pr = mean(1 * (y == d)))
          }
          Dev[l]  <- sum(deviance_y1)
          Dev1[l] <- sum(deviance_y2)
        }
        
        DEV1[i] <- ((2) * sum(Dev1)) / length(y)
        DEV[i]  <- ((2) * sum(Dev))  / length(y)
        
        if (is.null(t)) {
          t <- 0
        }
        big_t[q] <- t
        Mbeta[q] <- mbeta
        
        Y_hat[q] <- list(matrix(c(y, unlist(pr)), N, (max(y) + 1)))
        
        for (v in 1:max(y)) {
          nzero_beta <- BETA[[v]][, q]
          nzero_theta <- unlist(THETA[[v]][,, i])
          non_zero[i, v]       <- sum(nzero_beta != 0)
          non_zero_theta[i, v] <- sum(nzero_theta != 0)
        }
        
        Lambda[q] <- lambda
        print(c(q, max(non_zero[q, ]), big_t[q], DEV[q], DEV[q - 1]))
        
        my_q <- my_q + 1
        my_v <- my_v + 1
        q <- q + 1
        print(q)
        if (q > nlambda) {
          break
        }
      } # inner my_v loop
    } # q > fq
    
    my_v <- sv
    my_V <- for_v
    q <- q
    
    sumact <- matrix(0, max(y))
    for (nn in 1:max(y)) {
      act  <- (active_set1[, nn])
      sact <- (sec_active1[, nn])
      if (all(act == sact)) {
        sumact[nn] <- 1
      }
    }
    
    tolerance <- tol
    if (q > st) {
      if (max(sumact) >= 1 | q > nlambda) {
        my_ok <- 1
        break
      } else if (sum(sumact) < 1 & q <= nlambda) {
        sec_active1 <- active_set1
        sec_active2 <- active_set2
        my_ok <- 0
      } else {
        my_ok <- 0
      }
    } else if (q > nlambda) {
      my_ok <- 1
    } else {
      my_ok <- 0
    }
  } # while(my_ok / q)
  
  dev_percent <- matrix(0, nlambda)
  for (i in 1:nlambda) {
    dev_percent[i] <- 1 - ((DEV[i]) / DEV1[i])
  }
  
  nzero <- matrix(0, nrow = nlambda)
  nzero_int <- matrix(0, nrow = nlambda)
  for (s in 1:nlambda) {
    nzero_int[s] <- max(non_zero_theta[s, ])
    nzero[s]     <- max(non_zero[s, ])
  }
  
  pred <- data.frame(
    Lambda   = matrix(Lambda, ncol = 1),
    nzero    = nzero,
    nzero_int = nzero_int,
    DEV      = matrix(DEV,  ncol = 1),
    nullDEV  = matrix(DEV1, ncol = 1),
    Dev_rat  = dev_percent
  )
  
  return(list(
    beta0    = BETA0,
    beta     = BETA,
    theta0   = THETA0,
    theta    = THETA,
    y_hat    = Y_hat,
    path     = pred,
    Lambdas  = Lambda,
    big_T    = big_t,
    Mbeta    = Mbeta,
    non_zero = non_zero
  ))
}

############################################################
##  Prediction function (your original version, unchanged
##  w.r.t theta dimensions)
############################################################

predict_lasso <- function(object, X, Z, y, lambda = NULL) {
  
  lambda.arg <- lambda
  
  if (is.null(lambda.arg)) {
    lambda <- object$Lambdas
    isel <- 1:length(lambda)
  }
  
  if (!is.null(lambda.arg)) {
    isel <- as.numeric(knn1(
      matrix(object$Lambdas, ncol = 1),
      matrix(lambda.arg, ncol = 1),
      1:length(object$Lambdas)
    ))
  }
  
  N <- nrow(X)
  p <- ncol(X)
  K <- ncol(Z)
  
  yh <- array(0, length(isel))
  DEV <- matrix(NA, length(isel))
  
  pBETA0 <- lapply(
    seq_len(max(y)),
    function(j) (matrix(0, nrow = length(isel)))
  )
  pBETA <- lapply(
    seq_len(max(y)),
    function(j) (matrix(0, nrow = p, ncol = length(isel)))
  )
  pTHETA0 <- lapply(
    seq_len(max(y)),
    function(j) (matrix(0, nrow = K, ncol = length(isel)))
  )
  pTHETA <- lapply(
    seq_len(max(y)),
    function(j) (array(0, c(p, K, length(isel))))
  )
  
  iii <- 0
  for (m in isel) {
    iii <- iii + 1
    
    z <- m
    n_i <- lapply(
      seq_len(max(y)),
      function(j) (matrix(0, nrow = N))
    )
    pr <- lapply(
      seq_len(max(y)),
      function(j) (matrix(0, nrow = N))
    )
    
    for (x in 1:max(y)) {
      beta0 <- object$beta0[[x]][z]
      beta  <- object$beta[[x]][, z]
      theta <- as.matrix(object$theta[[x]][,, z])
      theta_transpose <- t(theta)
      theta0 <- object$theta0[[x]][, z]
      
      pBETA0[[x]][iii]  <- beta0
      pBETA[[x]][, iii] <- beta
      pTHETA[[x]][,, iii] <- theta
      pTHETA0[[x]][, iii] <- theta0
      
      n_i[x] <- list(model(beta0, theta0, beta, theta, X, Z))
    }
    
    E <- matrix(0, N, max(y))
    for (x in 1:max(y)) {
      E[, x] <- exp(unlist(n_i[x]))
    }
    for (x in 1:max(y)) {
      n_i_b <- matrix(unlist(n_i[x]), N)
      pr[x] <- list(matrix(as.numeric(exp(n_i_b) / (rowSums(E))), N))
    }
    
    Dev <- matrix(0, nrow = length(y))
    for (l in 1:length(Dev)) {
      deviance_y1 <- matrix(0, 1, max(y))
      for (d in 1:max(y)) {
        y_d <- 1 * (y[l] == d)
        prob_d <- matrix(unlist(pr[d]), N)
        deviance_y1[d] <- Log(y = y_d, pr = prob_d[l])
      }
      Dev[l] <- sum(deviance_y1)
    }
    
    DEV[iii] <- ((2) * sum(Dev)) / length(y)
    
    yh[iii] <- list(matrix(unlist(pr), N, max(y)))
  }
  
  return(list(
    y_hat  = yh,
    beta0  = pBETA0,
    beta   = pBETA,
    theta0 = pTHETA0,
    theta  = pTHETA,
    deviance = DEV
  ))
}
