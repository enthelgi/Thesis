##############################################
# iFORM: Interaction Screening (Linear + Logistic)
# Original linear iFORM by Kirk Gosik (Hao & Zhang; Gosik & Wu)
# Extended here with a logistic-regression version (iForm_logistic)
##############################################


#' Interaction Screening for Ultra-High Dimensional Data (Linear)
#'
#' Extended variable selection approaches to jointly model main and interaction
#' effects from high-dimensional data originally proposed by Hao and Zhang (2014)
#' and extended by Gosik and Wu (2016). Based on a greedy forward approach,
#' their model can identify all possible interaction effects through two algorithms,
#' iFORT and iFORM, which have been proved to possess sure screening property in
#' an ultrahigh-dimensional setting.
#'
#' @name iForm
#' @param formula an object of class formula: a symbolic description of the model
#'   to be fitted.
#' @param data data.frame of your data with the response and all p predictors
#' @param heredity a string specifying the heredity to be considered:
#'   \code{"none"}, \code{"weak"}, or \code{"strong"}.
#' @param higher_order logical; \code{TRUE} to include order-3 interactions
#'   in the search (default \code{FALSE}).
#' @return a linear model (\code{lm}) of the final selected model
#' @examples
#' \dontrun{
#'   iForm(formula = hp ~ ., data = mtcars, heredity = "strong",
#'         higher_order = FALSE)
#' }
#' @author Kirk Gosik
#' @details
#' Runs the iFORM selection procedure on the dataset and returns a linear model
#' of the final selected model.  The model is an R object of class "lm".
#' @seealso \code{lm}, \code{model.frame}
#' @export
#' @importFrom stats lm as.formula model.frame
iForm <- function(formula,
                  data,
                  heredity = "strong",
                  higher_order = FALSE) {

  dat <- model.frame(formula, data)

  y <- dat[, 1]
  x <- dat[, -1, drop = FALSE]
  p <- ncol(x)
  n <- nrow(x)
  C <- names(x)
  S <- NULL
  M <- NULL
  bic <- NULL

  fit <- iformselect(x, y, p, n, C, S, bic, heredity, higher_order)

  y   <- fit$y
  S   <- fit$S
  bic <- fit$bic

  model_formula <- as.formula(
    paste("y ~ 0 +", paste(S[1:which.min(bic)], collapse = "+"))
  )

  lm(model_formula, data = x)
}



#' Selection for iFORM procedure (Linear)
#'
#' Helper function to run the selection procedure under different heredity
#' principles and different levels of interactions included in the selection,
#' for a linear (Gaussian) response.
#'
#' @name iformselect
#' @param x data.frame or matrix of predictors
#' @param y vector of observed responses
#' @param p number of predictors
#' @param n number of observations
#' @param C vector of candidate predictors to consider in this step
#' @param S vector of solution predictors selected in previous steps
#' @param bic vector of BIC-like values calculated at each step
#' @param heredity a string specifying heredity: \code{"none"},
#'   \code{"weak"}, or \code{"strong"}.
#' @param higher_order logical; include order-3 interactions (default FALSE)
#' @return list with elements \code{y}, \code{S}, and \code{bic}
#' @author Kirk Gosik
#' @details
#' Runs the iFORM selection procedure for specified heredity and level of
#' interactions. It returns the solution to be fit from \code{iForm}.
#' @export
iformselect <- function(x, y, p, n, C, S, bic, heredity, higher_order) {

  repeat {

    RSS <- rss_map_func(C = C, S = S, y = y, data = x)

    S <- c(S, C[which.min(unlist(RSS))])
    C <- C[-which.min(unlist(RSS))]

    order2 <- switch(
      heredity,
      `none`   = NULL,
      `strong` = strong_order2(S = S, data = x),
      `weak`   = weak_order2(S = S, C = C, data = x)
    )

    C <- union(C, order2)

    if (higher_order) {

      order3 <- switch(
        heredity,
        `strong` = strong_order3(S = S, data = x),
        `weak`   = weak_order3(S = S, C = C, data = x)
      )

      C <- union(C, order3)
    }

    bic_val <- log(min(unlist(RSS)) / n) +
      length(S) * (log(n) + 2 * log(p)) / n

    bic <- append(bic, bic_val)

    if (length(bic) > n / log(n)) break
  }

  list(y = y, S = S, bic = bic)
}



#' Finding minimum RSS (Linear)
#'
#' Helper function to calculate the residual sum of squares for each candidate
#' predictor, given what has already been selected.
#'
#' @name rss_map_func
#' @param C vector of candidate predictors
#' @param S vector of solution predictors already selected
#' @param y vector of observed responses
#' @param data data.frame or matrix of predictors
#' @return a numeric vector of RSS values for each candidate predictor
#' @author Kirk Gosik
#' @details
#' Mapping function to calculate the residual sum of squares for each candidate
#' predictor.
#' @export
#' @importFrom stats model.matrix as.formula
rss_map_func <- function(C, S, y, data) {

  sapply(C, function(candidates) {

    var_names <- c(S, candidates)

    X <- model.matrix(
      as.formula(paste("~ 0 +", paste(var_names, collapse = "+"))),
      data = data
    )

    tryCatch({
      # OLS closed form
      beta_hat <- solve(t(X) %*% X, t(X) %*% y)
      sum((y - X %*% beta_hat) ^ 2)
    }, error = function(e) Inf)
  })
}



## ===========================
## Heredity Selection Helpers
## ===========================


#' Creating interactions under strong heredity (order 2)
#'
#' Finds all pairwise interactions among main effects currently in the
#' solution set, following the strong heredity principle.
#'
#' @name strong_order2
#' @param S vector of solution predictors selected so far
#' @param data data.frame or matrix of predictors
#' @return character vector of interaction term names
#' @author Kirk Gosik
#' @details
#' Finds all p choose 2 combinations of predictors in the solution set.
#' @seealso \code{combn}
#' @export
#' @importFrom utils combn
strong_order2 <- function(S, data) {

  tryCatch({
    main_effects <- sort(S[S %in% names(data)])
    combn(main_effects, 2, paste0, collapse = ":")
  }, error = function(e) NULL)
}



#' Creating interactions under weak heredity (order 2)
#'
#' Finds all possible pairwise interactions between the main effects already
#' selected and the remaining candidate main effects.
#'
#' @name weak_order2
#' @param S vector of solution predictors
#' @param C vector of candidate predictors
#' @param data data.frame or matrix of predictors
#' @return character vector of interaction term names
#' @author Kirk Gosik
#' @details
#' Finds all combinations between predictors in the solution set and predictors
#' in the candidate set.
#' @export
weak_order2 <- function(S, C, data) {

  tryCatch({
    main_effects <- sort(S[S %in% names(data)])
    as.vector(outer(main_effects, C[C %in% names(data)], paste, sep = ":"))
  }, error = function(e) NULL)
}



#' Creating interactions under strong heredity (order 3)
#'
#' Finds all third-order interactions among main effects currently in the
#' solution set, following the strong heredity principle.
#'
#' @name strong_order3
#' @param S vector of solution predictors
#' @param data data.frame or matrix of predictors
#' @return character vector of 3-way interaction term names
#' @author Kirk Gosik
#' @details
#' Finds all p choose 3 combinations of predictors in the solution set.
#' @export
#' @importFrom utils combn
strong_order3 <- function(S, data) {

  tryCatch({
    main_effects <- sort(S[S %in% names(data)])
    combn(main_effects, 3, paste0, collapse = ":")
  }, error = function(e) NULL)
}



#' Creating interactions under weak heredity (order 3)
#'
#' Finds all third-order interactions between existing 2-way interactions
#' in the solution set and the remaining candidate main effects, then
#' ensures canonical ordering.
#'
#' @name weak_order3
#' @param S vector of solution predictors
#' @param C vector of candidate predictors
#' @param data data.frame or matrix of predictors
#' @return character vector of 3-way interaction term names
#' @author Kirk Gosik
#' @details
#' Finds all p choose 3 combinations between predictors in the solution set
#' and predictors in the candidate set.
#' @export
weak_order3 <- function(S, C, data) {

  tryCatch({

    interaction_effects <- unlist(
      Map(
        function(int_term) paste0(int_term, collapse = ":"),
        Filter(function(vec) { length(vec) == 2 },
               strsplit(S, "[.]|[:]"))
      )
    )

    weak_three <- as.vector(
      outer(interaction_effects, C[C %in% names(data)], paste, sep = ":")
    )

    as.vector(
      unlist(
        Map(
          paste0, collapse = ":",
          Map(sort, strsplit(weak_three, ":"))
        )
      )
    )
  }, error = function(e) NULL)
}



## =========================================
## Logistic Regression Extension of iFORM
## =========================================


#' Interaction Screening for Ultra-High Dimensional Data (Logistic)
#'
#' Logistic-regression version of the iFORM procedure. Performs greedy forward
#' selection (with optional strong/weak heredity and higher-order interactions)
#' for a binary response.
#'
#' @name iForm_logistic
#' @param formula an object of class \code{formula}: a symbolic description
#'   of the model to be fitted. The response must be binary (0/1 or a 2-level
#'   factor).
#' @param data data.frame of your data with the response and all p predictors
#' @param heredity a string specifying the heredity to be considered:
#'   \code{"none"}, \code{"weak"}, or \code{"strong"}.
#' @param higher_order logical; \code{TRUE} to include order-3 interactions
#'   in the search (default \code{FALSE}).
#' @return a \code{glm} object (family = binomial) of the final selected model
#' @examples
#' \dontrun{
#'   iForm_logistic(formula = y ~ ., data = mydata,
#'                  heredity = "strong", higher_order = FALSE)
#' }
#' @author Adapted from Kirk Gosik (linear iForm) for logistic regression
#' @details
#' Runs the iFORM selection procedure on the dataset and returns a logistic
#' regression model (\code{glm} with \code{family = binomial}) of the final
#' selected model.
#' @seealso \code{glm}, \code{model.frame}
#' @export
#' @importFrom stats glm model.frame as.formula binomial
iForm_logistic <- function(formula,
                           data,
                           heredity = "strong",
                           higher_order = FALSE) {

  dat <- model.frame(formula, data)

  y <- dat[, 1]
  x <- dat[, -1, drop = FALSE]
  p <- ncol(x)
  n <- nrow(x)
  C <- names(x)
  S <- NULL
  bic <- NULL

  ## Ensure binary numeric response for glm.fit
  if (is.factor(y)) {
    if (nlevels(y) != 2L) {
      stop("Response must be binary (2 levels) for logistic iFORM.")
    }
    y <- as.numeric(y == levels(y)[2L])
  } else {
    uy <- sort(unique(y))
    if (length(uy) != 2L || !all(uy %in% c(0, 1))) {
      stop("Numeric response must be coded as 0/1 for logistic iFORM.")
    }
  }

  fit <- iformselect_logistic(x, y, p, n, C, S, bic, heredity, higher_order)

  y   <- fit$y
  S   <- fit$S
  bic <- fit$bic

  best_size <- which.min(bic)

  model_formula <- as.formula(
    paste("y ~ 0 +", paste(S[1:best_size], collapse = "+"))
  )

  glm(model_formula, data = x, family = binomial())
}



#' Selection for iFORM procedure (Logistic)
#'
#' Helper function to run the logistic iFORM selection procedure under
#' different heredity principles and different levels of interactions.
#'
#' @name iformselect_logistic
#' @param x data.frame or matrix of predictors
#' @param y numeric binary response (0/1)
#' @param p number of predictors
#' @param n number of observations
#' @param C vector of candidate predictors
#' @param S vector of solution predictors selected so far
#' @param bic vector of BIC-like values calculated at each step
#' @param heredity a string specifying heredity: \code{"none"},
#'   \code{"weak"}, or \code{"strong"}.
#' @param higher_order logical; include order-3 interactions in the search
#' @return list with elements \code{y}, \code{S}, and \code{bic}
#' @author Adapted from Kirk Gosik
#' @export
iformselect_logistic <- function(x, y, p, n, C, S, bic, heredity, higher_order) {

  repeat {

    DEV <- deviance_map_func(C = C, S = S, y = y, data = x)

    chosen_idx <- which.min(unlist(DEV))
    S <- c(S, C[chosen_idx])
    C <- C[-chosen_idx]

    order2 <- switch(
      heredity,
      `none`   = NULL,
      `strong` = strong_order2(S = S, data = x),
      `weak`   = weak_order2(S = S, C = C, data = x)
    )

    C <- union(C, order2)

    if (higher_order) {

      order3 <- switch(
        heredity,
        `strong` = strong_order3(S = S, data = x),
        `weak`   = weak_order3(S = S, C = C, data = x)
      )

      C <- union(C, order3)
    }

    ## EBIC-style criterion using deviance instead of RSS
    dev_min <- min(unlist(DEV))
    bic_val <- dev_min / n + length(S) * (log(n) + 2 * log(p)) / n

    bic <- append(bic, bic_val)

    if (length(bic) > n / log(n)) break
  }

  list(y = y, S = S, bic = bic)
}



#' Finding minimum deviance (Logistic)
#'
#' Helper function to calculate the model deviance for each candidate
#' predictor, given the variables already selected in the solution set.
#'
#' @name deviance_map_func
#' @param C vector of candidate predictors
#' @param S vector of solution predictors already selected
#' @param y numeric binary response (0/1)
#' @param data data.frame or matrix of predictors
#' @return a numeric vector of deviance values for each candidate predictor
#' @author Adapted from Kirk Gosik
#' @details
#' Mapping function to calculate the deviance (from logistic regression) for
#' each candidate predictor.
#' @export
#' @importFrom stats model.matrix as.formula glm.fit binomial
deviance_map_func <- function(C, S, y, data) {

  sapply(C, function(candidates) {

    var_names <- c(S, candidates)

    X <- model.matrix(
      as.formula(paste("~ 0 +", paste(var_names, collapse = "+"))),
      data = data
    )

    fit <- try(
      glm.fit(x = X, y = y, family = binomial()),
      silent = TRUE
    )

    if (inherits(fit, "try-error") ||
        is.null(fit$deviance) ||
        !isTRUE(fit$converged)) {
      Inf
    } else {
      fit$deviance
    }
  })
}
