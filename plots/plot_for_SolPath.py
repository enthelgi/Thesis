import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

# Optional: Change matplotlib style
plt.style.use('ggplot')

def costfunction(X, y, theta):
    '''OLS cost function'''
    m = np.size(y)
    h = X @ theta
    # Extract the scalar from the 1x1 result
    J = ((1. / (2 * m)) * (h - y).T @ (h - y))[0, 0]
    return J

def closed_form_solution(X, y):
    '''Linear regression closed form solution'''
    return np.linalg.inv(X.T @ X) @ X.T @ y

def cost_l1(x, y):
    '''L1 cost function'''
    return np.abs(x) + np.abs(y)

# ------------------- Data Generation -------------------
# Generate 40 equally spaced points in [0,1]
x = np.linspace(0, 1, 40)

# Generate noise from a uniform distribution
noise = np.random.uniform(size=40)

# True signal: sin(1.5π x)
y = np.sin(x * 1.5 * np.pi)

# Create noisy observations and reshape as a column vector
y_noise = (y + noise).reshape(-1, 1)

# Center the response by subtracting its mean
y_noise = y_noise - y_noise.mean()

# ------------------- Design Matrix -------------------
# Construct design matrix with two predictors: 2*x and x^2
X = np.vstack((2 * x, x**2)).T

# Normalize each column of the design matrix (using the L2 norm)
X = X / np.linalg.norm(X, axis=0)

# ------------------- Lambda Grid via Leave-One-Out Residuals -------------------
n, p = X.shape  # n: number of samples, p: number of predictors
'''
# For each predictor j, compute the leave-one-out residual.
lambda_candidates = []
for j in range(p):
    # Get the indices of the predictors except j
    idx = [k for k in range(p) if k != j]
    X_minus_j = X[:, idx]  # design matrix excluding j-th column

    # Fit the regression using all predictors except j.
    # We use lstsq to handle the case when p-1 = 1.
    theta_minus_j, _, _, _ = np.linalg.lstsq(X_minus_j, y_noise, rcond=None)
    
    # Compute the leave-one-out residual:
    r_minus_j = y_noise - X_minus_j @ theta_minus_j
    
    # Compute the correlation (inner product) between the j-th predictor and r(-j)
    # Since r_minus_j is n x 1 and X[:, j] is n x 1, the inner product is a scalar.
    lambda_j = np.abs(X[:, j].T @ r_minus_j) / n
    lambda_candidates.append(lambda_j)
'''
# Use the maximum value as lambda_max
#lambda_max = np.max(lambda_candidates)
lambda_max = np.max(np.abs(X.T @ y_noise)) / n

# Choose epsilon (a small fraction, e.g., 0.001) and set lambda_min accordingly.
epsilon = 0.0001
lambda_min = epsilon * lambda_max

# Create a grid of 100 logarithmically spaced lambda values between lambda_max and lambda_min.
lambda_range = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num=100)

# ------------------- Lasso Regression over Lambda Grid -------------------
theta_0_list_reg_l1 = []
theta_1_list_reg_l1 = []

for l in lambda_range:
    # Initialize and fit Lasso (without intercept)
    model_sk_reg = linear_model.Lasso(alpha=l, fit_intercept=False, max_iter=10000)
    model_sk_reg.fit(X, y_noise)
    t0, t1 = model_sk_reg.coef_
    theta_0_list_reg_l1.append(t0)
    theta_1_list_reg_l1.append(t1)

# ------------------- Meshgrid for Contours -------------------
# Create a grid over the coefficient space for contour plots
xx, yy = np.meshgrid(np.linspace(-2, 17, 100), np.linspace(-17, 3, 100))

# (Optional) Compute cost function values on the grid
Z_l1 = np.array([cost_l1(xi, yi) for xi, yi in zip(np.ravel(xx), np.ravel(yy))]).reshape(xx.shape)
Z_ls = np.array([costfunction(X, y_noise.reshape(-1, 1), 
                               np.array([t0, t1]).reshape(-1, 1)) 
                 for t0, t1 in zip(np.ravel(xx), np.ravel(yy))]).reshape(xx.shape)

# ------------------- Plotting -------------------
# Plot 1: Contour plot with the Lasso solution path in coefficient space
fig, ax = plt.subplots(figsize=(8, 7))

# Plot contours for L1 (diamond-shaped) and RSS (elliptical) cost functions
contour1 = ax.contour(xx, yy, Z_l1, levels=[.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
                      cmap='gist_gray')
contour2 = ax.contour(xx, yy, Z_ls, levels=[.01, .06, .09, .11, .15],
                      cmap='coolwarm')

ax.clabel(contour1, inline=True, fontsize=8, fmt="L1")
ax.clabel(contour2, inline=True, fontsize=8, fmt="RSS")

ax.set_xlabel(r'$\beta_1$')
ax.set_ylabel(r'$\beta_2$')
ax.set_title('Lasso solution as a function of $\lambda$: RSS and L1 contours')

# Compute the least squares solution (which corresponds to λ = 0)
min_ls = np.linalg.inv(X.T @ X) @ X.T @ y_noise
ax.plot(min_ls[0], min_ls[1], marker='x', color='red', markersize=10, label='Least Squares Min')

# Plot the Lasso solution path (as points connected by a line)
ax.plot(theta_0_list_reg_l1, theta_1_list_reg_l1, linestyle='-', marker='o', color='red', alpha=0.7, label='Lasso Path')

ax.legend()
plt.show()

# Plot 2: Solution path (coefficient values vs. lambda)
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.semilogx(lambda_range, theta_0_list_reg_l1,marker='o', label=r'$\beta_1$', color='green')
ax2.semilogx(lambda_range, theta_1_list_reg_l1,marker='o', label=r'$\beta_2$', color='brown')

ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel('Coefficient value')
ax2.set_title('Lasso Solution Path')
ax2.legend()
plt.show()

