import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Parameters
np.random.seed(0)
n_vectors = 200  # number of simulated residual correlation vectors
K = 3  # dimension of theta_j (modifying variables)
alphas = [0.2, 0.5, 0.8]
lambdas = [0.5, 1.0, 2.0]

# Simulate random residual correlations (first argument of soft-thresholding)
Wjr_residuals = np.random.randn(n_vectors, K)
Wjr_residuals = (Wjr_residuals - Wjr_residuals.mean(axis=0)) / Wjr_residuals.std(axis=0)

# Soft-thresholding function
def soft_threshold(vec, thresh):
    return np.sign(vec) * np.maximum(np.abs(vec) - thresh, 0)

# Create grid of subplots
fig, axes = plt.subplots(len(alphas), len(lambdas), figsize=(18, 10), sharex=True, sharey=True)

# Loop over combinations of alpha and lambda
for i, alpha in enumerate(alphas):
    for j, lambd in enumerate(lambdas):
        ax = axes[i, j]
        inner_thresh = alpha * lambd
        outer_thresh = 2 * (1 - alpha) * lambd

        norms = []
        for vec in Wjr_residuals:
            shrunk = soft_threshold(vec, inner_thresh)
            vec_norm = norm(shrunk, 2)
            norms.append(vec_norm)

        # Plot L2 norms after inner soft-thresholding
        ax.scatter(np.arange(len(norms)), norms, color='green', alpha=0.6, label='L2 Norm of Shrunk Vector', marker='x')
        ax.axhline(y=outer_thresh, color='red', linestyle='--', label=r'$2(1 - \alpha)\lambda$')

        ax.set_title(rf'$\alpha={alpha},\ \lambda={lambd}$')
        if j == 0:
            ax.set_ylabel("L2 Norm after Inner Shrinkage")
        if i == len(alphas) - 1:
            ax.set_xlabel("Vector Index")

        # Show legend only once
        if i == 0 and j == len(lambdas) - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.suptitle("Double Soft-Thresholding Across $\\alpha$ and $\\lambda$", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
