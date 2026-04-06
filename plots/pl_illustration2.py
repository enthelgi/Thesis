import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# --- Parameters ---
np.random.seed(0)
n_vectors = 200
K = 3
alphas = [0.2, 0.5, 0.8]
lambdas = [0.5, 1.0, 2.0]

# Simulated vectors
Wjr_residuals = np.random.randn(n_vectors, K)
theta_vec_id = 42  # fixed example vector
cor_vec = Wjr_residuals[theta_vec_id]

# Soft-thresholding function
def soft_threshold(vec, thresh):
    return np.sign(vec) * np.maximum(np.abs(vec) - thresh, 0)

# Label vertical positions (staggered)
y_positions = {
    r'$-\alpha\lambda$': 0.63,
    r'$-2(1-\alpha)\lambda$': 0.61,
    r'$2(1-\alpha)\lambda$': 0.59,
    r'$\alpha\lambda$': 0.57,
    r'vec.norm': 0.55
}

# --- Plotting setup ---
fig, axes = plt.subplots(len(alphas), len(lambdas), figsize=(18, 10), sharex=False, sharey=False)

for i, alpha in enumerate(alphas):
    for j, lambd in enumerate(lambdas):
        ax = axes[i, j]
        
        inner_thresh = round(alpha * lambd, 2)
        outer_thresh = round(2 * (1 - alpha) * lambd, 2)
        vec_norm = round(norm(soft_threshold(cor_vec, inner_thresh), 2), 2)
        survived = vec_norm > outer_thresh
        color = 'green' if survived else 'red'

        # All positions
        points = {
            r'$-\alpha\lambda$': -inner_thresh,
            r'$-2(1-\alpha)\lambda$': -outer_thresh,
            r'$2(1-\alpha)\lambda$': outer_thresh,
            r'$\alpha\lambda$': inner_thresh,
            r'vec.norm': vec_norm
        }

        all_x = list(points.values())
        x_min, x_max = min(all_x), max(all_x)
        buffer = 0.4 * (x_max - x_min)

        # Baseline
        ax.plot([x_min - buffer, x_max + buffer], [0.5, 0.5], color='black', linewidth=1)

        # Plot bullets, arrows, and labels
        for label, x in points.items():
            dot_color = color if label == r'vec.norm' else ('blue' if 'alpha' in label else 'purple')
            ax.plot(x, 0.5, 'o', color=dot_color, markersize=10)
            ax.text(x, y_positions[label] + 0.005, f"{x:.2f}", fontsize=9, ha='center', va='bottom')
            ax.text(x, y_positions[label] - 0.005, label, fontsize=9, ha='center', va='top', rotation=0)
            ax.annotate('', xy=(x, 0.5), xytext=(x, y_positions[label] - 0.01),
                        arrowprops=dict(arrowstyle='->', lw=1))

        ax.set_xlim(x_min - buffer, x_max + buffer)
        # Adjust y-limits to fit extreme thresholds
        ax.set_ylim(0.48, 0.67 + 0.01 * (abs(x_max) > 3))
        ax.set_title(rf'$\alpha={alpha},\ \lambda={lambd}$')
        ax.axis('off')

plt.suptitle("Visualizing Thresholds and Correlation Norm for various α,λ", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
