import numpy as np
import matplotlib.pyplot as plt

# Define an open range of alpha values between 0 and 1 
alpha_vals = np.linspace(0.01, 0.99, 500)

# Define f(alpha) = max(alpha, 2(1 - alpha))
f_alpha = np.maximum(alpha_vals, 2 * (1 - alpha_vals))

# Critical lambda = 1 / f(alpha)
lambda_threshold = 1 / f_alpha

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alpha_vals, lambda_threshold, label=r'$\lambda > \frac{1}{\max(\alpha,\ 2(1 - \alpha))}$', color='purple', linewidth=2)
plt.axhline(1, linestyle='--', color='gray', label=r'$\lambda = 1$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Minimum $\lambda$ needed')
plt.title(r'Critical $\lambda$ Threshold to Make At Least One Interval Wider Than $[-1, 1]$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
