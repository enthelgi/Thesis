import numpy as np
import matplotlib.pyplot as plt


r = np.array([1, -2, 1])  
W_j = np.array([[0.6, -0.1],
                [-0.2, 1.2],
                [0.4, -0.3]])  

N = 3
alpha = 0.6
lambda_val = 1.0
inner_thresh = alpha * lambda_val
outer_thresh = (2 * (1 - alpha) * lambda_val)**2

# Step 1: Compute W_j^T r / N
v = (W_j.T @ r) / N

# Step 2: Element-wise soft-thresholding
def soft_thresh(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

v_shrunk = soft_thresh(v, inner_thresh)

# Step 3: Norm of the shrunk vector
shrunk_norm = np.linalg.norm(v_shrunk, 2)

# Create visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.axhline(alpha, color='green', lw=1.5, linestyle=':', alpha=0.7)
ax.axhline(-alpha, color='green', lw=1.5, linestyle=':', alpha=0.7)
ax.axvline(alpha, color='green', lw=1.5, linestyle=':', alpha=0.7)
ax.axvline(-alpha, color='green', lw=1.5, linestyle=':', alpha=0.7)
ax.set_aspect('equal')

# Plot original vector (before soft-thresholding)
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Signal Vector')

# Plot soft-thresholded vector
ax.quiver(0, 0, v_shrunk[0], v_shrunk[1], angles='xy', scale_units='xy', scale=1, color='green', label='After Inner Soft-Thresholding')

# Outer threshold circle
circle = plt.Circle((0, 0), outer_thresh, color='red', fill=False, linestyle='--', label=r'$x^2+y^2 = (2(1-\alpha)\lambda)^2$')
ax.add_patch(circle)

# Formatting
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Nested Soft-Thresholding ")
ax.set_xlabel("coordinate 1")
ax.set_ylabel("coordinate 2")
ax.plot([], [], color='green', linestyle=':', lw=1.5, label=r'$\pm\alpha \lambda$ threshold')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display values 
v, v_shrunk, shrunk_norm, outer_thresh
