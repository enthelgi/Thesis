
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic data for x1, x2, and y
np.random.seed(0)
n=100
x1 = np.random.uniform(-10, 25, n)
x2 = np.random.uniform(-50, 25, n)
b0, b1, b2 = 1, 7, 7.5  # true coefficients
y_true = b0 + b1 * x1 + b2 * x2 + np.random.normal(0, 2, n)  # Add some noise

# Create a meshgrid for b1 and b2 to evaluate RSS
b1_values = np.linspace(-10, 20, 100)
b2_values = np.linspace(-2.5, 15, 100)
b1_grid, b2_grid = np.meshgrid(b1_values, b2_values)

# Compute the RSS for each pair of b1, b2
rss_grid = np.zeros_like(b1_grid)
for i in range(len(b1_values)):
    for j in range(len(b2_values)):
        y_pred = b0 + b1_grid[i, j] * x1 + b2_grid[i, j] * x2
        rss_grid[i, j] = np.sum((y_true - y_pred) ** 2)

# Create the combined plots
fig = plt.figure(figsize=(14, 6))

# 3D RSS Surface Plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(b1_grid, b2_grid, rss_grid, cmap='viridis',rstride=2, cstride=2)
ax1.set_xlabel('b1')
ax1.set_ylabel('b2')
ax1.set_zlabel('RSS')
ax1.set_title('3D Surface Plot of RSS')

# 2D Contour Plot of RSS
levels = [rss_grid.min(),50000.1315514815015, 700000.1315514815015, 1150000.1315514815015, 2190000.1315514815015, rss_grid.max()]
print(rss_grid.min(),rss_grid.max())
ax2 = fig.add_subplot(132)
contour = ax2.contour(b1_grid, b2_grid, rss_grid, levels=levels, cmap='viridis')
ax2.set_xlabel('b1')
ax2.set_ylabel('b2')
ax2.set_title('2D Contour Plot of RSS')
fig.colorbar(contour, ax=ax2, orientation='vertical', label='RSS')

# Overlay of 2D curves with LASSO
ax3 = fig.add_subplot(133)
ax3.set_title('LASSO with Elliptic Curves')
ax3.set_xlabel('b1')
ax3.set_ylabel('b2')

ax3.contour(b1_grid, b2_grid, rss_grid, levels=levels, cmap='viridis')

# Add LASSO constraint
ax3.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], color='red', linewidth=2, label='LASSO Constraint')
ax3.legend()

plt.tight_layout()
plt.show()

