import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the coefficients
b0 = 2  # intercept
b1 = 3  # coefficient of x1
b2 = 4  # coefficient of x2

# Create a meshgrid for x1 and x2
x1 = np.linspace(-10, 10, 100)  # Range for x1
x2 = np.linspace(-10, 10, 100)  # Range for x2
x1_grid, x2_grid = np.meshgrid(x1, x2)

# Calculate y based on the equation y = b0 + b1*x1 + b2*x2
y_grid = b0 + b1 * x1_grid + b2 * x2_grid

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x1_grid, x2_grid, y_grid, cmap='viridis')

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D Surface Plot of y = b0 + b1*x1 + b2*x2')

# Show the plot
plt.show()


np.random.seed(0)
x1 = np.random.uniform(-1, 1, 100)
x2 = np.random.uniform(-1, 1, 100)
# Create the design matrix
X = np.column_stack((np.ones_like(x1), x1, x2))
# Perform OLS estimation: b = (X^T X)^(-1) X^T y

true_beta = np.array([1.0, 2.0, -1.5])
noise = np.random.normal(0, 0.1, size=x1.shape)
#y = function_3d(true_beta, x1, x2) + noise

def function_3d(beta, x1, x2):
    #return beta[0] + beta[1] * x1 + beta[2] * x2
    X = np.column_stack((np.ones_like(x1), x1, x2))
    return X @ beta

y = function_3d(true_beta, x1, x2) + noise
beta = np.linalg.inv(X.T @ X) @ X.T @ y
def ols_estimation(x1, x2, y):
    # Create the design matrix
    X = np.column_stack((np.ones_like(x1), x1, x2))
    # Perform OLS estimation: b = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def plot_lasso_elliptic():
    # Generate synthetic data for x1, x2, and y

    true_beta = np.array([1.0, 2.0, -1.5])
    noise = np.random.normal(0, 0.1, size=x1.shape)
    y = function_3d(true_beta, x1, x2) + noise

    # Perform OLS estimation to get beta (b0, b1, b2)
    beta_estimated = ols_estimation(x1, x2, y)

    # Generate data for the 3D elliptical function
    b1_range = np.linspace(beta_estimated[1] - 1, beta_estimated[1] + 1, 100)
    b2_range = np.linspace(beta_estimated[2] - 1, beta_estimated[2] + 1, 100)
    b0_fixed = beta_estimated[0]  # Fix b0 at the estimated value
    b1_grid, b2_grid = np.meshgrid(b1_range, b2_range)

    # Quadratic form representing the 3D elliptical curve
    #y_surface = b0_fixed + b1_grid**2 / 2 - b2_grid**2 / 3

    # Plot the 3D elliptical curve
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131, projection='3d')

    ax.plot_surface( x1, x2, beta[0] + beta[1] * x1 + beta[2] * x2, cmap='viridis', alpha=0.8)
    ax.set_title('3D Elliptic Curve')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    # Generate and plot 2D elliptical projections in the b1-b2 plane
    ax2 = fig.add_subplot(132)
    ax2.set_title('Elliptic Projections in b1-b2 Plane')
    ax2.set_xlabel('b1')
    ax2.set_ylabel('b2')

    #levels = np.linspace(y_surface.min(), y_surface.max(), 10)
    ax2.contour(b1_grid, b2_grid, y, cmap='Blues')

    # Overlay of 2D curves with LASSO
    ax3 = fig.add_subplot(133)
    ax3.set_title('LASSO with Elliptic Curves')
    ax3.set_xlabel('b1')
    ax3.set_ylabel('b2')

    ax3.contour(b1_grid, b2_grid, y, cmap='Blues')

    # Add LASSO constraint (diamond shape)
    ax3.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], color='red', linewidth=2, label='LASSO Constraint')
    ax3.legend()

    plt.tight_layout()
    plt.show()

plot_lasso_elliptic()
