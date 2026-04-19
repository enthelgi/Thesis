import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Generate example data
np.random.seed(42)
n_samples, n_features = 50, 30
X = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features)
y = np.dot(X, true_coef) + np.random.normal(size=n_samples)

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Fit Lasso regression
lasso = Lasso(alpha=0.1) 
lasso.fit(X_std, y)
coefficients = lasso.coef_

# Soft thresholding function
def soft_threshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

# Apply soft thresholding to coefficients
alpha = 0.5  # Thresholding parameter
soft_thresholded_coef = soft_threshold(coefficients, alpha)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.stem(coefficients, linefmt='b-', markerfmt='bo', label='Original Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.ylim(-1.8, 1.8)  # Set y-axis limits
plt.title('Original Lasso Coefficients')
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(soft_thresholded_coef, linefmt='g-', markerfmt='go', label='Thresholded Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.ylim(-1.8, 1.8)  # Set y-axis limits
plt.title('Soft Thresholded Lasso Coefficients')
plt.legend()

plt.tight_layout()
plt.show()