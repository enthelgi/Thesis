import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Simulate data
np.random.seed(42)
n = 100
X1 = np.random.rand(n) * 10
X2 = np.random.rand(n) * 10
# Interaction effect: X1*X2 term
#Y = 5 + 2 * X1 + 3 * X2 + 1.5 * X1 * X2 + np.random.randn(n) * 2
Y = 2 * X1 + 3 * X2 + 1.5 * X1 * X2 + np.random.randn(n) * 2
#Y = 2 * X1 + 3 * X2 + 1.5 * X1 * X2 

# Create DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Fit models
# Model 1: Without interaction
model1 = LinearRegression().fit(data[['X1', 'X2']], data['Y'])

# Model 2: With interaction
data['X1_X2'] = data['X1'] * data['X2']
model2 = LinearRegression().fit(data[['X1', 'X2', 'X1_X2']], data['Y'])

# Generate grid data for contour plots
x1_range = np.linspace(data['X1'].min(), data['X1'].max(), 100)
x2_range = np.linspace(data['X2'].min(), data['X2'].max(), 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
grid_data = pd.DataFrame({'X1': X1_grid.ravel(), 'X2': X2_grid.ravel()})
grid_data['X1_X2'] = grid_data['X1'] * grid_data['X2']

# Predictions
grid_data['Y_pred_no_interaction'] = model1.predict(grid_data[['X1', 'X2']])
grid_data['Y_pred_with_interaction'] = model2.predict(grid_data[['X1', 'X2', 'X1_X2']])

# Reshape predictions for contour plots
Z_no_interaction = grid_data['Y_pred_no_interaction'].values.reshape(X1_grid.shape)
Z_with_interaction = grid_data['Y_pred_with_interaction'].values.reshape(X1_grid.shape)

# Visualization
plt.figure(figsize=(14, 6))

# Contour plot without interaction
plt.subplot(1, 2, 1)
contour1 = plt.contourf(X1_grid, X2_grid, Z_no_interaction, cmap='viridis')
plt.colorbar(contour1)
plt.scatter(data['X1'], data['X2'], c=data['Y'], edgecolors='w', linewidths=0.5)
plt.title('Model without Interaction')
plt.xlabel('X1')
plt.ylabel('X2')

# Contour plot with interaction
plt.subplot(1, 2, 2)
contour2 = plt.contourf(X1_grid, X2_grid, Z_with_interaction, cmap='viridis')
plt.colorbar(contour2)
plt.scatter(data['X1'], data['X2'], c=data['Y'], edgecolors='w', linewidths=0.5)
plt.title('Model with Interaction')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()

# Model performance
print("Model without Interaction:")
print(f"R^2: {r2_score(data['Y'], data['Y_pred_no_interaction'])}")
print(f"MSE: {mean_squared_error(data['Y'], data['Y_pred_no_interaction'])}")

print("\nModel with Interaction:")
print(f"R^2: {r2_score(data['Y'], data['Y_pred_with_interaction'])}")
print(f"MSE: {mean_squared_error(data['Y'], data['Y_pred_with_interaction'])}")
