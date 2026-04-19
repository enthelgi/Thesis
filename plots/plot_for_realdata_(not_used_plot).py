import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

# Parameters for the two sigmoid functions
L = 1  # maximum value for both sigmoids
k1, k2 = 0.5, 0.5  # steepness of the curves
x0_1, x0_2 = 16, 17  # midpoints of the curves

# Define the range for x values
x = np.linspace(0, 30, 300)

# Compute the sigmoid values
y1 = sigmoid(x, L, x0_1, k1)
y2 = sigmoid(x, L, x0_2, k2)

# Create the plot
plt.figure(figsize=(10, 6))

# Zoom in by adjusting the x and y limits
plt.xlim(10, 25)
plt.ylim(0, 1.2)

# Plot the sigmoid curves
plt.plot(x, y1, label='1')
plt.plot(x, y2, label='2')

# Add dotted lines at y=0.1 and the difference of the curves being 1
plt.axhline(y=0.1, color='grey', linestyle='--')
plt.plot([11.6, 11.6], [0, 0.1], 'k--', lw=0.5)
plt.plot([12.6, 12.6], [0, 0.1], 'k--', lw=0.5)

# Add a horizontal line 
plt.axhline(y=0.1, color='grey', linestyle='--')

# Annotate the points where the curves meet the dotted line 
plt.annotate('16 cycles', xy=(11.6, 0.1), xytext=(11.6, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('17 cycles', xy=(12.6, 0.1), xytext=(12.6, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Label the axes without scale
plt.xlabel('Cycles')
plt.ylabel('Fluorescence')
plt.xticks([], [])
plt.yticks([], [])
plt.legend()
plt.grid(True)
plt.title('Dilution by 2')
plt.show()
