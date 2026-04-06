import numpy as np
import matplotlib.pyplot as plt

# Define logistic and logit functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

# Range of scores
x = np.linspace(-10, 10, 500)
p = sigmoid(x)
log_odds = logit(p)

# Simulated sample points
sample_x = np.array([-8, -5, -3, 0, 1, 4, 7])
true_y = np.array([0, 0, 1, 0, 1, 1, 1])
sample_p = sigmoid(sample_x)

# Start plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# --- LEFT PLOT: Probability vs. Score ---
axs[0].plot(x, p, label="Probability of Success", color='green')
axs[0].scatter(sample_x, sample_p, color='red', label='Sample Points (True Outcomes)')
for xi, yi, pi in zip(sample_x, true_y, sample_p):
    axs[0].text(xi, pi + 0.05, f"True: {yi}", ha='center', fontsize=9)
axs[0].set_title("Logistic Model: Score to Probability")
axs[0].set_xlabel("$t_i$")
axs[0].set_ylabel("Probability")
axs[0].legend(loc='upper left')  

# --- MIDDLE PLOT: Log-Odds vs. Probability ---
axs[1].plot(p, log_odds, color='purple', label='L$t_i$ vs. Probability')
axs[1].set_title("Relationship: Score to Log-Odds")
axs[1].set_xlabel("$t_i$")
axs[1].set_ylabel("Log-Odds")
axs[1].legend(loc='upper left')  

# --- RIGHT PLOT: Score vs. Log-Odds ---
axs[2].plot(x, log_odds, color='blue', label='$t_i$')
axs[2].set_title("Linear version: Measurment to Score")
axs[2].set_xlabel("$X_i$")
axs[2].set_ylabel("$t_i$")
axs[2].legend(loc='upper left') 

plt.show()

