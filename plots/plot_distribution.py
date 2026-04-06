import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both datasets
df1 = pd.read_csv("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv")
df2 = pd.read_csv("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv")


# Exclude the 'MCI' column if present
if 'MCI' in df1.columns:
	df1 = df1.drop(columns=['MCI'])
if 'MCI' in df2.columns:
	df2 = df2.drop(columns=['MCI'])

# Set columns to integer indices for plotting
df1.columns = range(df1.shape[1])
df2.columns = range(df2.shape[1])

# Find global y-limits for both datasets
all_data = pd.concat([df1, df2])
ymin = all_data.min().min()
ymax = all_data.max().max()

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df1, showfliers=True)
plt.title("TGS's Dataset")
plt.ylim(ymin, ymax)
plt.xlabel("Feature Index")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
sns.boxplot(data=df2, showfliers=True)
plt.title("Balanced Dataset")
plt.ylim(ymin, ymax)
plt.xlabel("Feature Index")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig("boxplots_both_datasets.png")
plt.show()