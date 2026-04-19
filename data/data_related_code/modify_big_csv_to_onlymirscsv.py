import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv("ProAD_Dataset_Phase3_Marinaki_new_2.csv")

# Step 2: Modify the DataFrame as per the provided operations
# Modify specific value
df.loc[17, 'hsa_miR_X8h_5p'] = -5.57

# Create a copy of the DataFrame
dfc = df.copy()

# Filter rows based on 'group' column values and select relevant columns
df_grouped = dfc[dfc['group'].isin(["MCI","Control"])]

# Convert 'MCI' to 1 and 'Control' to 0 in the 'group' column
binary_column = df_grouped['group'].map({'MCI': 1, 'Control': 0})

mir_columns = df_grouped.filter(regex="^hsa")
mir_columns = mir_columns.drop("hsa_miR_X15o_3p", axis=1)

# Fill missing values with minimum values from respective columns
mir_columns["hsa_miR_X21t_5p"] = mir_columns["hsa_miR_X21t_5p"].fillna(0)
mir_columns["hsa_miR_X1a_3p"] = mir_columns["hsa_miR_X1a_3p"].fillna(0)
mir_columns["hsa_miR_X7g_3p"] = mir_columns["hsa_miR_X7g_3p"].fillna(0)
mir_columns["hsa_miR_X9i_5p"] = mir_columns["hsa_miR_X9i_5p"].fillna(0)

# Combine the filtered columns with the binary column
combined_columns = pd.concat([mir_columns, binary_column.rename('MCI')], axis=1)

# Print the resulting DataFrame
print(combined_columns)
output_csv = "modified_data.csv"  # Specify the path for the output CSV file
combined_columns.to_csv(output_csv, index=False)

print(f"Modified data saved to {output_csv}")



