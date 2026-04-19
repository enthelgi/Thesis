import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

'''
Checks if a dataset is normalized (mean ~0, variance ~1) and normalizes it if not.
Saves the normalized dataset to a new CSV file.
'''

def check_and_normalize_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print(df.head())
    # Exclude the binary MCI column if it exists
    if 'MCI' in df.columns:
        data = df.drop(columns=['MCI'])
        mci_column = df['MCI']
    else:
        data = df
        mci_column = None

    # Check if the dataset is normalized
    means = data.mean(axis=0)
    print("means",means)
    variances = data.var(axis=0)
    print("variaces",variances)
    normalized = np.allclose(means, 0, atol=0.0001) and np.allclose(variances, 1, atol=0.01)
    print(normalized)

    if normalized:
        print("The dataset is already normalized.")
        return df
    else:
        print("The dataset is not normalized. Normalizing now...")
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        df_normalized = pd.DataFrame(data_normalized, columns=data.columns)
        print(df_normalized.head())
        if mci_column is not None:
            df_normalized['MCI'] = mci_column

        # Save the normalized dataset
        output_path = file_path.replace('.csv', '_normalized.csv')
        df_normalized.to_csv(output_path, index=False)

        print(f"The dataset has been normalized and saved to {output_path}.")
        return df_normalized


#file_path = 'C:/Users/enthe/Documents/projectX/PliableLasso/modified_data_normalized.csv'
file_path = 'C:/Users/enthe/Documents/projectX/PliableLasso/modified_data.csv'
normalized_df = check_and_normalize_dataset(file_path)
