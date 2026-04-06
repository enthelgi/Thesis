import pandas as pd
from pathlib import Path

def balance_binary_dataset_downsample(
    input_csv: str,
    random_state: int = 42,
    shuffle: bool = True
) -> pd.DataFrame:
    """
    Load a CSV where the last column is the target y (binary classes),
    and create a balanced dataset via DOWN-SAMPLING the majority class.

    The new dataset is saved with the original filename + "_balanced.csv".

    Params
    ------
    input_csv : str
        Path to input CSV
    random_state : int
        RNG seed for reproducibility
    shuffle : bool
        Whether to shuffle the balanced rows before saving

    Returns
    -------
    df_balanced : pandas DataFrame
    """
    df = pd.read_csv(input_csv)
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (features + target).")

    y_col = df.columns[-1]
    classes = df[y_col].unique()

    if len(classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {len(classes)}: {classes!r}")

    # Split by class
    df_class0 = df[df[y_col] == classes[0]]
    df_class1 = df[df[y_col] == classes[1]]

    # Identify majority/minority
    if len(df_class0) > len(df_class1):
        majority, minority = df_class0, df_class1
    else:
        majority, minority = df_class1, df_class0

    # Downsample majority
    majority_down = majority.sample(n=len(minority), random_state=random_state)
    df_balanced = pd.concat([minority, majority_down], axis=0)

    if shuffle:
        df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Build output filename
    p = Path(input_csv)
    output_csv = p.with_name(p.stem + "_balanced.csv")

    df_balanced.to_csv(output_csv, index=False)

    print(f"Balanced dataset saved as: {output_csv}")
    print("Per-class counts:", df_balanced[y_col].value_counts().to_dict())

    return df_balanced


# --- Example usage ---
if __name__ == "__main__":
    balance_binary_dataset_downsample("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv")
