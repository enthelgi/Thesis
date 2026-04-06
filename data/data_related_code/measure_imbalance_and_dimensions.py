import pandas as pd
from collections import Counter

def analyze_dataset_from_file(file_path: str):
    """
    Loads a dataset (CSV) where the last column is the target y,
    and all preceding columns are features X.

    Reports:
      - Dimensions (samples, features)
      - Prevalence of each class
      - Class imbalance ratio
    """
    # Load dataset
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]   # all columns except last
    y = df.iloc[:, -1]    # last column

    # --- Dimensions ---
    n_samples, n_features = X.shape
    print("=== Dataset Dimensions ===")
    print(f"Samples: {n_samples}, Features: {n_features}")

    # --- Class counts & prevalence ---
    counts = Counter(y)
    prevalence = {cls: cnt / n_samples for cls, cnt in counts.items()}

    print("\n=== Class Prevalence ===")
    for cls, prev in prevalence.items():
        print(f"{cls}: {prev:.2%} ({counts[cls]} samples)")

    # --- Imbalance ratio ---
    majority = max(counts.values())
    minority = min(counts.values())
    imbalance_ratio = majority / minority if minority > 0 else float("inf")

    print("\n=== Class Imbalance ===")
    print(f"Majority class count: {majority}")
    print(f"Minority class count: {minority}")
    print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")

    return {
        "dimensions": {"samples": n_samples, "features": n_features},
        "prevalence": prevalence,
        "imbalance_ratio": imbalance_ratio,
    }


# --- Example usage ---
if __name__ == "__main__":
    report = analyze_dataset_from_file("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv")
