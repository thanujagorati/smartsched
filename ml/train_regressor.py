"""
Regression model: predicts expected average waiting time for a GIVEN
algorithm on a GIVEN workload -- a continuous number, not a category.

This complements the classifier (which only says "SJF wins") by
letting us show the actual MAGNITUDE of the difference between
algorithms -- e.g. "SJF: ~12 units, FCFS: ~18 units" -- which is much
more useful for a dashboard than just a winner label.

Approach: reshape the wide dataset (one column per algorithm's
avg_waiting_time) into a LONG format -- one row per
(workload, algorithm) pair, with the algorithm itself as an input
feature (one-hot encoded) and avg_waiting_time as the target. A single
regressor then learns "given these workload features AND this
algorithm choice, what's the expected wait time?" -- which lets it
generalize across all four algorithms instead of training four
separate models.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from train_classifier import FEATURE_COLUMNS


ALGORITHMS = ["FCFS", "SJF", "Priority", "RoundRobin"]


def reshape_to_long_format(df):
    """
    Convert the wide dataset (one row per workload, with
    FCFS_avg_waiting_time / SJF_avg_waiting_time / etc as separate
    columns) into long format: one row per (workload, algorithm) pair.

    Args:
        df: the original dataset (from scheduling_dataset.csv)

    Returns:
        DataFrame with FEATURE_COLUMNS + one-hot algorithm columns +
        'target_waiting_time'
    """
    rows = []
    for _, row in df.iterrows():
        for algo in ALGORITHMS:
            new_row = {col: row[col] for col in FEATURE_COLUMNS}
            new_row["algorithm"] = algo
            new_row["target_waiting_time"] = row[f"{algo}_avg_waiting_time"]
            rows.append(new_row)

    long_df = pd.DataFrame(rows)
    # One-hot encode algorithm -- turns one 'algorithm' column into 4
    # binary columns (is_FCFS, is_SJF, etc), since models need numeric
    # input, not raw text categories
    long_df = pd.get_dummies(long_df, columns=["algorithm"], prefix="algo")
    return long_df


def train_regressor(long_df, test_size=0.2, random_state=42):
    """
    Train and evaluate a regressor on the long-format dataset.

    Returns:
        (trained_model, feature_columns_used)
    """
    feature_cols = [c for c in long_df.columns if c != "target_waiting_time"]
    X = long_df[feature_cols]
    y = long_df["target_waiting_time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,     # capped depth -- keeps the saved model file
                           # small (learned this the hard way with the
                           # classifier hitting GitHub's 100MB limit)
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f} time units")
    print(f"R² score: {r2:.3f}  (1.0 = perfect, 0.0 = no better than guessing the mean)")

    return model, feature_cols


if __name__ == "__main__":
    df = pd.read_csv("../data/scheduling_dataset.csv")
    long_df = reshape_to_long_format(df)

    print(f"Reshaped {len(df)} workloads into {len(long_df)} (workload, algorithm) rows\n")

    model, feature_cols = train_regressor(long_df)

    joblib.dump({"model": model, "feature_columns": feature_cols},
                "../data/regression_model.joblib")
    print("\nModel saved to ../data/regression_model.joblib")
