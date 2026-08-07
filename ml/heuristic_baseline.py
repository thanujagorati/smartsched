"""
Heuristic baseline: a simple, hand-written rule for picking a
scheduling algorithm -- NO machine learning involved.

Why this matters: if the Random Forest classifier can't meaningfully
beat this simple rule, there's no real justification for using ML at
all. This comparison is the actual evidence behind "why ML instead of
just simulating/using a fixed rule" -- not just an assumption.

The rule is based on domain knowledge, not learned from data:
- High burst_time_cv (very uneven burst times) -> SJF tends to help
  the most, since it specifically targets short jobs getting stuck
  behind long ones
- Otherwise -> FCFS, since there isn't a strong enough signal to
  justify the complexity of prioritizing/preempting
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def heuristic_predict(row, cv_threshold=0.5):
    """
    A simple hand-written rule -- deliberately NOT using ML.

    Args:
        row: a row (dict-like) with at least 'burst_time_cv'
        cv_threshold: cutoff above which we guess SJF helps

    Returns:
        predicted algorithm name (string)
    """
    if row["burst_time_cv"] > cv_threshold:
        return "SJF"
    return "FCFS"


def evaluate_heuristic(df, cv_threshold=0.5):
    """
    Run the heuristic on every row and compare to the true label.

    Args:
        df: dataset with 'burst_time_cv' and 'best_algorithm' columns
        cv_threshold: passed through to heuristic_predict

    Returns:
        (accuracy, predictions list)
    """
    predictions = df.apply(
        lambda row: heuristic_predict(row, cv_threshold), axis=1
    )
    accuracy = accuracy_score(df["best_algorithm"], predictions)
    return accuracy, predictions


if __name__ == "__main__":
    df = pd.read_csv("../data/scheduling_dataset.csv")

    accuracy, predictions = evaluate_heuristic(df)
    print(f"Heuristic baseline accuracy: {accuracy:.3f}\n")
    print("Heuristic classification report:")
    print(classification_report(df["best_algorithm"], predictions, zero_division=0))

    print("\n--- For comparison, the Random Forest classifier achieved: ---")
    print("Overall accuracy: 0.934")
    print("(see ml/train_classifier.py output -- SJF f1=0.97, FCFS f1=0.30,")
    print("Priority f1=0.02, RoundRobin f1=0.12)")
    print("\nKey comparison point: does the RF meaningfully beat this simple")
    print("rule, or just match it? If a one-line if/else gets similar")
    print("accuracy, that's an honest signal the workload-to-algorithm")
    print("relationship may be simpler than a full ML pipeline suggests --")
    print("still worth reporting either way.")
