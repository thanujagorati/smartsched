"""
Random Forest classifier: predicts the best scheduling algorithm
for a given workload, based on its summary features (num_processes,
avg_burst_time, burst_time_cv, etc -- see dataset_builder.py's
extract_features()).

Why Random Forest specifically:
- Handles non-linear relationships between workload features and the
  best algorithm without manual feature engineering (e.g. it can learn
  that "high burst_time_cv AND moderate num_processes" favors SJF, a
  pattern that's hard to hand-code as a simple rule)
- Fairly robust to the mixed feature scales we have here (counts,
  means, ratios) without needing to normalize inputs first
- Gives us feature_importances_ for free -- useful for explaining
  WHY the model makes a given prediction (see also: SHAP, added later
  for a more rigorous version of this)

class_weight='balanced' directly addresses the label imbalance
documented in docs/limitations.md -- it up-weights the penalty for
misclassifying minority classes (RoundRobin, Priority) so the model
can't just learn to always predict SJF and call it a day.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


FEATURE_COLUMNS = [
    "num_processes",
    "avg_burst_time",
    "std_burst_time",
    "burst_time_cv",
    "min_burst_time",
    "max_burst_time",
    "avg_arrival_gap",
    "total_burst_time",
    "priority_variance",
    "priority_burst_corr",
]
LABEL_COLUMN = "best_algorithm"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return X, y


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Split data, train a Random Forest, and evaluate it properly --
    not just accuracy, since our classes are imbalanced.

    Returns:
        (trained_model, X_test, y_test, y_pred) -- test set and
        predictions returned too, so results can be inspected further
        (e.g. for the confusion matrix, or comparing specific rows)
    """
    # stratify=y keeps the class proportions roughly the same in both
    # train and test sets -- important with imbalanced data, otherwise
    # a random split could leave the test set with ZERO examples of
    # the rarest class (RoundRobin, currently only 1 row total)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,       # number of trees in the forest
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,               # use all available CPU cores
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("(Accuracy alone is misleading here -- see per-class report below)\n")

    print("Classification report (precision/recall/f1 PER class):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (rows=actual, columns=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    return model, X_test, y_test, y_pred


def show_feature_importance(model, feature_columns):
    """
    Print which features the model relies on most -- an early,
    simple form of explainability (SHAP comes later for something
    more rigorous, but this is useful right now).
    """
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False)
    print("\nFeature importance (higher = more influential in predictions):")
    print(importances)


if __name__ == "__main__":
    X, y = load_dataset("../data/scheduling_dataset.csv")

    print(f"Loaded {len(X)} rows")
    print(f"Class distribution:\n{y.value_counts()}\n")

    model, X_test, y_test, y_pred = train_and_evaluate(X, y)
    show_feature_importance(model, FEATURE_COLUMNS)

    joblib.dump(model, "../data/random_forest_model.joblib")
    print("\nModel saved to ../data/random_forest_model.joblib")
