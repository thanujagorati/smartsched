"""
SHAP explainability for the Random Forest classifier.

Feature_importances_ (used in train_classifier.py) tells you which
features matter most ON AVERAGE across the whole dataset. It does NOT
tell you why the model made one SPECIFIC prediction for one SPECIFIC
workload.

SHAP (SHapley Additive exPlanations) fixes this. Based on Shapley
values from game theory, it computes -- for a single prediction --
exactly how much each feature pushed the model's output toward or
away from each class. This turns "the model said SJF" into "the model
said SJF because burst_time_cv was unusually high (+0.31) and
num_processes was low (+0.08)" -- a genuinely explainable answer, not
a black box.

This is what the dashboard (Week 5) will use to show WHY a
recommendation was made, not just WHAT was recommended.
"""

import pandas as pd
import joblib
import shap

from train_classifier import FEATURE_COLUMNS


def load_model_and_data(model_path="../data/random_forest_model.joblib",
                         data_path="../data/scheduling_dataset.csv"):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    return model, X


def explain_single_prediction(model, X, row_index):
    """
    Explain ONE specific prediction in human-readable terms.

    Args:
        model: trained RandomForestClassifier
        X: full feature DataFrame
        row_index: which row to explain

    Returns:
        dict: predicted class, and a sorted list of
        (feature, shap_value) pairs for that predicted class,
        showing which features pushed the prediction most
    """
    row = X.iloc[[row_index]]
    predicted_class = model.predict(row)[0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    # shap_values shape depends on sklearn/shap version -- newer shap
    # returns a single array with an extra class dimension. Handle
    # both shapes defensively so this doesn't silently break later.
    class_index = list(model.classes_).index(predicted_class)
    if isinstance(shap_values, list):
        values_for_class = shap_values[class_index][0]
    else:
        values_for_class = shap_values[0, :, class_index]

    feature_impact = sorted(
        zip(FEATURE_COLUMNS, values_for_class),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    return {
        "predicted_class": predicted_class,
        "feature_impact": feature_impact,
    }


def print_explanation(explanation):
    print(f"Predicted algorithm: {explanation['predicted_class']}\n")
    print("Why (features ranked by influence on THIS prediction):")
    for feature, value in explanation["feature_impact"]:
        direction = "pushed TOWARD" if value > 0 else "pushed AWAY FROM"
        print(f"  {feature:<22} {direction} '{explanation['predicted_class']}'  (impact: {value:+.3f})")


if __name__ == "__main__":
    model, X = load_model_and_data()

    # Explain a mix of predictions -- including at least one non-SJF
    # case, since SJF dominates the dataset and an all-SJF demo would
    # look repetitive despite the tool working correctly either way
    predictions = model.predict(X)
    non_sjf_row = next(i for i, p in enumerate(predictions) if p != "SJF")

    for idx in [0, non_sjf_row]:
        print(f"=== Explaining row {idx} (predicted: {predictions[idx]}) ===")
        explanation = explain_single_prediction(model, X, idx)
        print_explanation(explanation)
        print()
