import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_model, save_model, compute_model_metrics, inference
from data import process_data
import numpy as np


def compute_metrics_by_slice(X, y, model, feature_names, encoder, lb, feature) -> str:
    """
    Compute model metrics by slices of data for a specific categorical feature.

    Args:
        X (np.ndarray): Features data.
        y (np.ndarray): Labels.
        model: Trained model.
        feature_names (list): List of feature names corresponding to columns in `X`.
        encoder: Fitted OneHotEncoder instance.
        lb: LabelBinarizer instance.
        feature (str): Name of the feature to compute the metrics by.

    Returns:
        str: A formatted string with the metrics for each unique value of the feature.
    """
    # Get the index of the feature in feature_names list
    feature_index = feature_names.index(feature)
    unique_values = np.unique(X[:, feature_index])

    results = []
    for value in unique_values:
        # Create a mask for the slice based on the feature's unique value
        mask = X[:, feature_index] == value
        X_slice = X[mask]
        y_slice = y[mask]

        # Predict on the slice
        preds_slice = inference(model, X_slice)

        # Compute metrics
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        results.append(
            f"{feature} = {value}: Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}"
        )

    return "\n".join(results)


def main():
    # Load and preprocess data
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save model
    model = train_model(X_train, y_train)
    save_model(model, "artifacts/model.joblib")
    save_model(encoder, "artifacts/encoder.joblib")
    save_model(lb, "artifacts/lb.joblib")

    # Process test data
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict and compute overall metrics
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    metrics_summary = f"Overall - Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}\n"

    # Compute metrics for slices
    for feature in cat_features:
        metrics_summary += (
            compute_metrics_by_slice(
                X_test, y_test, model, cat_features, encoder, lb, feature
            )
            + "\n"
        )

    print(metrics_summary)
    with open("artifacts/slice_output.txt", "w") as f:
        f.write(metrics_summary)


if __name__ == "__main__":
    main()
