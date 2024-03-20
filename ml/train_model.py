import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_model, save_model, compute_model_metrics, inference
from data import process_data
import random

# set seed
random.seed(42)


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

    model = train_model(X_train, y_train)
    save_model(model, "artifacts/model.joblib")
    save_model(encoder, "artifacts/encoder.joblib")
    save_model(lb, "artifacts/lb.joblib")

    print("Training done!")

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

    summary = f"Overall - Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}\n"

    print("Overall evaluation done!")

    # Compute metrics for slices
    for feature in cat_features:
        unique_feature_values = test[feature].unique()
        for v in unique_feature_values:
            # not ideal to do the slicing again, but to be improved in the future.
            X_test_slice, y_test_slice, _, _ = process_data(
                test[test[feature] == v],
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds_slice = inference(model, X_test_slice)
            precision, recall, fbeta = compute_model_metrics(y_test_slice, preds_slice)

            summary += f"{feature}={v} (n={len(X_test_slice)}, N={len(test)})- "
            summary += f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}\n"

    print(summary)
    with open("artifacts/slice_output.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
