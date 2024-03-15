import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_model, save_model, compute_model_metrics, inference
from data import process_data


def main():
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
    save_model(model, "model.joblib")
    save_model(encoder, "encoder.joblib")
    save_model(lb, "lb.joblib")

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")


if __name__ == "__main__":
    main()
