from ml.model import train_model, compute_model_metrics, save_model, load_model
from sklearn.datasets import make_classification
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_train_model_returns_correct_type():
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42
    )
    model = train_model(X, y)
    assert isinstance(
        model, RandomForestClassifier
    ), "train_model should return a RandomForestClassifier instance."


def test_compute_model_metrics_returns_floats():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert all(
        isinstance(metric, float) for metric in [precision, recall, fbeta]
    ), "compute_model_metrics should return a tuple of floats."


def test_save_and_load_model(tmp_path):
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42
    )
    model = train_model(X, y)
    file_path = tmp_path / "model.joblib"
    save_model(model, str(file_path))
    loaded_model = load_model(str(file_path))
    assert isinstance(
        loaded_model, RandomForestClassifier
    ), "The loaded object should be a RandomForestClassifier instance."
