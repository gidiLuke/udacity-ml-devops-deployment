from fastapi.testclient import TestClient
from ml.app import app
import pandas as pd

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the salary prediction API."}


def test_predict_single():
    sample_payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict/single", json=sample_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]


def test_predict_batch(tmp_path):
    training_file_path = "data/census.csv"

    df = pd.read_csv(training_file_path)

    df = df.drop(columns=["salary"])
    test_file_path = tmp_path / "test.csv"
    df.to_csv(test_file_path, index=False)

    with open(test_file_path, "rb") as test_file:
        files = {"file": (str(test_file_path), test_file, "text/csv")}
        response = client.post("/predict/batch", files=files)

    # Check the response status code
    assert response.status_code == 200

    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    content = response.content.decode("utf-8")

    assert "predicted" in content

    input_rows = sum(1 for _ in open(test_file_path)) - 1  # Exclude header row
    output_rows = (
        sum(1 for line in content.split("\n") if line) - 1
    )  # Exclude header row
    assert input_rows == output_rows
