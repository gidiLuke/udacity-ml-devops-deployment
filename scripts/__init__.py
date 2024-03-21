import requests


def post_single_prediction(api_url: str, payload: dict):
    """Post payload for a single prediction to the API and print response.

    Args:
        api_url (str): URL for the single prediction endpoint.
        payload (dict): The payload for the single prediction.
    """
    response = requests.post(f"{api_url}/predict/single", json=payload)
    print(f"Single Prediction Status Code: {response.status_code}")
    print(f"Single Prediction Response: {response.json()}")


def post_batch_prediction(api_url: str, csv_file_path: str):
    """Post CSV file for batch prediction to the API and print response.

    Args:
        api_url (str): URL for the batch prediction endpoint.
        csv_file_path (str): Path to the CSV file for batch prediction.
    """
    with open(csv_file_path, "rb") as file:
        files = {"file": (csv_file_path, file, "text/csv")}
        response = requests.post(f"{api_url}/predict/batch", files=files)
        print(f"Batch Prediction Status Code: {response.status_code}")
        print(
            f"Batch Prediction Response (first 1000 characters): {response.text[:1000]}"
        )


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

api_url = "https://udacity-ml-devops-deployment.onrender.com"
csv_file_path = "data/test_date.csv"

post_single_prediction(api_url, sample_payload)

post_batch_prediction(api_url, csv_file_path)
