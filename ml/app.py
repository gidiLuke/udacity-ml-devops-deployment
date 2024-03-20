from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from ml.data import process_data
import pandas as pd
import joblib
import io
from pydantic import BaseModel, Field, conint
from enum import Enum

app = FastAPI()

# Load the trained artifacts
model = joblib.load("artifacts/model.joblib")
encoder = joblib.load("artifacts/encoder.joblib")
lb = joblib.load("artifacts/lb.joblib")


class Workclass(str, Enum):
    private = "Private"
    local_gov = "Local-gov"
    self_emp_not_inc = "Self-emp-not-inc"
    state_gov = "State-gov"
    federal_gov = "Federal-gov"
    self_emp_inc = "Self-emp-inc"
    never_worked = "Never-worked"
    without_pay = "Without-pay"
    unknown = "?"


class Education(str, Enum):
    bachelors = "Bachelors"
    some_college = "Some-college"
    hs_grad = "HS-grad"
    masters = "Masters"
    assoc_voc = "Assoc-voc"
    assoc_acdm = "Assoc-acdm"
    prof_school = "Prof-school"
    doctorate = "Doctorate"
    preschol = "Preschool"
    first_4th = "1st-4th"
    fifth_6th = "5th-6th"
    seventh_8th = "7th-8th"
    ninth = "9th"
    tenth = "10th"
    eleventh = "11th"
    twelfth = "12th"


class MaritalStatus(str, Enum):
    never_married = "Never-married"
    married_civ_spouse = "Married-civ-spouse"
    divorced = "Divorced"
    separated = "Separated"
    widowed = "Widowed"
    married_spouse_absent = "Married-spouse-absent"
    married_af_spouse = "Married-AF-spouse"


class Occupation(str, Enum):
    adm_clerical = "Adm-clerical"
    exec_managerial = "Exec-managerial"
    craft_repair = "Craft-repair"
    sales = "Sales"
    prof_specialty = "Prof-specialty"
    handlers_cleaners = "Handlers-cleaners"
    machine_op_inspct = "Machine-op-inspct"
    other_service = "Other-service"
    farming_fishing = "Farming-fishing"
    tech_support = "Tech-support"
    protective_serv = "Protective-serv"
    armed_forces = "Armed-Forces"
    priv_house_serv = "Priv-house-serv"
    transport_moving = "Transport-moving"
    unknown = "?"


class Relationship(str, Enum):
    not_in_family = "Not-in-family"
    husband = "Husband"
    unmarried = "Unmarried"
    wife = "Wife"
    own_child = "Own-child"
    other_relative = "Other-relative"


class Race(str, Enum):
    white = "White"
    black = "Black"
    asian_pac_islander = "Asian-Pac-Islander"
    amer_indian_eskimo = "Amer-Indian-Eskimo"
    other = "Other"


class Sex(str, Enum):
    male = "Male"
    female = "Female"


class NativeCountry(str, Enum):
    united_states = "United-States"
    philippines = "Philippines"
    germany = "Germany"
    canada = "Canada"
    puerto_rico = "Puerto-Rico"
    el_salvador = "El-Salvador"
    india = "India"
    cuba = "Cuba"
    england = "England"
    jamaica = "Jamaica"
    south_korea = "South-Korea"
    china = "China"
    mexico = "Mexico"
    vietnam = "Vietnam"
    japan = "Japan"
    poland = "Poland"
    columbia = "Columbia"
    thailand = "Thailand"
    ecuador = "Ecuador"
    laos = "Laos"
    taiwan = "Taiwan"
    haiti = "Haiti"
    portugal = "Portugal"
    dominican_republic = "Dominican-Republic"
    iran = "Iran"
    greece = "Greece"
    nicaragua = "Nicaragua"
    peru = "Peru"
    france = "France"
    ireland = "Ireland"
    hong_kong = "Hong-Kong"
    trinidad_tobago = "Trinadad&Tobago"
    cambodia = "Cambodia"
    unknown = "?"


class SinglePredictionRequest(BaseModel):
    age: conint(gt=0) = Field(description="The person's age.", example=37)
    workclass: Workclass = Field(
        description="The work class category.", example="Private"
    )
    fnlgt: int = Field(description="Final weighting factor.", example=77516)
    education: Education = Field(
        description="The highest level of education achieved.", example="Bachelors"
    )
    education_num: conint(gt=0) = Field(
        description="The number of educational years completed.",
        example=13,
        alias="education-num",
    )
    marital_status: MaritalStatus = Field(
        description="Marital status of the person.",
        example="Never-married",
        alias="marital-status",
    )
    occupation: Occupation = Field(
        description="The person's occupation.", example="Exec-managerial"
    )
    relationship: Relationship = Field(
        description="The person's relationship status.", example="Husband"
    )
    race: Race = Field(description="The person's race.", example="White")
    sex: Sex = Field(description="The person's sex.", example="Male")
    capital_gain: conint(ge=0) = Field(
        description="The person's capital gains.", example=2174, alias="capital-gain"
    )
    capital_loss: conint(ge=0) = Field(
        description="The person's capital losses.", example=0, alias="capital-loss"
    )
    hours_per_week: conint(gt=0) = Field(
        description="Number of hours worked per week.",
        example=40,
        alias="hours-per-week",
    )
    native_country: NativeCountry = Field(
        description="The person's native country.",
        example="United-States",
        alias="native-country",
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by encoding categorical features.

    Args:
    df (pd.DataFrame): The input DataFrame containing the data to be processed.

    Returns:
    pd.DataFrame: The processed DataFrame with categorical features encoded.
    """
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
    data, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return data


@app.get("/")
async def read_root():
    return {"message": "Welcome to the salary prediction API."}


@app.post("/predict/single", response_model=dict, tags=["Prediction"])
async def predict_single(request: SinglePredictionRequest):
    try:
        data_dict = request.dict(by_alias=True)
        single_instance_df = pd.DataFrame([data_dict])

        X = preprocess_data(single_instance_df)

        prediction = model.predict(X)

        print(prediction)

        # Invert transform on the label encoder to get the original label
        prediction_label = lb.inverse_transform(prediction)

        return {"prediction": prediction_label[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", summary="Predict Salary from CSV data", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be in CSV format.")
    try:
        # Load the CSV data into a DataFrame
        data = pd.read_csv(file.file)

        # Preprocess the data
        X = preprocess_data(data)

        # Predict
        predictions = model.predict(X)

        # Add the predictions to the original DataFrame
        data["predicted"] = lb.inverse_transform(predictions)

        # Convert the DataFrame to CSV for the response
        stream = io.StringIO()
        data.to_csv(stream, index=False)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
