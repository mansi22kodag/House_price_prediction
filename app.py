from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FasAPI(title="House Price Predictor")

model = joblib.load("house_model.pkl")
model_columns = joblib.load("model_columns.pkl")

class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars : int
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int
    YearRemodAdd: int
    LotArea: int

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Predictor API!"}

@app.post("/predict")
def predict(features: HouseData):
    input_data = pd.DataFrame([features.dict()])

    full_input = pd.DataFrame(columns=model_columns)
    full_input = pd.concat([full_input,input_data],ignore_index=True)
    full_input = full_input.fillna(0)[model_columns]

    log_price = model.predict(full_input)[0]
    predicted_price = int(np.expm1(log_price))

    return{
        "predicted_price_usd" : predicted_price,
        "predicted_price_inr" : int(predicted_price * 82.5),
        "predicted_note" : f"Model R2 =0.91 on test data"
    }