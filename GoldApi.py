from fastapi import FastAPI
import joblib
import pandas as pd
from Predict.PredictPrice10Days import predict_gold_10_days
from Predict.PredictPrice30Days import predict_gold_30_days
from Predict.PredictPriceDay import predict_gold_day


app = FastAPI()

model = joblib.load("gold_model_target_10d.pkl")
features = joblib.load("features_target_10d.pkl")

@app.get("/predict_day")
def get_prediction():
    prediction = predict_gold_day()

    return {"prediction": prediction}

@app.get("/predict_10d")
def get_prediction():
    prediction = predict_gold_10_days()

    return {"prediction": prediction}

@app.get("/predict_30d")
def get_prediction():
    prediction = predict_gold_30_days()

    return {"prediction": prediction}