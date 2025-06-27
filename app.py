from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Загрузка модели
model = joblib.load("real_estate_model.joblib")

# Модель для POST запроса
class PropertyFeatures(BaseModel):
    lat: float
    lon: float
    total_square: float
    rooms: int
    floor: int

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# GET endpoint
@app.get("/predict_get")
def predict_get(lat: float, lon: float, total_square: float, rooms: int, floor: int):
    input_data = [[lat, lon, total_square, rooms, floor]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}

# POST endpoint
@app.post("/predict_post")
def predict_post(features: PropertyFeatures):
    input_data = [[
        features.lat, 
        features.lon, 
        features.total_square, 
        features.rooms, 
        features.floor
    ]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}