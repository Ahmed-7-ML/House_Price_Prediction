# --- EndPoint ---
from fastapi import FastAPI

# --- Schema and Data Validation
from pydantic import BaseModel, Field

# --- Prepare the Input to Model as Numpy Array
import numpy as np

# --- Load the Saved Model to be used
import joblib

# Instantiate the FastAPI to create Endpoint
app = FastAPI(title="House Price Prediction API")

# Load the Model
model = joblib.load('models/xgb_house_price_model.pkl')

# Define Input Schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description='median income in block group')
    HouseAge: float = Field(..., description='median house age in block group')
    AveRooms: float = Field(..., description='average number of rooms per household')
    AveBedrms: float = Field(..., description='average number of bedrooms per household')
    Population: float = Field(..., description='block group population')
    AveOccup: float = Field(..., description='average number of household members')
    Latitude: float = Field(..., description='block group latitude')
    Longitude: float = Field(..., description='block group longitude')

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
                    features.Population, features.AveOccup, features.Latitude, features.Longitude]])
    
    output = model.predict(data)[0]
    return {"Predicted Price": float(output * 100_000) }
