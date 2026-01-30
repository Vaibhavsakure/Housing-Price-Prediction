from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# ================== LOAD MODEL ==================
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# ================== FASTAPI APP ==================
app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0",
    description="Predict housing prices using a trained ML model"
)

# ================== CORS CONFIG ==================
# Allows frontend (HTML / JS) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (OK for demo)
    allow_credentials=True,
    allow_methods=["*"],          # allow GET, POST, OPTIONS
    allow_headers=["*"],
)

# ================== HOME ==================
@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is running"}

# ================== PREDICT ==================
@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON input with housing features
    """

    # Convert input JSON → DataFrame
    input_df = pd.DataFrame([data])

    # Transform using saved pipeline
    transformed_data = pipeline.transform(input_df)

    # Predict
    prediction = model.predict(transformed_data)

    return {
        "predicted_median_house_value": float(prediction[0])
    }
