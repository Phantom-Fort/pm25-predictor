from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import json
import os
import tempfile

# App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Corrected this line
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "pm25_scaler_params.json")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")

# Load model and parameters
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

try:
    with open(SCALER_PATH, "r") as f:
        scaler_params = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Scaler parameters not found at {SCALER_PATH}")

try:
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Feature names not found at {FEATURES_PATH}")

# One-hot encoded features
ONE_HOT_FEATURES = [
    "explosive_type_ANFO",
    "explosive_type_emulsion",
    "rock_type_granite",
    "rock_type_limestone",
    "rock_type_sandstone",
    "dust_suppression_none",
    "dust_suppression_water_spray",
    "dust_suppression_chemical",
]

@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_file:
            json.dump(payload, tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Load input from temp file
        with open(tmp_path, "r") as f:
            parsed = json.load(f)
            features = parsed.get("features")

        if not features or not isinstance(features, dict):
            raise HTTPException(status_code=400, detail="Missing or invalid 'features' object.")

        missing = [feat for feat in feature_names if feat not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        df = pd.DataFrame([features], columns=feature_names)

        for col in feature_names:
            value = df[col].iloc[0]
            if col in ONE_HOT_FEATURES:
                if value not in [0, 1]:
                    raise HTTPException(status_code=400, detail=f"Feature {col} must be 0 or 1.")
                continue

            if col not in scaler_params:
                raise HTTPException(status_code=500, detail=f"Scaler parameters missing for {col}")

            min_val = scaler_params[col].get("min")
            max_val = scaler_params[col].get("max")
            if min_val is None or max_val is None:
                raise HTTPException(status_code=500, detail=f"Incomplete scaler parameters for {col}")

            clamped = max(min(value, max_val), min_val)
            df[col] = (clamped - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0

        prediction = model.predict(df)[0]
        return {"pm25": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
