from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.api.schemas import (
    SensorInput,
    PredictionResponse,
    BatchInput,
    BatchPredictionResponse,
    ModelInfoResponse
)
from src.models.predict import AQIPredictor
from src.utils.logger import get_logger

logger = get_logger("api_main")

app = FastAPI(
    title="Air Quality & PM2.5 Prediction API",
    description="Production REST API for predicting PM2.5 concentrations, AQI hazard levels, and meteorological impacts.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None

def get_predictor() -> AQIPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = AQIPredictor()
            logger.info("Predictor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load predictor: {str(e)}")
    return _predictor

@app.get("/", tags=["Health"])
def read_root():
    return {
        "project": "End-to-End Air Quality ML System",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    pred = get_predictor()
    return {"status": "healthy", "model_loaded": pred is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(sensor_data: SensorInput):
    pred = get_predictor()
    try:
        data_dict = sensor_data.model_dump()
        res = pred.predict_sample(data_dict)
        return res
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_data: BatchInput):
    pred = get_predictor()
    try:
        df_input = pd.DataFrame([s.model_dump() for s in batch_data.samples])
        df_res = pred.predict_batch(df_input)
        preds = []
        for idx, row in df_res.iterrows():
            preds.append({
                "predicted_PM2.5": float(row["Predicted_PM2.5"]),
                "predicted_AQI_category": str(row["Predicted_AQI_Category"]),
                "unit": "µg/m³"
            })
        return {"predictions": preds}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    pred = get_predictor()
    return {
        "model_name": type(pred.reg_model).__name__,
        "num_features": len(pred.feature_names),
        "features": pred.feature_names,
        "status": "ready"
    }
