import pytest
import os
import pandas as pd
from src.models.predict import AQIPredictor
from src.data.loader import load_config

def test_predictor():
    config = load_config()
    assert os.path.exists(config["best_model_path"])
    
    predictor = AQIPredictor(config=config)
    sample_input = {
        "AT": 25.0,
        "RH": 60.0,
        "WD": 180.0,
        "RF": 0.0,
        "PM10": 100.0,
        "NO2": 30.0,
        "CO": 0.7,
        "OZONE": 20.0,
        "hour": 14,
        "month": 6
    }
    
    res = predictor.predict_sample(sample_input)
    assert "predicted_PM2.5" in res
    assert "predicted_AQI_category" in res
    assert isinstance(res["predicted_PM2.5"], float)
    assert res["predicted_PM2.5"] >= 0.0
