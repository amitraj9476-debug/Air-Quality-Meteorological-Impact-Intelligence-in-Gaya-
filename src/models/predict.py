import os
import joblib
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data.loader import load_config
from src.data.features import engineer_features
from src.data.preprocess import assign_aqi_category

logger = get_logger("model_predict")

class AQIPredictor:
    """Inference engine for PM2.5 prediction and AQI classification."""
    
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.config = config
        
        self.reg_model = joblib.load(config["best_model_path"])
        self.aqi_model = joblib.load(config["aqi_model_path"])
        self.scaler = joblib.load(config["scaler_path"])
        self.feature_names = joblib.load(config["features_path"])
        logger.info(f"Loaded predictor artifacts with {len(self.feature_names)} features.")

    def prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Processes raw user/sensor input into expected model feature matrix."""
        df = input_df.copy()
        
        if "datetime" not in df.columns and "From Date" in df.columns:
            df["datetime"] = pd.to_datetime(df["From Date"], errors="coerce")
        elif "datetime" not in df.columns:
            df["datetime"] = pd.Timestamp.now()
            
        df = engineer_features(df, include_lags=False, config=self.config)
        
        # Ensure all required features exist (fill default/0 if missing in single sample)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[self.feature_names]

    def predict_sample(self, data_dict: dict) -> dict:
        """
        Predicts PM2.5 and AQI category for a single sample dictionary.
        Input dictionary format example:
        {
           "AT": 25.5,
           "RH": 65.0,
           "WD": 180.0,
           "RF": 0.0,
           "PM10": 110.0,
           "NO2": 35.0,
           "CO": 0.8,
           "OZONE": 25.0,
           "hour": 14,
           "month": 5
        }
        """
        df = pd.DataFrame([data_dict])
        X = self.prepare_features(df)
        
        pm25_pred = float(self.reg_model.predict(X)[0])
        pm25_pred = max(0.0, pm25_pred) # Clip non-negative
        
        aqi_cat = assign_aqi_category(pm25_pred)
        
        return {
            "predicted_PM2.5": round(pm25_pred, 2),
            "predicted_AQI_category": aqi_cat,
            "unit": "µg/m³"
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Predicts PM2.5 and AQI category for a batch dataframe."""
        X = self.prepare_features(df_input)
        preds = self.reg_model.predict(X)
        preds = np.clip(preds, a_min=0, a_max=None)
        
        res_df = df_input.copy()
        res_df["Predicted_PM2.5"] = np.round(preds, 2)
        res_df["Predicted_AQI_Category"] = res_df["Predicted_PM2.5"].apply(assign_aqi_category)
        return res_df
