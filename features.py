import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data.loader import load_config

logger = get_logger("data_features")

def engineer_features(df: pd.DataFrame, include_lags: bool = True, config: dict = None) -> pd.DataFrame:
    """
    Engineers domain-specific meteorological, temporal, cyclic, lag, and rolling statistics.
    """
    if config is None:
        config = load_config()
    
    df = df.copy()
    
    # 1. Temporal & Cyclic Features
    if "datetime" in df.columns:
        dt = df["datetime"]
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        
        # Sine / Cosine transform for periodic time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    # 2. Meteorological Domain Features
    if "WD" in df.columns:
        # Wind Direction degrees into vector sin/cos components
        wd_rad = np.radians(df["WD"])
        df["WD_sin"] = np.sin(wd_rad)
        df["WD_cos"] = np.cos(wd_rad)
    
    if "AT" in df.columns and "RH" in df.columns:
        # Temperature-Humidity Comfort & Dispersion Index (THI)
        df["THI"] = df["AT"] - (0.55 - 0.0055 * df["RH"]) * (df["AT"] - 14.5)
        # Vapor Pressure / Moisture Interaction
        df["AT_RH_ratio"] = df["AT"] / (df["RH"] + 1e-5)
    
    if "RF" in df.columns:
        df["is_raining"] = (df["RF"] > 0).astype(int)
    
    # 3. Time Series Lag & Rolling Window Features
    if include_lags:
        target_col = config.get("target_column", "PM2.5")
        if target_col in df.columns:
            for lag in [1, 2, 24]:
                df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
            
            df[f"{target_col}_roll_mean_3h"] = df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
            df[f"{target_col}_roll_std_3h"] = df[target_col].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
            df[f"{target_col}_roll_mean_24h"] = df[target_col].shift(1).rolling(window=24, min_periods=1).mean()
        
        # Weather feature lags
        for w_col in ["RH", "AT", "WD"]:
            if w_col in df.columns:
                df[f"{w_col}_lag_1"] = df[w_col].shift(1)
        
        # Drop initial NaN rows created by lags
        df = df.dropna().reset_index(drop=True)
    
    logger.info(f"Feature engineering completed. Total columns: {len(df.columns)}")
    return df
