import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data.loader import load_config

logger = get_logger("data_preprocess")

def assign_aqi_category(pm25_val: float) -> str:
    """Categorizes PM2.5 level into AQI Category according to CPCB standards."""
    if pd.isna(pm25_val):
        return np.nan
    if pm25_val <= 30:
        return "Good"
    elif pm25_val <= 60:
        return "Satisfactory"
    elif pm25_val <= 90:
        return "Moderate"
    elif pm25_val <= 120:
        return "Poor"
    elif pm25_val <= 250:
        return "Very Poor"
    else:
        return "Severe"

def preprocess_data(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Time-series interpolation for missing sensor values
    - Assigns AQI classification labels
    - Ensures clean continuous data structure
    """
    if config is None:
        config = load_config()
    
    df = df.copy()
    pollutant_cols = config["pollutant_columns"]
    met_cols = config["meteorological_columns"]
    feature_cols = pollutant_cols + met_cols
    
    logger.info("Starting data preprocessing...")
    
    # Interpolate numerical features linearly over time
    for col in feature_cols:
        if col in df.columns:
            # Interpolate missing values (limit_direction='both' covers edge missing values)
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            # Fill any remaining NaNs with median
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Create AQI Category column based on PM2.5
    if config["target_column"] in df.columns:
        df[config["aqi_target_column"]] = df[config["target_column"]].apply(assign_aqi_category)
    
    logger.info(f"Preprocessing completed. Clean dataframe shape: {df.shape}")
    return df
