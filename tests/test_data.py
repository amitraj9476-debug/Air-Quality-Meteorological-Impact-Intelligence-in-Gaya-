import pytest
import pandas as pd
import numpy as np

from src.data.loader import load_config, load_raw_data
from src.data.preprocess import preprocess_data, assign_aqi_category
from src.data.features import engineer_features

def test_assign_aqi_category():
    assert assign_aqi_category(15.0) == "Good"
    assert assign_aqi_category(45.0) == "Satisfactory"
    assert assign_aqi_category(75.0) == "Moderate"
    assert assign_aqi_category(105.0) == "Poor"
    assert assign_aqi_category(180.0) == "Very Poor"
    assert assign_aqi_category(300.0) == "Severe"

def test_preprocess_data():
    config = load_config()
    raw_df = pd.DataFrame({
        "From Date": ["01-01-2025 00:00", "01-01-2025 01:00", "01-01-2025 02:00"],
        "To Date": ["01-01-2025 01:00", "01-01-2025 02:00", "01-01-2025 03:00"],
        "PM2.5": [25.0, np.nan, 35.0],
        "PM10": [50.0, 60.0, 70.0],
        "NO2": [20.0, 22.0, 24.0],
        "CO": [0.5, 0.6, 0.7],
        "OZONE": [15.0, 16.0, 17.0],
        "RH": [70.0, 72.0, 74.0],
        "WD": [180.0, 185.0, 190.0],
        "AT": [20.0, 21.0, 22.0],
        "RF": [0.0, 0.0, 0.0]
    })
    
    clean_df = preprocess_data(raw_df, config=config)
    assert clean_df["PM2.5"].isnull().sum() == 0
    assert "AQI_Category" in clean_df.columns
    assert len(clean_df) == 3

def test_feature_engineering():
    config = load_config()
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=30, freq="h"),
        "PM2.5": np.random.uniform(10, 100, 30),
        "PM10": np.random.uniform(50, 150, 30),
        "NO2": np.random.uniform(10, 50, 30),
        "CO": np.random.uniform(0.1, 1.5, 30),
        "OZONE": np.random.uniform(5, 30, 30),
        "RH": np.random.uniform(40, 90, 30),
        "WD": np.random.uniform(0, 360, 30),
        "AT": np.random.uniform(15, 35, 30),
        "RF": np.zeros(30)
    })
    
    feat_df = engineer_features(df, include_lags=True, config=config)
    assert "hour_sin" in feat_df.columns
    assert "WD_sin" in feat_df.columns
    assert "THI" in feat_df.columns
    assert "PM2.5_lag_1" in feat_df.columns
    assert len(feat_df) > 0
