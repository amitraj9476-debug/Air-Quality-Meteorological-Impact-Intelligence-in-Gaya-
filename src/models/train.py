import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger
from src.data.loader import load_raw_data, load_config
from src.data.preprocess import preprocess_data
from src.data.features import engineer_features
from src.models.evaluate import evaluate_regression, evaluate_classification
from src.models.explainability import get_feature_importances

logger = get_logger("model_train")

def get_feature_columns(df: pd.DataFrame, target_col: str, aqi_target_col: str) -> list:
    """Selects numerical feature columns excluding targets and datetimes."""
    exclude = [target_col, aqi_target_col, "From Date", "To Date", "datetime"]
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return feature_cols

def train_and_evaluate_models(config: dict = None) -> dict:
    """
    Main training pipeline:
    - Loads raw data
    - Preprocesses and engineers features
    - Performs temporal train/test split
    - Trains regression & classification algorithms
    - Saves trained model artifacts
    """
    if config is None:
        config = load_config()
    
    os.makedirs(config["model_dir"], exist_ok=True)
    
    # 1. Pipeline Data Ingestion & Preprocessing
    df_raw = load_raw_data(config=config)
    df_clean = preprocess_data(df_raw, config=config)
    df_feat = engineer_features(df_clean, include_lags=True, config=config)
    
    target_col = config["target_column"]
    aqi_col = config["aqi_target_column"]
    feature_cols = get_feature_columns(df_feat, target_col, aqi_col)
    
    logger.info(f"Training using {len(feature_cols)} features: {feature_cols}")
    
    X = df_feat[feature_cols]
    y_reg = df_feat[target_col]
    y_cls = df_feat[aqi_col]
    
    # 2. Temporal Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(df_feat) * (1 - config["model_training"]["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_cls_train, y_cls_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Training & Comparison for PM2.5 Regression
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1),
        "Ridge": Ridge(alpha=1.0)
    }
    
    best_name = None
    best_rmse = float("inf")
    best_model = None
    reg_results = {}
    
    for name, model in models.items():
        logger.info(f"Training regression model: {name}...")
        if name in ["Ridge"]:
            model.fit(X_train_scaled, y_reg_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_reg_train)
            preds = model.predict(X_test)
        
        metrics = evaluate_regression(y_reg_test.values, preds)
        reg_results[name] = metrics
        
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = name
            best_model = model
    
    logger.info(f"Best Regression Model: {best_name} with RMSE={best_rmse:.4f}")
    
    # 4. Train AQI Classifier
    logger.info("Training AQI Classification model...")
    aqi_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    aqi_model.fit(X_train, y_cls_train)
    cls_preds = aqi_model.predict(X_test)
    cls_metrics = evaluate_classification(y_cls_test.values, cls_preds)
    
    # 5. Save Artifacts
    joblib.dump(best_model, config["best_model_path"])
    joblib.dump(aqi_model, config["aqi_model_path"])
    joblib.dump(scaler, config["scaler_path"])
    joblib.dump(feature_cols, config["features_path"])
    
    # Save feature importances
    imp_df = get_feature_importances(best_model, feature_cols)
    imp_df.to_csv(os.path.join(config["model_dir"], "feature_importances.csv"), index=False)
    
    summary = {
        "best_regression_model": best_name,
        "regression_results": reg_results,
        "classification_results": cls_metrics,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "features": feature_cols
    }
    
    with open(os.path.join(config["model_dir"], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("Model training pipeline executed successfully. Artifacts saved.")
    return summary

if __name__ == "__main__":
    train_and_evaluate_models()
