# 🌫️ Air Quality & Meteorological Impact Intelligence - End-to-End ML Project

An end-to-end production Machine Learning pipeline, Explainable AI (XAI) engine, REST API, and interactive Streamlit web dashboard built using 1-year continuous hourly air quality monitoring data (2025) from the **Central Pollution Control Board (CPCB)** monitoring station at **Kareemganj, Gaya - BSPCB, Bihar**.

### 📍 Monitoring Station Context
- **Organization**: Central Pollution Control Board (CPCB) / Bihar State Pollution Control Board (BSPCB)
- **Station**: Kareemganj, Gaya, Bihar
- **Monitoring Type**: Continuous Ambient Air Quality Monitoring (CAAQM)
- **Averaging Period**: 1 Hour (1H)
- **Date Range**: Jan 01, 2025 – Dec 31, 2025 (8,585 hourly observations)

---

## 🌟 Key Features

1. **Modular Architecture (`src/`)**: Clean Python package separation for data loading, time-series preprocessing, feature engineering, model training, and API inference.
2. **Meteorological Impact Analysis & Explainability (XAI)**:
   - **Feature Importance & SHAP Analysis**: Quantifies how Temperature (`AT`), Humidity (`RH`), Wind Direction (`WD`), and Rainfall (`RF`) drive `PM2.5` concentrations.
   - **Partial Dependence Plots (PDP)**: Visualizes non-linear marginal impacts of weather factors.
   - **Wind Direction Pollution Radar**: Polar sector plot mapping regional pollutant influx angles.
3. **Multi-Model Machine Learning Pipeline**:
   - Random Forest Regressor
   - XGBoost Regressor
   - LightGBM Regressor
   - Ridge Regressor ($R^2 = 0.877$, $RMSE = 9.90 \mu g/m^3$)
   - Random Forest AQI Hazard Classifier ($Accuracy = 75.7\%$)
4. **FastAPI REST Server (`src/api/main.py`)**: Endpoints for real-time predictions (`/predict`), batch predictions (`/predict_batch`), and health monitoring.
5. **Interactive Web Dashboard (`app/dashboard.py`)**: 4 comprehensive tabs for EDA, Explainability, Weather Scenario Simulation, and Model Benchmarks.
6. **Automated Testing Suite (`tests/`)**: Pytest coverage for data preprocessing, model inference, and API endpoints.

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed, then run:
```bash
python -m pip install -r requirements.txt
```

### 2. Model Training Pipeline
Train the ML models and generate saved artifacts:
```bash
python main.py train
```

### 3. Launch Interactive Streamlit Dashboard
```bash
python main.py dashboard
```
Access the dashboard at `http://localhost:8501`.

### 4. Launch FastAPI REST Server
```bash
python main.py api
```
Access interactive API documentation at `http://127.0.0.1:8000/docs`.

### 5. Run Automated Tests
```bash
python main.py test
```

---

## 📁 Project Structure

```
d:/New folder/
├── config/
│   └── config.yaml               # Config file with hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── loader.py             # Data loading & temporal sorting
│   │   ├── preprocess.py         # Missing value imputation & AQI categorization
│   │   └── features.py           # Meteorological interaction & cyclic features
│   ├── models/
│   │   ├── train.py              # Model training pipeline
│   │   ├── evaluate.py           # Regression & classification metrics
│   │   ├── explainability.py     # SHAP & Partial dependence analysis
│   │   └── predict.py            # Predictor inference engine
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   └── schemas.py            # Pydantic validation schemas
│   └── utils/
│       └── logger.py             # Logging setup
├── app/
│   └── dashboard.py              # Streamlit Web UI application
├── models/                       # Saved model artifacts (.joblib & json summaries)
├── tests/                        # Pytest suite
├── requirements.txt              # Dependencies list
├── main.py                       # CLI Launcher
└── README.md                     # Documentation
```
