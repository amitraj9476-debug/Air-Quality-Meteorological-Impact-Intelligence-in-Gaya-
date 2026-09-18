from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_PM2.5" in data
    assert "predicted_AQI_category" in data

def test_api_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["status"] == "ready"
