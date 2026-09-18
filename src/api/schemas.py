from pydantic import BaseModel, Field
from typing import List, Optional

class SensorInput(BaseModel):
    AT: float = Field(..., description="Ambient Temperature (°C)", json_schema_extra={"example": 25.5})
    RH: float = Field(..., description="Relative Humidity (%)", json_schema_extra={"example": 65.0})
    WD: float = Field(..., description="Wind Direction (Degrees 0-360)", json_schema_extra={"example": 180.0})
    RF: float = Field(..., description="Rainfall (mm)", json_schema_extra={"example": 0.0})
    PM10: Optional[float] = Field(100.0, description="PM10 Concentration (µg/m³)")
    NO2: Optional[float] = Field(30.0, description="NO2 Concentration (µg/m³)")
    CO: Optional[float] = Field(0.7, description="CO Concentration (mg/m³)")
    OZONE: Optional[float] = Field(25.0, description="Ozone Concentration (µg/m³)")
    hour: Optional[int] = Field(14, description="Hour of day (0-23)")
    month: Optional[int] = Field(5, description="Month (1-12)")

class PredictionResponse(BaseModel):
    predicted_PM2_5: float = Field(..., alias="predicted_PM2.5")
    predicted_AQI_category: str
    unit: str = "µg/m³"

class BatchInput(BaseModel):
    samples: List[SensorInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model_name: str
    num_features: int
    features: List[str]
    status: str
