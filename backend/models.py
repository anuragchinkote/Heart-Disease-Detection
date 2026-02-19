"""
Pydantic models for data validation
"""

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """
    Pydantic model for patient input validation
    All values should be numeric as per dataset specification
    """
    
    age: float = Field(..., ge=0, le=150, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    chest_pain_type: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    resting_blood_pressure: float = Field(..., ge=0, le=250, description="Resting blood pressure in mmHg")
    cholesterol: float = Field(..., ge=0, le=600, description="Cholesterol in mg/dL")
    fasting_blood_sugar: float = Field(..., ge=0, le=1, description="Fasting blood sugar: 0=<120, 1=>120")
    resting_ecg: float = Field(..., ge=0, le=2, description="Resting ECG (0-2)")
    max_heart_rate: float = Field(..., ge=0, le=250, description="Max heart rate achieved")
    exercise_induced_angina: float = Field(..., ge=0, le=1, description="Exercise induced angina: 0=No, 1=Yes")
    st_depression: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    st_slope: float = Field(..., ge=0, le=2, description="ST slope (0-2)")
    num_major_vessels: float = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thalassemia: float = Field(..., ge=0, le=3, description="Thalassemia type (0-3)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45.0,
                "sex": 1.0,
                "chest_pain_type": 0.0,
                "resting_blood_pressure": 130.0,
                "cholesterol": 240.0,
                "fasting_blood_sugar": 0.0,
                "resting_ecg": 0.0,
                "max_heart_rate": 150.0,
                "exercise_induced_angina": 0.0,
                "st_depression": 1.0,
                "st_slope": 1.0,
                "num_major_vessels": 0.0,
                "thalassemia": 0.0
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response
    """
    prediction: str  # "Disease" or "No Disease"
    probability_no_disease: float
    probability_disease: float
    confidence: float
    risk_level: str  # "Low", "Medium", "High"
    message: str


class HealthCheckResponse(BaseModel):
    """
    Pydantic model for health check response
    """
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_names_loaded: bool
