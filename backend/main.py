"""
Heart Disease Detection - FastAPI Backend
API entry point for model predictions
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import joblib
import os
from pathlib import Path
import logging

from models import PatientData, PredictionResponse, HealthCheckResponse
from predict import predict_disease

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# FASTAPI APP INITIALIZATION
# =====================================================================
app = FastAPI(
    title="Heart Disease Detection API",
    description="ML API for predicting heart disease using clinical features",
    version="1.0.0"
)

# =====================================================================
# CORS CONFIGURATION
# =====================================================================
# Allow frontend to access backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# LOAD MODEL AND RESOURCES
# =====================================================================
def load_model_resources():
    """
    Load model, scaler, and feature names from files
    """
    try:
        base_path = Path(__file__).parent / "model"
        
        # Check if model directory exists
        if not base_path.exists():
            logger.error(f"Model directory not found at {base_path}")
            return None, None, None
        
        # Load model
        model_path = base_path / "model.pkl"
        model = joblib.load(model_path) if model_path.exists() else None
        if model:
            logger.info(f"✓ Model loaded from {model_path}")
        else:
            logger.warning(f"⚠️  Model not found at {model_path}")
        
        # Load scaler
        scaler_path = base_path / "scaler.pkl"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if scaler:
            logger.info(f"✓ Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"⚠️  Scaler not found at {scaler_path}")
        
        # Load feature names
        features_path = base_path / "feature_names.pkl"
        feature_names = joblib.load(features_path) if features_path.exists() else None
        if feature_names:
            logger.info(f"✓ Feature names loaded from {features_path}")
        else:
            logger.warning(f"⚠️  Feature names not found at {features_path}")
        
        return model, scaler, feature_names
    
    except Exception as e:
        logger.error(f"❌ Error loading model resources: {str(e)}")
        return None, None, None


# Global model resources
MODEL = None
SCALER = None
FEATURE_NAMES = None

@app.on_event("startup")
async def startup_event():
    """
    Load model resources on startup
    """
    global MODEL, SCALER, FEATURE_NAMES
    MODEL, SCALER, FEATURE_NAMES = load_model_resources()
    logger.info("="*70)
    logger.info("✅ Backend startup complete!")
    logger.info("="*70)


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Heart Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    model_ok = MODEL is not None
    scaler_ok = SCALER is not None
    features_ok = FEATURE_NAMES is not None
    
    all_ok = model_ok and scaler_ok and features_ok
    status = "healthy" if all_ok else "unhealthy"
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_ok,
        scaler_loaded=scaler_ok,
        feature_names_loaded=features_ok
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Make prediction for heart disease
    
    Input: Patient clinical and demographic features
    Output: Prediction and probability scores
    
    Example:
    ```json
    {
        "age": 45.0,
        "sex": 1.0,
        ...
    }
    ```
    """
    
    # Validate model is loaded
    if MODEL is None or SCALER is None or FEATURE_NAMES is None:
        logger.error("Model resources not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first: python training/train.py"
        )
    
    try:
        # Make prediction
        logger.info(f"Making prediction for patient...")
        
        prediction, probability_disease = predict_disease(
            patient, MODEL, SCALER, FEATURE_NAMES
        )
        
        probability_no_disease = 1 - probability_disease
        confidence = max(probability_no_disease, probability_disease)
        
        # Determine prediction label
        prediction_label = "Disease" if prediction == 1 else "No Disease"
        
        # Determine risk level
        if probability_disease >= 0.7:
            risk_level = "High"
            message = "⚠️  High risk of heart disease. Immediate medical consultation recommended."
        elif probability_disease >= 0.4:
            risk_level = "Medium"
            message = "⚠️  Moderate risk of heart disease. Consider consulting a healthcare provider."
        else:
            risk_level = "Low"
            message = "✓ Low risk of heart disease based on clinical features."
        
        logger.info(f"Prediction made: {prediction_label} (Risk: {risk_level})")
        
        return PredictionResponse(
            prediction=prediction_label,
            probability_no_disease=float(probability_no_disease),
            probability_disease=float(probability_disease),
            confidence=float(confidence),
            risk_level=risk_level,
            message=message
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    
    except TypeError as e:
        logger.error(f"Type error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Type error in prediction: {str(e)}")
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


# =====================================================================
# ERROR HANDLERS
# =====================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler - returns proper JSON with correct status code
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch all exceptions and return JSON error response
    """
    logger.error(f"Unexpected error: {type(exc).__name__}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": f"Internal server error: {type(exc).__name__}: {str(exc)}"
        }
    )


# =====================================================================
# IF RUNNING DIRECTLY
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Heart Disease Detection API...")
    print("\n" + "="*70)
    print("🚀 Starting Heart Disease Detection API")
    print("="*70)
    print("\n📍 API will be available at: http://localhost:8000")
    print("📚 Interactive API docs at: http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
