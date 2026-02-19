"""
Heart Disease Detection - Prediction Logic
Contains functions for making predictions with the trained model
"""

import numpy as np
import pandas as pd
from typing import Tuple
from models import PatientData


def predict_disease(
    patient: PatientData,
    model,
    scaler,
    feature_names: list
) -> Tuple[int, float]:
    """
    Make prediction for heart disease
    
    Args:
        patient: PatientData object with patient features
        model: Trained ML model
        scaler: StandardScaler object for normalization
        feature_names: List of feature names in correct order
    
    Returns:
        Tuple of (prediction: int, probability: float)
        prediction: 0 (No Disease) or 1 (Disease)
        probability: Probability of disease (0-1)
    """
    
    # Convert patient data to dictionary
    patient_dict = patient.dict()
    
    # Create DataFrame with patient data in correct feature order
    patient_df = pd.DataFrame([patient_dict])[feature_names]
    
    # Validate all required features are present
    missing_features = set(feature_names) - set(patient_df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Scale features
    patient_scaled = scaler.transform(patient_df)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    
    # Get probability
    if hasattr(model, 'predict_proba'):
        # For models with predict_proba (LR, DT, RF)
        probabilities = model.predict_proba(patient_scaled)[0]
        probability_disease = probabilities[1]
    else:
        # For SVM with probability=True
        try:
            probabilities = model.predict_proba(patient_scaled)[0]
            probability_disease = probabilities[1]
        except:
            # Fallback: use decision function
            decision = model.decision_function(patient_scaled)[0]
            # Convert decision function to probability using sigmoid
            probability_disease = 1 / (1 + np.exp(-decision))
    
    return int(prediction), float(probability_disease)


def get_risk_assessment(probability_disease: float) -> dict:
    """
    Get risk assessment based on disease probability
    
    Args:
        probability_disease: Probability of disease (0-1)
    
    Returns:
        Dictionary with risk assessment details
    """
    
    if probability_disease >= 0.7:
        risk_level = "High"
        recommendation = "Immediate medical consultation recommended"
        color = "red"
    elif probability_disease >= 0.4:
        risk_level = "Medium"
        recommendation = "Consider consulting a healthcare provider"
        color = "orange"
    else:
        risk_level = "Low"
        recommendation = "Continue healthy lifestyle practices"
        color = "green"
    
    return {
        "risk_level": risk_level,
        "probability": probability_disease,
        "recommendation": recommendation,
        "color": color
    }


def format_prediction_result(
    prediction: int,
    probability_disease: float
) -> dict:
    """
    Format prediction result for frontend display
    
    Args:
        prediction: 0 (No Disease) or 1 (Disease)
        probability_disease: Probability of disease
    
    Returns:
        Formatted prediction result
    """
    
    prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    probability_no_disease = 1 - probability_disease
    confidence = max(probability_no_disease, probability_disease)
    risk = get_risk_assessment(probability_disease)
    
    return {
        "prediction": prediction_label,
        "disease_probability": round(probability_disease * 100, 2),
        "no_disease_probability": round(probability_no_disease * 100, 2),
        "confidence": round(confidence * 100, 2),
        "risk_level": risk["risk_level"],
        "recommendation": risk["recommendation"],
        "color": risk["color"]
    }
