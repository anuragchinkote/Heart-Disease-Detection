"""
Heart Disease Detection - Preprocessing Logic
Contains functions for data preprocessing and validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def validate_feature_ranges(features: dict) -> Dict[str, Any]:
    """
    Validate that features are within expected ranges
    
    Args:
        features: Dictionary of features to validate
    
    Returns:
        Dictionary with validation results
    """
    
    validation_rules = {
        'age': {'min': 0, 'max': 150},
        'sex': {'min': 0, 'max': 1},
        'chest_pain_type': {'min': 0, 'max': 3},
        'resting_blood_pressure': {'min': 0, 'max': 250},
        'cholesterol': {'min': 0, 'max': 600},
        'fasting_blood_sugar': {'min': 0, 'max': 1},
        'resting_ecg': {'min': 0, 'max': 2},
        'max_heart_rate': {'min': 0, 'max': 250},
        'exercise_induced_angina': {'min': 0, 'max': 1},
        'st_depression': {'min': 0, 'max': 10},
        'st_slope': {'min': 0, 'max': 2},
        'num_major_vessels': {'min': 0, 'max': 4},
        'thalassemia': {'min': 0, 'max': 3}
    }
    
    errors = []
    warnings = []
    
    for feature_name, rules in validation_rules.items():
        if feature_name not in features:
            errors.append(f"Missing feature: {feature_name}")
            continue
        
        value = features[feature_name]
        
        # Check type
        if not isinstance(value, (int, float)):
            errors.append(f"{feature_name}: must be numeric, got {type(value)}")
            continue
        
        # Check range
        if value < rules['min'] or value > rules['max']:
            errors.append(
                f"{feature_name}: value {value} out of range [{rules['min']}, {rules['max']}]"
            )
        
        # Add warnings for suspicious values
        if feature_name == 'age' and (value < 18 or value > 120):
            warnings.append(f"{feature_name}: unusual age value {value}")
        
        if feature_name == 'cholesterol' and value == 0:
            warnings.append(f"{feature_name}: zero cholesterol is unusual")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def normalize_features(
    features: dict,
    feature_names: list
) -> np.ndarray:
    """
    Convert features to normalized array in correct order
    
    Args:
        features: Dictionary of feature values
        feature_names: List of feature names in correct order
    
    Returns:
        Numpy array of features
    """
    
    values = []
    for name in feature_names:
        if name not in features:
            raise ValueError(f"Missing feature: {name}")
        values.append(features[name])
    
    return np.array(values).reshape(1, -1)


def calculate_statistics(features: dict) -> dict:
    """
    Calculate statistics for input features
    
    Args:
        features: Dictionary of feature values
    
    Returns:
        Dictionary with statistics
    """
    
    stats = {
        'mean_age': features.get('age', 0),
        'has_angina': features.get('exercise_induced_angina', 0) == 1,
        'high_cholesterol': features.get('cholesterol', 0) > 200,
        'high_blood_pressure': features.get('resting_blood_pressure', 0) > 140,
        'elevated_resting_ecg': features.get('resting_ecg', 0) > 0,
        'elevated_st_depression': features.get('st_depression', 0) > 1
    }
    
    return stats


def get_feature_descriptions() -> dict:
    """
    Get descriptions of all features for frontend display
    """
    
    descriptions = {
        'age': 'Age of the patient in years',
        'sex': 'Sex of the patient (0=Female, 1=Male)',
        'chest_pain_type': 'Type of chest pain (0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic)',
        'resting_blood_pressure': 'Resting blood pressure in mmHg',
        'cholesterol': 'Serum cholesterol in mg/dL',
        'fasting_blood_sugar': 'Fasting blood sugar (0=<120 mg/dL, 1=>120 mg/dL)',
        'resting_ecg': 'Resting electrocardiographic results (0=Normal, 1=Abnormal, 2=Left Ventricular Hypertrophy)',
        'max_heart_rate': 'Maximum heart rate achieved during exercise',
        'exercise_induced_angina': 'Exercise-induced angina (0=No, 1=Yes)',
        'st_depression': 'ST depression induced by exercise relative to rest',
        'st_slope': 'Slope of the ST segment (0=Upsloping, 1=Flat, 2=Downsloping)',
        'num_major_vessels': 'Number of major vessels colored by fluoroscopy (0-4)',
        'thalassemia': 'Thalassemia type (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown)'
    }
    
    return descriptions


def get_feature_units() -> dict:
    """
    Get units for all features
    """
    
    units = {
        'age': 'years',
        'sex': 'categorical',
        'chest_pain_type': 'categorical',
        'resting_blood_pressure': 'mmHg',
        'cholesterol': 'mg/dL',
        'fasting_blood_sugar': 'categorical',
        'resting_ecg': 'categorical',
        'max_heart_rate': 'bpm',
        'exercise_induced_angina': 'categorical',
        'st_depression': 'mm',
        'st_slope': 'categorical',
        'num_major_vessels': 'count',
        'thalassemia': 'categorical'
    }
    
    return units


def get_normal_ranges() -> dict:
    """
    Get normal/healthy ranges for features (for informational purposes)
    """
    
    ranges = {
        'age': 'Any age',
        'sex': 'N/A',
        'chest_pain_type': '0 = Typical Angina',
        'resting_blood_pressure': '< 120 mmHg (Normal)',
        'cholesterol': '< 200 mg/dL (Desirable)',
        'fasting_blood_sugar': '0 = < 120 mg/dL',
        'resting_ecg': '0 = Normal',
        'max_heart_rate': 'Age-dependent',
        'exercise_induced_angina': '0 = No angina',
        'st_depression': '< 1 mm',
        'st_slope': '0 or 1 (better)',
        'num_major_vessels': '0 vessels (no blockage)',
        'thalassemia': '0 = Normal'
    }
    
    return ranges
