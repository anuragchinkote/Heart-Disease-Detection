"""
Heart Disease Detection - Streamlit Frontend
Interactive web UI for making heart disease predictions
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM STYLING
# =====================================================================

st.markdown("""
    <style>
    .main {
        padding-top: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }
    
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .disease-positive {
        background-color: #ffcccc;
        border-left: 4px solid #cc0000;
    }
    
    .disease-negative {
        background-color: #ccffcc;
        border-left: 4px solid #00cc00;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .header-title {
        text-align: center;
        color: #e74c3c;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# CONFIGURATION
# =====================================================================

# API URL - Use environment variable if available (for Docker/Render deployment)
# Default to localhost for local development
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Feature names in correct order
FEATURE_NAMES = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    'cholesterol', 'fasting_blood_sugar', 'resting_ecg',
    'max_heart_rate', 'exercise_induced_angina', 'st_depression',
    'st_slope', 'num_major_vessels', 'thalassemia'
]

FEATURE_DESCRIPTIONS = {
    'age': 'Age of the patient in years',
    'sex': 'Sex (0=Female, 1=Male)',
    'chest_pain_type': 'Type of chest pain (0-3)',
    'resting_blood_pressure': 'Resting blood pressure (mmHg)',
    'cholesterol': 'Serum cholesterol (mg/dL)',
    'fasting_blood_sugar': 'Fasting blood sugar (0=<120, 1=>120)',
    'resting_ecg': 'Resting ECG (0-2)',
    'max_heart_rate': 'Maximum heart rate achieved',
    'exercise_induced_angina': 'Exercise-induced angina (0=No, 1=Yes)',
    'st_depression': 'ST depression induced by exercise',
    'st_slope': 'ST segment slope (0-2)',
    'num_major_vessels': 'Number of major vessels (0-4)',
    'thalassemia': 'Thalassemia type (0-3)'
}

# =====================================================================
# HEADER SECTION
# =====================================================================

st.markdown("<h1 style='text-align: center; color: #e74c3c;'>Heart Disease Detection</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: #666;'>ML-Powered Clinical Decision Support System</p>", 
            unsafe_allow_html=True)
st.divider()

# =====================================================================
# API HEALTH CHECK
# =====================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200, response.json()
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to API"}
    except Exception as e:
        return False, {"error": str(e)}

# =====================================================================
# MAKE PREDICTION
# =====================================================================

def make_prediction(patient_data):
    """
    Send prediction request to backend API
    
    Args:
        patient_data: Dictionary with patient features
    
    Returns:
        Tuple of (success, response)
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to API. Make sure backend is running on port 8000"}
    except requests.exceptions.Timeout:
        return False, {"error": "API request timeout"}
    except Exception as e:
        return False, {"error": f"Error: {str(e)}"}


# =====================================================================
# SIDEBAR - CONFIGURATION
# =====================================================================

with st.sidebar:
    st.header("Configuration")
    
    # API Settings
    st.subheader("API Settings")
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input != API_URL:
        API_URL = api_url_input
    
    # Check API Status
    st.subheader("API Status")
    is_healthy, health_info = check_api_health()
    
    if is_healthy:
        st.success("API is running")
        with st.expander("Details"):
            st.json(health_info)
    else:
        st.error("API is not running")
        st.warning("Make sure backend is running: `uvicorn main:app --reload`")
        with st.expander("Error Details"):
            st.json(health_info)
    
    st.divider()
    
    # Instructions
    st.subheader("Instructions")
    st.markdown("""
    1. **Enter Patient Data**: Fill in the form with patient information
    2. **Clinical Features**: Provide medical measurements
    3. **Make Prediction**: Click "Predict Disease Risk"
    4. **View Results**: See prediction and risk assessment
    
    **Disclaimer**: This system is for educational purposes only. 
    Always consult qualified healthcare professionals for medical decisions.
    """)
    
    st.divider()
    
    # About
    st.subheader("About")
    st.markdown("""
    **Version**: 1.0.0
    
    **Model**: Ensemble of classification algorithms
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    
    **Dataset**: UCI Heart Disease Dataset
    """)


# =====================================================================
# MAIN CONTENT - TABS
# =====================================================================

tab1, tab2, tab3 = st.tabs(["Prediction", "Information", "FAQ"])

# =====================================================================
# TAB 1: PREDICTION
# =====================================================================

with tab1:
    st.header("Patient Information Input")
    
    # Create form for patient data
    with st.form("patient_form", clear_on_submit=True):
        
        # Demographics Section
        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=0,
                max_value=150,
                value=45,
                help=FEATURE_DESCRIPTIONS['age']
            )
        
        with col2:
            sex = st.radio(
                "Sex",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help=FEATURE_DESCRIPTIONS['sex']
            )
        
        st.divider()
        
        # Cardiac Symptoms Section
        st.subheader("Cardiac Symptoms")
        col1, col2 = st.columns(2)
        
        with col1:
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ][x],
                help=FEATURE_DESCRIPTIONS['chest_pain_type']
            )
        
        with col2:
            exercise_induced_angina = st.radio(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help=FEATURE_DESCRIPTIONS['exercise_induced_angina']
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st_depression = st.number_input(
                "ST Depression (mm)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help=FEATURE_DESCRIPTIONS['st_depression']
            )
        
        with col2:
            st_slope = st.selectbox(
                "ST Segment Slope",
                options=[0, 1, 2],
                format_func=lambda x: [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ][x],
                help=FEATURE_DESCRIPTIONS['st_slope']
            )
        
        st.divider()
        
        # Blood Pressure Section
        st.subheader("Blood Pressure & Heart Rate")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            resting_blood_pressure = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=0,
                max_value=250,
                value=130,
                help=FEATURE_DESCRIPTIONS['resting_blood_pressure']
            )
        
        with col2:
            max_heart_rate = st.number_input(
                "Max Heart Rate (bpm)",
                min_value=0,
                max_value=250,
                value=150,
                help=FEATURE_DESCRIPTIONS['max_heart_rate']
            )
        
        with col3:
            resting_ecg = st.selectbox(
                "Resting ECG",
                options=[0, 1, 2],
                format_func=lambda x: [
                    "Normal",
                    "Abnormal",
                    "LVH"
                ][x],
                help=FEATURE_DESCRIPTIONS['resting_ecg']
            )
        
        st.divider()
        
        # Blood Chemistry Section
        st.subheader("Blood Chemistry")
        col1, col2 = st.columns(2)
        
        with col1:
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=0,
                max_value=600,
                value=240,
                help=FEATURE_DESCRIPTIONS['cholesterol']
            )
        
        with col2:
            fasting_blood_sugar = st.radio(
                "Fasting Blood Sugar",
                options=[0, 1],
                format_func=lambda x: "< 120 mg/dL" if x == 0 else "> 120 mg/dL",
                help=FEATURE_DESCRIPTIONS['fasting_blood_sugar']
            )
        
        st.divider()
        
        # Vessel & Thalassemia Section
        st.subheader("Additional Tests")
        col1, col2 = st.columns(2)
        
        with col1:
            num_major_vessels = st.selectbox(
                "Major Vessels (by fluoroscopy)",
                options=[0, 1, 2, 3, 4],
                help=FEATURE_DESCRIPTIONS['num_major_vessels']
            )
        
        with col2:
            thalassemia = st.selectbox(
                "Thalassemia Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect",
                    "Unknown"
                ][x],
                help=FEATURE_DESCRIPTIONS['thalassemia']
            )
        
        st.divider()
        
        # Submit Button
        submit_button = st.form_submit_button(
            "Predict Disease Risk",
            use_container_width=True
        )
    
    # =====================================================================
    # MAKE PREDICTION AND DISPLAY RESULTS
    # =====================================================================
    
    if submit_button:
        # Prepare patient data
        patient_data = {
            "age": age,
            "sex": sex,
            "chest_pain_type": chest_pain_type,
            "resting_blood_pressure": resting_blood_pressure,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate,
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression,
            "st_slope": st_slope,
            "num_major_vessels": num_major_vessels,
            "thalassemia": thalassemia
        }
        
        # Make prediction
        with st.spinner("Analyzing patient data..."):
            success, response = make_prediction(patient_data)
        
        if success:
            # Extract prediction results
            prediction = response.get('prediction')
            prob_disease = response.get('probability_disease', 0)
            prob_no_disease = response.get('probability_no_disease', 0)
            confidence = response.get('confidence', 0)
            risk_level = response.get('risk_level')
            message = response.get('message')
            
            # Display Results
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            # Main Prediction Box
            if prediction == "Disease":
                color = "#ffcccc"
                border_color = "#cc0000"
                emoji = ""
            else:
                color = "#ccffcc"
                border_color = "#00cc00"
                emoji = ""
            
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; 
                        border-left: 4px solid {border_color};'>
                <h2 style='margin-top: 0; color: {border_color};'>{prediction}</h2>
                <p style='font-size: 16px; margin: 10px 0;'><strong>Risk Level:</strong> {risk_level}</p>
                <p style='font-size: 16px; margin: 10px 0;'><strong>Message:</strong> {message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Probabilities
            st.markdown("### Prediction Probabilities")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Disease Risk",
                    value=f"{prob_disease*100:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="No Disease",
                    value=f"{prob_no_disease*100:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.1f}%",
                    delta=None
                )
            
            # Probability Bar Chart
            prob_data = pd.DataFrame({
                'Category': ['Disease Risk', 'No Disease'],
                'Probability': [prob_disease, prob_no_disease]
            })
            
            st.bar_chart(prob_data.set_index('Category'))
            
            st.markdown("---")
            
            # Patient Summary
            st.markdown("### Patient Summary")
            
            summary_data = {
                'Feature': list(patient_data.keys()),
                'Value': list(patient_data.values())
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Save Result Option
            st.markdown("---")
            st.success("Prediction completed!")
            
        else:
            # Error handling
            st.error("Error making prediction")
            st.error(response.get('error', 'Unknown error occurred'))
            st.info("Make sure the backend API is running on port 8000")
            st.code("uvicorn main:app --reload", language="bash")

# =====================================================================
# TAB 2: INFORMATION
# =====================================================================

with tab2:
    st.header("Model & Feature Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.markdown("""
        **Ensemble Classification Model**
        
        The prediction model uses multiple algorithms:
        
        1. **Logistic Regression** - Linear classifier
        2. **Decision Tree** - Tree-based decision boundaries
        3. **Random Forest** - Ensemble of decision trees
        4. **Support Vector Machine** - Non-linear boundaries
        
        The results are evaluated on:
        - **Recall (Sensitivity)**: Ability to detect disease cases
        - **Precision**: Accuracy of positive predictions
        - **F1 Score**: Balance between precision and recall
        - **ROC-AUC**: Overall discrimination ability
        """)
    
    with col2:
        st.subheader("Model Performance")
        st.markdown("""
        **Evaluation Metrics**
        
        - **Accuracy**: Overall correctness of predictions
        - **Sensitivity**: True positive rate
        - **Specificity**: True negative rate
        - **ROC-AUC**: Area under the ROC curve (0.5-1.0)
        
        **Dataset**
        - Source: UCI Heart Disease Dataset
        - Samples: ~300
        - Features: 13
        - Target: Binary (Disease/No Disease)
        """)
    
    st.divider()
    
    st.subheader("Feature Descriptions")
    
    for feature in FEATURE_NAMES:
        with st.expander(f"{feature.replace('_', ' ').title()}"):
            st.write(FEATURE_DESCRIPTIONS[feature])
    
    st.divider()
    
    st.subheader("Important Notes")
    st.warning("""
    **Clinical Disclaimer**:
    - This tool is for **educational purposes only**
    - Not intended for actual medical diagnosis
    - Always consult qualified healthcare professionals
    - Model predictions should not replace clinical judgment
    - Results may have limitations based on training data
    """)

# =====================================================================
# TAB 3: FAQ
# =====================================================================

with tab3:
    st.header("Frequently Asked Questions")
    
    faq_items = [
        {
            "question": "How accurate is this prediction?",
            "answer": """
            The model is trained on the UCI Heart Disease dataset and evaluated using multiple metrics.
            However, this is an educational tool and should not be used for actual medical decisions.
            Always consult healthcare professionals for accurate diagnosis.
            """
        },
        {
            "question": "What does each feature mean?",
            "answer": """
            Each feature represents a clinical or demographic characteristic:
            - Age, Sex: Patient demographics
            - Chest pain type, Exercise-induced angina: Symptoms
            - Blood pressure, Heart rate: Vital signs
            - Cholesterol, Blood sugar: Blood chemistry
            - ECG, ST segment: Electrical heart activity
            - Vessels, Thalassemia: Additional clinical findings
            """
        },
        {
            "question": "What does the risk level mean?",
            "answer": """
            - **Low Risk** (< 40%): Patient likely does not have heart disease
            - **Medium Risk** (40-70%): Further evaluation recommended
            - **High Risk** (> 70%): Immediate medical consultation advised
            """
        },
        {
            "question": "How should I use these results?",
            "answer": """
            1. Use results as a reference only
            2. Do not make medical decisions based solely on this tool
            3. Always consult qualified healthcare professionals
            4. Results should complement, not replace, clinical evaluation
            """
        },
        {
            "question": "What is the backend API?",
            "answer": """
            The backend is a FastAPI server that:
            - Loads the trained ML model
            - Preprocesses patient data
            - Makes predictions
            - Returns probability scores
            
            Run it with: `uvicorn main:app --reload`
            """
        },
        {
            "question": "Can I use this on my own server?",
            "answer": """
            Yes! Generate the project files and:
            1. Set up the backend with FastAPI
            2. Run Streamlit frontend
            3. Configure the API URL in the sidebar
            4. Deploy on cloud platforms (Render, Heroku, etc.)
            """
        }
    ]
    
    for item in faq_items:
        with st.expander(f"{item['question']}"):
            st.write(item['answer'])
    
    st.divider()
    
    st.subheader("Support")
    st.markdown("""
    For technical issues:
    1. Check that backend API is running
    2. Verify API URL in sidebar settings
    3. Check API health status indicator
    4. Review error messages for debugging
    
    For medical questions:
    - Consult a healthcare professional
    - Do not rely solely on this tool for diagnosis
    """)

# =====================================================================
# FOOTER
# =====================================================================

st.divider()
st.markdown("""
<p style='text-align: center; color: #999; font-size: 12px;'>
    Heart Disease Detection System v1.0.0 | 
    Educational Use Only | 
    © 2024 All Rights Reserved
</p>
<p style='text-align: center; color: #999; font-size: 12px;'>
    Medical Disclaimer: This tool is not intended for actual medical diagnosis.
    Always consult qualified healthcare professionals for medical decisions.
</p>
""", unsafe_allow_html=True)
