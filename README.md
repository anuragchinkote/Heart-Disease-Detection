#  Heart Disease Detection System

A machine learning-powered web application for predicting heart disease using clinical and demographic features. Built with Python, FastAPI, and Streamlit.


---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Installation & Setup](#installation--setup)
- [Folder Structure Explanation](#folder-structure-explanation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Google Colab Setup](#google-colab-setup)
- [Deployment on Render](#deployment-on-render)
- [Docker Setup](#docker-setup)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

##  Overview

This project implements a complete machine learning pipeline for heart disease detection:

- **Data Preparation**: Loading, preprocessing, and feature scaling
- **Model Training**: Multiple classification algorithms compared
- **Evaluation**: Comprehensive metrics (Recall, Precision, F1, ROC-AUC)
- **API Backend**: FastAPI for serving predictions
- **Web Frontend**: Interactive Streamlit interface
- **Deployment**: Ready for cloud deployment (Render, Heroku, AWS)

The system achieves high accuracy in predicting heart disease risk and provides detailed probability scores and risk assessments.

---

##  Features

### Machine Learning
-  Multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, SVM)
-  Automatic model selection based on ROC-AUC score
-  Feature scaling and normalization
-  Train/Test split (80/20)
-  Comprehensive evaluation metrics
-  Confusion matrix and ROC curves

### Backend API
-  RESTful API with FastAPI
-  Input validation (Pydantic)
-  Error handling and logging
-  Model persistence (joblib)
-  CORS enabled for frontend integration
-  Health check endpoint

### Frontend UI
-  Interactive web interface (Streamlit)
-  Patient data input form
-  Real-time predictions
-  Risk level assessment
-  Probability visualization
-  Feature information and descriptions
-  FAQ and medical disclaimers

### DevOps
-  Modular folder structure
-  Requirements management (pip)
-  Docker containerization (optional)
-  GitHub integration
-  Render deployment ready

---

##  Project Structure

```
CapstoneProject/
│
├── frontend/                    # Streamlit Web Interface
│   ├── app.py                  # Main Streamlit application
│   └── requirements.txt         # Frontend dependencies
│
├── backend/                     # FastAPI Backend Server
│   ├── main.py                 # API entry point
│   ├── predict.py              # Prediction logic
│   ├── preprocess.py           # Data preprocessing
│   ├── requirements.txt         # Backend dependencies
│   └── model/                  # Model files directory
│       ├── model.pkl           # Trained ML model
│       ├── scaler.pkl          # Feature scaler
│       ├── feature_names.pkl   # Feature names list
│       ├── model_comparison.csv # Model metrics
│       ├── confusion_matrix.png # Confusion matrix plot
│       ├── roc_curve.png        # ROC curves
│       └── feature_importance.png # Feature importance
│
├── training/                    # Model Training
│   └── train.py                # Training script
│
├── dataset/                     # Data Directory
│   └── heart_disease_dataset.csv # Original dataset
│
├── model/                       # Model Storage
│   └── (model files go here)
│
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
└── Dockerfile                   # Docker configuration
```

---

##  Technical Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0
- **ML Libraries**: scikit-learn 1.3.2, pandas 2.1.3, numpy 1.26.2
- **Model Persistence**: joblib 1.3.2
- **Validation**: Pydantic 2.4.2

### Frontend
- **Framework**: Streamlit 1.29.0
- **HTTP Client**: requests 2.31.0
- **Data**: pandas 2.1.3, numpy 1.26.2

### ML Algorithms
1. **Logistic Regression** - Linear classifier
2. **Decision Tree** - Tree-based classification
3. **Random Forest** - Ensemble method (100 estimators)
4. **Support Vector Machine** - Non-linear boundaries

### Deployment
- **Container**: Docker
- **Cloud**: Render, Heroku, AWS
- **Version Control**: Git/GitHub

---

##  Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git
- (Optional) Docker

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd CapstoneProject
```

### Step 2: Create Virtual Environment
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows CMD
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies
```bash
cd ../frontend
pip install -r requirements.txt
```

### Step 5: Prepare Dataset
Place `heart_disease_dataset.csv` in the `dataset/` folder:
```bash
cp ../Data/heart_disease_dataset.csv ../dataset/
```

---

##  Folder Structure Explanation

### `/frontend`
**Purpose**: Streamlit web application for user interaction

- `app.py`: Main Streamlit app with:
  - Patient data input form
  - Prediction results display
  - Risk assessment visualization
  - Information and FAQ tabs
  - API health status

### `/backend`
**Purpose**: FastAPI server for ML predictions

- `main.py`: API endpoints:
  - `GET /` - Root endpoint
  - `GET /health` - Health check
  - `POST /predict` - Make prediction
  
- `predict.py`: Prediction functions:
  - Load model and scaler
  - Prepare features
  - Calculate probabilities
  - Risk assessment
  
- `preprocess.py`: Data utilities:
  - Feature validation
  - Normalization
  - Statistics calculation
  - Feature descriptions

### `/training`
**Purpose**: Model training and evaluation

- `train.py`: Complete training pipeline:
  - Data loading and EDA
  - Feature analysis
  - Model training (4 algorithms)
  - Evaluation metrics
  - Model comparison
  - Best model selection
  - Visualization generation

### `/dataset`
**Purpose**: Data storage

- `heart_disease_dataset.csv`: UCI Heart Disease dataset
  - 303 patients
  - 13 features
  - Binary target variable

### `/model`
**Purpose**: Trained model and artifacts

- `model.pkl`: Best trained model
- `scaler.pkl`: StandardScaler object
- `feature_names.pkl`: Feature names list
- `model_comparison.csv`: Model performance metrics
- PNG visualizations: Confusion matrix, ROC curve, Feature importance

---

##  Running the Application

### Option 1: Local Development

**Terminal 1 - Start Backend**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Terminal 2 - Start Frontend**
```bash
cd frontend
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### Option 2: Production Deployment

```bash
# Backend
python -m uvicorn main:app --port 8000

# Frontend
streamlit run app.py --logger.level=error
```

---

##  API Documentation

### Health Check Endpoint
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_names_loaded": true
}
```

### Prediction Endpoint
```bash
POST /predict
```

**Request**:
```json
{
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
```

**Response**:
```json
{
  "prediction": "No Disease",
  "probability_no_disease": 0.85,
  "probability_disease": 0.15,
  "confidence": 0.85,
  "risk_level": "Low",
  "message": "Low risk of heart disease based on clinical features."
}
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

##  Model Details

### Training Data
- **Source**: UCI Heart Disease Dataset
- **Samples**: ~303 patients
- **Features**: 13 clinical and demographic variables
- **Target**: Binary (0=No Disease, 1=Disease)
- **Train/Test Split**: 80/20

### Features
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Patient age | Numeric | 0-150 |
| sex | Sex (0=F, 1=M) | Categorical | 0-1 |
| chest_pain_type | Type of chest pain | Categorical | 0-3 |
| resting_blood_pressure | Resting BP | Numeric | 0-250 |
| cholesterol | Serum cholesterol | Numeric | 0-600 |
| fasting_blood_sugar | Fasting BS (0=<120, 1=>120) | Categorical | 0-1 |
| resting_ecg | Resting ECG results | Categorical | 0-2 |
| max_heart_rate | Max heart rate | Numeric | 0-250 |
| exercise_induced_angina | Exercise angina (0=No, 1=Yes) | Categorical | 0-1 |
| st_depression | ST depression | Numeric | 0-10 |
| st_slope | ST segment slope | Categorical | 0-2 |
| num_major_vessels | Major vessels | Numeric | 0-4 |
| thalassemia | Thalassemia type | Categorical | 0-3 |

### Model Performance Metrics
- **Accuracy**: % of correct predictions
- **Precision**: % of positive predictions that are correct
- **Recall**: % of actual positive cases detected
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5-1.0)
- **Specificity**: % of negative cases correctly identified

### Confusion Matrix Elements
- **TP**: True Positives (correctly predicted disease)
- **TN**: True Negatives (correctly predicted no disease)
- **FP**: False Positives (incorrectly predicted disease)
- **FN**: False Negatives (incorrectly missed disease)

---

##  Google Colab Setup

### Step 1: Prepare the Notebook
```python
# In Colab, install dependencies
!pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 2: Upload Dataset
```python
from google.colab import files
uploaded = files.upload()
```

### Step 3: Modify Training Script
Update the data path in `train.py`:
```python
# Change from:
df = pd.read_csv('../dataset/heart_disease_dataset.csv')

# To:
df = pd.read_csv('heart_disease_dataset.csv')
```

### Step 4: Run Training
```python
%run train.py
```

### Step 5: Download Model Files
```python
from google.colab import files
files.download('model.pkl')
files.download('scaler.pkl')
files.download('feature_names.pkl')
```

### Step 6: Upload to Backend
Place the downloaded files in `backend/model/` directory

---

##  Deployment on Render

### Prerequisites
- GitHub account with repository
- Render account (render.com)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Deploy Backend on Render

1. Go to [render.com](https://render.com)
2. Click "New+" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: heart-disease-api
   - **Runtime**: Python 3.10
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000`
   - **Environment Variables**:
     - `PYTHON_VERSION=3.10.13`

5. Click "Create Web Service"
6. Get your API URL (e.g., https://heart-disease-api.onrender.com)

### Step 3: Deploy Frontend on Render

1. Click "New+" → "Web Service"
2. Configure:
   - **Name**: heart-disease-ui
   - **Runtime**: Python 3.10
   - **Build Command**: `pip install -r frontend/requirements.txt`
   - **Start Command**: `cd frontend && streamlit run app.py --server.port=8501 --server.address=0.0.0.0`
   - **Environment Variables**:
     - `API_URL=https://heart-disease-api.onrender.com`

3. Click "Create Web Service"

### Step 4: Update Frontend Config
In `frontend/app.py`, set API URL:
```python
API_URL = "https://heart-disease-api.onrender.com"
```

---

##  Docker Setup

### Build Docker Image
```bash
docker build -t heart-disease-api .
```

### Run Container
```bash
docker run -p 8000:8000 heart-disease-api
```

### Docker Compose (Optional)
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./model:/app/model

  frontend:
    image: streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
```

Run:
```bash
docker-compose up
```

---

##  Testing

### Test API with cURL
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45.0, "sex": 1.0, ...}'
```

### Test with Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 45.0,
        "sex": 1.0,
        # ... all other features
    }
)

print(response.json())
```

### Test Frontend
1. Open http://localhost:8501
2. Fill in patient data
3. Click "Predict Disease Risk"
4. Verify results display

---

##  Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`
```bash
# Solution: Install requirements
pip install -r backend/requirements.txt
```

**Problem**: `Port 8000 already in use`
```bash
# Solution: Use different port
python -m uvicorn main:app --port 8001
```

**Problem**: `Model not found`
```bash
# Solution: Ensure model files are in backend/model/
# Run training script first: python training/train.py
```

### Frontend Issues

**Problem**: `Cannot connect to API`
```
1. Check backend is running
2. Verify API URL in sidebar
3. Check health status indicator
4. Review backend logs
```

**Problem**: `Streamlit not found`
```bash
pip install streamlit==1.29.0
```

### Model Loading Issues

**Problem**: `FileNotFoundError: model.pkl`
```bash
# Solution: Train model first
cd training
python train.py
```

---

##  Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Streamlit Frontend   │
          │   (Port 8501)         │
          │  - Input Form         │
          │  - Results Display    │
          │  - FAQ & Info         │
          └──────────┬────────────┘
                     │
                     │ HTTP POST /predict
                     ▼
          ┌───────────────────────┐
          │   FastAPI Backend     │
          │   (Port 8000)         │
          │  - Validation         │
          │  - Prediction Logic   │
          │  - Error Handling     │
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │   ML Model Files      │
          │  - model.pkl          │
          │  - scaler.pkl         │
          │  - feature_names.pkl  │
          └───────────────────────┘
```

---

##  Security Considerations

-  Input validation using Pydantic
-  Type hints for error detection
-  CORS enabled for frontend
-  Model not exposed directly
-  Error messages don't leak sensitive info

**For Production**:
- Restrict CORS origins
- Use HTTPS
- Implement authentication
- Add rate limiting
- Monitor API usage

---

##  Performance

- **Prediction Time**: < 100ms
- **Model Size**: ~2 MB
- **Memory Usage**: ~200 MB
- **API Response**: < 500ms
- **Scalability**: Renders free tier can handle ~100 concurrent users

---

##  Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

---

##  License

This project is licensed under the MIT License - see LICENSE file for details.

---

##  Disclaimer

### IMPORTANT MEDICAL DISCLAIMER

**This system is for educational purposes only and should NOT be used for actual medical diagnosis or treatment decisions.**

- This tool does not replace professional medical advice
- Predictions are based on a machine learning model trained on historical data
- Results may have limitations and errors
- Always consult qualified healthcare professionals for medical diagnosis
- Do not make medical decisions based solely on this system's output
- The developers are not responsible for any medical decisions made using this tool


---

