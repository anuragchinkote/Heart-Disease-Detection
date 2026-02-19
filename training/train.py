"""
Heart Disease Detection - ML Model Training Script
This script trains multiple classification models and evaluates them.
Compatible with Google Colab and local environments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, ConfusionMatrixDisplay
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: DATA LOADING
# =====================================================================
print("="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# For Google Colab - uncomment these lines and upload the CSV file
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('heart_disease_dataset.csv')

# For local environment - specify the path
df = pd.read_csv('../dataset/heart_disease_dataset.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Total Samples: {df.shape[0]}")
print(f"  Total Features: {df.shape[1]}")

# =====================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n First 5 rows:")
print(df.head())

print("\n Dataset Information:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe())

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Target Variable Distribution:")
print(df['heart_disease'].value_counts())
print(f"\nClass Balance:")
print(df['heart_disease'].value_counts(normalize=True) * 100)

# =====================================================================
# STEP 3: FEATURE ANALYSIS
# =====================================================================
print("\n" + "="*70)
print("STEP 3: FEATURE ANALYSIS")
print("="*70)

features = [col for col in df.columns if col != 'heart_disease']
print(f"\n✓ Total Features: {len(features)}")
print(f"Features: {features}")

print(f"\nFeature Types:")
print(df[features].dtypes)

print(f"\nFeature Ranges:")
for col in features:
    print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

# =====================================================================
# STEP 4: PREPARE DATA FOR TRAINING
# =====================================================================
print("\n" + "="*70)
print("STEP 4: DATA PREPARATION")
print("="*70)

# Separate features and target
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Split data - 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features standardized using StandardScaler")

# Save scaler for later use
joblib.dump(scaler, '../model/scaler.pkl')
print(f"✓ Scaler saved to model/scaler.pkl")

# =====================================================================
# STEP 5: TRAIN MULTIPLE MODELS
# =====================================================================
print("\n" + "="*70)
print("STEP 5: TRAINING MODELS")
print("="*70)

models = {}
predictions = {}

# 1. Logistic Regression
print("\n Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
predictions['Logistic Regression'] = lr_model.predict(X_test_scaled)
models['Logistic Regression'] = lr_model
print("   ✓ Logistic Regression trained successfully!")

# 2. Decision Tree
print("\n Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
predictions['Decision Tree'] = dt_model.predict(X_test_scaled)
models['Decision Tree'] = dt_model
print("   ✓ Decision Tree trained successfully!")

# 3. Random Forest
print("\n Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
)
rf_model.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf_model.predict(X_test_scaled)
models['Random Forest'] = rf_model
print("   ✓ Random Forest trained successfully!")

# 4. Support Vector Machine (SVM)
print("\n Training Support Vector Machine (SVM)...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
predictions['SVM'] = svm_model.predict(X_test_scaled)
models['SVM'] = svm_model
print("   ✓ SVM trained successfully!")

# =====================================================================
# STEP 6: EVALUATE MODELS
# =====================================================================
print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION")
print("="*70)

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f" {name.upper()} EVALUATION")
    print(f"{'='*50}")
    
    y_pred = predictions[name]
    
    # Get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'Specificity': specificity,
        'Model': model,
        'Probabilities': y_pred_proba
    }
    
    # Print metrics
    print(f"\n✓ Accuracy:   {accuracy:.4f}")
    print(f"✓ Precision:  {precision:.4f}")
    print(f"✓ Recall:     {recall:.4f}")
    print(f"✓ F1 Score:   {f1:.4f}")
    print(f"✓ ROC-AUC:    {roc_auc:.4f}")
    print(f"✓ Specificity:{specificity:.4f}")
    
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# =====================================================================
# STEP 7: MODEL COMPARISON TABLE
# =====================================================================
print("\n" + "="*70)
print("STEP 7: MODEL COMPARISON TABLE")
print("="*70)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.drop('Model', axis=1).drop('Probabilities', axis=1)
comparison_df = comparison_df.round(4)

print("\n Model Performance Comparison:")
print(comparison_df)

# Save comparison table
comparison_df.to_csv('../model/model_comparison.csv')
print("\n✓ Comparison table saved to model/model_comparison.csv")

# =====================================================================
# STEP 8: SELECT BEST MODEL
# =====================================================================
print("\n" + "="*70)
print("STEP 8: SELECT BEST MODEL")
print("="*70)

# Select based on ROC-AUC (primary metric)
best_model_name = comparison_df['ROC-AUC'].idxmax()
best_model = results[best_model_name]['Model']
best_score = comparison_df['ROC-AUC'].max()

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC Score: {best_score:.4f}")

# Save best model
joblib.dump(best_model, '../model/model.pkl')
print(f"\n✓ Best model saved to model/model.pkl")

# =====================================================================
# STEP 9: CONFUSION MATRIX
# =====================================================================
print("\n" + "="*70)
print("STEP 9: CONFUSION MATRIX")
print("="*70)

y_pred_best = predictions[best_model_name]
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n{best_model_name} Confusion Matrix:")
print(cm)

# Create visualization
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'{best_model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../model/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix plot saved to model/confusion_matrix.png")
plt.close()

# =====================================================================
# STEP 10: ROC CURVE
# =====================================================================
print("\n" + "="*70)
print("STEP 10: ROC CURVE")
print("="*70)

plt.figure(figsize=(10, 8))

for name, result in results.items():
    y_pred_proba = result['Probabilities']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = result['ROC-AUC']
    
    line_style = '--' if name != best_model_name else '-'
    line_width = 2 if name == best_model_name else 1.5
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', 
             linestyle=line_style, linewidth=line_width)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../model/roc_curve.png', dpi=300, bbox_inches='tight')
print("\n✓ ROC curve saved to model/roc_curve.png")
plt.close()

# =====================================================================
# STEP 11: FEATURE IMPORTANCE
# =====================================================================
print("\n" + "="*70)
print("STEP 11: FEATURE IMPORTANCE")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n✓ Top 10 Features ({best_model_name}):")
    print(feature_importance.head(10))
    
    # Plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'{best_model_name} - Top 10 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../model/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved to model/feature_importance.png")
    plt.close()
else:
    print(f"\  {best_model_name} does not have feature importance attribute")

# =====================================================================
# STEP 12: SAVE FEATURE NAMES
# =====================================================================
print("\n" + "="*70)
print("STEP 12: SAVE CONFIGURATION")
print("="*70)

# Save feature names for preprocessing
joblib.dump(X.columns.tolist(), '../model/feature_names.pkl')
print(f"\n✓ Feature names saved to model/feature_names.pkl")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)

print(f"\n Best Model: {best_model_name}")
print(f"   ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f}")
print(f"   Recall: {results[best_model_name]['Recall']:.4f}")
print(f"   Precision: {results[best_model_name]['Precision']:.4f}")
print(f"   F1 Score: {results[best_model_name]['F1 Score']:.4f}")

print(f"\n Saved Files:")
print(f"   ✓ model/model.pkl - Trained model")
print(f"   ✓ model/scaler.pkl - Feature scaler")
print(f"   ✓ model/feature_names.pkl - Feature names")
print(f"   ✓ model/model_comparison.csv - Model comparison table")
print(f"   ✓ model/confusion_matrix.png - Confusion matrix visualization")
print(f"   ✓ model/roc_curve.png - ROC curve visualization")
print(f"   ✓ model/feature_importance.png - Feature importance plot")

print(f"\n Next Steps:")
print(f"   1. Review model performance metrics")
print(f"   2. Download model.pkl, scaler.pkl, and feature_names.pkl")
print(f"   3. Move these files to the backend/model directory")
print(f"   4. Run backend API with: uvicorn main:app --reload")
print(f"   5. Run frontend with: streamlit run app.py")

print("\n" + "="*70)
