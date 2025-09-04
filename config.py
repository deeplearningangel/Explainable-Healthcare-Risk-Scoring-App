"""
Configuration file for Healthcare Risk Scoring App
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'cv_folds': 5,
    'models': {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'penalty': 'l2'
        }
    }
}

# Risk scoring thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}

# Feature categories for interpretability
FEATURE_CATEGORIES = {
    'demographics': ['age', 'gender', 'race', 'insurance_type'],
    'vital_signs': ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'bmi'],
    'lab_values': ['glucose', 'cholesterol', 'hemoglobin', 'creatinine', 'white_blood_cells'],
    'medical_history': ['diabetes', 'hypertension', 'heart_disease', 'stroke_history', 'cancer_history'],
    'lifestyle': ['smoking', 'alcohol_consumption', 'exercise_frequency'],
    'medications': ['num_medications', 'high_risk_medications'],
    'healthcare_utilization': ['emergency_visits', 'hospitalizations', 'specialist_visits']
}

# Flask app configuration
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'DEBUG': os.environ.get('FLASK_DEBUG', 'True').lower() == 'true',
    'HOST': os.environ.get('FLASK_HOST', '127.0.0.1'),
    'PORT': int(os.environ.get('FLASK_PORT', 5000))
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'app.log'
}
