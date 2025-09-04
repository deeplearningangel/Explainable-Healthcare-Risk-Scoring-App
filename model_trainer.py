"""
Model Training Module for Healthcare Risk Scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from config import MODELS_DIR, MODEL_CONFIG, STATIC_DIR

class HealthcareRiskModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
        self.feature_names = []
    
    def initialize_models(self):
        """Initialize machine learning models"""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                **MODEL_CONFIG['models']['xgboost'],
                random_state=MODEL_CONFIG['random_state']
            ),
            'random_forest': RandomForestClassifier(
                **MODEL_CONFIG['models']['random_forest'],
                random_state=MODEL_CONFIG['random_state']
            ),
            'logistic_regression': Logistic
