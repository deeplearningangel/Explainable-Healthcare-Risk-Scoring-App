"""
Data Preprocessing Module for Healthcare Risk Scoring
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import logging
from config import DATA_DIR, MODELS_DIR, MODEL_CONFIG

class HealthcareDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def identify_feature_types(self, df):
        """Identify categorical and numerical features"""
        # Exclude target variables
        feature_columns = [col for col in df.columns if col not in ['high_risk', 'risk_probability']]
        
        categorical_features = []
        numerical_features = []
        
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Handle binary features (should be treated as categorical)
        binary_features = [col for col in numerical_features 
                          if df[col].nunique() == 2 and set(df[col].unique()) <= {0, 1}]
        
        for col in binary_features:
            numerical_features.remove(col)
            categorical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        logging.info(f"Categorical features: {categorical_features}")
        logging.info(f"Numerical features: {numerical_features}")
        
        return categorical_features, numerical_features
    
    def create_preprocessing_pipeline(self, df):
        """Create preprocessing pipeline"""
        categorical_features, numerical_features = self.identify_feature_types(df)
        
        # Numerical features: StandardScaler
        numerical_transformer = StandardScaler()
        
        # Categorical features: OneHotEncoder
        categorical_transformer = OneHotEncoder(
            drop='first', 
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def fit_transform_data(self, df):
        """Fit preprocessor and transform data"""
        # Separate features and target
        X = df.drop(['high_risk', 'risk_probability'], axis=1)
        y = df['high_risk']
        
        # Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline(X)
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        self.feature_names = self.get_feature_names()
        
        return X_transformed, y
    
    def transform_data(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform_data first.")
        
        X = df.drop(['high_risk', 'risk_probability'], axis=1, errors='ignore')
        X_transformed = self.preprocessor.transform(X)
        
        return X_transformed
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        feature_names = []
        
        # Get numerical feature names
        num_features = self.preprocessor.named_transformers_['num'].get_feature_names_out(
            self.numerical_features
        ) if hasattr(self.preprocessor.named_transformers_['num'], 'get_feature_names_out') else self.numerical_features
        
        # Get categorical feature names
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
            self.categorical_features
        )
        
        feature_names = list(num_features) + list(cat_features)
        return feature_names
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using various techniques"""
        if method == 'smote':
            smote = SMOTE(random_state=MODEL_CONFIG['random_state'])
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info(f"SMOTE applied. Original: {y.value_counts().to_dict()}")
            logging.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
            return X_resampled, y_resampled
        else:
            return X, y
    
    def split_data(self, X, y, test_size=None, validation_size=None):
        """Split data into train, validation, and test sets"""
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
        if validation_size is None:
            validation_size = MODEL_CONFIG['validation_size']
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=MODEL_CONFIG['random_state']
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted,
            stratify=y_trainval, random_state=MODEL_CONFIG['random_state']
        )
        
        logging.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_complete_pipeline(self, df, handle_imbalance=True, save_preprocessor=True):
        """Complete preprocessing pipeline"""
        logging.info("Starting data preprocessing...")
        
        # Fit and transform data
        X_transformed, y = self.fit_transform_data(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_transformed, y)
        
        # Handle class imbalance on training data only
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)
        
        # Save preprocessor
        if save_preprocessor:
            self.save_preprocessor()
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        return data_splits
    
    def save_preprocessor(self, filename='preprocessor.joblib'):
        """Save the fitted preprocessor"""
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        
        filepath = MODELS_DIR / filename
        joblib.dump(preprocessor_data, filepath)
        logging.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filename='preprocessor.joblib'):
        """Load a saved preprocessor"""
        filepath = MODELS_DIR / filename
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor = preprocessor_data['preprocessor']
        self.feature_names = preprocessor_data['feature_names']
        self.categorical_features = preprocessor_data['categorical_features']
        self.numerical_features = preprocessor_data['numerical_features']
        
        logging.info(f"Preprocessor loaded from {filepath}")
    
    def get_feature_importance_mapping(self):
        """Get mapping between original features and transformed feature names"""
        mapping = {}
        
        # Numerical features (1-to-1 mapping)
        for orig_feature in self.numerical_features:
            transformed_name = f"num__{orig_feature}"
            if transformed_name in self.feature_names:
                mapping[orig_feature] = [transformed_name]
            else:
                # Fallback to original name
                mapping[orig_feature] = [orig_feature]
        
        # Categorical features (1-to-many mapping)
        cat_transformer = self.preprocessor.named_transformers_['cat']
        cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_features)
        
        for orig_feature in self.categorical_features:
            transformed_names = [name for name in cat_feature_names 
                               if name.startswith(f"{orig_feature}_")]
            if not transformed_names:
                # Handle binary features that might not have prefix
                transformed_names = [name for name in cat_feature_names 
                                   if orig_feature in name]
            mapping[orig_feature] = transformed_names
        
        return mapping

def preprocess_healthcare_data(data_file='healthcare_data.csv'):
    """Main function to preprocess healthcare data"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    data_path = DATA_DIR / data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logging.info(f"Loaded data with shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor()
    
    # Run complete preprocessing pipeline
    data_splits = preprocessor.preprocess_complete_pipeline(df)
    
    # Save processed data
    processed_data_path = DATA_DIR / 'processed_data.joblib'
    joblib.dump(data_splits, processed_data_path)
    logging.info(f"Processed data saved to {processed_data_path}")
    
    return data_splits, preprocessor

if __name__ == "__main__":
    data_splits, preprocessor = preprocess_healthcare_data()
