"""
Synthetic Healthcare Data Generator
Generates realistic healthcare data for risk scoring model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import DATA_DIR

class HealthcareDataGenerator:
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_demographics(self):
        """Generate demographic features"""
        age = np.random.gamma(shape=3, scale=20, size=self.n_samples).astype(int)
        age = np.clip(age, 18, 95)
        
        gender = np.random.choice(['Male', 'Female'], size=self.n_samples, p=[0.48, 0.52])
        
        race = np.random.choice([
            'White', 'Black', 'Hispanic', 'Asian', 'Other'
        ], size=self.n_samples, p=[0.6, 0.13, 0.18, 0.06, 0.03])
        
        insurance_type = np.random.choice([
            'Private', 'Medicare', 'Medicaid', 'Uninsured'
        ], size=self.n_samples, p=[0.55, 0.18, 0.20, 0.07])
        
        return pd.DataFrame({
            'age': age,
            'gender': gender,
            'race': race,
            'insurance_type': insurance_type
        })
    
    def generate_vital_signs(self, demographics):
        """Generate vital signs correlated with age"""
        age = demographics['age'].values
        
        # Blood pressure increases with age
        systolic_bp = 110 + age * 0.5 + np.random.normal(0, 15, self.n_samples)
        diastolic_bp = 70 + age * 0.2 + np.random.normal(0, 10, self.n_samples)
        
        # Heart rate with some age correlation
        heart_rate = 80 - age * 0.1 + np.random.normal(0, 12, self.n_samples)
        
        # Temperature (mostly normal)
        temperature = np.random.normal(98.6, 1.2, self.n_samples)
        
        # BMI with age correlation
        bmi = 22 + age * 0.08 + np.random.normal(0, 4, self.n_samples)
        bmi = np.clip(bmi, 15, 50)
        
        return pd.DataFrame({
            'systolic_bp': np.clip(systolic_bp, 90, 200),
            'diastolic_bp': np.clip(diastolic_bp, 50, 120),
            'heart_rate': np.clip(heart_rate, 50, 120),
            'temperature': np.clip(temperature, 96, 105),
            'bmi': bmi
        })
    
    def generate_lab_values(self, demographics):
        """Generate lab values with some correlations"""
        age = demographics['age'].values
        
        glucose = 90 + age * 0.3 + np.random.gamma(2, 8, self.n_samples)
        cholesterol = 160 + age * 0.8 + np.random.normal(0, 30, self.n_samples)
        hemoglobin = 14 - age * 0.02 + np.random.normal(0, 1.5, self.n_samples)
        creatinine = 0.8 + age * 0.005 + np.random.gamma(2, 0.2, self.n_samples)
        white_blood_cells = np.random.gamma(3, 2.5, self.n_samples)
        
        return pd.DataFrame({
            'glucose': np.clip(glucose, 60, 400),
            'cholesterol': np.clip(cholesterol, 120, 350),
            'hemoglobin': np.clip(hemoglobin, 8, 18),
            'creatinine': np.clip(creatinine, 0.5, 5.0),
            'white_blood_cells': np.clip(white_blood_cells, 2, 20)
        })
    
    def generate_medical_history(self, demographics, vital_signs, lab_values):
        """Generate medical history with realistic correlations"""
        age = demographics['age'].values
        bmi = vital_signs['bmi'].values
        glucose = lab_values['glucose'].values
        bp = vital_signs['systolic_bp'].values
        
        # Diabetes probability increases with age, BMI, and glucose
        diabetes_prob = 0.05 + (age - 18) * 0.002 + np.maximum(0, bmi - 25) * 0.01 + np.maximum(0, glucose - 100) * 0.001
        diabetes = np.random.binomial(1, np.clip(diabetes_prob, 0, 0.8), self.n_samples)
        
        # Hypertension probability
        hypertension_prob = 0.1 + (age - 18) * 0.005 + np.maximum(0, bp - 130) * 0.005
        hypertension = np.random.binomial(1, np.clip(hypertension_prob, 0, 0.9), self.n_samples)
        
        # Heart disease
        heart_disease_prob = 0.02 + (age - 18) * 0.003 + diabetes * 0.1 + hypertension * 0.08
        heart_disease = np.random.binomial(1, np.clip(heart_disease_prob, 0, 0.6), self.n_samples)
        
        # Stroke history
        stroke_prob = 0.01 + (age - 18) * 0.002 + hypertension * 0.05 + heart_disease * 0.1
        stroke_history = np.random.binomial(1, np.clip(stroke_prob, 0, 0.4), self.n_samples)
        
        # Cancer history
        cancer_prob = 0.02 + (age - 18) * 0.004
        cancer_history = np.random.binomial(1, np.clip(cancer_prob, 0, 0.5), self.n_samples)
        
        return pd.DataFrame({
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'stroke_history': stroke_history,
            'cancer_history': cancer_history
        })
    
    def generate_lifestyle_factors(self, demographics):
        """Generate lifestyle factors"""
        age = demographics['age'].values
        
        # Smoking decreases with age (modern trend)
        smoking_prob = 0.3 - (age - 18) * 0.003
        smoking = np.random.binomial(1, np.clip(smoking_prob, 0.05, 0.4), self.n_samples)
        
        alcohol_consumption = np.random.choice([
            'None', 'Light', 'Moderate', 'Heavy'
        ], size=self.n_samples, p=[0.3, 0.4, 0.25, 0.05])
        
        exercise_frequency = np.random.choice([
            'Never', 'Rarely', 'Sometimes', 'Often', 'Daily'
        ], size=self.n_samples, p=[0.2, 0.25, 0.3, 0.2, 0.05])
        
        return pd.DataFrame({
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption,
            'exercise_frequency': exercise_frequency
        })
    
    def generate_medications_utilization(self, demographics, medical_history):
        """Generate medication and healthcare utilization data"""
        age = demographics['age'].values
        
        # Number of medications increases with age and conditions
        base_meds = age * 0.05
        condition_meds = (medical_history[['diabetes', 'hypertension', 'heart_disease']].sum(axis=1) * 2)
        num_medications = base_meds + condition_meds + np.random.poisson(1, self.n_samples)
        num_medications = np.clip(num_medications, 0, 20)
        
        # High-risk medications
        high_risk_prob = 0.1 + (num_medications > 5) * 0.2
        high_risk_medications = np.random.binomial(1, high_risk_prob, self.n_samples)
        
        # Healthcare utilization
        emergency_visits = np.random.poisson(0.5 + medical_history.sum(axis=1) * 0.3, self.n_samples)
        hospitalizations = np.random.poisson(0.2 + medical_history.sum(axis=1) * 0.2, self.n_samples)
        specialist_visits = np.random.poisson(1 + medical_history.sum(axis=1) * 0.5, self.n_samples)
        
        return pd.DataFrame({
            'num_medications': num_medications.astype(int),
            'high_risk_medications': high_risk_medications,
            'emergency_visits': emergency_visits,
            'hospitalizations': hospitalizations,
            'specialist_visits': specialist_visits
        })
    
    def generate_risk_outcome(self, data):
        """Generate the target risk outcome based on all features"""
        # Create risk score based on multiple factors
        risk_score = 0
        
        # Age factor
        risk_score += (data['age'] - 18) * 0.01
        
        # Vital signs
        risk_score += np.maximum(0, data['systolic_bp'] - 130) * 0.005
        risk_score += np.maximum(0, data['bmi'] - 25) * 0.02
        
        # Lab values
        risk_score += np.maximum(0, data['glucose'] - 100) * 0.002
        risk_score += np.maximum(0, data['cholesterol'] - 200) * 0.001
        
        # Medical history (major impact)
        risk_score += data['diabetes'] * 0.3
        risk_score += data['hypertension'] * 0.2
        risk_score += data['heart_disease'] * 0.4
        risk_score += data['stroke_history'] * 0.35
        risk_score += data['cancer_history'] * 0.25
        
        # Lifestyle factors
        risk_score += data['smoking'] * 0.15
        
        # Healthcare utilization
        risk_score += data['emergency_visits'] * 0.05
        risk_score += data['hospitalizations'] * 0.1
        
        # Add some noise
        risk_score += np.random.normal(0, 0.1, self.n_samples)
        
        # Convert to probability and then binary outcome
        risk_prob = 1 / (1 + np.exp(-risk_score))
        high_risk = np.random.binomial(1, risk_prob, self.n_samples)
        
        return high_risk, risk_prob
    
    def generate_complete_dataset(self):
        """Generate complete synthetic healthcare dataset"""
        print("Generating demographics...")
        demographics = self.generate_demographics()
        
        print("Generating vital signs...")
        vital_signs = self.generate_vital_signs(demographics)
        
        print("Generating lab values...")
        lab_values = self.generate_lab_values(demographics)
        
        print("Generating medical history...")
        medical_history = self.generate_medical_history(demographics, vital_signs, lab_values)
        
        print("Generating lifestyle factors...")
        lifestyle = self.generate_lifestyle_factors(demographics)
        
        print("Generating medications and utilization...")
        med_util = self.generate_medications_utilization(demographics, medical_history)
        
        # Combine all features
        data = pd.concat([demographics, vital_signs, lab_values, medical_history, 
                         lifestyle, med_util], axis=1)
        
        print("Generating risk outcomes...")
        high_risk, risk_prob = self.generate_risk_outcome(data)
        
        data['high_risk'] = high_risk
        data['risk_probability'] = risk_prob
        
        return data
    
    def save_dataset(self, filename='healthcare_data.csv'):
        """Generate and save the dataset"""
        data = self.generate_complete_dataset()
        filepath = DATA_DIR / filename
        data.to_csv(filepath, index=False)
        
        # Save data summary
        summary = {
            'n_samples': self.n_samples,
            'n_features': len(data.columns) - 2,  # Exclude target variables
            'high_risk_rate': data['high_risk'].mean(),
            'columns': list(data.columns)
        }
        
        with open(DATA_DIR / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {data.shape}")
        print(f"High-risk rate: {data['high_risk'].mean():.3f}")
        
        return data

if __name__ == "__main__":
    generator = HealthcareDataGenerator(n_samples=10000)
    data = generator.save_dataset()
