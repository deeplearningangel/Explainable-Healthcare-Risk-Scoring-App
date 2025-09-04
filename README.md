# Healthcare Risk Scoring App with Explainable AI

A comprehensive machine learning application for predicting healthcare risks with built-in explainability features using SHAP and LIME. This app generates synthetic healthcare data, trains multiple ML models, and provides interpretable risk predictions through a web interface.

## 🌟 Features

- **Synthetic Data Generation**: Creates realistic healthcare datasets with proper correlations
- **Multiple ML Models**: XGBoost, Random Forest, and Logistic Regression
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **Web Interface**: Flask-based dashboard for risk prediction and visualization
- **Model Comparison**: Automated model evaluation and selection
- **Risk Categorization**: Low, Medium, and High risk classifications
- **Feature Importance**: Detailed analysis of contributing factors
- **Data Preprocessing**: Automated feature engineering and scaling

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for complete package list

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd healthcare-risk-scoring-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data_generator.py
```

This creates:
- `data/healthcare_data.csv`: Main dataset with 10,000 synthetic patient records
- `data/data_summary.json`: Dataset metadata and statistics

### 3. Preprocess Data

```bash
python data_preprocessor.py
```

This handles:
- Feature scaling and encoding
- Train/validation/test splits
- Class imbalance correction with SMOTE
- Saves preprocessor pipeline

### 4. Train Models

```bash
python model_trainer.py
```

This trains and evaluates:
- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression
- Saves best performing model

### 5. Run the Web Application

```bash
python app.py
```

Access the application at `http://localhost:5000`

## 📁 Project Structure

```
healthcare-risk-scoring-app/
├── README.md
├── requirements.txt
├── config.py                 # Configuration settings
├── data_generator.py         # Synthetic data generation
├── data_preprocessor.py      # Data preprocessing pipeline
├── model_trainer.py          # Model training and evaluation
├── risk_scorer.py           # Risk scoring with explanations
├── app.py                   # Flask web application
├── data/                    # Data directory
│   ├── healthcare_data.csv
│   ├── processed_data.joblib
│   └── data_summary.json
├── models/                  # Saved models and preprocessors
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── model_metrics.json
├── static/                  # Static files for web app
└── templates/               # HTML templates
    └── index.html
```

## 🏥 Dataset Features

The synthetic healthcare dataset includes:

### Demographics
- Age, Gender, Race, Insurance Type

### Vital Signs
- Systolic/Diastolic Blood Pressure
- Heart Rate, Temperature, BMI

### Laboratory Values
- Glucose, Cholesterol, Hemoglobin
- Creatinine, White Blood Cell Count

### Medical History
- Diabetes, Hypertension, Heart Disease
- Stroke History, Cancer History

### Lifestyle Factors
- Smoking Status, Alcohol Consumption
- Exercise Frequency

### Healthcare Utilization
- Number of Medications
- Emergency Visits, Hospitalizations
- Specialist Visits

## 🤖 Machine Learning Models

### XGBoost Classifier
- Gradient boosting algorithm
- Excellent for structured data
- Built-in feature importance

### Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Natural feature interpretability

### Logistic Regression
- Linear probabilistic model
- Highly interpretable
- Good baseline model

## 🔍 Explainability Features

### SHAP (SHapley Additive exPlanations)
- Global feature importance
- Individual prediction explanations
- Feature interaction analysis
- Waterfall plots for risk factors

### LIME (Local Interpretable Model-agnostic Explanations)
- Local explanations for individual predictions
- Feature contribution visualization
- Model-agnostic approach

## 📊 Web Interface Features

### Risk Prediction Dashboard
- Input patient information
- Real-time risk calculation
- Visual risk categorization (Low/Medium/High)

### Explanation Visualizations
- SHAP summary plots
- Feature importance rankings
- Individual prediction breakdowns
- Risk factor contributions

### Model Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC curves
- Confusion matrices
- Cross-validation results

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'models': {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    }
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}
```

## 📈 Model Performance

Expected performance on synthetic data:
- **Accuracy**: ~85-90%
- **ROC-AUC**: ~0.90-0.95
- **Precision**: ~80-85%
- **Recall**: ~80-85%

## 🔧 Advanced Usage

### Custom Model Training

```python
from model_trainer import HealthcareRiskModelTrainer
from data_preprocessor import preprocess_healthcare_data

# Load preprocessed data
data_splits, preprocessor = preprocess_healthcare_data()

# Initialize and train models
trainer = HealthcareRiskModelTrainer()
trainer.train_all_models(data_splits)
```

### Individual Risk Scoring

```python
from risk_scorer import HealthcareRiskScorer
import pandas as pd

# Initialize scorer
scorer = HealthcareRiskScorer()
scorer.load_model()

# Score individual patient
patient_data = pd.DataFrame({
    'age': [65],
    'systolic_bp': [150],
    'diabetes': [1],
    # ... other features
})

risk_score, explanation = scorer.score_patient_with_explanation(patient_data)
```

## 🧪 Testing

Run basic functionality tests:

```python
# Test data generation
python -c "from data_generator import HealthcareDataGenerator; g = HealthcareDataGenerator(100); g.save_dataset('test_data.csv')"

# Test preprocessing
python -c "from data_preprocessor import preprocess_healthcare_data; preprocess_healthcare_data('test_data.csv')"

# Test model training
python -c "from model_trainer import HealthcareRiskModelTrainer; t = HealthcareRiskModelTrainer(); print('Models initialized successfully')"
```

## 🚨 Important Notes

### Data Privacy
- This application uses **synthetic data only**
- Never use real patient data without proper authorization
- Ensure compliance with HIPAA and other healthcare regulations

### Model Limitations
- Models are trained on synthetic data
- Real-world performance may vary
- Always validate with domain experts
- Not intended for actual medical diagnosis

### Production Considerations
- Implement proper authentication and authorization
- Add input validation and sanitization
- Set up proper logging and monitoring
- Consider model versioning and A/B testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in each Python file
- Review the configuration options in `config.py`

## 🔮 Future Enhancements

- [ ] Integration with real EHR systems
- [ ] Advanced deep learning models
- [ ] Time-series risk prediction
- [ ] Multi-class risk categorization
- [ ] API endpoints for integration
- [ ] Automated model retraining
- [ ] Advanced visualization dashboards
- [ ] Mobile-responsive interface

## 📚 References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Disclaimer**: This application is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
