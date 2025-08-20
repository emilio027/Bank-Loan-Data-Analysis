"""
Advanced Credit Risk Analytics Engine
====================================

Sophisticated machine learning platform for credit risk assessment and portfolio optimization.
Implements ensemble methods, deep learning, and real-time risk monitoring capabilities.

Author: Emilio Cardenas
License: MIT
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Advanced ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization and Business Intelligence
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical Analysis
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

# Time Series and Financial Analysis
from datetime import datetime, timedelta
import yfinance as yf

class AdvancedCreditRiskEngine:
    """
    Enterprise-grade credit risk analytics platform with advanced ML capabilities.
    
    Features:
    - Ensemble machine learning models (XGBoost, LightGBM, Random Forest)
    - Deep neural networks for complex pattern recognition
    - Real-time risk scoring and monitoring
    - SHAP explainability for model interpretability
    - Advanced feature engineering and selection
    - Portfolio optimization and stress testing
    """
    
    def __init__(self, config=None):
        """Initialize the credit risk engine with configuration parameters."""
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def _default_config(self):
        """Default configuration for the credit risk engine."""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'early_stopping_rounds': 50,
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 1000,
            'neural_network_epochs': 100,
            'batch_size': 32
        }
    
    def load_and_preprocess_data(self, data_path=None, data=None):
        """
        Load and preprocess credit risk data with advanced feature engineering.
        
        Args:
            data_path (str): Path to the dataset
            data (pd.DataFrame): Pre-loaded dataset
            
        Returns:
            tuple: Processed features and target variables
        """
        if data is None and data_path:
            data = pd.read_csv(data_path)
        elif data is None:
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data()
        
        # Advanced feature engineering
        data = self._engineer_features(data)
        
        # Handle missing values with sophisticated imputation
        data = self._handle_missing_values(data)
        
        # Outlier detection and treatment
        data = self._handle_outliers(data)
        
        # Feature selection using statistical methods
        features, target = self._select_features(data)
        
        return features, target
    
    def _generate_synthetic_data(self, n_samples=10000):
        """Generate realistic synthetic credit data for demonstration."""
        np.random.seed(self.config['random_state'])
        
        data = pd.DataFrame({
            'loan_amount': np.random.lognormal(10, 1, n_samples),
            'annual_income': np.random.lognormal(11, 0.5, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
            'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
            'loan_term': np.random.choice([36, 60], n_samples, p=[0.7, 0.3]),
            'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.2, 0.4]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
                                           n_samples, p=[0.4, 0.2, 0.2, 0.2]),
            'delinq_2yrs': np.random.poisson(0.5, n_samples),
            'open_accounts': np.random.poisson(10, n_samples),
            'total_accounts': np.random.poisson(25, n_samples),
            'revolving_balance': np.random.lognormal(8, 1, n_samples),
            'revolving_utilization': np.random.beta(2, 3, n_samples) * 100
        })
        
        # Create target variable with realistic relationships
        risk_score = (
            -0.01 * data['credit_score'] +
            0.02 * data['debt_to_income'] +
            0.1 * data['delinq_2yrs'] +
            -0.00001 * data['annual_income'] +
            0.01 * data['revolving_utilization'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        data['default'] = (risk_score > np.percentile(risk_score, 85)).astype(int)
        
        return data
    
    def _engineer_features(self, data):
        """Advanced feature engineering for credit risk modeling."""
        # Financial ratios
        data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']
        data['credit_utilization_ratio'] = data['revolving_balance'] / (data['revolving_balance'] + 1000)
        data['accounts_ratio'] = data['open_accounts'] / data['total_accounts']
        
        # Risk indicators
        data['high_risk_score'] = (data['credit_score'] < 600).astype(int)
        data['high_debt_ratio'] = (data['debt_to_income'] > 40).astype(int)
        data['recent_delinquency'] = (data['delinq_2yrs'] > 0).astype(int)
        
        # Interaction features
        data['score_income_interaction'] = data['credit_score'] * np.log(data['annual_income'])
        data['debt_employment_interaction'] = data['debt_to_income'] * data['employment_length']
        
        # Categorical encoding
        le = LabelEncoder()
        for col in ['home_ownership', 'loan_purpose']:
            if col in data.columns:
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
        
        return data
    
    def _handle_missing_values(self, data):
        """Sophisticated missing value imputation."""
        from sklearn.impute import KNNImputer
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Use KNN imputation for numeric columns
        imputer = KNNImputer(n_neighbors=5)
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        return data
    
    def _handle_outliers(self, data):
        """Advanced outlier detection and treatment using IQR and Z-score methods."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'default':  # Don't modify target variable
                # IQR method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data
    
    def _select_features(self, data):
        """Advanced feature selection using statistical methods."""
        # Separate features and target
        target_col = 'default'
        feature_cols = [col for col in data.columns if col != target_col]
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        return X, y
    
    def train_ensemble_models(self, X, y):
        """Train multiple advanced ML models for ensemble prediction."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # 1. XGBoost Classifier
        print("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.config['n_estimators'],
            learning_rate=self.config['learning_rate'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state'],
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        
        # 2. LightGBM Classifier
        print("Training LightGBM model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.config['n_estimators'],
            learning_rate=self.config['learning_rate'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state'],
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        self.models['lightgbm'] = lgb_model
        
        # 3. Random Forest Classifier
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # 4. Deep Neural Network
        print("Training Deep Neural Network...")
        nn_model = self._build_neural_network(X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
        nn_model.fit(
            X_train_scaled, y_train,
            epochs=self.config['neural_network_epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        self.models['neural_network'] = nn_model
        
        # 5. Logistic Regression (baseline)
        print("Training Logistic Regression model...")
        lr_model = LogisticRegression(
            random_state=self.config['random_state'],
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        print("All models trained successfully!")
        return self.models
    
    def _build_neural_network(self, input_dim):
        """Build advanced deep neural network for credit risk prediction."""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def evaluate_models(self):
        """Comprehensive model evaluation with advanced metrics."""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            if name == 'neural_network':
                y_pred_proba = model.predict(self.X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            accuracy = (y_pred == self.y_test).mean()
            
            # Cross-validation score
            if name != 'neural_network':
                cv_scores = cross_val_score(model, self.X_test, self.y_test, cv=5, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean, cv_std = auc_score, 0  # Placeholder for neural network
            
            results[name] = {
                'auc_score': auc_score,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV AUC: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        self.performance_metrics = results
        return results
    
    def create_ensemble_prediction(self, weights=None):
        """Create ensemble prediction using weighted average of all models."""
        if weights is None:
            # Use AUC scores as weights
            weights = {name: metrics['auc_score'] for name, metrics in self.performance_metrics.items()}
            total_weight = sum(weights.values())
            weights = {name: w/total_weight for name, w in weights.items()}
        
        ensemble_proba = np.zeros(len(self.y_test))
        
        for name, weight in weights.items():
            ensemble_proba += weight * self.performance_metrics[name]['probabilities']
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ensemble_auc = roc_auc_score(self.y_test, ensemble_proba)
        ensemble_accuracy = (ensemble_pred == self.y_test).mean()
        
        print(f"\nEnsemble Model Performance:")
        print(f"AUC Score: {ensemble_auc:.4f}")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_pred, ensemble_proba, ensemble_auc
    
    def generate_power_bi_data(self, output_path="credit_risk_dashboard_data.csv"):
        """Generate comprehensive data for Power BI dashboard integration."""
        # Create comprehensive dataset for Power BI
        dashboard_data = []
        
        # Model performance metrics
        for model_name, metrics in self.performance_metrics.items():
            dashboard_data.append({
                'metric_type': 'model_performance',
                'model_name': model_name,
                'auc_score': metrics['auc_score'],
                'accuracy': metrics['accuracy'],
                'cv_mean': metrics['cv_mean'],
                'cv_std': metrics['cv_std'],
                'timestamp': datetime.now()
            })
        
        # Risk distribution data
        for i, (pred, prob) in enumerate(zip(self.performance_metrics['xgboost']['predictions'], 
                                  
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)