"""
Model Training Module for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This module implements:
1. Multiple ML algorithms for quality prediction
2. Anomaly detection models
3. Model evaluation and comparison
4. Hyperparameter tuning
5. Model persistence and loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from loguru import logger

from src.config import (
    MODEL_DIR, REPORTS_DIR, MODEL_CONFIG, QUALITY_THRESHOLDS, BUSINESS_METRICS
)

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model trainer for F&B process anomaly detection.
    Implements multiple algorithms and evaluation methods.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.anomaly_detector = None
        self.scaler = None
        self.model_config = MODEL_CONFIG
        self.training_results = {}
        self.feature_importance = {}
        logger.info("ModelTrainer initialized")
    
    def train_quality_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                            X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """
        Train quality prediction models one by one and compare performance.
        This approach is more memory efficient and stable.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary containing training results and best model
        """
        logger.info("Starting sequential model training (one by one for stability)")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': {},
            'best_model': None,
            'feature_importance': {},
            'cross_validation': {},
            'business_impact': {}
        }
        
        # Define models to train in order of preference (fastest first)
        models_to_train = [
            ('linear_regression', self._train_linear_regression),
            ('ridge_regression', self._train_ridge_regression),
            ('random_forest', self._train_random_forest),
            ('xgboost', self._train_xgboost),
            ('neural_network', self._train_neural_network)
        ]
        
        best_score = -np.inf
        best_model_name = None
        total_models = len(models_to_train)
        
        # Early stopping threshold - if we get a very good model, we can stop
        early_stop_threshold = 0.90  # R² > 0.90 is excellent
        
        # Train each model individually
        for i, (model_name, train_func) in enumerate(models_to_train, 1):
            logger.info(f"🤖 Training {model_name} ({i}/{total_models})...")
            try:
                # Train model
                model_result = train_func(X_train, y_train, X_test, y_test)
                
                # Sanitize result for JSON serialization (remove model objects)
                sanitized_result = self._sanitize_result(model_result, model_name)
                results['models_trained'][model_name] = sanitized_result
                
                # Get performance score
                if 'test_metrics' in model_result:
                    score = model_result['test_metrics']['overall']['avg_r2']
                    logger.info(f"✅ {model_name} - R² Score: {score:.4f}")
                    
                    # Check if this is the best model so far
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        # Store the best model separately (don't include in JSON results)
                        if 'model' in model_result:
                            self.best_model = model_result['model']
                        logger.info(f"🏆 New best model: {model_name} (R² = {score:.4f})")
                    
                    # Store model temporarily (don't include in JSON results)
                    if 'model' in model_result:
                        self.models[model_name] = model_result['model']
                    
                    # Early stopping - if we have an excellent model, stop training
                    if score >= early_stop_threshold:
                        logger.info(f"🎯 Early stopping! Excellent score achieved: {score:.4f}")
                        break
                else:
                    logger.warning(f"No test metrics for {model_name}")
                
                # Clear memory after each model (keep only the best)
                if model_name != best_model_name and model_name in self.models:
                    del self.models[model_name]
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                results['models_trained'][model_name] = {'error': str(e)}
                continue
        
        # Set best model
        if best_model_name:
            results['best_model'] = best_model_name
            results['best_score'] = best_score
            results['model_count'] = len(results['models_trained'])
            logger.info(f"🎉 Final best model: {best_model_name} with R² = {best_score:.4f}")
        else:
            logger.warning("No successful model training completed")
            return results
        
        # Calculate business impact
        results['business_impact'] = self._calculate_business_impact(results)
        
        # Store feature importance from best model
        try:
            if self.best_model and hasattr(self.best_model, 'feature_importances_'):
                feature_names = X_train.columns
                importance_scores = self.best_model.feature_importances_
                self.feature_importance = dict(zip(feature_names, importance_scores))
                results['feature_importance'] = self.feature_importance
            elif hasattr(self.best_model, 'estimators_') and hasattr(self.best_model.estimators_[0], 'feature_importances_'):
                # For MultiOutputRegressor
                feature_names = X_train.columns
                importance_scores = self.best_model.estimators_[0].feature_importances_
                self.feature_importance = dict(zip(feature_names, importance_scores))
                results['feature_importance'] = self.feature_importance
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        self.training_results = results
        return results
    
    def _sanitize_result(self, result: Dict, model_name: str) -> Dict:
        """Remove model objects from results to make them JSON serializable."""
        sanitized = result.copy()
        
        # Remove the actual model object
        if 'model' in sanitized:
            del sanitized['model']
        
        # Add model type info
        model_types = {
            'linear_regression': 'LinearRegression',
            'ridge_regression': 'RidgeRegression', 
            'random_forest': 'RandomForest',
            'xgboost': 'XGBoost',
            'neural_network': 'NeuralNetwork'
        }
        
        sanitized['model_type'] = model_types.get(model_name, model_name)
        return sanitized
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                            X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train Random Forest model with simplified hyperparameter tuning."""
        logger.info("Training Random Forest model")
        
        # Simplified parameter grid for faster training
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        }
        
        # Multi-output wrapper
        rf_model = RandomForestRegressor(random_state=self.model_config['random_state'], n_jobs=-1)
        multi_rf = MultiOutputRegressor(rf_model)
        
        # Simplified grid search
        grid_search = GridSearchCV(
            multi_rf,
            {'estimator__' + k: v for k, v in param_grid.items()},
            cv=3,  # Reduced CV folds for speed
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_metrics': metrics
        }
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train XGBoost model with simplified hyperparameter tuning."""
        logger.info("Training XGBoost model")
        
        # Simplified parameter grid for faster training
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3]
        }
        
        # Multi-output wrapper
        xgb_model = xgb.XGBRegressor(
            random_state=self.model_config['random_state'],
            n_jobs=-1,
            verbosity=0  # Suppress XGBoost warnings
        )
        multi_xgb = MultiOutputRegressor(xgb_model)
        
        # Simplified grid search
        grid_search = GridSearchCV(
            multi_xgb,
            {'estimator__' + k: v for k, v in param_grid.items()},
            cv=3,  # Reduced CV folds for speed
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_metrics': metrics
        }
    
    def _train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                                X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression model")
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': model,
            'test_metrics': metrics
        }
    
    def _train_ridge_regression(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                               X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train Ridge Regression model with hyperparameter tuning."""
        logger.info("Training Ridge Regression model")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
        # Multi-output wrapper
        ridge_model = Ridge(random_state=self.model_config['random_state'])
        multi_ridge = MultiOutputRegressor(ridge_model)
        
        # Grid search
        grid_search = GridSearchCV(
            multi_ridge,
            {'estimator__' + k: v for k, v in param_grid.items()},
            cv=self.model_config['cv_folds'],
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_metrics': metrics
        }
    
    def _train_svr(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train Support Vector Regression model."""
        logger.info("Training SVR model")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (using first target for simplicity)
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train_scaled, y_train.iloc[:, 0])
        
        # Evaluate
        y_pred = model.predict(X_test_scaled).reshape(-1, 1)
        metrics = self._calculate_metrics(y_test.iloc[:, 0:1], y_pred)
        
        return {
            'model': model,
            'test_metrics': metrics
        }
    
    def _train_neural_network(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                             X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Train Neural Network model."""
        logger.info("Training Neural Network model")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=self.model_config['random_state']
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return {
            'model': model,
            'test_metrics': metrics
        }
    
    def train_anomaly_detector(self, X_train: pd.DataFrame) -> Dict:
        """
        Train anomaly detection model using Isolation Forest.
        
        Args:
            X_train: Training features
        
        Returns:
            Dictionary containing anomaly detector and results
        """
        logger.info("Training anomaly detection model")
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=self.model_config['anomaly_contamination'],
            random_state=self.model_config['random_state'],
            n_jobs=-1
        )
        
        self.anomaly_detector.fit(X_train)
        
        # Calculate anomaly scores
        anomaly_scores = self.anomaly_detector.score_samples(X_train)
        threshold = np.percentile(anomaly_scores, 
                                self.model_config['anomaly_contamination'] * 100)
        
        results = {
            'model_type': 'IsolationForest',
            'threshold': float(threshold),
            'contamination': self.model_config['anomaly_contamination'],
            'training_samples': len(X_train),
            'anomaly_scores_stats': {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores))
            }
        }
        
        logger.info(f"Anomaly detector trained. Threshold: {threshold:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Convert predictions to DataFrame if needed
        if isinstance(y_pred, np.ndarray):
            y_pred_df = pd.DataFrame(y_pred, columns=y_true.columns)
        else:
            y_pred_df = y_pred
        
        # Calculate metrics for each target
        for i, col in enumerate(y_true.columns):
            true_vals = y_true[col]
            pred_vals = y_pred_df.iloc[:, i] if y_pred_df.shape[1] > 1 else y_pred_df.iloc[:, 0]
            
            metrics[col] = {
                'mse': float(mean_squared_error(true_vals, pred_vals)),
                'rmse': float(np.sqrt(mean_squared_error(true_vals, pred_vals))),
                'mae': float(mean_absolute_error(true_vals, pred_vals)),
                'r2': float(r2_score(true_vals, pred_vals)),
                'mape': float(np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100)
            }
        
        # Overall metrics
        metrics['overall'] = {
            'avg_r2': np.mean([metrics[col]['r2'] for col in y_true.columns]),
            'avg_rmse': np.mean([metrics[col]['rmse'] for col in y_true.columns]),
            'avg_mae': np.mean([metrics[col]['mae'] for col in y_true.columns])
        }
        
        return metrics
    
    def _select_best_model(self, model_results: Dict) -> Optional[str]:
        """Select the best model based on R² score."""
        best_score = -np.inf
        best_model = None
        
        for model_name, result in model_results.items():
            if 'error' not in result and 'test_metrics' in result:
                avg_r2 = result['test_metrics']['overall']['avg_r2']
                if avg_r2 > best_score:
                    best_score = avg_r2
                    best_model = model_name
        
        return best_model
    
    def _calculate_business_impact(self, results: Dict) -> Dict:
        """Calculate business impact metrics."""
        if not results.get('best_model'):
            return {}
        
        best_model_name = results['best_model']
        best_model_results = results['models_trained'][best_model_name]
        
        if 'test_metrics' not in best_model_results:
            return {}
        
        metrics = best_model_results['test_metrics']
        
        # Calculate business metrics
        weight_accuracy = 1 - (metrics.get('Final_Weight', {}).get('mape', 100) / 100)
        quality_accuracy = 1 - (metrics.get('Quality_Score', {}).get('mape', 100) / 100)
        
        # Business impact calculations
        annual_production = BUSINESS_METRICS['annual_production_value']
        waste_cost = annual_production * (BUSINESS_METRICS['waste_cost_percentage'] / 100)
        quality_cost = annual_production * (BUSINESS_METRICS['quality_cost_percentage'] / 100)
        
        # Potential savings
        waste_reduction = waste_cost * (BUSINESS_METRICS['waste_reduction_target'] / 100)
        quality_improvement = quality_cost * (BUSINESS_METRICS['quality_improvement_target'] / 100)
        total_savings = waste_reduction + quality_improvement
        
        # ROI calculation
        implementation_cost = annual_production * 0.05  # 5% of production value
        roi_percentage = ((total_savings - implementation_cost) / implementation_cost) * 100
        payback_months = (implementation_cost / total_savings) * 12 if total_savings > 0 else float('inf')
        
        return {
            'weight_accuracy': weight_accuracy,
            'quality_accuracy': quality_accuracy,
            'combined_accuracy': (weight_accuracy + quality_accuracy) / 2,
            'waste_reduction_potential': waste_reduction,
            'quality_improvement_potential': quality_improvement,
            'total_annual_savings': total_savings,
            'implementation_cost': implementation_cost,
            'roi_percentage': roi_percentage,
            'payback_months': payback_months,
            'model_performance_score': metrics['overall']['avg_r2']
        }
    
    def save_models(self, model_prefix: str = 'model') -> Dict:
        """
        Save trained models to disk.
        
        Args:
            model_prefix: Prefix for model filenames
        
        Returns:
            Dictionary containing saved file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        # Save best quality model
        if self.best_model:
            model_path = MODEL_DIR / f'{model_prefix}_quality_{timestamp}.pkl'
            joblib.dump(self.best_model, model_path)
            saved_files['quality_model'] = str(model_path)
            logger.info(f"Quality model saved to {model_path}")
        
        # Save anomaly detector
        if self.best_model:
            anomaly_path = MODEL_DIR / f'{model_prefix}_anomaly_{timestamp}.pkl'
            joblib.dump(self.anomaly_detector, anomaly_path)
            saved_files['anomaly_detector'] = str(anomaly_path)
            logger.info(f"Anomaly detector saved to {anomaly_path}")
        
        # Save training results
        if self.training_results:
            results_path = MODEL_DIR / f'{model_prefix}_results_{timestamp}.json'
            with open(results_path, 'w') as f:
                json.dump(self.training_results, f, indent=2, default=str)
            saved_files['training_results'] = str(results_path)
            logger.info(f"Training results saved to {results_path}")
        
        # Save feature importance
        if self.feature_importance:
            importance_path = MODEL_DIR / f'{model_prefix}_importance_{timestamp}.json'
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            saved_files['feature_importance'] = str(importance_path)
            logger.info(f"Feature importance saved to {importance_path}")
        
        return saved_files
    
    def load_models(self, model_path: str, anomaly_path: str = None) -> bool:
        """
        Load trained models from disk.
        
        Args:
            model_path: Path to quality model
            anomaly_path: Path to anomaly detector (optional)
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Load quality model
            self.best_model = joblib.load(model_path)
            logger.info(f"Quality model loaded from {model_path}")
            
            # Load anomaly detector if provided
            if anomaly_path:
                self.anomaly_detector = joblib.load(anomaly_path)
                logger.info(f"Anomaly detector loaded from {anomaly_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about trained models."""
        info = {
            'models_available': list(self.models.keys()),
            'best_model': None,
            'anomaly_detector_available': self.anomaly_detector is not None,
            'feature_importance_available': len(self.feature_importance) > 0
        }
        
        if self.best_model:
            info['best_model'] = type(self.best_model).__name__
        
        return info
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions using the best model.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary containing predictions and confidence
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Make predictions
        predictions = self.best_model.predict(X)
        
        # Detect anomalies if detector is available
        anomalies = None
        if self.anomaly_detector:
            anomaly_scores = self.anomaly_detector.score_samples(X)
            anomaly_labels = self.anomaly_detector.predict(X)
            anomalies = {
                'scores': anomaly_scores.tolist(),
                'labels': anomaly_labels.tolist(),
                'is_anomaly': (anomaly_labels == -1).tolist()
            }
        
        return {
            'predictions': predictions.tolist(),
            'anomalies': anomalies,
            'model_info': self.get_model_info()
        }
