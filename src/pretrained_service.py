"""
Pre-trained Model Service for F&B Process Anomaly Detection System
Module 2: Instant Prediction Service

This module provides:
1. Multiple pre-trained ML models for instant predictions
2. Model ensemble capabilities
3. Confidence scoring and uncertainty quantification
4. Fast batch prediction processing
5. Model performance comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from loguru import logger

from src.config import (
    MODEL_DIR, MODEL_DIR_MODULE2, REPORTS_DIR, MODEL_CONFIG, QUALITY_THRESHOLDS, BUSINESS_METRICS
)

warnings.filterwarnings('ignore')

class PretrainedModelService:
    """
    Service for handling multiple pre-trained models for instant predictions.
    Provides ensemble predictions with confidence scoring.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.ensemble_model = None
        self.is_loaded = False
        
        # Model configurations for training
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'Random Forest',
                'type': 'tree_based'
            },
            'xgboost': {
                'model': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'XGBoost',
                'type': 'tree_based'
            },
            'gradient_boosting': {
                'model': MultiOutputRegressor(GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )),
                'name': 'Gradient Boosting',
                'type': 'tree_based'
            },
            'neural_network': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                ),
                'name': 'Neural Network',
                'type': 'neural'
            },
            'support_vector': {
                'model': MultiOutputRegressor(SVR(
                    kernel='rbf',
                    C=100,
                    gamma='scale',
                    epsilon=0.1
                )),
                'name': 'Support Vector Regression',
                'type': 'kernel'
            },
            'ridge_regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'name': 'Ridge Regression',
                'type': 'linear'
            },
            'elastic_net': {
                'model': ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    random_state=42
                ),
                'name': 'Elastic Net',
                'type': 'linear'
            }
        }
        
        logger.info("PretrainedModelService initialized")
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models and save them for Module 2.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary containing training results and model performance
        """
        logger.info("Training all models for Module 2...")
        
        results = {
            'models_trained': [],
            'model_scores': {},
            'best_model': None,
            'best_score': -np.inf,
            'ensemble_score': None,
            'training_timestamp': datetime.now().isoformat()
        }
        
        trained_models = []
        model_scores = []
        
        # Train individual models
        for model_key, config in self.model_configs.items():
            try:
                logger.info(f"Training {config['name']}...")
                
                model = config['model']
                scaler = StandardScaler()
                
                # Scale features for certain model types
                if config['type'] in ['neural', 'kernel', 'linear']:
                    X_scaled = scaler.fit_transform(X_train)
                    model.fit(X_scaled, y_train)
                    self.scalers[model_key] = scaler
                else:
                    model.fit(X_train, y_train)
                    X_scaled = X_train
                
                # Evaluate model
                y_pred = model.predict(X_scaled)
                
                # Calculate metrics for each target
                model_metrics = {}
                for i, col in enumerate(y_train.columns):
                    r2 = r2_score(y_train.iloc[:, i], y_pred[:, i] if len(y_pred.shape) > 1 else y_pred)
                    mse = mean_squared_error(y_train.iloc[:, i], y_pred[:, i] if len(y_pred.shape) > 1 else y_pred)
                    mae = mean_absolute_error(y_train.iloc[:, i], y_pred[:, i] if len(y_pred.shape) > 1 else y_pred)
                    
                    model_metrics[col] = {
                        'r2_score': r2,
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse)
                    }
                
                # Average R2 score across all targets
                avg_r2 = np.mean([metrics['r2_score'] for metrics in model_metrics.values()])
                
                # Store model and metadata
                self.models[model_key] = model
                self.model_metadata[model_key] = {
                    'name': config['name'],
                    'type': config['type'],
                    'metrics': model_metrics,
                    'avg_r2_score': avg_r2,
                    'training_timestamp': datetime.now().isoformat()
                }
                
                # Save model to disk
                model_path = MODEL_DIR_MODULE2 / f"pretrained_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump({
                    'model': model,
                    'scaler': self.scalers.get(model_key),
                    'metadata': self.model_metadata[model_key]
                }, model_path)
                
                results['models_trained'].append(model_key)
                results['model_scores'][model_key] = avg_r2
                
                if avg_r2 > results['best_score']:
                    results['best_model'] = model_key
                    results['best_score'] = avg_r2
                
                trained_models.append((model_key, model))
                model_scores.append(avg_r2)
                
                logger.info(f"{config['name']} trained successfully - R² Score: {avg_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {config['name']}: {str(e)}")
                continue
        
        # Create ensemble model
        if len(trained_models) >= 3:
            try:
                logger.info("Creating ensemble model...")
                
                # Select top 3 models for ensemble
                top_models = sorted(zip(trained_models, model_scores), key=lambda x: x[1], reverse=True)[:3]
                
                ensemble_estimators = []
                for (model_key, model), score in top_models:
                    ensemble_estimators.append((model_key, model))
                
                # Create voting regressor
                self.ensemble_model = VotingRegressor(
                    estimators=ensemble_estimators,
                    n_jobs=-1
                )
                
                # For ensemble, we need to handle scaling properly
                # Use the same scaling approach as the best model
                best_model_key = results['best_model']
                if best_model_key in self.scalers:
                    X_ensemble = self.scalers[best_model_key].transform(X_train)
                else:
                    X_ensemble = X_train
                
                # Note: VotingRegressor doesn't work well with different input scales
                # So we'll implement a custom ensemble prediction method
                
                ensemble_predictions = []
                for model_key, model in trained_models:
                    if model_key in self.scalers:
                        X_scaled = self.scalers[model_key].transform(X_train)
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X_train)
                    ensemble_predictions.append(pred)
                
                # Average predictions
                ensemble_pred = np.mean(ensemble_predictions, axis=0)
                
                # Calculate ensemble metrics
                ensemble_metrics = {}
                for i, col in enumerate(y_train.columns):
                    r2 = r2_score(y_train.iloc[:, i], ensemble_pred[:, i] if len(ensemble_pred.shape) > 1 else ensemble_pred)
                    ensemble_metrics[col] = {'r2_score': r2}
                
                ensemble_r2 = np.mean([metrics['r2_score'] for metrics in ensemble_metrics.values()])
                results['ensemble_score'] = ensemble_r2
                
                logger.info(f"Ensemble model created - R² Score: {ensemble_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error creating ensemble model: {str(e)}")
        
        # Save results
        results_path = MODEL_DIR_MODULE2 / f"pretrained_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.is_loaded = True
        logger.info(f"All models trained successfully. Best model: {results['best_model']} (R² = {results['best_score']:.4f})")
        
        return results
    
    def load_pretrained_models(self, model_dir: Optional[Path] = None) -> bool:
        """
        Load pre-trained models from disk.
        
        Args:
            model_dir: Directory containing pre-trained models
            
        Returns:
            True if models loaded successfully
        """
        if model_dir is None:
            model_dir = MODEL_DIR_MODULE2
        
        logger.info("Loading pre-trained models...")
        
        try:
            # Find all pretrained model files (support both naming patterns)
            model_files = list(model_dir.glob("*pretrained_*.pkl"))
            
            if not model_files:
                logger.warning("No pre-trained model files found")
                logger.info(f"Searched in directory: {model_dir}")
                logger.info(f"Files in directory: {list(model_dir.glob('*.pkl'))}")
                return False
            
            logger.info(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")
            
            # Load the most recent models for each type
            model_groups = {}
            for file_path in model_files:
                # Extract model type from filename
                filename = file_path.stem
                
                # Handle different naming patterns: "pretrained_modeltype_timestamp" or "gcFnB_pretrained_modeltype_timestamp"
                if 'pretrained_' in filename:
                    # Find the part after 'pretrained_'
                    parts = filename.split('pretrained_')[1].split('_')
                    if len(parts) >= 1:
                        model_type = parts[0]
                        if model_type not in model_groups:
                            model_groups[model_type] = []
                        model_groups[model_type].append(file_path)
                        logger.info(f"Found model type '{model_type}' in file: {file_path.name}")
            
            # Load the most recent model for each type
            for model_type, files in model_groups.items():
                # Sort by modification time, get the newest
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                
                try:
                    model_data = joblib.load(latest_file)
                    self.models[model_type] = model_data['model']
                    if 'scaler' in model_data and model_data['scaler'] is not None:
                        self.scalers[model_type] = model_data['scaler']
                    if 'metadata' in model_data:
                        self.model_metadata[model_type] = model_data['metadata']
                    
                    logger.info(f"Loaded {model_type} model from {latest_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {str(e)}")
                    continue
            
            self.is_loaded = len(self.models) > 0
            
            if self.is_loaded:
                logger.info(f"Successfully loaded {len(self.models)} pre-trained models")
                return True
            else:
                logger.error("Failed to load any pre-trained models")
                return False
                
        except Exception as e:
            logger.error(f"Error loading pre-trained models: {str(e)}")
            return False
    
    def predict_single(self, features: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction using a single model.
        
        Args:
            features: Input features for prediction
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        if not self.is_loaded:
            raise ValueError("No models loaded. Please load models first.")
        
        # Validate input features
        if features is None or features.empty:
            raise ValueError("No features provided for prediction")
        
        if model_name is None:
            # Use the model with highest R² score
            if not self.model_metadata:
                model_name = list(self.models.keys())[0]
            else:
                model_name = max(self.model_metadata.keys(), 
                               key=lambda x: self.model_metadata[x].get('avg_r2_score', 0))
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in loaded models")
        
        model = self.models[model_name]
        
        # Log feature information for debugging
        logger.info(f"Making prediction with {model_name}")
        logger.info(f"Input features shape: {features.shape}")
        logger.info(f"Input features columns: {list(features.columns)}")
        logger.info(f"Input features sample: {features.head(1).to_dict()}")
        
        # Apply scaling if needed
        if model_name in self.scalers:
            features_scaled = self.scalers[model_name].transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        predictions = model.predict(features_scaled)
        
        # Ensure predictions is 2D
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Calculate confidence (using model-specific method)
        confidence_scores = self._calculate_confidence(model, features_scaled, predictions, model_name)
        
        return {
            'model_used': model_name,
            'model_display_name': self.model_metadata.get(model_name, {}).get('name', model_name),
            'predictions': predictions.tolist(),
            'confidence_scores': confidence_scores,
            'model_metadata': self.model_metadata.get(model_name, {}),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_ensemble(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make ensemble prediction using multiple models.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Dictionary containing ensemble predictions and individual model results
        """
        if not self.is_loaded:
            raise ValueError("No models loaded. Please load models first.")
        
        individual_predictions = []
        individual_confidences = []
        model_results = {}
        
        # Get predictions from all models
        for model_name in self.models.keys():
            try:
                result = self.predict_single(features, model_name)
                individual_predictions.append(result['predictions'])
                individual_confidences.append(result['confidence_scores'])
                model_results[model_name] = result
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {str(e)}")
                continue
        
        if not individual_predictions:
            raise ValueError("No models could make predictions")
        
        # Calculate ensemble prediction (weighted by confidence)
        predictions_array = np.array(individual_predictions)
        
        # Handle confidence arrays which may have different shapes
        try:
            confidences_array = np.array(individual_confidences)
        except ValueError:
            # If arrays have different shapes, use simple average
            ensemble_predictions = np.mean(predictions_array, axis=0)
            ensemble_confidence = [0.8] * len(ensemble_predictions[0]) if len(ensemble_predictions.shape) > 1 else [0.8]
        else:
            # Weighted average based on confidence scores
            if confidences_array.ndim > 1 and confidences_array.shape[1] > 1:
                weights = confidences_array / np.sum(confidences_array, axis=0)
                ensemble_predictions = np.sum(predictions_array * weights[:, :, np.newaxis], axis=0)
            else:
                # Simple average if confidence structure is inconsistent
                ensemble_predictions = np.mean(predictions_array, axis=0)
            
            # Calculate ensemble confidence (average of individual confidences)
            ensemble_confidence = np.mean(confidences_array, axis=0)
            if hasattr(ensemble_confidence, '__len__') and len(ensemble_confidence) == 1:
                ensemble_confidence = [ensemble_confidence[0]] * (len(ensemble_predictions[0]) if len(ensemble_predictions.shape) > 1 else 1)
        
        return {
            'ensemble_predictions': ensemble_predictions.tolist() if hasattr(ensemble_predictions, 'tolist') else ensemble_predictions,
            'ensemble_confidence': ensemble_confidence.tolist() if hasattr(ensemble_confidence, 'tolist') else ensemble_confidence,
            'individual_results': model_results,
            'models_used': list(model_results.keys()),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, features_batch: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Make batch predictions for multiple samples.
        
        Args:
            features_batch: Batch of input features
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Dictionary containing batch predictions
        """
        if use_ensemble:
            return self.predict_ensemble(features_batch)
        else:
            return self.predict_single(features_batch)
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """
        Get comparison of all loaded models.
        
        Returns:
            Dictionary containing model comparison data
        """
        if not self.model_metadata:
            return {'error': 'No model metadata available'}
        
        comparison_data = {
            'models': {},
            'best_model': None,
            'best_score': -float('inf')
        }
        
        for model_name, metadata in self.model_metadata.items():
            # Convert NumPy types to native Python types for JSON serialization
            avg_r2_score = metadata.get('avg_r2_score', 0)
            if hasattr(avg_r2_score, 'item'):  # Check if it's a NumPy type
                avg_r2_score = float(avg_r2_score.item())
            else:
                avg_r2_score = float(avg_r2_score)
            
            # Convert metrics to JSON-serializable format
            metrics = metadata.get('metrics', {})
            serializable_metrics = {}
            for target, target_metrics in metrics.items():
                serializable_metrics[target] = {}
                for metric_name, metric_value in target_metrics.items():
                    try:
                        if hasattr(metric_value, 'item'):  # Check if it's a NumPy type
                            serializable_metrics[target][metric_name] = float(metric_value.item())
                        elif isinstance(metric_value, (tuple, list)):
                            # Handle tuple/list values by taking the first element
                            serializable_metrics[target][metric_name] = float(metric_value[0]) if metric_value else 0.0
                        else:
                            serializable_metrics[target][metric_name] = float(metric_value)
                    except (ValueError, TypeError, IndexError):
                        # If conversion fails, use 0.0 as fallback
                        serializable_metrics[target][metric_name] = 0.0
            
            comparison_data['models'][model_name] = {
                'display_name': metadata.get('name', model_name),
                'type': metadata.get('type', 'unknown'),
                'avg_r2_score': avg_r2_score,
                'metrics': serializable_metrics,
                'training_timestamp': metadata.get('training_timestamp', '')
            }
            
            if avg_r2_score > comparison_data['best_score']:
                comparison_data['best_model'] = model_name
                comparison_data['best_score'] = avg_r2_score
        
        return comparison_data
    
    def _calculate_confidence(self, model, features: np.ndarray, predictions: np.ndarray, model_name: str) -> List[float]:
        """
        Calculate confidence scores for predictions.
        
        Args:
            model: The trained model
            features: Input features
            predictions: Model predictions
            model_name: Name of the model
            
        Returns:
            List of confidence scores
        """
        try:
            # For tree-based models, use variance of tree predictions
            if hasattr(model, 'estimators_'):
                tree_predictions = []
                for estimator in model.estimators_[:min(50, len(model.estimators_))]:  # Use subset for speed
                    if hasattr(estimator, 'predict'):
                        pred = estimator.predict(features)
                        tree_predictions.append(pred)
                
                if tree_predictions:
                    tree_predictions = np.array(tree_predictions)
                    # Calculate variance across tree predictions
                    prediction_variance = np.var(tree_predictions, axis=0)
                    # Convert variance to confidence (lower variance = higher confidence)
                    confidence = 1.0 / (1.0 + prediction_variance)
                    return confidence.tolist()
            
            # For other models, use a simple heuristic based on prediction magnitude
            # Higher predictions typically have lower confidence in quality prediction
            confidence = np.ones(len(predictions)) * 0.8  # Base confidence
            
            return confidence.tolist()
            
        except Exception as e:
            logger.warning(f"Error calculating confidence for {model_name}: {str(e)}")
            # Return default confidence scores
            return [0.7] * len(predictions)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available models.
        
        Returns:
            List of model information dictionaries
        """
        models_info = []
        
        for model_name in self.models.keys():
            metadata = self.model_metadata.get(model_name, {})
            
            # Convert NumPy types to native Python types for JSON serialization
            avg_r2_score = metadata.get('avg_r2_score', 0)
            if hasattr(avg_r2_score, 'item'):  # Check if it's a NumPy type
                avg_r2_score = float(avg_r2_score.item())
            else:
                avg_r2_score = float(avg_r2_score)
            
            models_info.append({
                'key': model_name,
                'name': metadata.get('name', model_name),
                'type': metadata.get('type', 'unknown'),
                'avg_r2_score': avg_r2_score,
                'training_timestamp': metadata.get('training_timestamp', '')
            })
        
        # Sort by R² score (best first)
        models_info.sort(key=lambda x: x['avg_r2_score'], reverse=True)
        
        return models_info
