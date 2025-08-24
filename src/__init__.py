"""
F&B Process Anomaly Detection System
A comprehensive system for detecting anomalies in food and beverage manufacturing processes.
"""

__version__ = "1.0.0"
__author__ = "Honeywell Hackathon Team"

# Import main classes for easier access
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Predictor'
]
