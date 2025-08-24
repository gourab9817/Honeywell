"""
Configuration for F&B Process Anomaly Detection System
Honeywell Hackathon Solution
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'models'
MODEL_DIR_MODULE2 = DATA_DIR / 'model_module2'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'

# Create directories
for dir_path in [DATA_DIR, MODEL_DIR, MODEL_DIR_MODULE2, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'raw_data_file': 'FnB_Process_Data_Batch_Wise.csv',
    'process_sheet': 'Sheet1',  # Not used for CSV
    'quality_sheet': None,      # No separate quality sheet
    'batch_id_col': 'Batch_ID',
    'time_col': 'Time',
    'target_cols': ['Final_Weight', 'Quality_Score']  # Will be calculated
}

# Process parameters (from actual data analysis)
PROCESS_PARAMS = {
    'Flour (kg)': {'ideal': 10.0, 'tolerance': 0.5, 'unit': 'kg', 'critical': True},
    'Sugar (kg)': {'ideal': 5.0, 'tolerance': 0.3, 'unit': 'kg', 'critical': True},
    'Yeast (kg)': {'ideal': 2.0, 'tolerance': 0.15, 'unit': 'kg', 'critical': True},
    'Salt (kg)': {'ideal': 1.0, 'tolerance': 0.08, 'unit': 'kg', 'critical': True},
    'Water Temp (C)': {'ideal': 26.5, 'tolerance': 1.5, 'unit': '°C', 'critical': True},
    'Mixer Speed (RPM)': {'ideal': 150.0, 'tolerance': 10.0, 'unit': 'RPM', 'critical': True},
    'Mixing Temp (C)': {'ideal': 38.0, 'tolerance': 2.0, 'unit': '°C', 'critical': True},
    'Fermentation Temp (C)': {'ideal': 37.0, 'tolerance': 0.5, 'unit': '°C', 'critical': True},
    'Oven Temp (C)': {'ideal': 180.0, 'tolerance': 2.0, 'unit': '°C', 'critical': True},
    'Oven Humidity (%)': {'ideal': 45.0, 'tolerance': 2.0, 'unit': '%', 'critical': True}
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'weight': {
        'min': 48.0,  # kg
        'max': 52.0,  # kg
        'ideal': 50.0,  # kg
        'critical_low': 47.0,
        'critical_high': 53.0
    },
    'quality_score': {
        'min': 80.0,  # %
        'ideal': 90.0,  # %
        'critical': 75.0  # %
    }
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'anomaly_contamination': 0.1,
    'feature_selection_threshold': 0.01
}

# Flask configuration
FLASK_CONFIG = {
    'SECRET_KEY': os.getenv('SECRET_KEY', 'honeywell-hackathon-2024'),
    'DEBUG': os.getenv('DEBUG', 'True').lower() == 'true',
    'HOST': os.getenv('HOST', '0.0.0.0'),
    'PORT': int(os.getenv('PORT', 5000)),
    'UPLOAD_FOLDER': str(DATA_DIR / 'uploads'),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'window_sizes': [5, 10, 15],  # For rolling statistics
    'lag_features': [1, 2, 3],     # Lag periods
    'polynomial_degree': 2,         # For polynomial features
    'interaction_features': True,
    'deviation_features': True,
    'stability_features': True
}

# Alert configuration
ALERT_CONFIG = {
    'critical_deviation': 15,  # % deviation from ideal
    'warning_deviation': 10,   # % deviation from ideal
    'alert_cooldown': 300,     # seconds between same alerts
    'max_alerts_per_hour': 10
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    'file': str(REPORTS_DIR / 'system.log')
}

# Business impact metrics
BUSINESS_METRICS = {
    'waste_reduction_target': 15,  # %
    'quality_improvement_target': 10,  # %
    'downtime_reduction_target': 25,  # %
    'annual_production_value': 1000000,  # USD
    'waste_cost_percentage': 5,  # % of production value
    'quality_cost_percentage': 2  # % of production value
}
