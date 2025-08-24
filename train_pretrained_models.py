"""
Training Script for Pre-trained Models (Module 2)
F&B Process Anomaly Detection System

This script trains multiple ML models that will be used for instant predictions
in Module 2 of the application.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.pretrained_service import PretrainedModelService
from src.config import DATA_CONFIG, MODEL_CONFIG

def main():
    """Main training pipeline for Module 2 pre-trained models."""
    
    logger.info("🚀 Starting Module 2 Pre-trained Models Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        pretrained_service = PretrainedModelService()
        
        # Load and process data
        logger.info("Loading and processing data...")
        data_file = Path("data/raw/FnB_Process_Data_Batch_Wise.csv")
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return False
        
        # Load data
        process_data, quality_data = data_processor.load_data(str(data_file))
        logger.info(f"Loaded process data: {process_data.shape}")
        logger.info(f"Loaded quality data: {quality_data.shape}")
        
        # Clean data
        clean_process_data, clean_quality_data = data_processor.clean_data(
            process_data, quality_data
        )
        logger.info(f"Cleaned process data: {clean_process_data.shape}")
        logger.info(f"Cleaned quality data: {clean_quality_data.shape}")
        
        # Generate comprehensive quality report
        quality_report = data_processor.get_quality_report()
        outlier_report = data_processor.get_outlier_report()
        
        logger.info("Data quality summary:")
        logger.info(f"- Missing values handled: {quality_report.get('missing_values_handled', 'N/A')}")
        logger.info(f"- Outliers detected: {outlier_report.get('total_outliers', 'N/A')}")
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        features_df = feature_engineer.extract_batch_features(
            clean_process_data, clean_quality_data
        )
        logger.info(f"Extracted features: {features_df.shape}")
        
        # Select optimal features
        selected_features_df = feature_engineer.select_features(features_df)
        logger.info(f"Selected features: {selected_features_df.shape}")
        
        # Prepare training data
        logger.info("Preparing training data...")
        
        # Check what columns are available
        logger.info(f"Available columns: {list(selected_features_df.columns)}")
        
        # Look for target columns with different possible names
        possible_weight_cols = ['Final_Weight_kg', 'Final Weight (kg)', 'final_weight', 'weight']
        possible_quality_cols = ['Quality_Score_percent', 'Quality Score (%)', 'quality_score', 'quality']
        
        weight_col = None
        quality_col = None
        
        # Find weight column
        for col in selected_features_df.columns:
            if any(weight_name.lower() in col.lower() for weight_name in possible_weight_cols):
                weight_col = col
                break
        
        # Find quality column  
        for col in selected_features_df.columns:
            if any(quality_name.lower() in col.lower() for quality_name in possible_quality_cols):
                quality_col = col
                break
        
        # Create target columns list
        target_cols = []
        if weight_col:
            target_cols.append(weight_col)
            logger.info(f"Found weight column: {weight_col}")
        if quality_col:
            target_cols.append(quality_col)
            logger.info(f"Found quality column: {quality_col}")
        
        if not target_cols:
            logger.error("No target columns found in the data")
            logger.error("Please ensure your data contains weight and quality columns")
            return False
        
        y_train = selected_features_df[target_cols]
        
        # Features (X) - exclude target columns
        X_train = selected_features_df.drop(target_cols, axis=1, errors='ignore')
        
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Training targets shape: {y_train.shape}")
        logger.info(f"Target columns: {target_cols}")
        
        # Train all models
        logger.info("Training multiple ML models for Module 2...")
        logger.info("-" * 40)
        
        training_results = pretrained_service.train_all_models(X_train, y_train)
        
        # Display results
        logger.info("Training Results:")
        logger.info("-" * 40)
        logger.info(f"Models trained: {len(training_results['models_trained'])}")
        
        for model_name in training_results['models_trained']:
            score = training_results['model_scores'][model_name]
            logger.info(f"✓ {model_name}: R² = {score:.4f}")
        
        logger.info(f"\nBest Model: {training_results['best_model']}")
        logger.info(f"Best Score: {training_results['best_score']:.4f}")
        
        if training_results.get('ensemble_score'):
            logger.info(f"Ensemble Score: {training_results['ensemble_score']:.4f}")
        
        # Test predictions
        logger.info("\nTesting predictions...")
        
        # Test single model prediction
        test_sample = X_train.head(1)
        single_result = pretrained_service.predict_single(test_sample)
        logger.info(f"Single model prediction successful: {single_result['model_used']}")
        
        # Test ensemble prediction
        ensemble_result = pretrained_service.predict_ensemble(test_sample)
        logger.info(f"Ensemble prediction successful: {len(ensemble_result['models_used'])} models used")
        
        # Get model comparison
        comparison = pretrained_service.get_model_comparison()
        logger.info(f"\nModel comparison generated for {len(comparison['models'])} models")
        
        # Save feature columns for later use
        feature_columns_file = Path("data/processed/feature_columns_module2.json")
        import json
        with open(feature_columns_file, 'w') as f:
            json.dump({
                'feature_columns': X_train.columns.tolist(),
                'target_columns': target_cols,
                'training_timestamp': training_results['training_timestamp']
            }, f, indent=2)
        
        logger.info(f"Feature columns saved to: {feature_columns_file}")
        
        logger.info("=" * 60)
        logger.info("✅ Module 2 Pre-trained Models Training Complete!")
        logger.info(f"✅ {len(training_results['models_trained'])} models ready for instant predictions")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in training pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
