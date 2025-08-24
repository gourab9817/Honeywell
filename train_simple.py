"""
Simple Training Script for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This script focuses on training one good model efficiently.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import *
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

def main():
    """Simple training pipeline focusing on efficiency."""
    print("🏭 F&B Process Anomaly Detection System - Simple Training")
    print("=" * 60)
    
    try:
        # Step 1: Data Loading
        print("\n📊 Step 1: Loading Data")
        print("-" * 40)
        
        data_processor = DataProcessor()
        process_data, quality_data = data_processor.load_data()
        print(f"✅ Loaded {len(process_data)} process rows, {len(quality_data)} batches")
        
        # Step 2: Data Cleaning (simplified)
        print("\n🧹 Step 2: Data Cleaning")
        print("-" * 40)
        
        # Simple cleaning without complex outlier detection
        clean_process_data = process_data.copy()
        clean_quality_data = quality_data.copy()
        
        # Fill missing values with median
        for col in PROCESS_PARAMS.keys():
            if col in clean_process_data.columns:
                median_val = clean_process_data[col].median()
                clean_process_data[col].fillna(median_val, inplace=True)
        
        print(f"✅ Data cleaned: {len(clean_process_data)} rows")
        
        # Step 3: Feature Engineering (simplified)
        print("\n⚙️ Step 3: Feature Engineering")
        print("-" * 40)
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.extract_batch_features(clean_process_data, clean_quality_data)
        print(f"✅ Extracted {len(features_df.columns)} features for {len(features_df)} batches")
        
        # Step 4: Model Training (single best model)
        print("\n🤖 Step 4: Training Best Model")
        print("-" * 40)
        
        # Prepare data
        features_df = features_df.dropna(subset=['Final_Weight', 'Quality_Score'])
        if len(features_df) < 10:
            print("❌ Insufficient data for training")
            return
        
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Batch_ID', 'Final_Weight', 'Quality_Score']]
        X = features_df[feature_cols]
        y = features_df[['Final_Weight', 'Quality_Score']]
        
        # Split and scale
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)
        
        print(f"✅ Training data: {len(X_train)} samples, {len(X_train.columns)} features")
        
        # Train only Random Forest (usually performs well)
        model_trainer = ModelTrainer()
        print("🌲 Training Random Forest model...")
        
        try:
            rf_result = model_trainer._train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
            r2_score = rf_result['test_metrics']['overall']['avg_r2']
            print(f"✅ Random Forest R² Score: {r2_score:.4f}")
            
            # Save the model
            model_trainer.best_model = rf_result['model']
            saved_files = model_trainer.save_models('simple_fnb')
            print(f"✅ Model saved: {len(saved_files)} files")
            
        except Exception as e:
            print(f"❌ Error training Random Forest: {e}")
            return
        
        # Step 5: Test Prediction
        print("\n📈 Step 5: Testing Prediction")
        print("-" * 40)
        
        try:
            predictor = Predictor()
            test_batch = features_df.iloc[:1]
            result = predictor.predict_batch(test_batch)
            
            print(f"✅ Sample prediction:")
            print(f"   Weight: {result['predictions']['weight']:.2f} kg")
            print(f"   Quality: {result['predictions']['quality']:.2f}%")
            print(f"   Status: {result['quality_assessment']['overall_status']}")
            
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
        
        # Success
        print("\n🎉 SIMPLE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📈 Model Performance: R² = {r2_score:.4f}")
        print(f"💾 Model saved successfully")
        print(f"🚀 Ready to run: python app/app.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
