"""
Complete Training Script for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This script demonstrates the complete ML pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Anomaly detection
5. Business impact analysis
6. Report generation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import *
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

def main():
    """Main training pipeline."""
    print("🏭 F&B Process Anomaly Detection System - Training Pipeline")
    print("=" * 70)
    
    # Step 1: Data Loading and Preprocessing
    print("\n📊 Step 1: Data Loading and Preprocessing")
    print("-" * 50)
    
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Load data
        print("Loading data from Excel file...")
        process_data, quality_data = data_processor.load_data()
        print(f"✅ Loaded {len(process_data)} process data rows")
        print(f"✅ Loaded quality data for {len(quality_data)} batches")
        
        # Clean data
        print("Cleaning and preprocessing data...")
        clean_process_data, clean_quality_data = data_processor.clean_data(process_data, quality_data)
        print(f"✅ Cleaned data: {len(clean_process_data)} rows remaining")
        
        # Data quality analysis
        print("Performing data quality analysis...")
        quality_report = data_processor.get_quality_report()
        outlier_report = data_processor.get_outlier_report()
        
        print(f"✅ Data quality score: {quality_report['data_quality_score']:.3f}")
        print(f"✅ Outliers detected: {outlier_report['outlier_counts']['total']}")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return
    
    # Step 2: Feature Engineering
    print("\n⚙️ Step 2: Feature Engineering")
    print("-" * 50)
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Extract features
        print("Extracting comprehensive features...")
        features_df = feature_engineer.extract_batch_features(clean_process_data, clean_quality_data)
        print(f"✅ Extracted {len(features_df.columns)} features for {len(features_df)} batches")
        
        # Feature selection
        print("Selecting most important features...")
        selected_features_df = feature_engineer.select_features(features_df)
        print(f"✅ Selected {len(selected_features_df.columns)} features")
        
        # Save features
        feature_engineer.save_features(selected_features_df)
        print("✅ Features saved to disk")
        
        # Feature summary
        feature_summary = feature_engineer.get_feature_summary(selected_features_df)
        print(f"✅ Feature summary: {feature_summary['total_features']} total features")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training")
    print("-" * 50)
    
    try:
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Prepare training data
        print("Preparing training data...")
        features_df = selected_features_df.copy()
        features_df = features_df.dropna(subset=['Final_Weight', 'Quality_Score'])
        
        if len(features_df) < 10:
            print("❌ Insufficient data for training. Need at least 10 samples.")
            return
        
        # Prepare features and targets
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Batch_ID', 'Final_Weight', 'Quality_Score']]
        X = features_df[feature_cols]
        y = features_df[['Final_Weight', 'Quality_Score']]
        
        print(f"✅ Training data: {len(X)} samples, {len(X.columns)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        print(f"✅ Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)
        print("✅ Features scaled")
        
        # Train quality models
        print("Training quality prediction models...")
        training_results = model_trainer.train_quality_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Train anomaly detector
        print("Training anomaly detection model...")
        anomaly_results = model_trainer.train_anomaly_detector(X_train_scaled)
        
        # Save models
        print("Saving trained models...")
        saved_files = model_trainer.save_models('honeywell_fnb')
        print(f"✅ Models saved: {len(saved_files)} files")
        
        # Print results
        best_model = training_results.get('best_model')
        if best_model:
            best_results = training_results['models_trained'][best_model]
            print(f"✅ Best model: {best_model}")
            print(f"✅ Best R² score: {best_results['test_metrics']['overall']['avg_r2']:.4f}")
        
        # Business impact
        business_impact = training_results.get('business_impact', {})
        if business_impact:
            print(f"✅ Potential annual savings: ${business_impact.get('total_annual_savings', 0):,.0f}")
            print(f"✅ ROI: {business_impact.get('roi_percentage', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 4: Model Evaluation and Testing
    print("\n📈 Step 4: Model Evaluation and Testing")
    print("-" * 50)
    
    try:
        # Initialize predictor with trained models
        predictor = Predictor()
        
        # Test predictions
        print("Testing model predictions...")
        test_batch = selected_features_df.iloc[:1]
        prediction_result = predictor.predict_batch(test_batch)
        
        print(f"✅ Sample prediction:")
        print(f"   Weight: {prediction_result['predictions']['weight']:.2f} kg")
        print(f"   Quality: {prediction_result['predictions']['quality']:.2f}%")
        print(f"   Anomaly: {prediction_result['anomalies']['is_anomaly']}")
        print(f"   Risk Level: {prediction_result['anomalies']['risk_level']}")
        
        # Test real-time prediction
        print("Testing real-time prediction...")
        sample_process_data = clean_process_data[clean_process_data['Batch_ID'] == 1].tail(1)
        realtime_result = predictor.predict_realtime(sample_process_data)
        print(f"✅ Real-time prediction successful")
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        return
    
    # Step 5: Generate Comprehensive Report
    print("\n📋 Step 5: Generating Comprehensive Report")
    print("-" * 50)
    
    try:
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'project': 'F&B Process Anomaly Detection System - Honeywell Hackathon',
            'data_summary': {
                'process_data_rows': len(clean_process_data),
                'quality_batches': len(clean_quality_data),
                'features_extracted': len(features_df.columns),
                'features_selected': len(selected_features_df.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'data_quality': quality_report,
            'outlier_analysis': outlier_report,
            'model_training': training_results,
            'anomaly_detection': anomaly_results,
            'business_impact': business_impact,
            'model_performance': {
                'best_model': best_model,
                'best_r2_score': best_results['test_metrics']['overall']['avg_r2'] if best_model else None,
                'feature_importance': model_trainer.feature_importance
            },
            'system_status': predictor.get_system_status()
        }
        
        # Save report
        report_file = REPORTS_DIR / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved to: {report_file}")
        
        # Print summary
        print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Data processed: {len(clean_process_data)} rows")
        print(f"⚙️ Features engineered: {len(selected_features_df.columns)}")
        print(f"🤖 Models trained: {len(training_results['models_trained'])}")
        print(f"📈 Best model: {best_model}")
        print(f"💰 Potential savings: ${business_impact.get('total_annual_savings', 0):,.0f}")
        print(f"📋 Report saved: {report_file}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return

def create_visualizations():
    """Create visualizations for the report."""
    print("\n📊 Creating Visualizations")
    print("-" * 50)
    
    try:
        # Load data for visualizations
        data_processor = DataProcessor()
        process_data, quality_data = data_processor.load_data()
        clean_process_data, clean_quality_data = data_processor.clean_data(process_data, quality_data)
        
        # Create visualizations directory
        viz_dir = REPORTS_DIR / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Process Parameters Distribution
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(list(PROCESS_PARAMS.keys())[:6], 1):
            plt.subplot(2, 3, i)
            if param in clean_process_data.columns:
                clean_process_data[param].hist(bins=30, alpha=0.7)
                plt.axvline(PROCESS_PARAMS[param]['ideal'], color='red', linestyle='--', label='Ideal')
                plt.title(f'{param} Distribution')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / 'process_parameters_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Process parameters distribution plot saved")
        
        # 2. Quality Metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        quality_data['Final Weight (kg)'].hist(bins=20, alpha=0.7, color='skyblue')
        plt.axvline(QUALITY_THRESHOLDS['weight']['ideal'], color='red', linestyle='--', label='Ideal')
        plt.title('Final Weight Distribution')
        plt.xlabel('Weight (kg)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        quality_data['Quality Score (%)'].hist(bins=20, alpha=0.7, color='lightgreen')
        plt.axvline(QUALITY_THRESHOLDS['quality_score']['ideal'], color='red', linestyle='--', label='Ideal')
        plt.title('Quality Score Distribution')
        plt.xlabel('Quality Score (%)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'quality_metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Quality metrics distribution plot saved")
        
        # 3. Correlation Matrix
        plt.figure(figsize=(12, 10))
        numeric_data = clean_process_data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Process Parameters Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Correlation matrix plot saved")
        
        print(f"✅ All visualizations saved to: {viz_dir}")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

if __name__ == "__main__":
    # Run main training pipeline
    main()
    
    # Create visualizations
    create_visualizations()
    
    print("\n🚀 Training pipeline completed! You can now run the Flask app:")
    print("   python app/app.py")
    print("\n📊 Access the dashboard at: http://localhost:5000")
