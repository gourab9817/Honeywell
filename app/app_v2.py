"""
Flask Application for F&B Process Anomaly Detection System - Version 2
Dual Module Architecture: Module 1 (Train-on-demand) + Module 2 (Pre-trained Instant Predictions)

This application provides:
1. Module 1: Upload → Train → Predict workflow (existing functionality)
2. Module 2: Instant predictions using pre-trained models
3. Modern responsive UI with module selection
4. Model comparison and performance analytics
5. Batch prediction capabilities
6. Confidence scoring and uncertainty quantification
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
import io
import base64

from src.config import FLASK_CONFIG, DATA_CONFIG, PROCESS_PARAMS, QUALITY_THRESHOLDS, REPORTS_DIR
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from src.pretrained_service import PretrainedModelService
from src.prediction_pipeline import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)
CORS(app)

# Initialize components
data_processor = DataProcessor()
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()  # Module 1
predictor = Predictor()  # Module 1
pretrained_service = PretrainedModelService()  # Module 2
prediction_pipeline = None  # Will be initialized when Module 2 loads

# Global variables
current_data = None
current_features = None
training_status = {}
module2_status = {'loaded': False, 'models_count': 0}

# Create upload directory
upload_dir = Path(FLASK_CONFIG['UPLOAD_FOLDER'])
upload_dir.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'csv'}

# Initialize Module 2 on startup
def init_module2():
    """Initialize Module 2 with pre-trained models."""
    global module2_status, prediction_pipeline
    try:
        success = pretrained_service.load_pretrained_models()
        if success:
            available_models = pretrained_service.get_available_models()
            module2_status = {
                'loaded': True,
                'models_count': len(available_models),
                'models': available_models,
                'last_loaded': datetime.now().isoformat()
            }
            
            # Initialize prediction pipeline
            prediction_pipeline = PredictionPipeline()
            
            print(f"✅ Module 2 initialized with {len(available_models)} pre-trained models")
            print(f"✅ Prediction pipeline ready for comprehensive analysis")
        else:
            module2_status = {'loaded': False, 'models_count': 0, 'error': 'No pre-trained models found'}
            print("⚠️ Module 2: No pre-trained models found. Please run train_pretrained_models.py first.")
    except Exception as e:
        module2_status = {'loaded': False, 'models_count': 0, 'error': str(e)}
        print(f"❌ Module 2 initialization failed: {str(e)}")

# Routes
@app.route('/')
def index():
    """Render main dashboard with module selection."""
    return render_template('index_v2.html', module2_status=module2_status)

@app.route('/module1')
def module1():
    """Module 1: Train-on-demand interface."""
    return render_template('module1.html')

@app.route('/module2')
def module2():
    """Module 2: Pre-trained instant predictions interface."""
    return render_template('module2.html', module2_status=module2_status)

@app.route('/dashboard')
def dashboard():
    """Legacy dashboard route - redirect to main page."""
    return redirect(url_for('index'))

@app.route('/reports')
def reports():
    """Reports page."""
    return render_template('reports.html')

@app.route('/graphs')
def graphs():
    """Analysis graphs page."""
    return render_template('graphs_page.html')

# API Routes - System Status
@app.route('/api/status')
def get_status():
    """Get system status for both modules."""
    try:
        # Module 1 status
        module1_status = {
            'data_loaded': current_data is not None,
            'features_ready': current_features is not None,
            'models_trained': len(training_status) > 0
        }
        
        return jsonify({
            'status': 'operational',
            'module1': module1_status,
            'module2': module2_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# API Routes - Module 1 (Train-on-demand)
@app.route('/api/module1/upload', methods=['POST'])
def module1_upload():
    """Module 1: Upload and process data file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = upload_dir / filename
            file.save(filepath)
            
            # Process uploaded file
            global current_data, current_features
            
            # Load and clean data
            process_data, quality_data = data_processor.load_data(str(filepath))
            clean_process_data, clean_quality_data = data_processor.clean_data(process_data, quality_data)
            
            # Extract features
            features_df = feature_engineer.extract_batch_features(clean_process_data, clean_quality_data)
            selected_features_df = feature_engineer.select_features(features_df)
            
            # Store current data
            current_data = {
                'process_data': clean_process_data,
                'quality_data': clean_quality_data,
                'features': selected_features_df
            }
            current_features = selected_features_df
            
            # Save features
            feature_engineer.save_features(selected_features_df)
            
            # Get data quality report
            quality_report = data_processor.get_quality_report()
            outlier_report = data_processor.get_outlier_report()
            
            return jsonify({
                'success': True,
                'message': 'File uploaded and processed successfully',
                'data_info': {
                    'rows_processed': len(selected_features_df),
                    'features_extracted': len(selected_features_df.columns),
                    'quality_score': quality_report.get('overall_quality_score', 'N/A'),
                    'outliers_detected': outlier_report.get('total_outliers', 0)
                },
                'quality_report': quality_report,
                'outlier_report': outlier_report
            })
        
        return jsonify({'error': 'Invalid file format. Please upload Excel or CSV file.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/module1/train', methods=['POST'])
def module1_train():
    """Module 1: Train models with uploaded data."""
    try:
        global training_status
        
        if current_features is None:
            return jsonify({'error': 'No data available. Please upload data first.'}), 400
        
        # Prepare training data
        X_train = current_features.drop(['Final_Weight_kg', 'Quality_Score_percent'], axis=1, errors='ignore')
        
        # Determine target columns
        target_cols = []
        if 'Final_Weight_kg' in current_features.columns:
            target_cols.append('Final_Weight_kg')
        if 'Quality_Score_percent' in current_features.columns:
            target_cols.append('Quality_Score_percent')
        
        if not target_cols:
            return jsonify({'error': 'No target columns found in the data'}), 400
        
        y_train = current_features[target_cols]
        
        # Train models
        results = model_trainer.train_models(X_train, y_train)
        training_status = results
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'Models trained successfully. Best model: {results.get("best_model", "N/A")}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

@app.route('/api/module1/predict', methods=['POST'])
def module1_predict():
    """Module 1: Make predictions with trained models."""
    try:
        if not training_status:
            return jsonify({'error': 'No trained models available. Please train models first.'}), 400
        
        # Get prediction data from request
        data = request.get_json()
        
        if 'features' in data:
            # Direct feature input
            features_df = pd.DataFrame([data['features']])
        else:
            # Use current data for batch prediction
            if current_features is None:
                return jsonify({'error': 'No data available for prediction'}), 400
            features_df = current_features.drop(['Final_Weight_kg', 'Quality_Score_percent'], axis=1, errors='ignore')
        
        # Make predictions
        predictions = predictor.predict_batch(features_df)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'model_info': training_status
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making predictions: {str(e)}'}), 500

# API Routes - Module 2 (Pre-trained)
@app.route('/api/module2/models')
def module2_get_models():
    """Get available pre-trained models."""
    try:
        if not module2_status['loaded']:
            return jsonify({'error': 'Module 2 not initialized. No pre-trained models available.'}), 400
        
        models = pretrained_service.get_available_models()
        comparison = pretrained_service.get_model_comparison()
        
        return jsonify({
            'success': True,
            'models': models,
            'comparison': comparison,
            'status': module2_status
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting models: {str(e)}'}), 500

@app.route('/api/module2/predict', methods=['POST'])
def module2_predict():
    """Module 2: Make instant predictions with pre-trained models."""
    try:
        if not module2_status['loaded']:
            return jsonify({'error': 'Module 2 not initialized. Please ensure pre-trained models are available.'}), 400
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Save uploaded file temporarily
                filename = secure_filename(file.filename)
                filepath = upload_dir / filename
                file.save(filepath)
                
                # Process the file through the prediction pipeline
                if prediction_pipeline is None:
                    return jsonify({'error': 'Prediction pipeline not initialized'}), 500
                
                # Get prediction parameters
                use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
                model_name = request.form.get('model_name')  # None for ensemble
                
                # Use the prediction pipeline to process the file
                report = prediction_pipeline.process_file_and_predict(str(filepath), use_ensemble)
                
                if not report.get('success', False):
                    return jsonify({'error': f'File processing failed: {report.get("error", "Unknown error")}'}), 500
                
                # Extract predictions from the report
                predictions = report.get('predictions', {})
                if 'ensemble_predictions' in predictions:
                    result = {
                        'ensemble_predictions': predictions['ensemble_predictions'],
                        'ensemble_confidence': predictions.get('ensemble_confidence', [0.8, 0.8]),
                        'models_used': predictions.get('models_used', [])
                    }
                elif 'individual_predictions' in predictions:
                    result = {
                        'predictions': predictions.get('individual_predictions', []),
                        'confidence_scores': predictions.get('confidence_scores', [0.8, 0.8]),
                        'model_display_name': model_name or 'Selected Model'
                    }
                else:
                    # Fallback: use the first available prediction format
                    result = predictions
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Invalid file format. Please upload Excel or CSV file.'}), 400
        
        # Handle JSON data (for manual input)
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided for prediction'}), 400
        
        # Validate input
        if 'features' not in data:
            return jsonify({'error': 'No features provided for prediction'}), 400
        
        # Create features DataFrame
        if isinstance(data['features'], dict):
            # Single sample
            features_df = pd.DataFrame([data['features']])
        elif isinstance(data['features'], list):
            # Multiple samples
            features_df = pd.DataFrame(data['features'])
        else:
            return jsonify({'error': 'Invalid features format'}), 400
        
        # Validate features DataFrame
        if features_df.empty:
            return jsonify({'error': 'Empty features DataFrame created'}), 400
        
        print(f"DEBUG: Features DataFrame shape: {features_df.shape}")
        print(f"DEBUG: Features DataFrame columns: {list(features_df.columns)}")
        print(f"DEBUG: Features DataFrame sample: {features_df.head(1).to_dict()}")
        
        # Get prediction parameters
        model_name = data.get('model_name')  # None for ensemble
        use_ensemble = data.get('use_ensemble', True)
        
        # Make predictions
        if use_ensemble and model_name is None:
            result = pretrained_service.predict_ensemble(features_df)
        else:
            result = pretrained_service.predict_single(features_df, model_name)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making predictions: {str(e)}'}), 500

@app.route('/api/module2/predict/batch', methods=['POST'])
def module2_predict_batch():
    """Module 2: Batch predictions with pre-trained models."""
    try:
        if not module2_status['loaded']:
            return jsonify({'error': 'Module 2 not initialized'}), 400
        
        # Handle file upload for batch prediction
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Read uploaded file
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                # Process the data through feature engineering
                # Note: This assumes the uploaded file has raw process data
                process_data = df
                
                # Create dummy quality data if not present
                if 'Final_Weight_kg' not in df.columns:
                    quality_data = pd.DataFrame({
                        'Batch_ID': range(len(df)),
                        'Final_Weight_kg': [0] * len(df),
                        'Quality_Score_percent': [0] * len(df)
                    })
                else:
                    quality_data = df[['Final_Weight_kg', 'Quality_Score_percent']].copy()
                
                # Extract features
                features_df = feature_engineer.extract_batch_features(process_data, quality_data)
                
                # Remove target columns for prediction
                prediction_features = features_df.drop(['Final_Weight_kg', 'Quality_Score_percent'], 
                                                     axis=1, errors='ignore')
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        else:
            # JSON data
            data = request.get_json()
            if 'features' not in data:
                return jsonify({'error': 'No features provided'}), 400
            
            prediction_features = pd.DataFrame(data['features'])
        
        # Make batch predictions
        use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
        result = pretrained_service.predict_batch(prediction_features, use_ensemble)
        
        return jsonify({
            'success': True,
            'result': result,
            'batch_size': len(prediction_features),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in batch prediction: {str(e)}'}), 500

@app.route('/api/module2/retrain', methods=['POST'])
def module2_retrain():
    """Retrain Module 2 models with new data."""
    try:
        # This would trigger retraining of all pre-trained models
        # For now, return a message indicating manual retraining is needed
        return jsonify({
            'success': False,
            'message': 'Model retraining requires running train_pretrained_models.py script',
            'instruction': 'Please run: python train_pretrained_models.py'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error retraining models: {str(e)}'}), 500

@app.route('/api/module2/comprehensive-analysis', methods=['POST'])
def module2_comprehensive_analysis():
    """Module 2: Comprehensive analysis with predictions, anomalies, and graphs."""
    try:
        if not module2_status['loaded'] or prediction_pipeline is None:
            return jsonify({'error': 'Module 2 not initialized. Please ensure pre-trained models are available.'}), 400
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided for analysis'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Please upload Excel or CSV file.'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = upload_dir / filename
        file.save(filepath)
        
        # Get analysis parameters
        use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
        
        print(f"🔍 Starting comprehensive analysis for file: {filename}")
        
        # Run comprehensive analysis
        report = prediction_pipeline.process_file_and_predict(str(filepath), use_ensemble)
        
        if not report.get('success', False):
            return jsonify({'error': f'Analysis failed: {report.get("error", "Unknown error")}'}), 500
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"comprehensive_analysis_{timestamp}.json"
        report_path = REPORTS_DIR / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comprehensive analysis completed. Report saved: {report_filename}")
        
        return jsonify({
            'success': True,
            'message': 'Comprehensive analysis completed successfully',
            'report': report,
            'report_file': report_filename,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in comprehensive analysis: {str(e)}'}), 500

# Utility Routes
@app.route('/api/module2/load-report', methods=['POST'])
def module2_load_report():
    """Load a saved comprehensive analysis report from JSON file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Please upload a JSON file'}), 400
        
        # Read the JSON file
        import json
        report_data = json.load(file)
        
        if not report_data.get('success', False):
            return jsonify({'error': 'Invalid report file'}), 400
        
        return jsonify({
            'success': True,
            'report': report_data,
            'message': 'Report loaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error loading report: {str(e)}'}), 500

@app.route('/api/export/report')
def export_report():
    """Export comprehensive report."""
    try:
        report_type = request.args.get('type', 'summary')
        report_format = request.args.get('format', 'json')
        
        report = {
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'module1_status': {
                'data_loaded': current_data is not None,
                'models_trained': len(training_status) > 0,
                'training_results': training_status
            },
            'module2_status': module2_status
        }
        
        if current_data:
            report['data_summary'] = {
                'process_data_shape': current_data['process_data'].shape,
                'quality_data_shape': current_data['quality_data'].shape,
                'features_shape': current_data['features'].shape
            }
        
        if module2_status['loaded']:
            report['module2_models'] = pretrained_service.get_model_comparison()
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

# Initialize Module 2 on startup
with app.app_context():
    init_module2()

if __name__ == '__main__':
    print("🚀 Starting F&B Anomaly Detection System - Version 2")
    print("📊 Module 1: Train-on-demand predictions")
    print("⚡ Module 2: Instant pre-trained predictions")
    print("-" * 50)
    
    app.run(
        host=FLASK_CONFIG.get('HOST', '127.0.0.1'),
        port=FLASK_CONFIG.get('PORT', 5000),
        debug=FLASK_CONFIG.get('DEBUG', True)
    )
