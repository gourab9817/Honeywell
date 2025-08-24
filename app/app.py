"""
Flask Application for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This application provides:
1. Real-time process monitoring dashboard
2. Quality prediction API
3. Anomaly detection alerts
4. Data upload and processing
5. Model training and evaluation
6. Comprehensive reporting
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

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

from src.config import FLASK_CONFIG, DATA_CONFIG, PROCESS_PARAMS, QUALITY_THRESHOLDS
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)
CORS(app)

# Initialize components
data_processor = DataProcessor()
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()
predictor = Predictor()

# Global variables
current_data = None
current_features = None
training_status = {}

# Create upload directory
upload_dir = Path(FLASK_CONFIG['UPLOAD_FOLDER'])
upload_dir.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'csv'}

@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render monitoring dashboard."""
    return render_template('dashboard.html')

@app.route('/reports')
def reports():
    """Render reports page."""
    return render_template('reports.html')

@app.route('/api/status')
def get_status():
    """Get system status."""
    try:
        system_status = predictor.get_system_status()
        return jsonify({
            'status': 'success',
            'data': system_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process data file."""
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
                'filename': filename,
                'data_summary': {
                    'process_rows': len(clean_process_data),
                    'quality_batches': len(clean_quality_data),
                    'features_count': len(selected_features_df.columns),
                    'batches_processed': len(selected_features_df)
                },
                'quality_report': quality_report,
                'outlier_report': outlier_report
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'File processing error: {str(e)}'}), 500
    
@app.route('/api/train', methods=['POST'])
def train_models():
    """Train quality prediction and anomaly detection models."""
    try:
        if current_features is None:
            return jsonify({'error': 'No data loaded. Please upload data first.'}), 400
        
        # Prepare training data
        features_df = current_features.copy()
        
        # Remove rows with missing targets
        features_df = features_df.dropna(subset=['Final_Weight', 'Quality_Score'])
        
        if len(features_df) < 10:
            return jsonify({'error': 'Insufficient data for training. Need at least 10 samples.'}), 400
        
        # Prepare features and targets
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Batch_ID', 'Final_Weight', 'Quality_Score']]
        X = features_df[feature_cols]
        y = features_df[['Final_Weight', 'Quality_Score']]
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)
        
        # Train quality models
        training_results = model_trainer.train_quality_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Train anomaly detector
        anomaly_results = model_trainer.train_anomaly_detector(X_train_scaled)
        
        # Save models
        saved_files = model_trainer.save_models('honeywell_fnb')
        
        # Reload predictor with new models
        global predictor
        predictor = Predictor()
        
        # Update training status
        global training_status
        training_status = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'results': training_results,
            'anomaly_results': anomaly_results,
            'saved_files': saved_files
        }
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': training_results,
            'anomaly_results': anomaly_results,
            'saved_files': saved_files
        })
        
    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make quality prediction for a batch."""
    try:
        if current_features is None:
            return jsonify({'error': 'No data loaded. Please upload data first.'}), 400
        
        data = request.json
        batch_id = data.get('batch_id', current_features['Batch_ID'].iloc[-1])
        
        # Get features for the specified batch
        batch_features = current_features[current_features['Batch_ID'] == batch_id]
        
        if batch_features.empty:
            return jsonify({'error': f'Batch {batch_id} not found'}), 404
        
        # Make prediction
        result = predictor.predict_batch(batch_features)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/predict/realtime', methods=['POST'])
def realtime_prediction():
    """Make real-time prediction for streaming data."""
    try:
        data = request.json
        
        if not data or 'process_data' not in data:
            return jsonify({'error': 'Process data required'}), 400
        
        # Convert to DataFrame
        process_df = pd.DataFrame(data['process_data'])
        
        # Make real-time prediction
        result = predictor.predict_realtime(process_df)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Real-time prediction error: {str(e)}'}), 500

@app.route('/api/process-parameters')
def get_process_parameters():
    """Get current process parameters."""
    try:
        if current_data is None or current_data['process_data'] is None:
            # Return simulated data
            simulated_data = {}
            for param, config in PROCESS_PARAMS.items():
                ideal = config['ideal']
                current_val = ideal + np.random.normal(0, config['tolerance'] * 0.5)
                deviation = abs(current_val - ideal)
                
                status = 'normal'
                if deviation > config['tolerance'] * 2:
                    status = 'critical'
                elif deviation > config['tolerance']:
                    status = 'warning'
                
            simulated_data[param] = {
                    'current': round(current_val, 2),
                'ideal': ideal,
                'tolerance': config['tolerance'],
                'unit': config['unit'],
                    'status': status,
                    'deviation_percentage': round((deviation / ideal) * 100, 1)
            }
            
        return jsonify(simulated_data)
    
    # Get latest values from actual data
        process_data = current_data['process_data']
        latest_batch = process_data['Batch_ID'].max()
        latest_data = process_data[process_data['Batch_ID'] == latest_batch].iloc[-1]
        
        parameters = {}
        for param, config in PROCESS_PARAMS.items():
            if param in latest_data:
                current_val = float(latest_data[param])
                ideal = config['ideal']
                deviation = abs(current_val - ideal)
                
                status = 'normal'
                if deviation > config['tolerance'] * 2:
                    status = 'critical'
                elif deviation > config['tolerance']:
                    status = 'warning'
                
                parameters[param] = {
                    'current': round(current_val, 2),
                    'ideal': ideal,
                    'tolerance': config['tolerance'],
                    'unit': config['unit'],
                    'status': status,
                    'deviation_percentage': round((deviation / ideal) * 100, 1)
                }
        
        return jsonify(parameters)
        
    except Exception as e:
        return jsonify({'error': f'Error getting parameters: {str(e)}'}), 500

@app.route('/api/trend-data')
def get_trend_data():
    """Get trend data for charts."""
    try:
        hours = request.args.get('hours', 1, type=int)
        
        if current_data is None or current_data['process_data'] is None:
            # Return simulated trend data
            time_points = list(range(60))
            trends = {
                'timestamps': time_points,
                'parameters': {}
            }
            
            for param, config in list(PROCESS_PARAMS.items())[:5]:  # Limit to 5 parameters
                ideal = config['ideal']
                values = ideal + np.random.normal(0, config['tolerance'], 60)
                trends['parameters'][param] = values.tolist()
            
            trends['predicted_quality'] = (85 + np.random.normal(0, 3, 60)).tolist()
            
            return jsonify(trends)
        
            # Use actual data
        process_data = current_data['process_data']
        latest_batch = process_data['Batch_ID'].max()
        batch_data = process_data[process_data['Batch_ID'] == latest_batch]
        
        # Limit to requested time range
        if hours == 1:
            batch_data = batch_data.tail(60)
            
            trends = {
                'timestamps': batch_data['Time'].tolist(),
                'parameters': {}
            }
            
        # Add parameter trends
        for param in PROCESS_PARAMS.keys():
            if param in batch_data.columns:
                trends['parameters'][param] = batch_data[param].tolist()
        
        # Add quality predictions if available
        if current_features is not None:
            batch_features = current_features[current_features['Batch_ID'] == latest_batch]
            if not batch_features.empty:
                try:
                    result = predictor.predict_batch(batch_features)
                    quality_pred = result['predictions']['quality']
                    trends['predicted_quality'] = [quality_pred] * len(batch_data)
                except:
                    trends['predicted_quality'] = [85] * len(batch_data)
            
            return jsonify(trends)
        
    except Exception as e:
        return jsonify({'error': f'Error getting trend data: {str(e)}'}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    try:
        limit = request.args.get('limit', 10, type=int)
        alerts = predictor.get_alert_history(limit)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'total_alerts': len(predictor.alert_history)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting alerts: {str(e)}'}), 500

@app.route('/api/batch-summary')
def get_batch_summary():
    """Get summary of all batches."""
    try:
        if current_data is None or current_data['quality_data'] is None:
            return jsonify({'error': 'No quality data available'}), 404
    
        quality_data = current_data['quality_data']
        summary = []
        
        for _, row in quality_data.iterrows():
            batch_id = row['Batch_ID']
            weight = row['Final_Weight']
            quality = row['Quality_Score']
            
            batch_summary = {
                'batch_id': int(batch_id),
                'actual_weight': round(weight, 2),
                'actual_quality': round(quality, 2),
                'weight_status': 'pass' if QUALITY_THRESHOLDS['weight']['min'] <= weight <= QUALITY_THRESHOLDS['weight']['max'] else 'fail',
                'quality_status': 'pass' if quality >= QUALITY_THRESHOLDS['quality_score']['min'] else 'fail'
            }
            
            # Add predictions if available
            if current_features is not None:
                batch_features = current_features[current_features['Batch_ID'] == batch_id]
                if not batch_features.empty:
                    try:
                        result = predictor.predict_batch(batch_features)
                        batch_summary['predicted_weight'] = round(result['predictions']['weight'], 2)
                        batch_summary['predicted_quality'] = round(result['predictions']['quality'], 2)
                        batch_summary['anomaly_detected'] = result['anomalies']['is_anomaly']
                    except:
                        pass
            
            summary.append(batch_summary)
        
        # Calculate overall metrics
        total_batches = len(summary)
        pass_rate = sum(1 for s in summary 
                       if s['weight_status'] == 'pass' and s['quality_status'] == 'pass') / total_batches * 100
        
        return jsonify({
            'success': True,
            'batches': summary,
            'total_batches': total_batches,
            'pass_rate': round(pass_rate, 2)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting batch summary: {str(e)}'}), 500

@app.route('/api/reports/quality')
def get_quality_report():
    """Get data quality report."""
    try:
        quality_report = data_processor.get_quality_report()
        outlier_report = data_processor.get_outlier_report()
        
        return jsonify({
            'success': True,
            'quality_report': quality_report,
            'outlier_report': outlier_report
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting quality report: {str(e)}'}), 500

@app.route('/api/reports/training')
def get_training_report():
    """Get model training report."""
    try:
        if not training_status.get('completed', False):
            return jsonify({'error': 'No training completed yet'}), 404
        
        return jsonify({
            'success': True,
            'training_status': training_status
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting training report: {str(e)}'}), 500

@app.route('/api/export/features')
def export_features():
    """Export engineered features."""
    try:
        if current_features is None:
            return jsonify({'error': 'No features available'}), 404
        
        # Create CSV in memory
        output = io.StringIO()
        current_features.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'engineered_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Error exporting features: {str(e)}'}), 500

@app.route('/api/export/report')
def export_report():
    """Export comprehensive report."""
    try:
        # Generate comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': predictor.get_system_status(),
            'data_summary': {
                'process_data_loaded': current_data is not None,
                'features_loaded': current_features is not None
            },
            'training_status': training_status,
            'quality_report': data_processor.get_quality_report() if current_data else None,
            'outlier_report': data_processor.get_outlier_report() if current_data else None
        }
        
        # Convert to JSON
        output = io.StringIO()
        json.dump(report, output, indent=2, default=str)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'fnb_anomaly_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    except Exception as e:
        return jsonify({'error': f'Error exporting report: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting F&B Process Anomaly Detection System...")
    print(f"📊 System Status: {predictor.get_system_status()}")
    print(f"🌐 Dashboard available at: http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=FLASK_CONFIG['DEBUG']
    )
