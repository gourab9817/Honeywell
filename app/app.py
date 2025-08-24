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
dashboard_data = {
    'last_training': None,
    'current_batch': None,
    'quality_score': None,
    'predicted_weight': None,
    'anomaly_risk': 'Low',
    'process_parameters': {},
    'alerts': [],
    'batch_summary': {},
    'training_report': {}
}

# Create upload directory
upload_dir = Path(FLASK_CONFIG['UPLOAD_FOLDER'])
upload_dir.mkdir(parents=True, exist_ok=True)

# Create dashboard data directory
dashboard_data_dir = Path('data/dashboard')
dashboard_data_dir.mkdir(parents=True, exist_ok=True)

def save_dashboard_data():
    """Save dashboard data to JSON file."""
    try:
        dashboard_file = dashboard_data_dir / 'dashboard_data.json'
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving dashboard data: {e}")

def load_dashboard_data():
    """Load dashboard data from JSON file."""
    try:
        dashboard_file = dashboard_data_dir / 'dashboard_data.json'
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                data = json.load(f)
                global dashboard_data
                dashboard_data.update(data)
    except Exception as e:
        print(f"Error loading dashboard data: {e}")

def load_latest_training_report():
    """Load the latest training report from the models directory."""
    try:
        models_dir = Path('data/models')
        if not models_dir.exists():
            return None
        
        # Find the latest training results file
        result_files = list(models_dir.glob('*_results_*.json'))
        if not result_files:
            return None
        
        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            report_data = json.load(f)
        
        return report_data
    except Exception as e:
        print(f"Error loading training report: {e}")
        return None

def update_dashboard_metrics(training_results=None, prediction_results=None):
    """Update dashboard metrics with new data."""
    global dashboard_data
    
    # Load latest training report
    training_report = load_latest_training_report()
    
    # Initialize with default values only if we have meaningful data
    dashboard_data = {
        'quality_score': None,
        'predicted_weight': None,
        'anomaly_risk': None,
        'current_batch': None,
        'process_parameters': {},
        'batch_summary': {},
        'training_report': {},
        'last_training': {}
    }
    
    if training_report:
        # Extract key metrics from training report
        if 'training_status' in training_report and 'results' in training_report['training_status']:
            results = training_report['training_status']['results']
            
            # Get best model performance
            best_model = results.get('best_model', 'Unknown')
            best_score = results.get('best_score', 0)
            
            # Get business impact metrics
            business_impact = results.get('business_impact', {})
            
            # Only add training data if we have meaningful values
            if best_model != 'Unknown' and best_score != 0:
                dashboard_data['last_training'] = {
                    'timestamp': training_report.get('timestamp', ''),
                    'best_model': best_model,
                    'best_score': best_score,
                    'model_scores': results.get('model_scores', {}),
                    'business_impact': business_impact,
                    'feature_importance': results.get('feature_importance', {})
                }
                
                # Update quality score from business impact only if meaningful
                if business_impact and business_impact.get('quality_accuracy', 0) > 0:
                    quality_accuracy = business_impact.get('quality_accuracy', 0) * 100
                    if quality_accuracy > 0:
                        dashboard_data['quality_score'] = round(quality_accuracy, 1)
                    
                    # Calculate predicted weight from weight accuracy only if meaningful
                    weight_accuracy = business_impact.get('weight_accuracy', 0)
                    if weight_accuracy > 0:
                        # Assuming ideal weight is around 45kg
                        dashboard_data['predicted_weight'] = round(45 * weight_accuracy, 1)
                    
                    # Set anomaly risk based on model performance only if meaningful
                    if best_score > 0:
                        if best_score > 0.8:
                            dashboard_data['anomaly_risk'] = 'Low'
                        elif best_score > 0.6:
                            dashboard_data['anomaly_risk'] = 'Medium'
                        else:
                            dashboard_data['anomaly_risk'] = 'High'
    
    if training_results:
        # Only update if we have meaningful training results
        if training_results.get('best_model') and training_results.get('best_score', 0) > 0:
            dashboard_data['last_training'] = {
                'timestamp': datetime.now().isoformat(),
                'best_model': training_results.get('best_model', 'Unknown'),
                'best_score': training_results.get('best_score', 0),
                'model_scores': training_results.get('model_scores', {})
            }
    
    if prediction_results:
        # Only update if we have meaningful prediction results
        quality = prediction_results.get('quality', 0)
        weight = prediction_results.get('weight', 0)
        anomaly_risk = prediction_results.get('anomaly_risk', 'Unknown')
        
        if quality > 0:
            dashboard_data['quality_score'] = quality
        if weight > 0:
            dashboard_data['predicted_weight'] = weight
        if anomaly_risk != 'Unknown':
            dashboard_data['anomaly_risk'] = anomaly_risk
    
    # Update process parameters if we have current data
    if current_data and current_data['process_data'] is not None:
        try:
            latest_data = current_data['process_data'].iloc[-1]
            dashboard_data['process_parameters'] = {}
            
            for param, config in PROCESS_PARAMS.items():
                if param in latest_data:
                    current_val = float(latest_data[param])
                    ideal = config['ideal']
                    deviation = abs(current_val - ideal)
                    
                    # Only include if we have meaningful values
                    if current_val > 0:
                        status = 'normal'
                        if deviation > config['tolerance'] * 2:
                            status = 'critical'
                        elif deviation > config['tolerance']:
                            status = 'warning'
                        
                        dashboard_data['process_parameters'][param] = {
                            'current': round(current_val, 2),
                            'ideal': ideal,
                            'tolerance': config['tolerance'],
                            'unit': config['unit'],
                            'status': status,
                            'deviation_percentage': round((deviation / ideal) * 100, 1)
                        }
        except Exception as e:
            print(f"Error updating process parameters: {e}")
    
    # Update batch summary if we have quality data
    if current_data and current_data['quality_data'] is not None:
        try:
            quality_data = current_data['quality_data']
            total_batches = len(quality_data)
            
            if total_batches > 0:
                pass_count = sum(1 for _, row in quality_data.iterrows() 
                               if QUALITY_THRESHOLDS['weight']['min'] <= row['Final_Weight'] <= QUALITY_THRESHOLDS['weight']['max'] 
                               and row['Quality_Score'] >= QUALITY_THRESHOLDS['quality_score']['min'])
                
                pass_rate = round((pass_count / total_batches) * 100, 2) if total_batches > 0 else 0
                
                # Only include if we have meaningful batch data
                if total_batches > 0 and pass_rate > 0:
                    dashboard_data['batch_summary'] = {
                        'total_batches': total_batches,
                        'pass_rate': pass_rate,
                        'batches': quality_data.tail(5).to_dict('records')
                    }
        except Exception as e:
            print(f"Error updating batch summary: {e}")
    
    # Add training report data to dashboard only if meaningful
    if training_report:
        quality_report = training_report.get('quality_report', {})
        outlier_report = training_report.get('outlier_report', {})
        system_status = training_report.get('system_status', {})
        
        # Only include if we have meaningful data
        if quality_report or outlier_report or system_status:
            dashboard_data['training_report'] = {
                'quality_report': quality_report,
                'outlier_report': outlier_report,
                'system_status': system_status
            }
    
    save_dashboard_data()

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
            
            # Update dashboard with new data
            update_dashboard_metrics()
            
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
        
        # Update dashboard with training results
        update_dashboard_metrics(training_results=training_results)
        
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
        
        # Update dashboard with prediction results
        update_dashboard_metrics(prediction_results=result)
        
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
        # Load latest dashboard data
        load_dashboard_data()
        
        if dashboard_data['process_parameters']:
            return jsonify(dashboard_data['process_parameters'])
        else:
            # Return simulated data if no real data available
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
        # Load latest dashboard data
        load_dashboard_data()
        
        if dashboard_data['batch_summary']:
            return jsonify({
                'success': True,
                'batches': dashboard_data['batch_summary'].get('batches', []),
                'total_batches': dashboard_data['batch_summary'].get('total_batches', 0),
                'pass_rate': dashboard_data['batch_summary'].get('pass_rate', 0)
            })
        
        # Fallback to current data if no dashboard data
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

@app.route('/api/dashboard/metrics')
def get_dashboard_metrics():
    """Get current dashboard metrics."""
    try:
        # Load latest dashboard data
        load_dashboard_data()
        
        # Only include meaningful data - filter out null/zero/undefined values
        metrics = {}
        
        # Only include quality_score if it's meaningful
        quality_score = dashboard_data.get('quality_score')
        if quality_score is not None and quality_score > 0:
            metrics['quality_score'] = quality_score
        
        # Only include predicted_weight if it's meaningful
        predicted_weight = dashboard_data.get('predicted_weight')
        if predicted_weight is not None and predicted_weight > 0:
            metrics['predicted_weight'] = predicted_weight
        
        # Only include anomaly_risk if it's meaningful
        anomaly_risk = dashboard_data.get('anomaly_risk')
        if anomaly_risk is not None and anomaly_risk != 'Unknown':
            metrics['anomaly_risk'] = anomaly_risk
        
        # Only include current_batch if it's meaningful
        current_batch = dashboard_data.get('current_batch')
        if current_batch is not None and current_batch > 0:
            metrics['current_batch'] = current_batch
        
        # Only include last_training if it has meaningful data
        last_training = dashboard_data.get('last_training', {})
        if (last_training and 
            last_training.get('best_model') and 
            last_training.get('best_model') != 'Unknown' and
            last_training.get('best_score', 0) > 0):
            metrics['last_training'] = last_training
        
        # Only include system_status if predictor is available
        try:
            if predictor:
                system_status = predictor.get_system_status()
                if system_status and any(system_status.values()):
                    metrics['system_status'] = system_status
        except:
            pass  # Skip if predictor is not available
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting dashboard metrics: {str(e)}'}), 500

@app.route('/api/dashboard/quality-gauge')
def get_quality_gauge_data():
    """Get quality gauge data for dashboard."""
    try:
        # Load latest dashboard data
        load_dashboard_data()
        
        quality_score = dashboard_data.get('quality_score', 87)
        
        return jsonify({
            'success': True,
            'quality_score': quality_score,
            'gauge_data': [quality_score, 100 - quality_score]
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting quality gauge data: {str(e)}'}), 500

@app.route('/api/dashboard/training-analysis')
def get_training_analysis():
    """Get comprehensive training analysis data."""
    try:
        # Load latest dashboard data
        load_dashboard_data()
        
        training_report = dashboard_data.get('training_report', {})
        last_training = dashboard_data.get('last_training', {})
        
        # Only include meaningful data
        analysis_data = {}
        
        # Model performance - only if we have meaningful data
        if (last_training and 
            last_training.get('best_model') and 
            last_training.get('best_model') != 'Unknown' and
            last_training.get('best_score', 0) > 0):
            
            analysis_data['model_performance'] = {
                'best_model': last_training.get('best_model'),
                'best_score': last_training.get('best_score'),
                'model_scores': last_training.get('model_scores', {})
            }
        
        # Business impact - only if we have meaningful data
        business_impact = last_training.get('business_impact', {})
        if business_impact and any(v != 0 for v in business_impact.values() if isinstance(v, (int, float))):
            analysis_data['business_impact'] = business_impact
        
        # Data quality - only if we have meaningful data
        quality_report = training_report.get('quality_report', {})
        outlier_report = training_report.get('outlier_report', {})
        
        if quality_report or outlier_report:
            data_quality = {}
            
            data_overview = quality_report.get('data_overview', {})
            if data_overview and any(v != 0 for v in data_overview.values() if isinstance(v, (int, float))):
                data_quality['data_overview'] = data_overview
            
            data_quality_score = quality_report.get('data_quality_score', 0)
            if data_quality_score > 0:
                data_quality['data_quality_score'] = data_quality_score
            
            outlier_percentage = outlier_report.get('outlier_percentages', {}).get('total', 0)
            if outlier_percentage > 0:
                data_quality['outlier_percentage'] = outlier_percentage
            
            if data_quality:
                analysis_data['data_quality'] = data_quality
        
        # Feature importance - only if we have meaningful data
        feature_importance = last_training.get('feature_importance', {})
        if feature_importance and any(v > 0 for v in feature_importance.values() if isinstance(v, (int, float))):
            analysis_data['feature_importance'] = feature_importance
        
        # System status - only if we have meaningful data
        system_status = training_report.get('system_status', {})
        if system_status and any(system_status.values()):
            analysis_data['system_status'] = system_status
        
        return jsonify({
            'success': True,
            'analysis': analysis_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting training analysis: {str(e)}'}), 500

@app.route('/api/dashboard/model-comparison')
def get_model_comparison():
    """Get model comparison data for charts."""
    try:
        # Load latest dashboard data
        load_dashboard_data()
        
        last_training = dashboard_data.get('last_training', {})
        model_scores = last_training.get('model_scores', {})
        
        # Prepare data for charts
        models = []
        scores = []
        colors = []
        
        for model_name, model_data in model_scores.items():
            if 'test_metrics' in model_data and 'overall' in model_data['test_metrics']:
                avg_r2 = model_data['test_metrics']['overall'].get('avg_r2', 0)
                models.append(model_name.replace('_', ' ').title())
                scores.append(round(avg_r2 * 100, 1))  # Convert to percentage
                
                # Color coding based on performance
                if avg_r2 > 0.8:
                    colors.append('#10b981')  # Green
                elif avg_r2 > 0.6:
                    colors.append('#f59e0b')  # Orange
                else:
                    colors.append('#ef4444')  # Red
        
        return jsonify({
            'success': True,
            'models': models,
            'scores': scores,
            'colors': colors
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting model comparison: {str(e)}'}), 500

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
    
    # Load existing dashboard data on startup
    load_dashboard_data()
    print("📈 Dashboard data loaded from previous sessions")
    
    print(f"🌐 Dashboard available at: http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=FLASK_CONFIG['DEBUG']
    )
