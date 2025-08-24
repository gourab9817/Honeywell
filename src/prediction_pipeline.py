"""
Comprehensive Prediction Pipeline for Module 2
Handles data processing, predictions, anomaly detection, and report generation
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.pretrained_service import PretrainedModelService
from src.config import MODEL_DIR_MODULE2, REPORTS_DIR

class PredictionPipeline:
    """
    Comprehensive prediction pipeline for Module 2
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.pretrained_service = PretrainedModelService()
        
        # Load models
        self.models_loaded = self.pretrained_service.load_pretrained_models()
        if not self.models_loaded:
            raise ValueError("Failed to load pre-trained models")
    
    def process_file_and_predict(self, file_path: str, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: Process file → Extract features → Make predictions → Detect anomalies → Generate report
        
        Args:
            file_path: Path to the uploaded CSV/Excel file
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Comprehensive prediction report with analysis
        """
        try:
            # Step 1: Load and process data
            print("📊 Loading and processing data...")
            process_data, quality_data = self.data_processor.load_data(file_path)
            clean_process_data, clean_quality_data = self.data_processor.clean_data(process_data, quality_data)
            
            # Step 2: Extract features
            print("🔧 Extracting features...")
            features_df = self.feature_engineer.extract_batch_features(clean_process_data, clean_quality_data)
            selected_features_df = self.feature_engineer.select_features_for_module2(features_df)
            
            # Step 3: Prepare prediction features
            print("🎯 Preparing prediction features...")
            prediction_features = selected_features_df.copy()
            
            # Remove target columns if they exist
            target_cols = ['Final_Weight', 'Quality_Score', 'Final_Weight_kg', 'Quality_Score_percent']
            for col in target_cols:
                if col in prediction_features.columns:
                    prediction_features = prediction_features.drop(col, axis=1)
            
            # Step 4: Make predictions
            print("🤖 Making predictions...")
            if use_ensemble:
                prediction_result = self.pretrained_service.predict_ensemble(prediction_features)
            else:
                prediction_result = self.pretrained_service.predict_single(prediction_features)
            
            # Step 5: Detect anomalies
            print("🚨 Detecting anomalies...")
            anomaly_results = self._detect_anomalies(clean_process_data, prediction_result)
            
            # Step 6: Generate analysis graphs
            print("📈 Generating analysis graphs...")
            graph_data = self._generate_analysis_graphs(clean_process_data, prediction_result, anomaly_results)
            
            # Step 7: Create comprehensive report
            print("📋 Creating comprehensive report...")
            report = self._create_comprehensive_report(
                file_path, clean_process_data, clean_quality_data, 
                selected_features_df, prediction_result, anomaly_results, graph_data
            )
            
            return report
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _detect_anomalies(self, process_data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in process data and predictions
        
        Args:
            process_data: Clean process data
            predictions: Model predictions
            
        Returns:
            Anomaly detection results
        """
        anomalies = {
            'process_anomalies': [],
            'prediction_anomalies': [],
            'quality_warnings': [],
            'critical_issues': []
        }
        
        # Process data anomalies
        for column in process_data.select_dtypes(include=[np.number]).columns:
            if column in ['Batch_ID', 'Time']:
                continue
                
            values = process_data[column].dropna()
            if len(values) == 0:
                continue
            
            # Calculate statistics
            mean_val = values.mean()
            std_val = values.std()
            
            # Detect outliers (beyond 3 standard deviations)
            outliers = values[(values < mean_val - 3*std_val) | (values > mean_val + 3*std_val)]
            
            if len(outliers) > 0:
                anomalies['process_anomalies'].append({
                    'parameter': column,
                    'anomaly_type': 'outlier',
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(values)) * 100,
                    'outlier_values': outliers.tolist(),
                    'normal_range': [mean_val - 2*std_val, mean_val + 2*std_val]
                })
        
        # Prediction anomalies
        if 'ensemble_predictions' in predictions:
            pred_values = np.array(predictions['ensemble_predictions'])
            
            # Check for extreme predictions
            for i, target in enumerate(['Final_Weight', 'Quality_Score']):
                target_preds = pred_values[:, i] if pred_values.ndim > 1 else pred_values
                
                # Define reasonable ranges based on training data
                if target == 'Final_Weight':
                    min_reasonable, max_reasonable = 40, 50  # kg (based on training data range)
                else:  # Quality_Score
                    min_reasonable, max_reasonable = 85, 100  # percentage (based on training data range)
                
                extreme_preds = target_preds[(target_preds < min_reasonable) | (target_preds > max_reasonable)]
                
                if len(extreme_preds) > 0:
                    anomalies['prediction_anomalies'].append({
                        'target': target,
                        'anomaly_type': 'extreme_prediction',
                        'count': len(extreme_preds),
                        'extreme_values': extreme_preds.tolist(),
                        'expected_range': [min_reasonable, max_reasonable]
                    })
        
        # Quality warnings based on process parameters
        critical_params = {
            'Water Temp (C)': {'min': 20, 'max': 35, 'critical': True},
            'Oven Temp (C)': {'min': 150, 'max': 200, 'critical': True},
            'Fermentation Temp (C)': {'min': 35, 'max': 40, 'critical': True}
        }
        
        for param, limits in critical_params.items():
            if param in process_data.columns:
                values = process_data[param].dropna()
                out_of_range = values[(values < limits['min']) | (values > limits['max'])]
                
                if len(out_of_range) > 0:
                    warning_level = 'CRITICAL' if limits['critical'] else 'WARNING'
                    anomalies['quality_warnings'].append({
                        'parameter': param,
                        'warning_level': warning_level,
                        'issue': f"Values outside safe range ({limits['min']}-{limits['max']})",
                        'count': len(out_of_range),
                        'problematic_values': out_of_range.tolist()
                    })
        
        return anomalies
    
    def _generate_analysis_graphs(self, process_data: pd.DataFrame, predictions: Dict[str, Any], 
                                anomalies: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate analysis graphs and return base64 encoded images
        
        Args:
            process_data: Clean process data
            predictions: Model predictions
            anomalies: Anomaly detection results
            
        Returns:
            Dictionary with base64 encoded graph images
        """
        graphs = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Process Parameters Trend
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Process Parameters Analysis', fontsize=16, fontweight='bold')
            
            # Temperature trends
            temp_cols = [col for col in process_data.columns if 'Temp' in col and col in process_data.columns]
            for i, col in enumerate(temp_cols[:2]):
                if col in process_data.columns:
                    axes[0, i].plot(process_data[col].values, linewidth=2)
                    axes[0, i].set_title(f'{col} Trend')
                    axes[0, i].set_xlabel('Sample')
                    axes[0, i].set_ylabel('Temperature (°C)')
                    axes[0, i].grid(True, alpha=0.3)
            
            # Other parameters
            other_cols = [col for col in process_data.columns if 'Temp' not in col and 'kg' in col][:2]
            for i, col in enumerate(other_cols):
                if col in process_data.columns:
                    axes[1, i].plot(process_data[col].values, linewidth=2, color='green')
                    axes[1, i].set_title(f'{col} Trend')
                    axes[1, i].set_xlabel('Sample')
                    axes[1, i].set_ylabel('Weight (kg)')
                    axes[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to base64
            import io
            import base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            graphs['process_trends'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 2. Prediction Distribution
            if 'ensemble_predictions' in predictions:
                pred_values = np.array(predictions['ensemble_predictions'])
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle('Prediction Analysis', fontsize=16, fontweight='bold')
                
                # Final Weight distribution
                if pred_values.ndim > 1:
                    weight_preds = pred_values[:, 0]
                    quality_preds = pred_values[:, 1]
                else:
                    weight_preds = pred_values
                    quality_preds = pred_values
                
                axes[0].hist(weight_preds, bins=20, alpha=0.7, color='blue', edgecolor='black')
                axes[0].set_title('Final Weight Distribution')
                axes[0].set_xlabel('Weight (kg)')
                axes[0].set_ylabel('Frequency')
                axes[0].grid(True, alpha=0.3)
                
                # Quality Score distribution
                axes[1].hist(quality_preds, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1].set_title('Quality Score Distribution')
                axes[1].set_xlabel('Quality Score (%)')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                graphs['prediction_distribution'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            # 3. Anomaly Summary
            if anomalies['process_anomalies'] or anomalies['prediction_anomalies']:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Count anomalies by type
                anomaly_counts = {
                    'Process Outliers': len(anomalies['process_anomalies']),
                    'Extreme Predictions': len(anomalies['prediction_anomalies']),
                    'Quality Warnings': len(anomalies['quality_warnings'])
                }
                
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                bars = ax.bar(anomaly_counts.keys(), anomaly_counts.values(), color=colors, alpha=0.8)
                
                ax.set_title('Anomaly Summary', fontsize=16, fontweight='bold')
                ax.set_ylabel('Number of Issues')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, anomaly_counts.values()):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(value), ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                graphs['anomaly_summary'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
        except Exception as e:
            print(f"Error generating graphs: {str(e)}")
            graphs['error'] = f"Failed to generate graphs: {str(e)}"
        
        return graphs
    
    def _create_comprehensive_report(self, file_path: str, process_data: pd.DataFrame, 
                                   quality_data: pd.DataFrame, features_df: pd.DataFrame,
                                   predictions: Dict[str, Any], anomalies: Dict[str, Any],
                                   graphs: Dict[str, str]) -> Dict[str, Any]:
        """
        Create comprehensive prediction report
        
        Args:
            file_path: Original file path
            process_data: Clean process data
            quality_data: Clean quality data
            features_df: Extracted features
            predictions: Model predictions
            anomalies: Anomaly detection results
            graphs: Generated graph data
            
        Returns:
            Comprehensive report dictionary
        """
        # Calculate summary statistics
        if 'ensemble_predictions' in predictions:
            pred_values = np.array(predictions['ensemble_predictions'])
            if pred_values.ndim > 1:
                weight_stats = {
                    'mean': float(np.mean(pred_values[:, 0])),
                    'std': float(np.std(pred_values[:, 0])),
                    'min': float(np.min(pred_values[:, 0])),
                    'max': float(np.max(pred_values[:, 0])),
                    'median': float(np.median(pred_values[:, 0]))
                }
                quality_stats = {
                    'mean': float(np.mean(pred_values[:, 1])),
                    'std': float(np.std(pred_values[:, 1])),
                    'min': float(np.min(pred_values[:, 1])),
                    'max': float(np.max(pred_values[:, 1])),
                    'median': float(np.median(pred_values[:, 1]))
                }
            else:
                weight_stats = quality_stats = {
                    'mean': float(np.mean(pred_values)),
                    'std': float(np.std(pred_values)),
                    'min': float(np.min(pred_values)),
                    'max': float(np.max(pred_values)),
                    'median': float(np.median(pred_values))
                }
        else:
            weight_stats = quality_stats = {}
        
        # Convert numpy values to native Python types for JSON serialization
        def convert_numpy_values(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_values(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_values(item) for item in obj]
            else:
                return obj
        
        # Convert predictions to native Python types
        predictions_converted = convert_numpy_values(predictions)
        
        # Create report
        report = {
            'success': True,
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_file': Path(file_path).name,
                'pipeline_version': '2.0',
                'models_used': predictions.get('models_used', ['ensemble'])
            },
            'data_summary': {
                'process_data_shape': process_data.shape,
                'quality_data_shape': quality_data.shape,
                'features_shape': features_df.shape,
                'total_samples': len(process_data),
                'total_features': len(features_df.columns)
            },
            'predictions': {
                'summary': {
                    'final_weight': weight_stats,
                    'quality_score': quality_stats
                },
                'raw_predictions': predictions_converted,
                'confidence_scores': predictions_converted.get('ensemble_confidence', [])
            },
            'anomaly_analysis': {
                'summary': {
                    'total_process_anomalies': len(anomalies['process_anomalies']),
                    'total_prediction_anomalies': len(anomalies['prediction_anomalies']),
                    'total_quality_warnings': len(anomalies['quality_warnings']),
                    'total_critical_issues': len(anomalies['critical_issues'])
                },
                'details': anomalies
            },
            'recommendations': self._generate_recommendations(anomalies, predictions),
            'graphs': graphs,  # Include the actual graph data
            'quality_assessment': {
                'overall_quality_score': self._calculate_quality_score(anomalies, predictions),
                'risk_level': self._assess_risk_level(anomalies),
                'confidence_level': self._assess_confidence_level(predictions)
            }
        }
        
        # Convert all numpy values to native Python types
        report = convert_numpy_values(report)
        
        # Also convert any remaining numpy values in the report
        import json
        
        def deep_convert_numpy(obj):
            """Recursively convert all numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: deep_convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert_numpy(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'dtype'):  # catch any other numpy types
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return float(obj)
            else:
                return obj
        
        # Convert all numpy values
        report = deep_convert_numpy(report)
        
        # Final validation
        try:
            json.dumps(report)
        except TypeError as e:
            print(f"Warning: Still have JSON serialization issues: {e}")
            # Last resort: convert everything to strings
            def stringify(obj):
                if isinstance(obj, dict):
                    return {k: stringify(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [stringify(item) for item in obj]
                else:
                    return str(obj)
            
            report = stringify(report)
        
        return report
    
    def _generate_recommendations(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Process anomalies
        if anomalies['process_anomalies']:
            recommendations.append("⚠️ Process anomalies detected - Review process parameters for consistency")
        
        # Quality warnings
        if anomalies['quality_warnings']:
            recommendations.append("🚨 Quality warnings found - Check critical temperature and pressure settings")
        
        # Prediction anomalies
        if anomalies['prediction_anomalies']:
            recommendations.append("📊 Extreme predictions detected - Verify input data quality and model assumptions")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Process appears to be running within normal parameters")
        
        recommendations.append("📈 Monitor key process parameters regularly")
        recommendations.append("🔍 Consider implementing real-time monitoring for critical parameters")
        
        return recommendations
    
    def _calculate_quality_score(self, anomalies: Dict[str, Any], predictions: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for anomalies
        base_score -= len(anomalies['process_anomalies']) * 5
        base_score -= len(anomalies['prediction_anomalies']) * 10
        base_score -= len(anomalies['quality_warnings']) * 15
        base_score -= len(anomalies['critical_issues']) * 20
        
        return max(0.0, min(100.0, base_score))
    
    def _assess_risk_level(self, anomalies: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        total_issues = (len(anomalies['process_anomalies']) + 
                       len(anomalies['prediction_anomalies']) + 
                       len(anomalies['quality_warnings']) + 
                       len(anomalies['critical_issues']))
        
        if total_issues == 0:
            return "LOW"
        elif total_issues <= 3:
            return "MEDIUM"
        elif total_issues <= 6:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _assess_confidence_level(self, predictions: Dict[str, Any]) -> str:
        """Assess prediction confidence level"""
        if 'ensemble_confidence' in predictions:
            avg_confidence = np.mean(predictions['ensemble_confidence'])
            if avg_confidence >= 0.8:
                return "HIGH"
            elif avg_confidence >= 0.6:
                return "MEDIUM"
            else:
                return "LOW"
        return "UNKNOWN"
