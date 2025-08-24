"""
Data Processing Module for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This module handles:
1. Data quality analysis and documentation
2. Outlier detection and removal with graphical demonstration
3. Statistical preprocessing methods
4. Data validation and cleaning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib
from loguru import logger
import json
from datetime import datetime

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR,
    DATA_CONFIG, PROCESS_PARAMS, MODEL_CONFIG, QUALITY_THRESHOLDS
)

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processor for F&B manufacturing data.
    Implements statistical methods for data quality analysis and outlier detection.
    """
    
    def __init__(self):
        self.scaler = None
        self.process_params = list(PROCESS_PARAMS.keys())
        self.data_config = DATA_CONFIG
        self.quality_report = {}
        self.outlier_report = {}
        logger.info("DataProcessor initialized")
    
    def load_data(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from CSV file and create quality targets.
        
        Args:
            file_path: Path to the CSV file (optional)
        
        Returns:
            Tuple of (process_data, quality_data)
        """
        if file_path is None:
            file_path = RAW_DATA_DIR / self.data_config['raw_data_file']
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Determine file type and load accordingly
            if str(file_path).endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                # For Excel files, just load the first sheet
                data = pd.read_excel(file_path, sheet_name=0)
            
            logger.info(f"Loaded raw data: {data.shape}")
            
            # Clean column names
            data.columns = data.columns.str.strip()
            
            # Extract process data
            relevant_cols = [self.data_config['batch_id_col'], self.data_config['time_col']] + self.process_params
            available_cols = [col for col in relevant_cols if col in data.columns]
            process_data = data[available_cols].copy()
            
            # Create quality data from process parameters
            quality_data = self._create_quality_targets(process_data)
            
            logger.info(f"Process data columns: {process_data.columns.tolist()}")
            logger.info(f"Quality data shape: {quality_data.shape}")
            
            return process_data, quality_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _create_quality_targets(self, process_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create realistic quality targets based on process parameters.
        
        Args:
            process_data: Process data with batch information
            
        Returns:
            DataFrame with batch-level quality targets
        """
        logger.info("Creating quality targets from process parameters")
        
        # Group by batch to calculate batch-level metrics
        batch_data = []
        
        for batch_id in process_data['Batch_ID'].unique():
            batch_process = process_data[process_data['Batch_ID'] == batch_id]
            
            # Calculate deviations from ideal conditions
            total_deviation = 0
            param_count = 0
            
            for param, config in PROCESS_PARAMS.items():
                if param in batch_process.columns:
                    param_values = batch_process[param]
                    ideal = config['ideal']
                    tolerance = config['tolerance']
                    
                    # Calculate average deviation for this batch
                    avg_deviation = abs(param_values.mean() - ideal) / ideal
                    total_deviation += min(avg_deviation / (tolerance / ideal), 1.0)  # Cap at 1.0
                    param_count += 1
            
            # Calculate quality metrics
            if param_count > 0:
                avg_deviation = total_deviation / param_count
                
                # Quality Score: Higher when deviations are lower
                base_quality = 95.0  # Base quality score
                quality_penalty = avg_deviation * 20  # Up to 20 point penalty
                quality_score = max(base_quality - quality_penalty + np.random.normal(0, 2), 70.0)
                
                # Final Weight: Based on ingredient quantities with some process impact
                flour_avg = batch_process['Flour (kg)'].mean() if 'Flour (kg)' in batch_process.columns else 10.0
                sugar_avg = batch_process['Sugar (kg)'].mean() if 'Sugar (kg)' in batch_process.columns else 5.0
                yeast_avg = batch_process['Yeast (kg)'].mean() if 'Yeast (kg)' in batch_process.columns else 2.0
                salt_avg = batch_process['Salt (kg)'].mean() if 'Salt (kg)' in batch_process.columns else 1.0
                
                # Base weight from ingredients + process efficiency factor
                base_weight = (flour_avg + sugar_avg + yeast_avg + salt_avg) * 2.5  # Typical bread yield
                process_efficiency = 1.0 - (avg_deviation * 0.1)  # Up to 10% loss from poor process
                final_weight = base_weight * process_efficiency + np.random.normal(0, 1.0)
                
            else:
                # Fallback values
                quality_score = 85.0 + np.random.normal(0, 5)
                final_weight = 45.0 + np.random.normal(0, 3)
            
            batch_data.append({
                'Batch_ID': int(batch_id),
                'Final_Weight': round(final_weight, 2),
                'Quality_Score': round(min(max(quality_score, 0), 100), 2)
            })
        
        quality_df = pd.DataFrame(batch_data)
        logger.info(f"Created quality targets for {len(quality_df)} batches")
        
        return quality_df
    
    def analyze_data_quality(self, process_data: pd.DataFrame, quality_data: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality analysis using statistical methods.
        
        Args:
            process_data: Process parameters data
            quality_data: Quality outcomes data
        
        Returns:
            Dictionary containing quality metrics and analysis
        """
        logger.info("Starting comprehensive data quality analysis")
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'data_overview': {},
            'missing_values': {},
            'data_types': {},
            'statistical_summary': {},
            'data_quality_score': 0.0,
            'recommendations': []
        }
        
        # Data overview
        quality_report['data_overview'] = {
            'process_data_shape': process_data.shape,
            'quality_data_shape': quality_data.shape,
            'total_batches': process_data[self.data_config['batch_id_col']].nunique(),
            'time_points_per_batch': process_data.groupby(self.data_config['batch_id_col']).size().mean(),
            'process_parameters': len(self.process_params),
            'quality_metrics': len(self.data_config['target_cols'])
        }
        
        # Missing values analysis
        process_missing = process_data.isnull().sum()
        quality_missing = quality_data.isnull().sum()
        
        quality_report['missing_values'] = {
            'process_data': {
                col: {
                    'count': int(count),
                    'percentage': float(count / len(process_data) * 100)
                }
                for col, count in process_missing.items() if count > 0
            },
            'quality_data': {
                col: {
                    'count': int(count),
                    'percentage': float(count / len(quality_data) * 100)
                }
                for col, count in quality_missing.items() if count > 0
            }
        }
        
        # Data types analysis
        quality_report['data_types'] = {
            'process_data': dict(process_data.dtypes.astype(str)),
            'quality_data': dict(quality_data.dtypes.astype(str))
        }
        
        # Statistical summary
        numeric_process = process_data.select_dtypes(include=[np.number])
        numeric_quality = quality_data.select_dtypes(include=[np.number])
        
        quality_report['statistical_summary'] = {
            'process_data': {
                col: {
                    'mean': float(numeric_process[col].mean()),
                    'std': float(numeric_process[col].std()),
                    'min': float(numeric_process[col].min()),
                    'max': float(numeric_process[col].max()),
                    'median': float(numeric_process[col].median()),
                    'q1': float(numeric_process[col].quantile(0.25)),
                    'q3': float(numeric_process[col].quantile(0.75)),
                    'skewness': float(stats.skew(numeric_process[col].dropna())),
                    'kurtosis': float(stats.kurtosis(numeric_process[col].dropna()))
                }
                for col in numeric_process.columns
            },
            'quality_data': {
                col: {
                    'mean': float(numeric_quality[col].mean()),
                    'std': float(numeric_quality[col].std()),
                    'min': float(numeric_quality[col].min()),
                    'max': float(numeric_quality[col].max()),
                    'median': float(numeric_quality[col].median())
                }
                for col in numeric_quality.columns
            }
        }
        
        # Calculate data quality score
        missing_score = 1.0 - (process_data.isnull().sum().sum() / process_data.size)
        type_score = 1.0  # Assuming proper data types
        range_score = 1.0  # Assuming reasonable value ranges
        
        quality_report['data_quality_score'] = (missing_score + type_score + range_score) / 3
        
        # Generate recommendations
        if missing_score < 0.95:
            quality_report['recommendations'].append("High missing values detected - implement imputation strategy")
        if quality_report['data_quality_score'] < 0.8:
            quality_report['recommendations'].append("Overall data quality needs improvement")
        
        self.quality_report = quality_report
        logger.info(f"Data quality analysis completed. Score: {quality_report['data_quality_score']:.3f}")
        
        return quality_report
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'isolation_forest') -> Dict:
        """
        Detect outliers using multiple statistical methods.
        
        Args:
            data: Input DataFrame
            method: 'isolation_forest', 'iqr', 'zscore', or 'combined'
        
        Returns:
            Dictionary containing outlier analysis results
        """
        logger.info(f"Detecting outliers using {method} method")
        
        outlier_report = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'outlier_counts': {},
            'outlier_indices': {},
            'outlier_percentages': {},
            'before_after_comparison': {}
        }
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Initialize outlier_indices
        outlier_indices = []
        
        if method == 'isolation_forest':
            # Isolation Forest for multivariate outlier detection
            iso_forest = IsolationForest(
                contamination=MODEL_CONFIG['anomaly_contamination'],
                random_state=MODEL_CONFIG['random_state']
            )
            outlier_labels = iso_forest.fit_predict(numeric_data)
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
        
        elif method == 'iqr':
            # IQR method for each column
            outlier_indices = set()
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)].index
                outlier_indices.update(col_outliers)
            outlier_indices = list(outlier_indices)
        
        elif method == 'zscore':
            # Z-score method
            outlier_indices = set()
            for col in numeric_data.columns:
                z_scores = np.abs(stats.zscore(numeric_data[col].dropna()))
                col_outliers = numeric_data[z_scores > 3].index
                outlier_indices.update(col_outliers)
            outlier_indices = list(outlier_indices)
        
        elif method == 'combined':
            # Combine multiple methods
            iso_outliers = self.detect_outliers(data, 'isolation_forest')['outlier_indices']['all']
            iqr_outliers = self.detect_outliers(data, 'iqr')['outlier_indices']['all']
            zscore_outliers = self.detect_outliers(data, 'zscore')['outlier_indices']['all']
            
            # Union of all detected outliers (convert to sets for proper union)
            outlier_indices = list(set(iso_outliers) | set(iqr_outliers) | set(zscore_outliers))
        
        else:
            # Default to isolation forest
            outlier_indices = self.detect_outliers(data, 'isolation_forest')['outlier_indices']['all']
        
        # Store results
        outlier_report['outlier_indices']['all'] = outlier_indices
        outlier_report['outlier_counts']['total'] = len(outlier_indices)
        outlier_report['outlier_percentages']['total'] = len(outlier_indices) / len(data) * 100
        
        # Per-column analysis
        for col in numeric_data.columns:
            col_outliers = data.loc[outlier_indices, col]
            outlier_report['outlier_counts'][col] = len(col_outliers)
            outlier_report['outlier_percentages'][col] = len(col_outliers) / len(data) * 100
        
        # Before/after comparison
        outlier_report['before_after_comparison'] = {
            'before_cleaning': {
                'total_rows': len(data),
                'statistical_summary': numeric_data.describe().to_dict()
            }
        }
        
        self.outlier_report = outlier_report
        logger.info(f"Outlier detection completed. Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(data)*100:.2f}%)")
        
        return outlier_report
    
    def remove_outliers(self, data: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        """
        Remove outliers and provide before/after comparison.
        
        Args:
            data: Input DataFrame
            outlier_indices: Indices of outliers to remove
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Removing {len(outlier_indices)} outliers")
        
        # Store before statistics
        numeric_data = data.select_dtypes(include=[np.number])
        before_stats = numeric_data.describe()
        
        # Remove outliers
        cleaned_data = data.drop(index=outlier_indices).reset_index(drop=True)
        
        # Store after statistics
        cleaned_numeric = cleaned_data.select_dtypes(include=[np.number])
        after_stats = cleaned_numeric.describe()
        
        # Update outlier report with after comparison
        self.outlier_report['before_after_comparison']['after_cleaning'] = {
            'total_rows': len(cleaned_data),
            'rows_removed': len(data) - len(cleaned_data),
            'removal_percentage': (len(data) - len(cleaned_data)) / len(data) * 100,
            'statistical_summary': after_stats.to_dict()
        }
        
        logger.info(f"Outlier removal completed. Remaining data: {len(cleaned_data)} rows")
        
        return cleaned_data
    
    def clean_data(self, process_data: pd.DataFrame, quality_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            process_data: Raw process data
            quality_data: Raw quality data
        
        Returns:
            Tuple of (cleaned_process_data, cleaned_quality_data)
        """
        logger.info("Starting comprehensive data cleaning")
        
        # Step 1: Data quality analysis
        quality_report = self.analyze_data_quality(process_data, quality_data)
        
        # Step 2: Handle missing values
        process_cleaned = process_data.copy()
        quality_cleaned = quality_data.copy()
        
        # Forward fill then backward fill for time series continuity
        process_cleaned = process_cleaned.groupby(self.data_config['batch_id_col']).apply(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        # Median imputation for remaining missing values
        for col in self.process_params:
            if col in process_cleaned.columns:
                median_val = process_cleaned[col].median()
                process_cleaned[col].fillna(median_val, inplace=True)
        
        # Step 3: Outlier detection and removal
        outlier_report = self.detect_outliers(process_cleaned, method='combined')
        outlier_indices = outlier_report['outlier_indices']['all']
        
        if len(outlier_indices) > 0:
            process_cleaned = self.remove_outliers(process_cleaned, outlier_indices)
        
        # Step 4: Data validation
        process_cleaned = self._validate_process_data(process_cleaned)
        quality_cleaned = self._validate_quality_data(quality_cleaned)
        
        # Step 5: Save reports
        self._save_quality_reports(quality_report, outlier_report)
        
        logger.info("Data cleaning completed successfully")
        
        return process_cleaned, quality_cleaned
    
    def _validate_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate process data against known constraints."""
        validated_data = data.copy()
        
        for param, config in PROCESS_PARAMS.items():
            if param in validated_data.columns:
                # Check for physically impossible values
                if 'Temp' in param:
                    validated_data[param] = validated_data[param].clip(-50, 300)
                elif 'Humidity' in param:
                    validated_data[param] = validated_data[param].clip(0, 100)
                elif 'Speed' in param:
                    validated_data[param] = validated_data[param].clip(0, 1000)
                elif 'kg' in param:
                    validated_data[param] = validated_data[param].clip(0, 50)
        
        return validated_data
    
    def _validate_quality_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate quality data against known constraints."""
        validated_data = data.copy()
        
        if 'Final Weight (kg)' in validated_data.columns:
            validated_data['Final Weight (kg)'] = validated_data['Final Weight (kg)'].clip(30, 70)
        
        if 'Quality Score (%)' in validated_data.columns:
            validated_data['Quality Score (%)'] = validated_data['Quality Score (%)'].clip(0, 100)
        
        return validated_data
    
    def _save_quality_reports(self, quality_report: Dict, outlier_report: Dict):
        """Save quality and outlier reports to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save quality report
        quality_file = REPORTS_DIR / f'data_quality_report_{timestamp}.json'
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Save outlier report
        outlier_file = REPORTS_DIR / f'outlier_analysis_report_{timestamp}.json'
        with open(outlier_file, 'w') as f:
            json.dump(outlier_report, f, indent=2, default=str)
        
        logger.info(f"Quality reports saved to {REPORTS_DIR}")
    
    def split_data(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into training and testing sets")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=None
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard' or 'robust'
        
        Returns:
            Tuple of (scaled_train, scaled_test)
        """
        logger.info(f"Scaling features using {method} scaler")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Save scaler
        scaler_path = MODEL_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled
    
    def get_quality_report(self) -> Dict:
        """Get the latest quality report."""
        return self.quality_report
    
    def get_outlier_report(self) -> Dict:
        """Get the latest outlier report."""
        return self.outlier_report
