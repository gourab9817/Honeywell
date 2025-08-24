"""
Feature Engineering Module for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This module creates comprehensive features from multi-variable process data including:
1. Statistical features (mean, std, min, max, etc.)
2. Time-series features (trends, stability, volatility)
3. Deviation features (from ideal conditions)
4. Interaction features between parameters
5. Process-specific features for F&B manufacturing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.feature_selection import SelectKBest, f_regression
from loguru import logger
import json
from datetime import datetime

from src.config import (
    PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR,
    PROCESS_PARAMS, FEATURE_CONFIG, QUALITY_THRESHOLDS
)

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineer for F&B manufacturing process data.
    Creates features that capture process variations and anomalies.
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.feature_config = FEATURE_CONFIG
        logger.info("FeatureEngineer initialized")
    
    def extract_batch_features(self, process_data: pd.DataFrame, 
                              quality_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for each batch.
        
        Args:
            process_data: Process parameters data
            quality_data: Quality outcomes data
        
        Returns:
            DataFrame with engineered features for each batch
        """
        logger.info("Starting comprehensive feature extraction")
        
        batch_features = []
        batches = process_data['Batch_ID'].unique()
        
        logger.info(f"Processing {len(batches)} batches")
        
        for batch_id in batches:
            batch_data = process_data[process_data['Batch_ID'] == batch_id]
            
            if batch_data.empty:
                continue
            
            features = {'Batch_ID': batch_id}
            
            # Extract features for each process parameter
            for param in PROCESS_PARAMS.keys():
                if param in batch_data.columns:
                    param_features = self._extract_parameter_features(batch_data, param)
                    features.update(param_features)
            
            # Extract time-series features
            time_features = self._extract_time_features(batch_data)
            features.update(time_features)
            
            # Extract process interaction features
            interaction_features = self._extract_interaction_features(batch_data)
            features.update(interaction_features)
            
            # Extract deviation features
            deviation_features = self._extract_deviation_features(batch_data)
            features.update(deviation_features)
            
            # Extract stability features
            stability_features = self._extract_stability_features(batch_data)
            features.update(stability_features)
            
            # Add quality targets if available
            if batch_id in quality_data['Batch_ID'].values:
                batch_quality = quality_data[quality_data['Batch_ID'] == batch_id].iloc[0]
                features['Final_Weight'] = batch_quality['Final_Weight']
                features['Quality_Score'] = batch_quality['Quality_Score']
            else:
                features['Final_Weight'] = np.nan
                features['Quality_Score'] = np.nan
            
            batch_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(batch_features)
        
        # Handle missing values
        features_df = self._handle_missing_features(features_df)
        
        logger.info(f"Feature extraction completed. Shape: {features_df.shape}")
        
        return features_df
    
    def _extract_parameter_features(self, batch_data: pd.DataFrame, param: str) -> Dict:
        """Extract comprehensive features for a single parameter."""
        features = {}
        param_clean = param.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        
        if param not in batch_data.columns:
            return features
        
        data = pd.to_numeric(batch_data[param], errors='coerce').dropna()
        
        if len(data) == 0:
            return features
        
        # Basic statistical features
        features[f'{param_clean}_mean'] = data.mean()
        features[f'{param_clean}_std'] = data.std() if len(data) > 1 else 0
        features[f'{param_clean}_min'] = data.min()
        features[f'{param_clean}_max'] = data.max()
        features[f'{param_clean}_median'] = data.median()
        features[f'{param_clean}_range'] = data.max() - data.min()
        
        # Percentile features
        features[f'{param_clean}_q25'] = data.quantile(0.25)
        features[f'{param_clean}_q75'] = data.quantile(0.75)
        features[f'{param_clean}_iqr'] = features[f'{param_clean}_q75'] - features[f'{param_clean}_q25']
        
        # Distribution features
        features[f'{param_clean}_skewness'] = stats.skew(data) if len(data) > 2 else 0
        features[f'{param_clean}_kurtosis'] = stats.kurtosis(data) if len(data) > 2 else 0
        
        # Coefficient of variation
        if features[f'{param_clean}_mean'] != 0:
            features[f'{param_clean}_cv'] = features[f'{param_clean}_std'] / features[f'{param_clean}_mean']
        else:
            features[f'{param_clean}_cv'] = 0
        
        # Trend features
        if len(data) > 1:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            features[f'{param_clean}_trend_slope'] = slope
            features[f'{param_clean}_trend_r2'] = r_value ** 2
            features[f'{param_clean}_trend_pvalue'] = p_value
        
        # Peak and valley features
        try:
            peaks, _ = find_peaks(data.values, height=data.mean())
            valleys, _ = find_peaks(-data.values, height=-data.mean())
            features[f'{param_clean}_num_peaks'] = len(peaks)
            features[f'{param_clean}_num_valleys'] = len(valleys)
        except:
            features[f'{param_clean}_num_peaks'] = 0
            features[f'{param_clean}_num_valleys'] = 0
        
        return features
    
    def _extract_time_features(self, batch_data: pd.DataFrame) -> Dict:
        """Extract time-series specific features."""
        features = {}
        
        if 'Time' in batch_data.columns:
            try:
                time_data = pd.to_numeric(batch_data['Time'], errors='coerce').dropna()
                if len(time_data) > 0:
                    features['batch_duration'] = time_data.max() - time_data.min()
                    features['time_steps'] = len(batch_data)
                    features['avg_time_step'] = features['batch_duration'] / features['time_steps'] if features['time_steps'] > 0 else 0
            
                    # Time-based stability
                    if len(time_data) > 1:
                        time_diff = np.diff(time_data)
                        features['time_consistency'] = np.std(time_diff)
                        features['time_regularity'] = 1 / (1 + features['time_consistency'])
            except:
                features['batch_duration'] = len(batch_data)
                features['time_steps'] = len(batch_data)
                features['avg_time_step'] = 1
                features['time_consistency'] = 0
                features['time_regularity'] = 1
        
        return features
    
    def _extract_interaction_features(self, batch_data: pd.DataFrame) -> Dict:
        """Extract interaction features between parameters."""
        features = {}
        
        # Temperature interactions
        temp_params = [col for col in batch_data.columns if 'Temp' in col]
        if len(temp_params) >= 2:
            for i, param1 in enumerate(temp_params):
                for param2 in temp_params[i+1:]:
                    if param1 in batch_data.columns and param2 in batch_data.columns:
                        data1 = pd.to_numeric(batch_data[param1], errors='coerce').dropna()
                        data2 = pd.to_numeric(batch_data[param2], errors='coerce').dropna()
                        
                        if len(data1) > 0 and len(data2) > 0:
                            min_len = min(len(data1), len(data2))
                            temp_diff = data1.iloc[:min_len].values - data2.iloc[:min_len].values
                            
                            param1_clean = param1.replace(' ', '_').replace('(', '').replace(')', '')
                            param2_clean = param2.replace(' ', '_').replace('(', '').replace(')', '')
                            
                            features[f'{param1_clean}_{param2_clean}_diff_mean'] = np.mean(temp_diff)
                            features[f'{param1_clean}_{param2_clean}_diff_std'] = np.std(temp_diff) if len(temp_diff) > 1 else 0
                            features[f'{param1_clean}_{param2_clean}_correlation'] = np.corrcoef(data1.iloc[:min_len], data2.iloc[:min_len])[0, 1] if min_len > 1 else 0
            
            # Process efficiency features
        if 'Mixer Speed (RPM)' in batch_data.columns and 'Flour (kg)' in batch_data.columns:
            rpm_data = pd.to_numeric(batch_data['Mixer Speed (RPM)'], errors='coerce').dropna()
            flour_data = pd.to_numeric(batch_data['Flour (kg)'], errors='coerce').dropna()
            
            if len(rpm_data) > 0 and len(flour_data) > 0:
                rpm_mean = rpm_data.mean()
                flour_mean = flour_data.mean()
                features['rpm_per_kg_flour'] = rpm_mean / flour_mean if flour_mean != 0 else 0
        
        return features
    
    def _extract_deviation_features(self, batch_data: pd.DataFrame) -> Dict:
        """Extract deviation features from ideal conditions."""
        features = {}
        
        for param, config in PROCESS_PARAMS.items():
            if param in batch_data.columns:
                data = pd.to_numeric(batch_data[param], errors='coerce').dropna()
                ideal = config['ideal']
                tolerance = config['tolerance']
                
                if len(data) > 0:
                    param_clean = param.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                    
                    # Deviation statistics
                    deviations = (data - ideal) / ideal * 100
                    features[f'{param_clean}_mean_deviation'] = deviations.mean()
                    features[f'{param_clean}_deviation_std'] = deviations.std() if len(deviations) > 1 else 0
                    features[f'{param_clean}_max_deviation'] = deviations.abs().max()
                    
                    # Out of tolerance ratios
                    out_of_tolerance = np.abs(data - ideal) > tolerance
                    features[f'{param_clean}_out_of_tolerance_ratio'] = out_of_tolerance.mean()
                    
                    # Critical deviation ratio
                    critical_deviation = np.abs(data - ideal) > 2 * tolerance
                    features[f'{param_clean}_critical_deviation_ratio'] = critical_deviation.mean()
                    
                    # Consecutive out of tolerance
                    if len(out_of_tolerance) > 1:
                        consecutive_oot = 0
                        max_consecutive = 0
                        for is_oot in out_of_tolerance:
                            if is_oot:
                                consecutive_oot += 1
                                max_consecutive = max(max_consecutive, consecutive_oot)
                            else:
                                consecutive_oot = 0
                        features[f'{param_clean}_max_consecutive_oot'] = max_consecutive
                    else:
                        features[f'{param_clean}_max_consecutive_oot'] = 0
        
        return features
    
    def _extract_stability_features(self, batch_data: pd.DataFrame) -> Dict:
        """Extract process stability features."""
        features = {}
        
        for param in PROCESS_PARAMS.keys():
            if param in batch_data.columns:
                data = pd.to_numeric(batch_data[param], errors='coerce').dropna()
                param_clean = param.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                
                if len(data) > 1:
                    # Rolling statistics for stability
                    rolling_std = data.rolling(window=min(5, len(data)), min_periods=1).std()
                    features[f'{param_clean}_stability'] = 1 / (1 + rolling_std.mean())
                    
                    # Change rate
                    if len(data) > 1:
                        change_rate = np.abs(np.diff(data)) / data.iloc[:-1]
                        features[f'{param_clean}_mean_change_rate'] = change_rate.mean()
                        features[f'{param_clean}_change_rate_std'] = change_rate.std()
                    
                    # Volatility
                    features[f'{param_clean}_volatility'] = data.std() / data.mean() if data.mean() != 0 else 0
        
        return features
    
    def _handle_missing_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature DataFrame."""
        # Fill missing values with median for numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Batch_ID', 'Final_Weight', 'Quality_Score']:
                median_val = features_df[col].median()
                features_df[col].fillna(median_val, inplace=True)
        
        # Fill target columns with 0 if missing (for prediction scenarios)
        if 'Final_Weight' in features_df.columns:
            features_df['Final_Weight'].fillna(0, inplace=True)
        if 'Quality_Score' in features_df.columns:
            features_df['Quality_Score'].fillna(0, inplace=True)
        
        return features_df
    
    def select_features(self, features_df: pd.DataFrame, 
                       target_cols: List[str] = None) -> pd.DataFrame:
        """
        Select the most important features using statistical methods.
        
        Args:
            features_df: Feature DataFrame
            target_cols: Target column names
        
        Returns:
            DataFrame with selected features
        """
        if target_cols is None:
            target_cols = ['Final_Weight', 'Quality_Score']
        
        logger.info("Starting feature selection")
        
        # Remove non-predictive columns
        exclude_cols = ['Batch_ID'] + target_cols
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            logger.warning("No features available for selection")
            return features_df
        
        # Select numeric features only
        numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            logger.warning("No numeric features available")
            return features_df
        
        # Remove features with too many missing values or constant values
        valid_features = []
        for col in numeric_features.columns:
            if (numeric_features[col].notna().sum() > len(numeric_features) * 0.5 and  # At least 50% non-NaN
                numeric_features[col].nunique() > 1):  # Not constant
                valid_features.append(col)
        
        if len(valid_features) == 0:
            logger.warning("No valid features after filtering")
            return features_df
        
        # Use SelectKBest for feature selection
        try:
            # Prepare target for feature selection (use first target if multiple)
            target = features_df[target_cols[0]].fillna(features_df[target_cols[0]].median())
            
            # Select top features
            k = min(50, len(valid_features))  # Select top 50 features or all if less
            selector = SelectKBest(score_func=f_regression, k=k)
            
            X_selected = selector.fit_transform(numeric_features[valid_features], target)
            selected_features = [valid_features[i] for i in selector.get_support(indices=True)]
            
            # Store feature importance scores
            self.feature_importance = dict(zip(valid_features, selector.scores_))
            
            # Create final DataFrame
            final_cols = ['Batch_ID'] + selected_features + target_cols
            final_cols = [col for col in final_cols if col in features_df.columns]
            
            selected_df = features_df[final_cols].copy()
            self.selected_features = selected_features
            
            logger.info(f"Feature selection completed. Selected {len(selected_features)} features")
            
            return selected_df
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return features_df
    
    def select_features_for_module2(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features specifically for Module 2 to ensure compatibility with pre-trained models.
        This method ensures the exact same features that were used during model training.
        
        Args:
            features_df: DataFrame with all extracted features
            
        Returns:
            DataFrame with selected features matching Module 2 model requirements
        """
        # Expected features for Module 2 models (from training)
        expected_features = [
            "Batch_ID",
            "Flour_kg_mean",
            "Flour_kg_max", 
            "Flour_kg_num_valleys",
            "Sugar_kg_mean",
            "Sugar_kg_std",
            "Sugar_kg_min",
            "Sugar_kg_max",
            "Sugar_kg_median",
            "Sugar_kg_range",
            "Sugar_kg_q25",
            "Sugar_kg_q75",
            "Sugar_kg_iqr",
            "Sugar_kg_cv",
            "Sugar_kg_trend_slope",
            "Sugar_kg_trend_r2",
            "Yeast_kg_trend_pvalue",
            "Salt_kg_iqr",
            "Salt_kg_kurtosis",
            "Salt_kg_trend_r2",
            "Water_Temp_C_iqr",
            "Water_Temp_C_kurtosis",
            "Water_Temp_C_trend_r2",
            "Water_Temp_C_num_peaks",
            "Water_Temp_C_num_valleys",
            "Mixer_Speed_RPM_std",
            "Mixer_Speed_RPM_min",
            "Mixer_Speed_RPM_range",
            "Mixer_Speed_RPM_iqr",
            "Mixer_Speed_RPM_cv",
            "Mixer_Speed_RPM_num_valleys",
            "Mixing_Temp_C_num_peaks",
            "Fermentation_Temp_C_range",
            "Fermentation_Temp_C_skewness",
            "Fermentation_Temp_C_num_peaks",
            "Fermentation_Temp_C_num_valleys",
            "Oven_Temp_C_skewness",
            "Oven_Temp_C_num_valleys",
            "Oven_Humidity_pct_num_peaks",
            "Oven_Humidity_pct_num_valleys",
            "time_steps",
            "rpm_per_kg_flour",
            "Flour_kg_mean_deviation",
            "Sugar_kg_mean_deviation",
            "Sugar_kg_deviation_std",
            "Salt_kg_max_deviation",
            "Mixer_Speed_RPM_deviation_std",
            "Sugar_kg_volatility",
            "Mixer_Speed_RPM_mean_change_rate",
            "Mixer_Speed_RPM_volatility",
            "Fermentation_Temp_C_mean_change_rate"
        ]
        
        # Check which expected features are available
        available_features = []
        missing_features = []
        
        for feature in expected_features:
            if feature in features_df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features for Module 2: {missing_features}")
            logger.warning("Creating dummy values for missing features")
            
            # Create dummy values for missing features
            for feature in missing_features:
                if 'mean' in feature or 'median' in feature or 'std' in feature:
                    features_df[feature] = 0.0
                elif 'num_' in feature or 'steps' in feature:
                    features_df[feature] = 0
                elif 'trend' in feature or 'slope' in feature or 'r2' in feature:
                    features_df[feature] = 0.0
                elif 'pvalue' in feature:
                    features_df[feature] = 1.0
                elif 'deviation' in feature or 'volatility' in feature:
                    features_df[feature] = 0.0
                elif 'rate' in feature:
                    features_df[feature] = 0.0
                else:
                    features_df[feature] = 0.0
        
        # Select only the expected features in the correct order
        selected_features = features_df[expected_features].copy()
        
        logger.info(f"Selected {len(selected_features.columns)} features for Module 2")
        logger.info(f"Feature columns: {list(selected_features.columns)}")
        
        return selected_features
    
    def save_features(self, features_df: pd.DataFrame, filename: str = None):
        """Save engineered features to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'engineered_features_{timestamp}.csv'
        
        filepath = PROCESSED_DATA_DIR / filename
        features_df.to_csv(filepath, index=False)
        logger.info(f"Features saved to {filepath}")
        
        # Save selected features list
        if self.selected_features:
            features_list_file = PROCESSED_DATA_DIR / 'selected_features.txt'
            with open(features_list_file, 'w') as f:
                for feature in self.selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"Selected features list saved to {features_list_file}")
        
        # Save feature importance
        if self.feature_importance:
            importance_file = PROCESSED_DATA_DIR / 'feature_importance.json'
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            logger.info(f"Feature importance saved to {importance_file}")
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary of engineered features."""
        summary = {
            'total_features': len(features_df.columns),
            'numeric_features': len(features_df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(features_df.select_dtypes(include=['object']).columns),
            'missing_values': features_df.isnull().sum().sum(),
            'feature_types': {}
        }
        
        for col in features_df.columns:
            if col in ['Batch_ID', 'Final_Weight', 'Quality_Score']:
                summary['feature_types'][col] = 'target_or_id'
            elif features_df[col].dtype in ['int64', 'float64']:
                summary['feature_types'][col] = 'numeric'
            else:
                summary['feature_types'][col] = 'categorical'
        
        return summary
