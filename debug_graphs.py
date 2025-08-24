"""
Debug script to test graph generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def test_graph_generation():
    """Test if graph generation works"""
    print("🧪 Testing Graph Generation")
    print("=" * 40)
    
    try:
        # Create sample data
        print("📊 Creating sample data...")
        process_data = pd.DataFrame({
            'Flour_kg': np.random.normal(50, 2, 25),
            'Sugar_kg': np.random.normal(12, 1, 25),
            'Water_Temp': np.random.normal(25, 3, 25),
            'Mixer_Speed': np.random.normal(120, 10, 25),
            'Mixing_Temp': np.random.normal(38, 2, 25)
        })
        
        predictions = {
            'ensemble_predictions': [
                [45.2, 95.1],
                [44.8, 94.5],
                [45.5, 95.8],
                [44.9, 94.2],
                [45.1, 95.3]
            ]
        }
        
        anomalies = {
            'process_anomalies': [
                {'parameter': 'Water_Temp', 'count': 2, 'percentage': 8.0}
            ],
            'prediction_anomalies': [],
            'quality_warnings': [],
            'critical_issues': []
        }
        
        print("📈 Testing matplotlib and seaborn...")
        
        # Test 1: Basic matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(process_data['Flour_kg'].values)
        plt.title('Test Plot')
        plt.close()
        print("✅ Basic matplotlib works")
        
        # Test 2: Seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(process_data['Flour_kg'])
        plt.title('Test Histogram')
        plt.close()
        print("✅ Seaborn works")
        
        # Test 3: Base64 encoding
        plt.figure(figsize=(10, 6))
        plt.plot(process_data['Flour_kg'].values)
        plt.title('Test Plot for Base64')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        print(f"✅ Base64 encoding works (length: {len(base64_data)})")
        
        # Test 4: Full graph generation
        print("🎨 Testing full graph generation...")
        graphs = {}
        
        # Process Parameters Trend
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Process Parameters Analysis', fontsize=16, fontweight='bold')
        
        # Temperature trends
        temp_cols = [col for col in process_data.columns if 'Temp' in col]
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
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        graphs['process_trends'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        print(f"✅ Process trends graph generated (length: {len(graphs['process_trends'])})")
        
        # Prediction Distribution
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
            
            print(f"✅ Prediction distribution graph generated (length: {len(graphs['prediction_distribution'])})")
        
        # Anomaly Summary
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
            
            print(f"✅ Anomaly summary graph generated (length: {len(graphs['anomaly_summary'])})")
        
        print(f"\n🎉 Success! Generated {len(graphs)} graphs:")
        for graph_name, graph_data in graphs.items():
            print(f"   ✅ {graph_name}: {len(graph_data)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_graph_generation()
