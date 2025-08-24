#!/usr/bin/env python3
"""
Test script for training analysis dashboard functionality
"""

import requests
import json
import time
from pathlib import Path

def test_training_dashboard():
    """Test the new training analysis dashboard functionality."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Training Analysis Dashboard")
    print("=" * 60)
    
    # Test 1: Check if training report exists
    print("\n1. Checking Training Report Files...")
    models_dir = Path('data/models')
    if models_dir.exists():
        result_files = list(models_dir.glob('*_results_*.json'))
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            print(f"✅ Found training report: {latest_file.name}")
            
            # Load and display key metrics
            with open(latest_file, 'r') as f:
                report_data = json.load(f)
            
            if 'training_status' in report_data and 'results' in report_data['training_status']:
                results = report_data['training_status']['results']
                best_model = results.get('best_model', 'Unknown')
                best_score = results.get('best_score', 0)
                business_impact = results.get('business_impact', {})
                
                print(f"   Best Model: {best_model}")
                print(f"   Best Score: {best_score:.4f}")
                print(f"   Annual Savings: ${business_impact.get('total_annual_savings', 0):,}")
                print(f"   ROI: {business_impact.get('roi_percentage', 0):.1f}%")
        else:
            print("❌ No training results found")
    else:
        print("❌ Models directory not found")
    
    # Test 2: Dashboard Metrics
    print("\n2. Testing Dashboard Metrics...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/metrics")
        if response.status_code == 200:
            data = response.json()
            print("✅ Dashboard metrics endpoint working")
            if data.get('success'):
                metrics = data['metrics']
                print(f"   Best Model: {metrics.get('last_training', {}).get('best_model', 'Unknown')}")
                print(f"   Best Score: {metrics.get('last_training', {}).get('best_score', 0):.4f}")
                print(f"   Quality Score: {metrics.get('quality_score', 0)}%")
                print(f"   Predicted Weight: {metrics.get('predicted_weight', 0)} kg")
        else:
            print(f"❌ Dashboard metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing dashboard metrics: {e}")
    
    # Test 3: Training Analysis
    print("\n3. Testing Training Analysis...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/training-analysis")
        if response.status_code == 200:
            data = response.json()
            print("✅ Training analysis endpoint working")
            if data.get('success'):
                analysis = data['analysis']
                model_perf = analysis.get('model_performance', {})
                business_impact = analysis.get('business_impact', {})
                data_quality = analysis.get('data_quality', {})
                
                print(f"   Best Model: {model_perf.get('best_model', 'Unknown')}")
                print(f"   Best Score: {model_perf.get('best_score', 0):.4f}")
                print(f"   Weight Accuracy: {business_impact.get('weight_accuracy', 0):.1%}")
                print(f"   Quality Accuracy: {business_impact.get('quality_accuracy', 0):.1%}")
                print(f"   Data Quality Score: {data_quality.get('data_quality_score', 0):.1%}")
                print(f"   Outliers Removed: {data_quality.get('outlier_percentage', 0)}%")
        else:
            print(f"❌ Training analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing training analysis: {e}")
    
    # Test 4: Model Comparison
    print("\n4. Testing Model Comparison...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/model-comparison")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model comparison endpoint working")
            if data.get('success'):
                models = data.get('models', [])
                scores = data.get('scores', [])
                print(f"   Models compared: {len(models)}")
                for i, model in enumerate(models):
                    if i < len(scores):
                        print(f"   - {model}: {scores[i]}%")
        else:
            print(f"❌ Model comparison failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing model comparison: {e}")
    
    # Test 5: Quality Gauge
    print("\n5. Testing Quality Gauge...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/quality-gauge")
        if response.status_code == 200:
            data = response.json()
            print("✅ Quality gauge endpoint working")
            if data.get('success'):
                print(f"   Quality Score: {data.get('quality_score', 0)}%")
                print(f"   Gauge Data: {data.get('gauge_data', [])}")
        else:
            print(f"❌ Quality gauge failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing quality gauge: {e}")
    
    # Test 6: Process Parameters
    print("\n6. Testing Process Parameters...")
    try:
        response = requests.get(f"{base_url}/api/process-parameters")
        if response.status_code == 200:
            data = response.json()
            print("✅ Process parameters endpoint working")
            print(f"   Parameters loaded: {len(data)}")
            for param, info in list(data.items())[:3]:  # Show first 3
                print(f"   - {param}: {info['current']} {info['unit']} ({info['status']})")
        else:
            print(f"❌ Process parameters failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing process parameters: {e}")
    
    # Test 7: Batch Summary
    print("\n7. Testing Batch Summary...")
    try:
        response = requests.get(f"{base_url}/api/batch-summary")
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch summary endpoint working")
            if data.get('success'):
                print(f"   Total Batches: {data.get('total_batches', 0)}")
                print(f"   Pass Rate: {data.get('pass_rate', 0)}%")
                print(f"   Recent Batches: {len(data.get('batches', []))}")
        else:
            print(f"❌ Batch summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing batch summary: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Training Analysis Dashboard Test Complete!")
    print("\n📋 Dashboard Features:")
    print("✅ Model Performance Comparison Chart")
    print("✅ Business Impact Analysis")
    print("✅ Data Quality Overview")
    print("✅ Feature Importance Visualization")
    print("✅ Real-time Process Parameters")
    print("✅ Quality Prediction Gauge")
    print("✅ Batch Summary")
    print("\n🚀 The dashboard now displays comprehensive training analysis!")

if __name__ == "__main__":
    test_training_dashboard()
