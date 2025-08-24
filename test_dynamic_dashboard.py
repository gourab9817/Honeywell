#!/usr/bin/env python3
"""
Test script for dynamic dashboard functionality
"""

import requests
import json
import time

def test_dashboard_endpoints():
    """Test the new dashboard API endpoints."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Dynamic Dashboard Endpoints")
    print("=" * 50)
    
    # Test 1: Dashboard Metrics
    print("\n1. Testing Dashboard Metrics...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/metrics")
        if response.status_code == 200:
            data = response.json()
            print("✅ Dashboard metrics endpoint working")
            print(f"   Quality Score: {data['metrics']['quality_score']}")
            print(f"   Predicted Weight: {data['metrics']['predicted_weight']}")
            print(f"   Anomaly Risk: {data['metrics']['anomaly_risk']}")
        else:
            print(f"❌ Dashboard metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing dashboard metrics: {e}")
    
    # Test 2: Quality Gauge Data
    print("\n2. Testing Quality Gauge Data...")
    try:
        response = requests.get(f"{base_url}/api/dashboard/quality-gauge")
        if response.status_code == 200:
            data = response.json()
            print("✅ Quality gauge endpoint working")
            print(f"   Quality Score: {data['quality_score']}")
            print(f"   Gauge Data: {data['gauge_data']}")
        else:
            print(f"❌ Quality gauge failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing quality gauge: {e}")
    
    # Test 3: Process Parameters
    print("\n3. Testing Process Parameters...")
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
    
    # Test 4: Batch Summary
    print("\n4. Testing Batch Summary...")
    try:
        response = requests.get(f"{base_url}/api/batch-summary")
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch summary endpoint working")
            print(f"   Total Batches: {data['total_batches']}")
            print(f"   Pass Rate: {data['pass_rate']}%")
            print(f"   Recent Batches: {len(data['batches'])}")
        else:
            print(f"❌ Batch summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing batch summary: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Dynamic Dashboard Test Complete!")
    print("\n📋 Next Steps:")
    print("1. Upload data through the home page")
    print("2. Train models to generate real data")
    print("3. Check dashboard to see dynamic updates")
    print("4. Data will persist between sessions")

if __name__ == "__main__":
    test_dashboard_endpoints()
