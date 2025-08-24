"""
Test script to verify graph generation is working
"""

import requests
from pathlib import Path
import json

def test_comprehensive_analysis_with_graphs():
    """Test comprehensive analysis with graph generation"""
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Comprehensive Analysis with Graph Generation")
    print("=" * 60)
    
    # Test file
    test_file = Path("data/raw/App_testing_data.csv")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"📁 Testing with file: {test_file.name}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/csv')}
            data = {'use_ensemble': 'true'}
            
            print("   🚀 Sending comprehensive analysis request...")
            response = requests.post(f"{base_url}/api/module2/comprehensive-analysis", 
                                   files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Comprehensive analysis completed successfully!")
                    
                    # Check if graphs were generated
                    report = result.get('report', {})
                    graphs = report.get('graphs', {})
                    
                    print(f"   📊 Graphs generated: {len(graphs)}")
                    for graph_name in graphs.keys():
                        if graph_name != 'error':
                            print(f"      ✅ {graph_name}")
                        else:
                            print(f"      ❌ {graphs[graph_name]}")
                    
                    # Check if graphs contain base64 data
                    has_graph_data = any(
                        isinstance(graphs.get(key), str) and len(graphs.get(key)) > 1000
                        for key in ['process_trends', 'prediction_distribution', 'anomaly_summary']
                    )
                    
                    if has_graph_data:
                        print("   🎨 Graph data (base64) is present!")
                    else:
                        print("   ⚠️ Graph data is missing or incomplete")
                    
                    # Save the result for inspection
                    output_file = "test_comprehensive_analysis_result.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"   💾 Result saved to: {output_file}")
                    
                    return True
                else:
                    print(f"❌ Analysis failed: {result.get('error')}")
                    return False
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_comprehensive_analysis_with_graphs()
