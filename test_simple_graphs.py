"""
Simple test to verify the graphs page works
"""

import json
from pathlib import Path

def test_graphs_page():
    """Test if the graphs page can load graph data"""
    print("🧪 Testing Graphs Page")
    print("=" * 40)
    
    # Check if we have any comprehensive analysis results
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("❌ Reports directory not found")
        return False
    
    # Look for comprehensive analysis files
    comprehensive_files = list(reports_dir.glob("comprehensive_analysis_*.json"))
    
    if not comprehensive_files:
        print("❌ No comprehensive analysis files found")
        print("   Please run a comprehensive analysis first")
        return False
    
    # Use the most recent file
    latest_file = max(comprehensive_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Found analysis file: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Check if graphs are present
        report = data.get('report', {})
        graphs = report.get('graphs', {})
        
        print(f"📊 Graphs found: {len(graphs)}")
        
        for graph_name in graphs.keys():
            if graph_name != 'error':
                graph_data = graphs[graph_name]
                if isinstance(graph_data, str) and len(graph_data) > 1000:
                    print(f"   ✅ {graph_name}: {len(graph_data)} characters (base64 data)")
                else:
                    print(f"   ⚠️ {graph_name}: Invalid or missing data")
            else:
                print(f"   ❌ {graph_name}: {graphs[graph_name]}")
        
        # Check if we have valid graph data
        valid_graphs = [
            name for name, data in graphs.items() 
            if name != 'error' and isinstance(data, str) and len(data) > 1000
        ]
        
        if valid_graphs:
            print(f"\n✅ Found {len(valid_graphs)} valid graphs!")
            print("   You can now view these graphs in the web interface.")
            print("   1. Go to Module 2")
            print("   2. Run a comprehensive analysis")
            print("   3. Click 'View Graphs' button")
            print("   4. Or navigate to /graphs page directly")
            return True
        else:
            print("\n❌ No valid graph data found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return False

if __name__ == "__main__":
    test_graphs_page()
