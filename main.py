#!/usr/bin/env python3
"""
Main Application Launcher for F&B Process Anomaly Detection System
Honeywell Hackathon Solution

This script provides a unified interface to run both the original app.py
and the new dual-module app_v2.py applications.

Usage:
    python main.py                    # Run with interactive menu
    python main.py --app1             # Run original app.py only
    python main.py --app2             # Run app_v2.py only (dual-module)
    python main.py --both             # Run both apps on different ports
    python main.py --help             # Show help
"""

import os
import sys
import argparse
import subprocess
import threading
import time
import signal
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

def print_banner():
    """Print application banner."""
    print("=" * 80)
    print("🚀 F&B Process Anomaly Detection System - Main Launcher")
    print("=" * 80)
    print("Honeywell Hackathon Solution")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import pandas
        import numpy
        import joblib
        import matplotlib
        import seaborn
        import xgboost
        import sklearn
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if required model files exist."""
    model_dir = Path("data/model_module2")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.joblib"))
        if model_files:
            print(f"✅ Found {len(model_files)} pre-trained models in {model_dir}")
            return True
        else:
            print(f"⚠️  No model files found in {model_dir}")
            return False
    else:
        print(f"⚠️  Model directory {model_dir} not found")
        return False

def run_app1():
    """Run the original app.py (Module 1: Train-on-demand)."""
    print("\n🚀 Starting Original Application (app.py)...")
    print("📊 Module 1: Train-on-demand functionality")
    print("🌐 Available at: http://localhost:5000")
    print("=" * 50)
    
    try:
        # Change to app directory and run app.py
        os.chdir("app")
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Original application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running original application: {e}")
    finally:
        os.chdir("..")

def run_app2():
    """Run the new app_v2.py (Dual-module architecture)."""
    print("\n🚀 Starting Dual-Module Application (app_v2.py)...")
    print("📊 Module 1: Train-on-demand functionality")
    print("📊 Module 2: Pre-trained prediction functionality")
    print("🌐 Available at: http://localhost:5001")
    print("=" * 50)
    
    try:
        # Change to app directory and run app_v2.py
        os.chdir("app")
        subprocess.run([sys.executable, "app_v2.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dual-module application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dual-module application: {e}")
    finally:
        os.chdir("..")

def run_both_apps():
    """Run both applications on different ports."""
    print("\n🚀 Starting Both Applications...")
    print("📊 Original App (app.py): http://localhost:5000")
    print("📊 Dual-Module App (app_v2.py): http://localhost:5001")
    print("=" * 50)
    
    processes = []
    
    try:
        # Start app.py on port 5000
        os.chdir("app")
        process1 = subprocess.Popen([sys.executable, "app.py"])
        processes.append(process1)
        print("✅ Original app started on port 5000")
        
        # Wait a moment for the first app to start
        time.sleep(3)
        
        # Start app_v2.py on port 5001 using environment variable
        env = os.environ.copy()
        env['PORT'] = '5001'
        process2 = subprocess.Popen([sys.executable, "app_v2.py"], env=env)
        processes.append(process2)
        print("✅ Dual-module app started on port 5001")
        
        print("\n🌐 Both applications are running:")
        print("   • Original App: http://localhost:5000")
        print("   • Dual-Module App: http://localhost:5001")
        print("\n⏹️  Press Ctrl+C to stop both applications")
        
        # Wait for both processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping both applications...")
        for process in processes:
            process.terminate()
        print("✅ Both applications stopped")
    except Exception as e:
        print(f"❌ Error running applications: {e}")
    finally:
        os.chdir("..")

def interactive_menu():
    """Show interactive menu for application selection."""
    while True:
        print("\n" + "=" * 50)
        print("🎯 Choose Application to Run:")
        print("=" * 50)
        print("1. Original App (app.py) - Module 1: Train-on-demand")
        print("2. Dual-Module App (app_v2.py) - Both modules")
        print("3. Run Both Apps (different ports)")
        print("4. Check System Status")
        print("5. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            run_app1()
        elif choice == "2":
            run_app2()
        elif choice == "3":
            run_both_apps()
        elif choice == "4":
            print_system_status()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

def print_system_status():
    """Print system status and information."""
    print("\n" + "=" * 50)
    print("📊 System Status")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model files
    models_ok = check_model_files()
    
    # Check data directories
    data_dir = Path("data")
    if data_dir.exists():
        print(f"✅ Data directory exists: {data_dir}")
    else:
        print(f"❌ Data directory missing: {data_dir}")
    
    # Check app files
    app1_exists = Path("app/app.py").exists()
    app2_exists = Path("app/app_v2.py").exists()
    
    print(f"{'✅' if app1_exists else '❌'} Original app: app/app.py")
    print(f"{'✅' if app2_exists else '❌'} Dual-module app: app/app_v2.py")
    
    # Check templates
    templates_dir = Path("app/templates")
    if templates_dir.exists():
        template_files = list(templates_dir.glob("*.html"))
        print(f"✅ Templates directory: {len(template_files)} HTML files")
    else:
        print(f"❌ Templates directory missing: {templates_dir}")
    
    print("\n📋 Application Features:")
    print("   Original App (app.py):")
    print("   • Real-time process monitoring")
    print("   • Data upload and processing")
    print("   • Model training and evaluation")
    print("   • Quality prediction API")
    print("   • Anomaly detection alerts")
    print("   • Comprehensive reporting")
    
    print("\n   Dual-Module App (app_v2.py):")
    print("   • Module 1: Train-on-demand (same as original)")
    print("   • Module 2: Pre-trained prediction")
    print("   • Instant anomaly prediction")
    print("   • Multiple ML models (XGBoost, Random Forest, Neural Network)")
    print("   • Comprehensive analysis with graphs")
    print("   • JSON report generation")
    
    print("\n" + "=" * 50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="F&B Process Anomaly Detection System - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive menu
  python main.py --app1             # Run original app only
  python main.py --app2             # Run dual-module app only
  python main.py --both             # Run both apps
  python main.py --status           # Check system status
        """
    )
    
    parser.add_argument("--app1", action="store_true", 
                       help="Run original app.py (Module 1: Train-on-demand)")
    parser.add_argument("--app2", action="store_true", 
                       help="Run app_v2.py (Dual-module architecture)")
    parser.add_argument("--both", action="store_true", 
                       help="Run both applications on different ports")
    parser.add_argument("--status", action="store_true", 
                       help="Check system status and dependencies")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check system status if requested
    if args.status:
        print_system_status()
        return
    
    # Run based on arguments
    if args.app1:
        run_app1()
    elif args.app2:
        run_app2()
    elif args.both:
        run_both_apps()
    else:
        # No arguments provided, show interactive menu
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
