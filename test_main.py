#!/usr/bin/env python3
"""
Test script for main.py functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_main_import():
    """Test that main.py can be imported."""
    try:
        import main
        print("✅ main.py imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import main.py: {e}")
        return False

def test_functions():
    """Test that main functions exist."""
    try:
        import main
        
        # Test that required functions exist
        required_functions = [
            'print_banner',
            'check_dependencies', 
            'check_model_files',
            'run_app1',
            'run_app2',
            'run_both_apps',
            'interactive_menu',
            'print_system_status',
            'main'
        ]
        
        for func_name in required_functions:
            if hasattr(main, func_name):
                print(f"✅ Function {func_name} exists")
            else:
                print(f"❌ Function {func_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        return False

def test_dependencies():
    """Test dependency checking."""
    try:
        import main
        result = main.check_dependencies()
        print(f"✅ Dependencies check completed: {result}")
        return result
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

def test_model_files():
    """Test model file checking."""
    try:
        import main
        result = main.check_model_files()
        print(f"✅ Model files check completed: {result}")
        return result
    except Exception as e:
        print(f"❌ Error checking model files: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing main.py functionality...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_main_import),
        ("Functions Test", test_functions),
        ("Dependencies Test", test_dependencies),
        ("Model Files Test", test_model_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! main.py is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
