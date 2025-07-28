# test_import.py - Quick test to verify imports work

import sys
import os

print("🧪 Testing imports...")

try:
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Test the problematic import
    from src.visualize_data import visualize_stock_data

    print("✅ visualize_stock_data imported successfully!")

    # Test other imports
    from src.visualize_data import create_visualizations

    print("✅ create_visualizations imported successfully!")

    from src.visualize_data import simple_plot

    print("✅ simple_plot imported successfully!")

    print("\n🎉 All imports successful! Your main.py should work now.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Check that the file exists and has no syntax errors")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()

print("\n📁 Checking file structure...")
files_to_check = [
    "src/visualize_data.py",
    "src/fetch_data.py",
    "src/process_data.py",
    "src/analyze_data.py",
    "main.py"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")

print("\n🔍 Checking visualize_data.py content...")
try:
    with open("src/visualize_data.py", "r") as f:
        content = f.read()
        if "def visualize_stock_data" in content:
            print("✅ visualize_stock_data function found")
        else:
            print("❌ visualize_stock_data function not found")

        if "def create_visualizations" in content:
            print("✅ create_visualizations function found")
        else:
            print("❌ create_visualizations function not found")

except Exception as e:
    print(f"❌ Error reading file: {e}")