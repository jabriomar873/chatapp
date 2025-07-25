#!/usr/bin/env python3
"""
Script to fix NumPy 2.x compatibility issues
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Fix NumPy compatibility issues"""
    print("🔧 Fixing NumPy 2.x compatibility issues...")
    
    # Step 1: Uninstall problematic packages
    print("\n📦 Uninstalling packages with NumPy conflicts...")
    packages_to_uninstall = [
        "numpy",
        "matplotlib", 
        "tensorflow",
        "tf-keras",
        "torch",
        "torchvision",
        "torchaudio"
    ]
    
    for package in packages_to_uninstall:
        run_command(f"pip uninstall -y {package}")
    
    # Step 2: Install compatible NumPy first
    print("\n🔢 Installing compatible NumPy version...")
    run_command("pip install 'numpy<2.0.0,>=1.24.0'")
    
    # Step 3: Reinstall other packages
    print("\n📦 Reinstalling compatible packages...")
    compatible_packages = [
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0"
    ]
    
    for package in compatible_packages:
        run_command(f"pip install '{package}'")
    
    # Step 4: Install requirements
    print("\n📋 Installing updated requirements...")
    if os.path.exists("requirements.txt"):
        run_command("pip install -r requirements.txt")
    
    print("\n✅ NumPy compatibility fix completed!")
    print("🔄 Please restart your application.")

if __name__ == "__main__":
    main()
