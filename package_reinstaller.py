"""
Quick fix for missing packages in existing PDF Chat installation
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request

# Configuration
APP_NAME = "PDFChat"
INSTALL_DIR = Path.home() / f".{APP_NAME.lower()}"

def download_file(url, destination):
    """Download file"""
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return False

def main():
    print("=== PDF Chat Package Reinstaller ===")
    print(f"Install directory: {INSTALL_DIR}")
    
    python_exe = INSTALL_DIR / "python" / "python.exe"
    
    if not python_exe.exists():
        print("❌ Python not found. Please run the main installer first.")
        input("Press Enter to exit...")
        return
    
    print("✅ Python found")
    
    # Setup pip
    print("🔧 Setting up pip...")
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_file = INSTALL_DIR / "get-pip.py"
    
    try:
        print("📥 Downloading pip installer...")
        if download_file(get_pip_url, get_pip_file):
            print("🔧 Installing pip...")
            subprocess.run([str(python_exe), str(get_pip_file)], check=True)
            print("✅ Pip ready")
            get_pip_file.unlink(missing_ok=True)
        else:
            print("❌ Failed to setup pip")
            return
    except Exception as e:
        print(f"❌ Pip installation failed: {e}")
    
    # Install packages
    packages = [
        "streamlit",
        "langchain", 
        "langchain-community", 
        "langchain-ollama",
        "pypdf", 
        "pymupdf", 
        "pytesseract", 
        "pillow", 
        "faiss-cpu",
        "scikit-learn", 
        "numpy", 
        "python-dotenv"
    ]
    
    print("📦 Installing packages...")
    success_count = 0
    
    for package in packages:
        try:
            print(f"  📦 Installing {package}...")
            cmd = [str(python_exe), "-m", "pip", "install", package, "--no-warn-script-location", "--upgrade"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ {package}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {package} failed")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"    Error: {e.stderr[:200]}...")
    
    print(f"\n📊 Results: {success_count}/{len(packages)} packages installed")
    
    # Test critical packages
    print("\n🔍 Testing installation...")
    critical_tests = [
        ("streamlit", "import streamlit as st; print('Streamlit OK')"),
        ("langchain", "import langchain; print('LangChain OK')"),
        ("pypdf", "import pypdf; print('PyPDF OK')"),
        ("sklearn", "import sklearn; print('Scikit-learn OK')"),
        ("numpy", "import numpy; print('NumPy OK')")
    ]
    
    working_packages = 0
    for name, test_code in critical_tests:
        try:
            cmd = [str(python_exe), "-c", test_code]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ {name}")
            working_packages += 1
        except subprocess.CalledProcessError:
            print(f"  ❌ {name}")
    
    print(f"\n📊 Working packages: {working_packages}/{len(critical_tests)}")
    
    if working_packages >= 4:
        print("✅ Installation looks good! Try running PDF Chat again.")
    else:
        print("❌ Some packages are still missing. You may need to:")
        print("   - Check your internet connection")
        print("   - Run as administrator")
        print("   - Temporarily disable antivirus")
    
    print("\nDone!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
