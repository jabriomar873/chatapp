"""
Advanced package installer for embedded Python environments
This version handles the specific requirements of embedded Python distributions
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile

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

def setup_embedded_python():
    """Setup embedded Python to work with pip properly"""
    python_dir = INSTALL_DIR / "python"
    python_exe = python_dir / "python.exe"
    
    # Create pth file to enable site-packages
    pth_file = python_dir / "python311._pth"
    if pth_file.exists():
        print("🔧 Configuring embedded Python...")
        content = pth_file.read_text()
        if "import site" not in content:
            # Add site import to enable pip functionality
            content += "\nimport site\n"
            pth_file.write_text(content)
            print("✅ Python configuration updated")
    
    # Ensure Scripts directory exists
    scripts_dir = python_dir / "Scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    return python_exe

def install_packages_direct():
    """Install packages using direct wheel downloads for embedded Python"""
    python_exe = setup_embedded_python()
    
    if not python_exe.exists():
        print("❌ Python executable not found")
        return False
    
    # Test basic Python functionality
    try:
        result = subprocess.run([str(python_exe), "-c", "print('Python OK')"], 
                              capture_output=True, text=True, check=True)
        print("✅ Python executable working")
    except Exception as e:
        print(f"❌ Python executable test failed: {e}")
        return False
    
    # Install pip using ensurepip (built into Python)
    print("🔧 Installing pip using ensurepip...")
    try:
        result = subprocess.run([str(python_exe), "-m", "ensurepip", "--upgrade"], 
                              capture_output=True, text=True, check=True)
        print("✅ Pip installed via ensurepip")
    except Exception as e:
        print(f"⚠️ ensurepip failed, trying alternative method...")
        
        # Alternative: download and install pip manually
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_file = INSTALL_DIR / "get-pip.py"
        
        if download_file(get_pip_url, get_pip_file):
            try:
                result = subprocess.run([str(python_exe), str(get_pip_file)], 
                                      capture_output=True, text=True, check=True)
                print("✅ Pip installed via get-pip.py")
                get_pip_file.unlink(missing_ok=True)
            except Exception as e2:
                print(f"❌ get-pip.py also failed: {e2}")
                return False
        else:
            print("❌ Could not download get-pip.py")
            return False
    
    # Test pip installation
    try:
        result = subprocess.run([str(python_exe), "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Pip version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Pip test failed: {e}")
        return False
    
    # Install packages with specific flags for embedded Python
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
    
    print("📦 Installing packages with embedded Python compatibility...")
    success_count = 0
    
    for package in packages:
        try:
            print(f"  📦 Installing {package}...")
            cmd = [
                str(python_exe), "-m", "pip", "install", package,
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--user"  # Install to user directory for embedded Python
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  check=True, timeout=300)
            print(f"  ✅ {package}")
            success_count += 1
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {package} - timeout")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {package} failed")
            if e.stderr:
                # Show only first line of error for brevity
                error_line = e.stderr.strip().split('\n')[0]
                print(f"    Error: {error_line[:100]}...")
    
    print(f"\n📊 Results: {success_count}/{len(packages)} packages installed")
    
    # Test critical packages with user site-packages
    print("\n🔍 Testing installation...")
    critical_tests = [
        ("streamlit", "import streamlit; print('Streamlit:', streamlit.__version__)"),
        ("langchain", "import langchain; print('LangChain: OK')"),
        ("pypdf", "import pypdf; print('PyPDF: OK')"),
        ("sklearn", "import sklearn; print('Scikit-learn:', sklearn.__version__)"),
        ("numpy", "import numpy; print('NumPy:', numpy.__version__)")
    ]
    
    working_packages = 0
    for name, test_code in critical_tests:
        try:
            # Set PYTHONPATH to include user site-packages
            env = os.environ.copy()
            env['PYTHONPATH'] = str(python_exe.parent)
            
            cmd = [str(python_exe), "-c", test_code]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  check=True, env=env, timeout=10)
            print(f"  ✅ {name}: {result.stdout.strip()}")
            working_packages += 1
        except Exception as e:
            print(f"  ❌ {name}")
    
    print(f"\n📊 Working packages: {working_packages}/{len(critical_tests)}")
    
    if working_packages >= 3:
        print("✅ Installation successful! Core packages are working.")
        return True
    else:
        print("❌ Installation incomplete. Some critical packages missing.")
        return False

def main():
    print("=== PDF Chat Advanced Package Installer ===")
    print(f"Install directory: {INSTALL_DIR}")
    
    if not INSTALL_DIR.exists():
        print("❌ PDF Chat not installed. Please run the main launcher first.")
        input("Press Enter to exit...")
        return
    
    print("✅ PDF Chat directory found")
    
    if install_packages_direct():
        print("\n🎉 Installation completed successfully!")
        print("🚀 You can now try running PDF Chat again.")
    else:
        print("\n❌ Installation failed.")
        print("💡 Troubleshooting suggestions:")
        print("   - Try running as administrator")
        print("   - Check your internet connection")
        print("   - Temporarily disable antivirus")
        print("   - Try deleting .pdfchat folder and starting fresh")
    
    print("\nDone!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
