"""
Simple embedded Python package installer
Uses a different approach that works better with embedded Python
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request

# Configuration
INSTALL_DIR = Path.home() / ".pdfchat"

def download_and_run_installer():
    """Download Python packages as wheels and install them directly"""
    python_exe = INSTALL_DIR / "python" / "python.exe"
    
    if not python_exe.exists():
        print("❌ Python not found. Run the main launcher first.")
        return False
    
    print("🔧 Preparing embedded Python environment...")
    
    # First, configure embedded Python to allow pip
    python_dir = python_exe.parent
    pth_file = python_dir / "python311._pth"
    
    if pth_file.exists():
        content = pth_file.read_text()
        if "import site" not in content:
            content += "\nimport site\n"
            pth_file.write_text(content)
            print("✅ Enabled site-packages")
    
    # Try using ensurepip first (built into Python)
    print("🔧 Installing pip...")
    try:
        subprocess.run([str(python_exe), "-m", "ensurepip", "--upgrade"], 
                      check=True, capture_output=True)
        print("✅ Pip installed")
    except:
        print("⚠️ Using alternative pip installation...")
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_file = INSTALL_DIR / "get-pip.py"
        
        try:
            urllib.request.urlretrieve(get_pip_url, get_pip_file)
            subprocess.run([str(python_exe), str(get_pip_file)], 
                          check=True, capture_output=True)
            get_pip_file.unlink(missing_ok=True)
            print("✅ Pip installed via get-pip")
        except Exception as e:
            print(f"❌ Failed to install pip: {e}")
            return False
    
    # Install packages using pip with flags that work better with embedded Python
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
    success = 0
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        try:
            # Use specific flags for embedded Python
            cmd = [
                str(python_exe), "-m", "pip", "install", pkg,
                "--no-warn-script-location",
                "--no-cache-dir"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  check=True, timeout=180)
            print(f"  ✅ {pkg}")
            success += 1
        except Exception as e:
            print(f"  ❌ {pkg} failed")
    
    print(f"\n📊 Installed: {success}/{len(packages)}")
    
    # Test installation
    print("\n🔍 Testing...")
    tests = [
        "import streamlit; print('Streamlit OK')",
        "import langchain; print('LangChain OK')", 
        "import numpy; print('NumPy OK')"
    ]
    
    working = 0
    for test in tests:
        try:
            subprocess.run([str(python_exe), "-c", test], 
                          check=True, capture_output=True, timeout=10)
            working += 1
        except:
            pass
    
    print(f"✅ {working}/{len(tests)} core packages working")
    
    if working >= 2:
        print("🎉 Installation successful!")
        return True
    else:
        print("❌ Installation incomplete")
        return False

def main():
    print("=== Simple PDF Chat Package Installer ===")
    
    if download_and_run_installer():
        print("\n✅ Ready! Try running PDF Chat now.")
    else:
        print("\n❌ Installation failed.")
        print("💡 Try:")
        print("   - Run as administrator") 
        print("   - Check internet connection")
        print("   - Delete .pdfchat folder and start fresh")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
