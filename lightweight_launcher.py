"""
Lightweight launcher for PDF Chat
This approach uses a small executable that downloads dependencies on first run
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path
import webbrowser
import time
import threading

# Configuration
APP_NAME = "PDFChat"
APP_VERSION = "1.0"
INSTALL_DIR = Path.home() / f".{APP_NAME.lower()}"
PYTHON_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
REPO_URL = "https://github.com/jabriomar873/chatapp/archive/refs/heads/main.zip"

def show_console_message(message):
    """Show message in console with formatting"""
    print("=" * 60)
    print(f"🚀 {APP_NAME} - {message}")
    print("=" * 60)

def check_installation():
    """Check if app is already installed"""
    return (INSTALL_DIR / "app.py").exists()

def download_file(url, destination, description):
    """Download file with progress"""
    print(f"📥 Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded {description}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"❌ Failed to extract: {e}")
        return False

def install_app():
    """Install the application"""
    show_console_message("First Time Setup")
    
    # Create install directory
    INSTALL_DIR.mkdir(exist_ok=True)
    
    # Download portable Python
    python_zip = INSTALL_DIR / "python.zip"
    if not download_file(PYTHON_URL, python_zip, "Portable Python"):
        return False
    
    # Extract Python
    python_dir = INSTALL_DIR / "python"
    if not extract_zip(python_zip, python_dir):
        return False
    
    # Download app source
    app_zip = INSTALL_DIR / "app.zip"
    if not download_file(REPO_URL, app_zip, "PDF Chat App"):
        return False
    
    # Extract app
    if not extract_zip(app_zip, INSTALL_DIR):
        return False
    
    # Move files from extracted folder
    extracted_folder = INSTALL_DIR / "chatapp-main"
    if extracted_folder.exists():
        for item in extracted_folder.iterdir():
            if item.is_file():
                item.rename(INSTALL_DIR / item.name)
            elif item.is_dir():
                # Move directory contents
                target_dir = INSTALL_DIR / item.name
                target_dir.mkdir(exist_ok=True)
                for subitem in item.iterdir():
                    subitem.rename(target_dir / subitem.name)
    
    # Install pip packages
    python_exe = python_dir / "python.exe"
    pip_install_cmd = [
        str(python_exe), "-m", "pip", "install",
        "streamlit", "langchain", "langchain-community", "langchain-ollama",
        "pypdf", "pymupdf", "pytesseract", "pillow", "faiss-cpu",
        "scikit-learn", "numpy", "python-dotenv"
    ]
    
    print("📦 Installing Python packages...")
    try:
        subprocess.run(pip_install_cmd, check=True, capture_output=True)
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False
    
    # Cleanup
    python_zip.unlink(missing_ok=True)
    app_zip.unlink(missing_ok=True)
    if extracted_folder.exists():
        import shutil
        shutil.rmtree(extracted_folder)
    
    print("✅ Installation completed!")
    return True

def start_app():
    """Start the PDF Chat application"""
    python_exe = INSTALL_DIR / "python" / "python.exe"
    app_script = INSTALL_DIR / "app.py"
    
    if not python_exe.exists() or not app_script.exists():
        print("❌ Installation files missing. Please reinstall.")
        return False
    
    # Start Streamlit app
    cmd = [str(python_exe), "-m", "streamlit", "run", str(app_script), "--server.headless", "true"]
    
    try:
        print("🚀 Starting PDF Chat...")
        process = subprocess.Popen(cmd, cwd=str(INSTALL_DIR))
        
        # Wait a moment then open browser
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
        
        print("🌐 PDF Chat is running at: http://localhost:8501")
        print("📝 Close this window to stop the application")
        
        # Wait for process to finish
        process.wait()
        
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    show_console_message("Launcher")
    
    if not check_installation():
        print("🔧 First time setup required...")
        if not install_app():
            print("❌ Installation failed")
            input("Press Enter to exit...")
            return
    else:
        print("✅ Application already installed")
    
    # Check Ollama
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        print("✅ Ollama detected")
    except:
        print("⚠️  Ollama not found!")
        print("📥 Please install Ollama from: https://ollama.ai")
        print("🔧 Then run: ollama pull llama3.2:1b")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Start the app
    start_app()

if __name__ == "__main__":
    main()
