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
    
    # Setup pip for embedded Python
    python_exe = python_dir / "python.exe"
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_file = INSTALL_DIR / "get-pip.py"
    
    print("🔧 Setting up pip...")
    try:
        # Download get-pip.py
        if not download_file(get_pip_url, get_pip_file, "pip installer"):
            return False
        
        # Install pip
        subprocess.run([str(python_exe), str(get_pip_file)], check=True, capture_output=True)
        print("✅ Pip installed successfully")
        
        # Remove get-pip.py
        get_pip_file.unlink(missing_ok=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install pip: {e}")
        return False
    
    # Install packages one by one for better error handling
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
    
    print("📦 Installing Python packages...")
    for package in packages:
        try:
            print(f"  Installing {package}...")
            cmd = [str(python_exe), "-m", "pip", "install", package, "--no-warn-script-location"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            print(f"  Error output: {e.stderr if hasattr(e, 'stderr') else 'No error details'}")
            # Continue with other packages
    
    # Verify critical packages
    print("🔍 Verifying installation...")
    critical_packages = ["streamlit", "langchain"]
    for package in critical_packages:
        try:
            cmd = [str(python_exe), "-c", f"import {package}; print(f'{package} OK')"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ {package} verified")
        except subprocess.CalledProcessError:
            print(f"  ❌ {package} failed verification")
    
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
        time.sleep(3)
        return False
    
    # Check if port 8501 is available
    import socket
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except:
                return False
    
    port = 8501
    if not is_port_available(port):
        print("⚠️  Port 8501 is already in use. Trying port 8502...")
        port = 8502
        if not is_port_available(port):
            print("❌ Ports 8501 and 8502 are busy. Please close other Streamlit apps.")
            time.sleep(3)
            return False
    
    # Start Streamlit app with better configuration
    cmd = [
        str(python_exe), "-m", "streamlit", "run", str(app_script),
        "--server.headless", "true",
        "--server.address", "localhost",
        "--server.port", str(port),
        "--global.developmentMode", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    try:
        print("🚀 Starting PDF Chat...")
        print(f"📡 Using port: {port}")
        
        # Start process with output capture
        process = subprocess.Popen(
            cmd, 
            cwd=str(INSTALL_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        # Wait for Streamlit to start properly
        print("⏳ Waiting for server to start...")
        max_wait = 30  # Wait up to 30 seconds
        for i in range(max_wait):
            try:
                # Check if process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"❌ Process failed to start:")
                    print(f"Error: {stderr[:500]}")
                    time.sleep(3)
                    return False
                
                # Try to connect to the server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    if s.connect_ex(('localhost', port)) == 0:
                        print("✅ Server is ready!")
                        break
                time.sleep(1)
                if i % 5 == 0:
                    print(f"⏳ Still waiting... ({i+1}/{max_wait})")
            except:
                time.sleep(1)
        else:
            print("❌ Server failed to start within 30 seconds")
            process.terminate()
            time.sleep(3)
            return False
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"🌐 Opening browser: {url}")
        webbrowser.open(url)
        
        print("✅ PDF Chat is running!")
        print(f"🌐 Access at: {url}")
        print("🔄 Keep this window open to keep the app running")
        print("❌ Close this window to stop the app")
        
        # Wait for process to finish
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            process.terminate()
            process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        time.sleep(3)
        return False
    
    return True

def main():
    """Main launcher function"""
    try:
        show_console_message("Launcher")
        
        if not check_installation():
            print("🔧 First time setup required...")
            if not install_app():
                print("❌ Installation failed")
                print("⚠️ Setup failed. Please check your internet connection and try again.")
                print("🔧 Troubleshooting:")
                print("   - Check internet connection")
                print("   - Try running as administrator")
                print("   - Temporarily disable antivirus")
                time.sleep(5)
                return
        else:
            print("✅ Application already installed")
        
        # Check Ollama
        ollama_available = False
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, check=True, timeout=5)
            print("✅ Ollama detected")
            ollama_available = True
        except:
            print("⚠️  Ollama not found!")
            print("📥 Please install Ollama from: https://ollama.ai")
            print("🔧 Then run: ollama pull llama3.2:1b")
            print("🚀 Starting app anyway - you'll need Ollama for AI features")
            time.sleep(2)
        
        # Start the app
        if not start_app():
            print("❌ Failed to start the application")
            print("🔧 Troubleshooting:")
            print("   - Make sure no other Streamlit apps are running")
            print("   - Try restarting your computer")
            print("   - Check Windows Firewall settings")
            if not ollama_available:
                print("   - Install Ollama for full functionality")
            time.sleep(5)
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🔧 Please try running as administrator or contact support")
        time.sleep(5)

if __name__ == "__main__":
    main()
