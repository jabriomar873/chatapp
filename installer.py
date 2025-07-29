"""
Smart PDF Chat Installer
Downloads and sets up everything automatically
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import webbrowser
import time
import threading

APP_NAME = "PDFChat"
INSTALL_DIR = Path.home() / ".pdfchat"
GITHUB_REPO = "jabriomar873/chatapp"

def print_banner():
    print("=" * 60)
    print("🚀 PDF Chat - Smart Installer")
    print("   AI-Powered Document Q&A")
    print("=" * 60)

def check_python():
    """Check if Python is available"""
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python found: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("❌ Python not found")
    return False

def download_with_progress(url, destination, description):
    """Download file with simple progress"""
    print(f"📥 Downloading {description}...")
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if percent % 10 == 0:  # Show every 10%
                    print(f"   {percent}% complete")
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"✅ Downloaded {description}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def install_packages():
    """Install required Python packages"""
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
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def download_app():
    """Download app from GitHub"""
    app_url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"
    app_zip = INSTALL_DIR / "app.zip"
    
    if not download_with_progress(app_url, app_zip, "PDF Chat App"):
        return False
    
    # Extract app
    print("📂 Extracting application...")
    try:
        with zipfile.ZipFile(app_zip, 'r') as zip_ref:
            zip_ref.extractall(INSTALL_DIR)
        
        # Move files from extracted folder
        extracted_folder = INSTALL_DIR / "chatapp-main"
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(INSTALL_DIR / item.name))
            shutil.rmtree(extracted_folder)
        
        app_zip.unlink()
        print("✅ Application extracted")
        return True
    except Exception as e:
        print(f"❌ Failed to extract: {e}")
        return False

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama detected")
            return True
    except:
        pass
    
    print("⚠️  Ollama not found")
    print("📥 Please install Ollama from: https://ollama.ai")
    print("🔧 Then run: ollama pull llama3.2:1b")
    return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows)"""
    try:
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "PDF Chat.bat"
        
        launcher_content = f'''@echo off
cd /d "{INSTALL_DIR}"
python -m streamlit run app.py --server.headless true
pause'''
        
        with open(shortcut_path, 'w') as f:
            f.write(launcher_content)
        
        print(f"✅ Desktop shortcut created: {shortcut_path}")
        return True
    except:
        print("⚠️  Could not create desktop shortcut")
        return False

def start_app():
    """Start the application"""
    app_file = INSTALL_DIR / "app.py"
    if not app_file.exists():
        print("❌ App file not found")
        return False
    
    print("🚀 Starting PDF Chat...")
    
    # Start browser opening in background
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.headless", "true"]
        subprocess.run(cmd, cwd=str(INSTALL_DIR))
    except KeyboardInterrupt:
        print("\n👋 PDF Chat stopped")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        return False
    
    return True

def main():
    """Main installer function"""
    print_banner()
    
    # Check if already installed
    if (INSTALL_DIR / "app.py").exists():
        print("✅ PDF Chat already installed")
        response = input("🔄 Update to latest version? (y/N): ")
        if response.lower() != 'y':
            if input("🚀 Start PDF Chat now? (Y/n): ").lower() != 'n':
                start_app()
            return
    
    # Create install directory
    INSTALL_DIR.mkdir(exist_ok=True)
    
    # Check Python
    if not check_python():
        print("💡 Please install Python from: https://python.org")
        input("Press Enter to exit...")
        return
    
    # Install packages
    if not install_packages():
        input("Press Enter to exit...")
        return
    
    # Download app
    if not download_app():
        input("Press Enter to exit...")
        return
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Create shortcut
    create_desktop_shortcut()
    
    print("\n🎉 Installation completed!")
    print(f"📁 Installed to: {INSTALL_DIR}")
    
    if not ollama_ok:
        print("\n⚠️  Please install Ollama before using:")
        print("1. Visit: https://ollama.ai")
        print("2. Run: ollama pull llama3.2:1b")
        
        if input("\n📖 Open Ollama website? (Y/n): ").lower() != 'n':
            webbrowser.open("https://ollama.ai")
        
        input("\nPress Enter after installing Ollama...")
    
    # Start the app
    if input("\n🚀 Start PDF Chat now? (Y/n): ").lower() != 'n':
        start_app()
    else:
        print("\n💡 You can start PDF Chat anytime using the desktop shortcut!")

if __name__ == "__main__":
    main()
