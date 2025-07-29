"""
Launcher script for standalone PDF Chat executable
This script ensures proper Streamlit startup and browser opening
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def start_ollama():
    """Start Ollama service if needed"""
    try:
        subprocess.Popen(['ollama', 'serve'], 
                        creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(3)  # Give Ollama time to start
    except:
        pass

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)  # Wait for Streamlit to start
    webbrowser.open('http://localhost:8501')

def main():
    """Main launcher function"""
    print("🚀 Starting PDF Chat...")
    
    # Check Ollama installation
    if not check_ollama():
        print("⚠️  Ollama not found. Please install from: https://ollama.ai")
        print("📥 After installation, run: ollama pull llama3.2:1b")
        input("Press Enter after installing Ollama...")
    
    # Start Ollama service
    start_ollama()
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Start Streamlit
    try:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", "app.py", "--server.headless", "true"]
        stcli.main()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Please check if all dependencies are properly installed")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
