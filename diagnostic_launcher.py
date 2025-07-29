"""
Simple diagnostic launcher for PDF Chat
This version shows detailed output and doesn't hide the console
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import webbrowser
import time
import socket

# Configuration
APP_NAME = "PDFChat"
INSTALL_DIR = Path.home() / f".{APP_NAME.lower()}"

def check_port(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except:
        return False

def test_connection(port, timeout=10):
    """Test if we can connect to the port"""
    for i in range(timeout):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(('localhost', port)) == 0:
                    return True
        except:
            pass
        time.sleep(1)
        print(f"  Attempt {i+1}/{timeout}...")
    return False

def main():
    print("=== PDF Chat Diagnostic Launcher ===")
    print(f"Install directory: {INSTALL_DIR}")
    
    # Check installation
    python_exe = INSTALL_DIR / "python" / "python.exe"
    app_script = INSTALL_DIR / "app.py"
    
    print(f"\nChecking files:")
    print(f"  Python: {python_exe.exists()} - {python_exe}")
    print(f"  App: {app_script.exists()} - {app_script}")
    
    if not python_exe.exists() or not app_script.exists():
        print("❌ Required files missing. Please run the main launcher first.")
        input("Press Enter to exit...")
        return
    
    # Check ports
    print(f"\nChecking ports:")
    port = 8501
    if check_port(port):
        print(f"  Port {port}: Available ✅")
    else:
        print(f"  Port {port}: In use ❌")
        port = 8502
        if check_port(port):
            print(f"  Port {port}: Available ✅")
        else:
            print(f"  Port {port}: In use ❌")
            print("  Both common ports are busy!")
    
    # Check Ollama
    print(f"\nChecking Ollama:")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  Ollama: Available ✅")
            models = result.stdout.strip().split('\n')[1:]
            print(f"  Models: {len(models)} found")
        else:
            print("  Ollama: Not working ❌")
    except subprocess.TimeoutExpired:
        print("  Ollama: Timeout ⏰")
    except FileNotFoundError:
        print("  Ollama: Not installed ❌")
    except Exception as e:
        print(f"  Ollama: Error - {e}")
    
    # Start Streamlit
    cmd = [
        str(python_exe), "-m", "streamlit", "run", str(app_script),
        "--server.headless", "true",
        "--server.address", "localhost",
        "--server.port", str(port)
    ]
    
    print(f"\nStarting Streamlit on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(INSTALL_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("\nStreamlit output:")
        print("-" * 50)
        
        # Show real-time output
        start_time = time.time()
        server_ready = False
        
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
                if "Network URL:" in output or "Local URL:" in output:
                    server_ready = True
                    break
            
            if process.poll() is not None:
                print("Process terminated early")
                break
                
            if time.time() - start_time > 30:
                print("Timeout waiting for server")
                break
        
        if server_ready:
            print(f"\n✅ Server appears to be ready!")
            print(f"🌐 Testing connection to localhost:{port}...")
            
            if test_connection(port):
                print("✅ Connection successful!")
                url = f"http://localhost:{port}"
                print(f"🚀 Opening: {url}")
                webbrowser.open(url)
                
                print("\n✅ App should be running in your browser!")
                print("🔄 Keep this window open")
                print("Press Ctrl+C to stop")
                
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n👋 Stopping...")
                    process.terminate()
            else:
                print("❌ Could not connect to server")
        else:
            print("❌ Server failed to start properly")
            
    except Exception as e:
        print(f"❌ Error starting app: {e}")
    
    print("\nDone. Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()
