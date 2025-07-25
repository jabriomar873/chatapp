#!/usr/bin/env python3
"""
Clean startup script for Enhanced PDF Chat
Suppresses common warnings that don't affect functionality
"""

import os
import sys
import warnings
import subprocess

def setup_environment():
    """Set up environment variables to reduce warnings"""
    # Suppress torch warnings
    os.environ['TORCH_LOGS'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set Python warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")

def main():
    """Start the application with clean output"""
    print("🚀 Starting Enhanced PDF Chat Application...")
    print("📊 Initializing with optimized settings...")
    
    # Setup environment
    setup_environment()
    
    try:
        # Start Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✅ Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
