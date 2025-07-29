#!/usr/bin/env python3
"""
Build script to create standalone executable for PDF Chat App
Run this script to create a distributable .exe file
"""

import os
import subprocess
import sys
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed successfully")

def create_spec_file():
    """Create PyInstaller spec file for better control"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('htmlTemplates.py', '.'),
        ('.env', '.') if os.path.exists('.env') else None,
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.web.cli',
        'streamlit.runtime.scriptrunner.script_runner',
        'click',
        'altair',
        'pandas',
        'numpy',
        'scikit-learn',
        'faiss-cpu',
        'langchain',
        'langchain_community',
        'langchain_ollama',
        'pypdf',
        'fitz',
        'pytesseract',
        'PIL',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'tensorflow',
        'transformers',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDFChat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' if os.path.exists('app_icon.ico') else None,
)
'''
    
    with open('PDFChat.spec', 'w') as f:
        f.write(spec_content)
    print("✅ Created PDFChat.spec file")

def build_executable():
    """Build the standalone executable"""
    print("🚀 Building standalone executable...")
    
    # Build using spec file
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "PDFChat.spec"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ Build completed successfully!")
        print("📁 Executable location: dist/PDFChat.exe")
        print("📦 Distribution folder: dist/")
        
        # Create instructions
        create_distribution_instructions()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("💡 Try installing missing dependencies or check the build log")

def create_distribution_instructions():
    """Create instructions for distributing the app"""
    instructions = '''# 📦 PDF Chat - Standalone Distribution

## 📁 Files to Distribute

Copy the entire `dist/` folder to share your application:

```
📁 dist/
└── PDFChat.exe          # Main executable (click to run)
```

## 🚀 Installation Instructions for End Users

### Requirements for Users:
1. **Ollama** (AI Engine) - Download from: https://ollama.ai
2. **Tesseract OCR** (for scanned PDFs) - Auto-installs on Windows

### Setup Steps:
1. Download and extract the `dist` folder
2. Install Ollama: https://ollama.ai
3. Install an AI model: `ollama pull llama3.2:1b`
4. Double-click `PDFChat.exe` to run

### Usage:
- The app will open in your default web browser
- Upload PDF files and start chatting!

## 📝 Notes:
- No Python installation required for end users
- All dependencies are bundled in the executable
- Private and secure - everything runs locally
'''

    with open('DISTRIBUTION.md', 'w') as f:
        f.write(instructions)
    print("✅ Created DISTRIBUTION.md with user instructions")

def main():
    """Main build process"""
    print("🔨 PDF Chat - Executable Builder")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory")
        print("💡 Please run this script from the project root directory")
        return
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    build_executable()
    
    print("\n🎉 Build process completed!")
    print("📁 Check the 'dist' folder for your executable")
    print("📖 Read DISTRIBUTION.md for sharing instructions")

if __name__ == "__main__":
    main()
