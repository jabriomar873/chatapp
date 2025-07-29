# 📦 PDF Chat - Standalone Executable

## 🎯 Create Distributable .exe File

This project can be packaged as a standalone executable (.exe) that users can run without installing Python or dependencies.

## 🚀 Build Instructions

### Option 1: Quick Build (Recommended)
```bash
# Run the automated build script
build.bat
```

### Option 2: Manual Build
```bash
# Install PyInstaller
pip install pyinstaller

# Run the build script
python build_exe.py
```

## 📁 Distribution

After building, you'll get:
- `dist/PDFChat.exe` - Standalone executable
- `DISTRIBUTION.md` - Instructions for end users

## 📋 End User Requirements

Users only need to install:
1. **Ollama** (AI Engine) - https://ollama.ai
2. **AI Model**: `ollama pull llama3.2:1b`
3. **Tesseract OCR** (optional, for scanned PDFs)

## 🎯 Alternative Distribution Methods

### Method 1: Portable App
- Package as a portable application
- Include Ollama portable version
- Single-click installation

### Method 2: Installer Package
- Create MSI installer with dependencies
- Automated Ollama installation
- Desktop shortcuts and file associations

### Method 3: Docker Container
- Containerized application
- Cross-platform compatibility
- Easy deployment

## 🔧 Build Configuration

The build process:
- ✅ Bundles all Python dependencies
- ✅ Excludes torch/tensorflow (torch-free design)
- ✅ Includes OCR libraries
- ✅ Creates single executable file
- ✅ Windows-optimized

## 📊 Expected File Sizes
- **Executable**: ~150-200 MB
- **Distribution**: Complete standalone app
- **Requirements**: Ollama (separate download)

## 🚀 Usage for End Users

1. Download `PDFChat.exe`
2. Install Ollama from https://ollama.ai
3. Run: `ollama pull llama3.2:1b`
4. Double-click `PDFChat.exe`
5. App opens in browser automatically!
