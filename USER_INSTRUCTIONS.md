# 📋 PDF Chat - User Instructions

## 🚀 Quick Start

### Option 1: Download Launcher (Recommended)
1. **Download** `PDFChat-Launcher.exe` from GitHub
2. **Run** the launcher (Windows may show security warning - click "More info" → "Run anyway")
3. **First run**: App will automatically install (takes 2-3 minutes)
4. **Install Ollama**: Download from https://ollama.ai
5. **Install AI model**: Open command prompt and run: `ollama pull llama3.2:1b`
6. **Done!** App will open in your browser

### Option 2: Portable Version
1. **Download** the full project folder from GitHub
2. **Run** `portable_app.bat`
3. **Install Ollama** and AI model (same as above)

## 💬 How to Use

1. **Upload PDFs**: Drag and drop your PDF files
2. **Process**: Click "Process Documents" to analyze your files
3. **Chat**: Ask questions about your documents
4. **Get Answers**: AI responds based on your documents

## 🔧 Requirements

- **Windows 10/11**
- **Ollama** (AI engine) - Download from https://ollama.ai
- **AI Model**: `llama3.2:1b` (1GB download)
- **Optional**: Tesseract OCR for scanned PDFs (auto-installs)

## 🔍 Features

- ✅ **Multiple PDF upload**
- ✅ **OCR for scanned documents**
- ✅ **Natural language questions**
- ✅ **Private & secure** (everything runs locally)
- ✅ **No cloud dependencies**
- ✅ **Recent conversations at top**

## ⚠️ Troubleshooting

### "Windows protected your PC" warning
- Click "More info" → "Run anyway"
- This is normal for new executables

### "Ollama not found" error
- Install Ollama from https://ollama.ai
- Restart the app after installation

### OCR not working for scanned PDFs
- Run: `winget install UB-Mannheim.TesseractOCR`
- Or download from: https://github.com/UB-Mannheim/tesseract/wiki

### App won't start
- Make sure you have internet connection (for first-time setup)
- Run as administrator if needed
- Check Windows firewall settings

## 🆘 Support

For issues or questions:
- Check troubleshooting section above
- Create an issue on GitHub: https://github.com/jabriomar873/chatapp/issues
- Make sure Ollama is installed and AI model is downloaded

## 🎯 Tips for Best Experience

1. **PDF Quality**: Clear, readable PDFs work best
2. **File Size**: No strict limits, but smaller files process faster  
3. **Questions**: Be specific about what you want to know
4. **Models**: Try different Ollama models for different needs:
   - `llama3.2:1b` - Fast, good for most tasks
   - `llama3.2:3b` - Better quality, slower
   - `llama3.1:8b` - Best quality, requires more RAM

Enjoy using PDF Chat! 🚀
