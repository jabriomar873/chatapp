# 🛠️ Troubleshooting Guide - Enhanced PDF Chat

## 🚨 Common Issues and Solutions

### 1. OpenAI API Issues

#### **Error: Quota Exceeded (429)**
```
Error code: 429 - You exceeded your current quota
```
**Solutions:**
- Check your OpenAI billing: https://platform.openai.com/account/billing
- Add a payment method or upgrade your plan
- Remove the OpenAI API key to use free models
- Monitor usage at: https://platform.openai.com/usage

#### **Error: Invalid API Key (401)**
```
Error code: 401 - Invalid API key
```
**Solutions:**
- Verify your API key at: https://platform.openai.com/api-keys
- Ensure the key is correctly set in `.env` file
- Check for extra spaces or characters in the key

#### **Error: Rate Limit (429)**
```
Rate limit reached
```
**Solutions:**
- Wait 1-2 minutes before trying again
- Upgrade to a higher tier plan
- Use shorter questions to reduce token usage

### 2. LangChain/Retrieval Issues

#### **Error: BaseRetriever Validation**
```
Input should be a valid dictionary or instance of BaseRetriever
```
**Status:** ✅ **FIXED** - Enhanced retriever now properly inherits from BaseRetriever

#### **Error: Memory Deprecation Warning**
```
LangChainDeprecationWarning: Please see the migration guide
```
**Status:** ⚠️ **Warning Only** - App still functions normally

### 3. NumPy/Package Compatibility Issues

#### **Error: NumPy 2.x Compatibility**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.1
```
**Status:** ✅ **FIXED** - requirements.txt updated to use `numpy<2.0.0,>=1.24.0`

**Manual Fix if needed:**
```bash
pip uninstall numpy
pip install "numpy<2.0.0,>=1.24.0"
pip install -r requirements.txt
```

#### **Error: Torch RuntimeError**
```
RuntimeError: Tried to instantiate class '__path__._path'
```
**Status:** ⚠️ **Warning Only** - Related to Streamlit file watching, doesn't affect functionality

### 4. PDF Processing Issues

#### **Error: No Text Found**
```
No readable text found in the uploaded PDFs
```
**Solutions:**
- Ensure PDFs contain selectable text (not just images)
- Install Tesseract for OCR: `winget install UB-Mannheim.TesseractOCR`
- Try with different PDF files
- Check if Tesseract is detected in OCR settings

#### **Error: Chunking Failed**
```
Could not create text chunks
```
**Solutions:**
- Verify PDFs have readable content
- Try smaller PDF files first
- Check for corrupted PDF files

### 5. Model Loading Issues

#### **Error: HuggingFace Model Download**
```
Failed to load model
```
**Solutions:**
- Ensure stable internet connection
- Wait for model download to complete (first time only)
- Clear HuggingFace cache: `~/.cache/huggingface/`
- Restart the application

### 6. Memory/Performance Issues

#### **Error: Out of Memory**
```
CUDA out of memory / System memory error
```
**Solutions:**
- Use smaller PDF files
- Reduce chunk size in settings
- Restart the application
- Close other memory-intensive applications

## 🔧 Quick Fixes

### Reset Application State
```bash
# Stop the application (Ctrl+C)
# Clear session state
rm -rf __pycache__/
# Restart
streamlit run app.py
```

### Reinstall Dependencies
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Clear Model Cache
```bash
# Windows
rmdir /s "%USERPROFILE%\.cache\huggingface"
# or manually delete: C:\Users\[username]\.cache\huggingface\
```

## 📊 System Requirements

### Minimum Requirements:
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB free space for models
- **Python**: 3.8 or higher
- **Internet**: Required for first-time model download

### Recommended Setup:
- **RAM**: 16GB or more
- **CPU**: Multi-core processor
- **Internet**: Stable connection for OpenAI API
- **Storage**: SSD for better performance

## 🆘 Getting Help

### Error Reporting
When reporting issues, include:
1. **Error message** (full traceback)
2. **PDF type** (text-based or scanned)
3. **Model used** (OpenAI or HuggingFace)
4. **System info** (Windows version, Python version)
5. **Steps to reproduce**

### Logs Location
- **Streamlit logs**: Terminal/console output
- **Model cache**: `~/.cache/huggingface/`
- **Temporary files**: `__pycache__/`

### Debug Mode
Add this to your `.env` file for more detailed logging:
```
LANGCHAIN_VERBOSE=true
STREAMLIT_LOGGER_LEVEL=debug
```

## ✅ Health Checks

### Verify Installation
```python
# Run in Python to check key components
import streamlit
import langchain
import numpy
print(f"Streamlit: {streamlit.__version__}")
print(f"LangChain: {langchain.__version__}")
print(f"NumPy: {numpy.__version__}")
```

### Test Models
1. **HuggingFace**: Upload a simple PDF and ask "What is this about?"
2. **OpenAI**: Ensure API key is set and has quota
3. **OCR**: Test with a scanned PDF

### Performance Check
- **PDF Processing**: Should complete within 1-2 minutes
- **First Question**: May take 30-60 seconds (model loading)
- **Subsequent Questions**: Should respond within 10-30 seconds

## 🔄 Version Compatibility

### Tested Versions:
- **Python**: 3.8-3.12
- **Streamlit**: 1.28.0+
- **LangChain**: 0.1.0+
- **NumPy**: 1.24.0-1.26.x (not 2.x)
- **PyTorch**: 2.0.0+

### Known Issues:
- **NumPy 2.x**: Not compatible (use <2.0.0)
- **Python 3.13**: Not tested
- **macOS ARM**: May need additional setup for some models

## 📈 Performance Optimization

### For Better Speed:
1. **Use OpenAI models** (faster than local models)
2. **Smaller PDFs** (under 10MB per file)
3. **Text-based PDFs** (avoid OCR when possible)
4. **Close other applications** to free RAM

### For Better Accuracy:
1. **OpenAI GPT-4** (best reasoning)
2. **Clean, well-structured PDFs**
3. **Specific questions** rather than broad ones
4. **Multiple document processing** for context

---

**Last Updated**: July 24, 2025
**Status**: All critical issues resolved ✅
