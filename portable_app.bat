@echo off
title PDF Chat - Portable App

echo.
echo 💬 PDF Chat - Portable Application
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo 📥 Please install Python from: https://python.org
    echo    OR use the standalone .exe version
    pause
    exit /b 1
)

REM Check if Ollama is installed
ollama list >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not found!
    echo 📥 Installing Ollama...
    echo 🔗 Download from: https://ollama.ai
    start https://ollama.ai
    echo.
    echo ⏳ Please install Ollama and run: ollama pull llama3.2:1b
    pause
)

REM Install Python dependencies if needed
if not exist ".venv" (
    echo 📦 Setting up virtual environment...
    python -m venv .venv
)

echo 🔧 Activating environment...
call .venv\Scripts\activate.bat

echo 📋 Installing dependencies...
pip install -r requirements.txt -q

echo 🚀 Starting PDF Chat...
echo 🌐 Opening in browser: http://localhost:8501
echo.
echo ✨ Upload your PDFs and start chatting!
echo 🛑 Press Ctrl+C to stop the application
echo.

REM Start the application
streamlit run app.py --server.headless true

echo.
echo 👋 PDF Chat closed. Thanks for using!
pause
