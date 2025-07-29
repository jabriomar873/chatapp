@echo off
echo 🔨 PDF Chat - Executable Builder
echo ========================================

echo 📦 Installing PyInstaller...
pip install pyinstaller

echo 🚀 Building standalone executable...
python build_exe.py

echo.
echo ✅ Build completed!
echo 📁 Check the 'dist' folder for PDFChat.exe
echo 📖 Read DISTRIBUTION.md for distribution instructions

pause
