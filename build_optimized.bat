@echo off
echo 🚀 PDF Chat - Optimized Build
echo =============================

echo 📦 Building optimized executable...
echo ⏱️  This will take 5-10 minutes...

REM Build lightweight launcher (FAST - ~2 minutes)
echo.
echo 🔹 Building Lightweight Launcher (RECOMMENDED)
pyinstaller --onefile --name "PDFChat-Launcher" --noconsole lightweight_launcher.py

REM Build optimized main app (SLOWER - ~8 minutes)
echo.
echo 🔹 Building Optimized Main App (Optional)
pyinstaller --onefile --name "PDFChat" ^
  --exclude-module torch ^
  --exclude-module tensorflow ^
  --exclude-module transformers ^
  --exclude-module matplotlib ^
  --exclude-module tkinter ^
  --exclude-module IPython ^
  --exclude-module jupyter ^
  --exclude-module notebook ^
  --add-data "htmlTemplates.py;." ^
  app.py

echo.
echo ✅ Build completed!
echo 📁 Results in 'dist' folder:
echo    🔸 PDFChat-Launcher.exe (~5MB) - RECOMMENDED
echo    🔸 PDFChat.exe (~120MB) - Full app
echo.
echo 💡 Recommendation: Share PDFChat-Launcher.exe
echo    - Smaller download
echo    - Auto-updates from GitHub
echo    - Easier distribution

pause
