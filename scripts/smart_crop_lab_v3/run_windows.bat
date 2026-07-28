@echo off
setlocal
cd /d "%~dp0"

echo =============================================
echo Smart Crop ^& Degradation Lab - installation
echo =============================================

where python >nul 2>nul
if errorlevel 1 (
  echo Python introuvable. Installe Python 3.10 ou plus recent et coche "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creation de l'environnement virtuel...
  python -m venv .venv
  if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :error
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo L'interface va s'ouvrir sur http://127.0.0.1:7860
python app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Une erreur est survenue. Copie le message affiche dans cette fenetre.
pause
exit /b 1
