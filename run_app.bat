@echo off
title Continual Learning System
echo ============================================
echo   Continual Learning System
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

echo Installing or updating dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting Continual Learning Web Interface...
echo The app will open in your browser automatically.
echo.
python -m streamlit run app.py
pause
