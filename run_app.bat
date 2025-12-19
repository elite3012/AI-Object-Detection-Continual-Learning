@echo off
title Continual Learning System
echo ============================================
echo   Continual Learning System - First Run
echo ============================================
echo.

REM Update pip to the latest version
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [ERROR] Failed to update pip
    pause
    exit /b 1
)

REM Install precompiled numpy to avoid build issues
pip install numpy --only-binary=:all:
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install numpy
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo [INSTALL] Installing dependencies from requirements.txt...
    echo This may take several minutes on first run...
    echo.
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
    echo [OK] All dependencies installed successfully
) else (
    echo [OK] Dependencies already installed
)

echo.
echo Starting Continual Learning Web Interface...
echo The app will open in your browser automatically.
echo.
streamlit run app.py
pause
