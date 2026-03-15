@echo off
title bonziPONY Setup
color 0d
echo.
echo  ============================================
echo     bonziPONY v1.69 — One-Click Setup
echo  ============================================
echo.

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Download it from https://www.python.org/downloads/
    echo  MAKE SURE you check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found.

:: ── Create venv if it doesn't exist ──────────────────────────
if not exist "venv\" (
    echo.
    echo  Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv. Continuing without it...
        goto :skip_venv
    )
    echo  [OK] Virtual environment created.
)

:: ── Activate venv ────────────────────────────────────────────
call venv\Scripts\activate.bat 2>nul
echo  [OK] Virtual environment activated.

:skip_venv

:: ── Upgrade pip ──────────────────────────────────────────────
echo.
echo  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  [OK] pip upgraded.

:: ── Install dependencies ─────────────────────────────────────
echo.
echo  Installing dependencies (this may take a while)...
echo  If torch/transformers are slow, be patient — they're big.
echo.
pip install -r requirements-lock.txt
if errorlevel 1 (
    echo.
    echo  [WARN] Lockfile install had issues, trying loose requirements...
    pip install -r requirements.txt
)
echo.
echo  [OK] Dependencies installed.

:: ── Copy config if needed ────────────────────────────────────
if not exist "config.yaml" (
    echo.
    echo  No config.yaml found — copying from example...
    copy config.yaml.example config.yaml >nul
    echo  [OK] config.yaml created.
    echo.
    echo  !! IMPORTANT: Edit config.yaml with your API keys !!
    echo  Or use the right-click menu after launch to set them.
    echo.
)

:: ── Create empty dirs ────────────────────────────────────────
if not exist "memory" mkdir memory
if not exist "diary" mkdir diary
if not exist "logs" mkdir logs

:: ── Launch ───────────────────────────────────────────────────
echo.
echo  ============================================
echo     Setup complete! Launching bonziPONY...
echo  ============================================
echo.
echo  Right-click the pony for settings.
echo  Double-click or say the wake word to chat.
echo  Close this window to kill the pony.
echo.

python main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] bonziPONY crashed. Check the error above.
    echo.
    pause
)
