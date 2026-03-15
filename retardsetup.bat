@echo off
title bonziPONY Setup
color 0d
echo.
echo  ============================================
echo     bonziPONY v1.69 — One-Click Setup
echo  ============================================
echo.

:: ── Find the script's own directory and cd into it ───────────
cd /d "%~dp0"

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
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv. Continuing without it...
        set PY=python
        goto :install
    )
    echo  [OK] Virtual environment created.
)

:: ── Use venv python directly (more reliable than activate) ───
set PY=venv\Scripts\python.exe
set PIP=venv\Scripts\pip.exe
echo  [OK] Using venv Python.
goto :install

:install

:: ── Set PIP fallback if not using venv ───────────────────────
if not defined PIP set PIP=pip

:: ── Upgrade pip ──────────────────────────────────────────────
echo.
echo  Upgrading pip...
%PY% -m pip install --upgrade pip --quiet
echo  [OK] pip upgraded.

:: ── Install dependencies ─────────────────────────────────────
echo.
echo  Installing dependencies (this may take a while)...
echo  If torch/transformers are slow, be patient — they're big.
echo.
%PIP% install -r requirements-lock.txt
if errorlevel 1 (
    echo.
    echo  [WARN] Lockfile install had issues, trying loose requirements...
    %PIP% install -r requirements.txt
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

%PY% main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] bonziPONY crashed. Check the error above.
    echo.
    pause
)
