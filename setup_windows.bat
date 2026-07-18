@echo off
REM One-time setup for local QLoRA training on Windows (RTX 3060 12GB).
REM Double-click this file, or run it from a terminal in this folder.
setlocal
cd /d "%~dp0"

echo ============================================
echo  Step 1/4: Creating Python virtual environment
echo ============================================
python -m venv myenv
if errorlevel 1 (
  echo.
  echo ERROR: Python not found. Install Python 3.10 or newer from python.org
  echo and tick "Add Python to PATH" during install, then run this again.
  pause
  exit /b 1
)
call myenv\Scripts\activate.bat

echo.
echo ============================================
echo  Step 2/4: Installing CUDA PyTorch (for your GPU)
echo ============================================
REM cu130 wheels match a CUDA 13.x driver. If your driver is older, pick the
REM matching index from https://pytorch.org/get-started/locally/ (e.g. cu128).
pip install torch --index-url https://download.pytorch.org/whl/cu130

echo.
echo ============================================
echo  Step 3/4: Installing the other requirements
echo ============================================
pip install -r requirements.txt

echo.
echo ============================================
echo  Step 4/4: Preflight check (GPU, libraries, token)
echo ============================================
python preflight_check.py

echo.
echo Setup finished. If the preflight said PASS, run train_windows.bat next.
pause
