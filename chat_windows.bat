@echo off
REM Chat with your fine-tuned model to test it. Run train first so the adapter exists.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

if "%HF_TOKEN%"=="" set /p HF_TOKEN=Enter your Hugging Face token:

echo Loading your fine-tuned model. Type 'exit' to quit.
python chat.py --adapter_dir ./cyber-qa-out/adapter
pause
