@echo off
REM Start the fine-tuned model API on Windows. Run setup_windows.bat first,
REM and make sure Ollama is running with your model loaded (see deploy_hf_ollama.md).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

REM Install the API dependencies (safe to run every time; skips if present)
pip install -r requirements-api.txt

if "%API_KEY%"=="" set /p API_KEY=Set an API key clients must send:
if "%API_MODEL%"=="" set API_MODEL=cyber-qa

echo.
echo API starting at http://127.0.0.1:8000   (interactive docs at /docs)
echo Press Ctrl+C to stop.
uvicorn api_server:app --host 127.0.0.1 --port 8000
pause
