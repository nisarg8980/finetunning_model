@echo off
REM Run QLoRA training on your local GPU. Run setup_windows.bat first.
REM Your Hugging Face token is NEVER stored in this file; it is asked for at run time.
setlocal
cd /d "%~dp0"

if not exist "myenv\Scripts\activate.bat" (
  echo ERROR: virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
call myenv\Scripts\activate.bat

if "%HF_TOKEN%"=="" set /p HF_TOKEN=Enter your Hugging Face token (input is visible):

echo.
echo ============================================
echo  Training Mistral-7B QLoRA on your 3060
echo  This can take 45 to 90 minutes, plus a one-time ~15 GB model download.
echo  Tip: close browsers/games so the GPU has its full 12 GB.
echo ============================================
python finetune_mistral_qlora.py ^
  --train_file data/cybersecurity_qa_train.jsonl ^
  --eval_file data/cybersecurity_qa_val.jsonl ^
  --output_dir ./cyber-qa-out

echo.
if errorlevel 1 (
  echo Training reported an error above. Copy the message and send it for a fix.
) else (
  echo Done. Your fine-tuned adapter is in:  cyber-qa-out\adapter
  echo Training graphs:      cyber-qa-out\training_curves.png
  echo Before/after metrics: cyber-qa-out\metrics_report.md  ^(Accuracy, F1, Response Quality^)
  echo Next: check hallucination with the command in CYBER_QA_RUNBOOK.md
  echo To publish to Hugging Face, re-run with:  --push_repo your-username/name --push_private
)
pause
