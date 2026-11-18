@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m streamlit run app.py
) else (
  echo Run setup_windows.bat first to install dependencies and train the model.
  pause
)
