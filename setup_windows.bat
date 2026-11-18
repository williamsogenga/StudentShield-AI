@echo off
setlocal
cd /d "%~dp0"
echo ================== StudentShield AI Setup ==================
where py >nul 2>&1 && (py -3.10 -m venv .venv) || (python -m venv .venv)
if not exist .venv\Scripts\python.exe (
  echo Could not create environment. Install Python 3.10+ and retry.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist data\uci_students.csv (
  echo Downloading authentic UCI student dataset ...
  .venv\Scripts\python.exe download_data.py
  if errorlevel 1 (
    echo UCI dataset could not be downloaded. Check internet and rerun setup.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe train.py
if errorlevel 1 (
 echo Model training failed.
 pause
 exit /b 1
)
.venv\Scripts\python.exe make_submission.py
.venv\Scripts\python.exe -m streamlit run app.py
pause
