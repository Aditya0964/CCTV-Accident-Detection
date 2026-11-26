@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0outputs\runtime" set "PYTHONPATH=%~dp0outputs\runtime;%PYTHONPATH%"
set "CCTV_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%CCTV_PYTHON%" set "CCTV_PYTHON=%USERPROFILE%\OneDrive\Desktop\accident-detection-starter\.venv\Scripts\python.exe"
if exist "%CCTV_PYTHON%" (
    "%CCTV_PYTHON%" -B -m src.demo_app %*
) else (
    py -3.11 -B -m src.demo_app %*
)
if errorlevel 1 pause
