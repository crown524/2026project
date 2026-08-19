@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   DAIC Depression Screening System - Quick Start
echo ============================================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo [1/2] Virtual environment ready
    goto :start_service
)

echo [1/2] First run - creating virtual environment...
where python >nul 2>nul
if %errorlevel%==0 (
    python setup_and_run.py
    goto :start_service
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 setup_and_run.py
    goto :start_service
)

echo [ERROR] Python not found.
echo.
echo Please install Python 3.10+ from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:start_service
echo [2/2] Starting Streamlit interface...
echo.
echo Service will start at: http://localhost:8501
echo Press Ctrl+C to stop the service
echo.
".venv\Scripts\python.exe" -m streamlit run app_ui.py

:done
echo.
pause
