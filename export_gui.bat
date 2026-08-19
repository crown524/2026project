@echo off
REM DAIC dataset export tool - GUI launcher
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" export_tool.py
    goto :eof
)

where pythonw 1>nul 2>nul
if %errorlevel%==0 (
    start "" pythonw export_tool.py
    goto :eof
)

echo [ERROR] Python not found.
echo Please run setup_and_run.bat first to create the environment.
echo.
pause
