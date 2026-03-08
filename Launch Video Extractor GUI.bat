@echo off
echo Starting Video Screenshot Extractor GUI...
echo.

cd /d "%~dp0"

REM Try to run the GUI
python launch_gui.py

REM If that fails, try python3
if %errorlevel% neq 0 (
    echo.
    echo Trying with python3...
    python3 launch_gui.py
)

REM If still failing, try py launcher
if %errorlevel% neq 0 (
    echo.
    echo Trying with py launcher...
    py launch_gui.py
)

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Could not start the application.
    echo Please make sure Python is installed and in your PATH.
    echo.
    echo You can try running: python launch_gui.py
    echo.
    pause
)