@echo off
set VENV_DIR=.venv

set PYTHON_CMD=
py -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
) else (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if %errorlevel% == 0 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
    echo Python ^>= 3.9 is required. Install it from https://www.python.org/
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" %PYTHON_CMD% -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip -q
python -m pip install -e ".[train]" -q

needle --help
