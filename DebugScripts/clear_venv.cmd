@echo off
REM Clear Python virtual environment dependencies
REM Keeps only pip, setuptools, and wheel

echo Clearing Python virtual environment dependencies...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at .venv
    echo Please create a virtual environment first.
    pause
    exit /b 1
)

REM Kill any running streamlit processes that might lock files
echo Checking for running streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
if %errorlevel% equ 0 (
    echo Streamlit process terminated.
) else (
    echo No streamlit process found.
)
echo.

REM Save list of installed packages
echo Saving list of installed packages...
.venv\Scripts\python.exe -m pip freeze > installed_packages_temp.txt
if %errorlevel% neq 0 (
    echo Error: Failed to get package list
    pause
    exit /b 1
)
echo.

REM Uninstall all packages
echo Uninstalling all packages (this may take a few minutes)...
.venv\Scripts\python.exe -m pip uninstall -y -r installed_packages_temp.txt
echo.

REM Clean up temp file
del installed_packages_temp.txt 2>nul

REM Clear pip cache
echo Clearing pip cache...
.venv\Scripts\python.exe -m pip cache purge
echo.

REM Show final package list
echo.
echo Final package list:
.venv\Scripts\python.exe -m pip list
echo.

echo Done! Python environment cleared successfully.
echo Only pip, setuptools, and wheel remain.
pause
