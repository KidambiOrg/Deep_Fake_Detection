@echo off
REM Complete cleanup and fresh installation of Python dependencies
REM Clears virtual environment, pip cache, and reinstalls from requirements.txt

echo ========================================
echo Python Environment Complete Cleanup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at .venv
    echo Please create a virtual environment first.
    pause
    exit /b 1
)

REM Step 1: Kill any running processes that might lock files
echo [1/6] Checking for running processes...
taskkill /F /IM streamlit.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul
echo Done.
echo.

REM Step 2: Save and uninstall all packages
echo [2/6] Uninstalling all packages from virtual environment...
.venv\Scripts\python.exe -m pip freeze > temp_packages.txt
.venv\Scripts\python.exe -m pip uninstall -y -r temp_packages.txt
del temp_packages.txt 2>nul
echo Done.
echo.

REM Step 3: Clear pip cache
echo [3/6] Clearing pip cache...
.venv\Scripts\python.exe -m pip cache purge
echo Done.
echo.

REM Step 4: Upgrade pip, setuptools, and wheel
echo [4/6] Upgrading pip, setuptools, and wheel...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
echo Done.
echo.

REM Step 5: Clean up corrupted package directories
echo [5/6] Cleaning up corrupted package directories...
if exist ".venv\Lib\site-packages\~treamlit" rmdir /S /Q ".venv\Lib\site-packages\~treamlit"
if exist ".venv\Lib\site-packages\~harset_normalizer" rmdir /S /Q ".venv\Lib\site-packages\~harset_normalizer"
if exist ".venv\Lib\site-packages\~umpy" rmdir /S /Q ".venv\Lib\site-packages\~umpy"
if exist ".venv\Lib\site-packages\~umpy.libs" rmdir /S /Q ".venv\Lib\site-packages\~umpy.libs"
if exist ".venv\Lib\site-packages\~il" rmdir /S /Q ".venv\Lib\site-packages\~il"
echo Done.
echo.

REM Step 6: Install all packages from requirements.txt
echo [6/6] Installing packages from requirements.txt...
echo This may take several minutes...
echo.
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.

REM Show final status
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Final package list:
.venv\Scripts\python.exe -m pip list
echo.

pause
