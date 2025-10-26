@echo off
echo Starting Deepfake Detection Web UI...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt
echo.

REM Check if secrets.toml exists
if not exist .streamlit\secrets.toml (
    echo Warning: .streamlit\secrets.toml not found!
    echo Using default local endpoint: http://localhost:7071/v1.0
    echo.
)

REM Start Streamlit
echo Starting Streamlit app...
@echo off
echo Starting Streamlit Deepfake Detection App...
echo.
streamlit run app.py

pause
