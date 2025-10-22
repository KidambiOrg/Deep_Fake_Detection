@echo off
echo ========================================
echo EfficientNet-B7 Conversion Script
echo ========================================
echo.

cd /d "%~dp0.."

echo Running conversion script...
python Scripts\convert_efficientnet_b7_tf_to_pytorch.py

echo.
echo ========================================
echo Conversion complete!
echo ========================================
echo.
pause
