@echo off
cd /d "%~dp0src"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
python demo.py %*
echo.
pause
