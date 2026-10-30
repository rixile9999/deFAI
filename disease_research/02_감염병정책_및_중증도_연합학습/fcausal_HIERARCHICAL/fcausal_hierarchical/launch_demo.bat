@echo off
REM CES 2026 데모 런처 배치 파일

setlocal
set PYTHON_EXE=C:\Users\Haneol\Anaconda3\envs\flwr\python.exe
set PYTHONIOENCODING=utf-8

echo.
echo ========================================================================
echo   CES 2026 DEMO LAUNCHER
echo   Hierarchical Federated Learning Demonstration
echo ========================================================================
echo.

"%PYTHON_EXE%" demo_launcher.py

pause
