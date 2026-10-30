@echo off
REM CES 2026 Demo Launcher - Simple Version
REM All ASCII characters only (no Korean to avoid encoding issues)

setlocal
set PYTHON_EXE=C:\Users\Haneol\Anaconda3\envs\flwr\python.exe
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore::DeprecationWarning
set FLWR_TELEMETRY_ENABLED=0

echo.
echo ========================================================================
echo   CES 2026 DEMO - Simple Launcher
echo   Hierarchical Federated Learning Demo
echo ========================================================================
echo.
echo Screen Layout Guide (3840x2160):
echo.
echo     +-------------+-------------+-------------+
echo     ^|   (Empty)   ^|    KDCA     ^|  Seongnam   ^|  (Top Row)
echo     +-------------+-------------+-------------+
echo     ^|    Seoul    ^|   (Empty)   ^|  Gyeonggi   ^|  (Middle Row)
echo     +-------------+-------------+-------------+
echo     ^|   Gangnam   ^|   Seocho    ^|    Suwon    ^|  (Bottom Row)
echo     +-------------+-------------+-------------+
echo.
echo Each window: approximately 1280x720 pixels
echo.
echo Press any key to start the three-stage federated learning agents for 7 local governments...
pause >nul

echo.
echo.
echo ========================================================================
echo   STEP 1: Launching Tier 0 (KDCA Central Server)
echo ========================================================================
start "Tier 0 - KDCA" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier0.py"
echo    [OK] Tier 0 started - Waiting 3 seconds for server to be ready...
timeout /t 3 /nobreak >nul

echo.
echo ========================================================================
echo   STEP 2: Launching Tier 1 (Provincial Servers)
echo ========================================================================
echo    Starting Seoul Province (Port 9001)...
start "Tier 1 - Seoul" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier1.py --port 9001 --region Seoul --treated --min-districts 2"
timeout /t 2 /nobreak >nul

echo    Starting Gyeonggi Province (Port 9002)...
start "Tier 1 - Gyeonggi" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier1.py --port 9002 --region Gyeonggi --min-districts 2"
echo    [OK] Tier 1 servers started - Waiting 5 seconds for ports to open...
timeout /t 5 /nobreak >nul

echo.
echo ========================================================================
echo   STEP 3: Launching Tier 2 (District Clients)
echo ========================================================================
echo    Connecting districts to provincial servers...
start "Tier 2 - Gangnam" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001"
timeout /t 1 /nobreak >nul

start "Tier 2 - Seocho" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier2.py --data data/districts/seoul_seocho.xlsx --target 127.0.0.1:9001"
timeout /t 1 /nobreak >nul

start "Tier 2 - Suwon" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier2.py --data data/districts/gyeonggi_suwon.xlsx --target 127.0.0.1:9002"
timeout /t 1 /nobreak >nul

start "Tier 2 - Seongnam" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set PYTHONWARNINGS=ignore::DeprecationWarning && set FLWR_TELEMETRY_ENABLED=0 && "%PYTHON_EXE%" run_tier2.py --data data/districts/gyeonggi_seongnam.xlsx --target 127.0.0.1:9002"
echo    [OK] All districts launched!

echo.
echo ========================================================================
echo   ALL 7 TERMINALS LAUNCHED SUCCESSFULLY!
echo ========================================================================
echo.
echo What happens next (automatic):
echo.
echo   1. Tier 2 districts send encrypted data to Tier 1 (5-10 sec)
echo      -^> Windows will auto-close after showing completion message
echo.
echo   2. Tier 1 provinces perform blind aggregation (10-15 sec)
echo      -^> Windows stay open showing "Mission complete!"
echo.
echo   3. Tier 0 KDCA performs DiD analysis (15-20 sec)
echo      -^> Window stays open showing "-71.2%% reduction"
echo.
echo Manual Window Arrangement:
echo    - Arrange the 7 windows in 3x3 grid (see layout at top)
echo    - Each cell: approximately 1280x720 pixels
echo    - Use Win+Arrow keys for quick window snapping
echo.
echo Recording Recommendation:
echo    - Start OBS screen recording NOW if not already started
echo    - Resolution: 3840x2160 (4K) or 1920x1080 (FHD)
echo    - The federated learning will complete in about 20 seconds
echo.
echo Waiting 25 seconds for federated learning to complete...
timeout /t 25 /nobreak >nul

echo.
echo Press any key to play visualization video...
pause >nul

echo.
echo Opening visualization video...
start "" demo1_hierarchical.mp4

echo.
echo Demo complete!
echo.
pause
