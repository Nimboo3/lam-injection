@echo off
REM Quick demo runner for Windows
REM Runs both baseline and defense experiments automatically

echo.
echo ================================================================================
echo                    DEMO AUTOMATION SCRIPT
echo ================================================================================
echo.
echo This script will run BOTH experiments:
echo   1. Baseline (no defense)
echo   2. With defenses enabled
echo.
echo Total estimated time: ~7 minutes
echo.
pause

REM Save original directory
set ORIGINAL_DIR=%CD%

REM Navigate to project root
cd /d "%~dp0.."

echo.
echo ================================================================================
echo STEP 1/2: Running BASELINE experiment (no defense)
echo ================================================================================
echo.

REM Run baseline
python demo\run_demo.py

if errorlevel 1 (
    echo.
    echo ERROR: Baseline demo failed!
    pause
    exit /b 1
)

echo.
echo ✓ Baseline complete!
echo.
echo Waiting 3 seconds before starting defense demo...
timeout /t 3 /nobreak > nul

echo.
echo ================================================================================
echo STEP 2/2: ENABLING DEFENSES and running again
echo ================================================================================
echo.

# Create temporary script with defense enabled
echo import sys > demo\temp_defense_demo.py
echo from pathlib import Path >> demo\temp_defense_demo.py
echo sys.path.insert(0, str(Path(__file__).parent.parent)) >> demo\temp_defense_demo.py
echo. >> demo\temp_defense_demo.py
echo # Read original file >> demo\temp_defense_demo.py
echo with open("demo/run_demo.py", "r", encoding="utf-8") as f: >> demo\temp_defense_demo.py
echo     code = f.read() >> demo\temp_defense_demo.py
echo. >> demo\temp_defense_demo.py
echo # Replace USE_DEFENSE flag >> demo\temp_defense_demo.py
echo code = code.replace("USE_DEFENSE = False", "USE_DEFENSE = True") >> demo\temp_defense_demo.py
echo. >> demo\temp_defense_demo.py
echo # Execute >> demo\temp_defense_demo.py
echo exec(code) >> demo\temp_defense_demo.py

python demo\temp_defense_demo.py

if errorlevel 1 (
    echo.
    echo ERROR: Defense demo failed!
    del demo\temp_defense_demo.py
    pause
    exit /b 1
)

REM Cleanup
del demo\temp_defense_demo.py

echo.
echo ================================================================================
echo SUCCESS! Both experiments complete
echo ================================================================================
echo.
echo Results saved to:
echo   • data\demo_results\no_defense\
echo   • data\demo_results\with_defense\
echo.
echo Next steps:
echo   1. View plots: explorer data\demo_results
echo   2. Compare results: jupyter notebook notebooks\demo_comparison.ipynb
echo.
echo ================================================================================
pause
