@echo off
REM Quick Research Experiment Runner for Windows
REM This script helps you run the multi-model comparison experiment

echo ================================================================================
echo Multi-Model Research Experiment Runner
echo ================================================================================
echo.

REM Check if Ollama is running
echo [1/3] Checking Ollama service...
ollama list >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Ollama service is not running!
    echo.
    echo Please open a new terminal and run:
    echo   ollama serve
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)
echo   [OK] Ollama is running

REM Check if models are downloaded
echo.
echo [2/3] Checking downloaded models...
ollama list | findstr /i "phi3" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: phi3 model not found
    echo.
    set /p download="Download models now? (y/n): "
    if /i "%download%"=="y" (
        echo.
        echo Running model downloader...
        python scripts\setup_ollama_models.py
        if errorlevel 1 (
            echo.
            echo ERROR: Model download failed
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Skipping model download. Experiment may fail.
        echo.
    )
) else (
    echo   [OK] Models are ready
)

REM Run the experiment
echo.
echo [3/3] Running multi-model comparison experiment...
echo.
echo This will take approximately 10-15 minutes.
echo Press Ctrl+C to cancel.
echo.
pause

python experiments\multi_model_comparison.py

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Experiment failed!
    echo ================================================================================
    echo.
    echo Check the error messages above for details.
    echo.
    echo Common issues:
    echo   - GEMINI_API_KEY not set in .env file
    echo   - Ollama service stopped during experiment
    echo   - Network connection issues
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Experiment completed!
echo ================================================================================
echo.
echo Results saved to: data\multi_model_comparison\
echo.
echo Files generated:
echo   - multi_model_results_*.csv  (raw data)
echo   - asr_comparison.png         (ASR curves)
echo   - utility_comparison.png     (goal-reaching rates)
echo   - combined_comparison.png    (side-by-side plots)
echo   - vulnerability_ranking.png  (bar chart)
echo.
echo Next steps:
echo   1. Open data\multi_model_comparison\ folder
echo   2. View the plots
echo   3. Analyze the CSV data
echo   4. Use results in your research paper
echo.
echo See RESEARCH_QUICKSTART.md for paper writing tips!
echo.
pause
