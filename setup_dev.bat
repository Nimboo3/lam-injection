@echo off
REM Setup script for development environment (Windows)

echo ============================================================
echo Setting up Agentic Prompt-Injection Benchmark
echo ============================================================
echo.

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Virtual environment not activated!
    echo Please run: .venv\Scripts\activate
    exit /b 1
)

echo [1/3] Installing package in editable mode...
pip install -e .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install package
    exit /b 1
)

echo.
echo [2/3] Installing development dependencies...
pip install pytest pytest-cov
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dev dependencies
    exit /b 1
)

echo.
echo [3/3] Verifying installation...
python -c "import envs.gridworld; print('✓ envs.gridworld imported successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to import modules
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete! You can now run:
echo   - pytest tests/test_gridworld.py -v
echo   - python scripts/demo_gridworld.py
echo ============================================================
