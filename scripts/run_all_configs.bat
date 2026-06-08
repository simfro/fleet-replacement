@echo off
setlocal

set REPO=%~dp0..
set PYTHON=%REPO%\.venv\Scripts\python.exe
set SCRIPT=%REPO%\scripts\run_batch.py
set ENV_CFG=%REPO%\configs\env.yaml

echo === lookahead / baseline forecast ===
"%PYTHON%" "%SCRIPT%" --config "%ENV_CFG%" --lookahead-config "%REPO%\configs\lookahead_baseline.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo === lookahead / conservative forecast ===
"%PYTHON%" "%SCRIPT%" --config "%ENV_CFG%" --lookahead-config "%REPO%\configs\lookahead_conservative.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo === lookahead / optimistic forecast ===
"%PYTHON%" "%SCRIPT%" --config "%ENV_CFG%" --lookahead-config "%REPO%\configs\lookahead_optimistic.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo === myopic ===
"%PYTHON%" "%SCRIPT%" --config "%ENV_CFG%" --agent myopic --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo.
echo All batches completed.
goto :eof

:error
echo.
echo ERROR: batch run failed (exit code %errorlevel%).
exit /b %errorlevel%
