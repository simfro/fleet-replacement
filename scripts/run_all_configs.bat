@echo off
setlocal

set REPO=%~dp0..
set PYTHON=%REPO%\.venv\Scripts\python.exe
set SCRIPT=%REPO%\scripts\run_batch.py

echo === baseline ===
"%PYTHON%" "%SCRIPT%" --config "%REPO%\configs\baseline.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo === conservative ===
"%PYTHON%" "%SCRIPT%" --config "%REPO%\configs\conservative.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo === optimistic ===
"%PYTHON%" "%SCRIPT%" --config "%REPO%\configs\optimistic.yaml" --n-episodes 200 --seed-start 0
if errorlevel 1 goto :error

echo.
echo All batches completed.
goto :eof

:error
echo.
echo ERROR: batch run failed (exit code %errorlevel%).
exit /b %errorlevel%
