@echo off
rem Start the Shop the Look server on Windows (same as run.sh). HOST/PORT/DEVICE are optional (DEVICE=cpu|cuda).
cd /d "%~dp0app"
set "PY=..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8000"
"%PY%" -m uvicorn server:app --host %HOST% --port %PORT%
