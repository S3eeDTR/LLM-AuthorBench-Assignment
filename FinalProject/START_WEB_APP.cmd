@echo off
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PROJECT_PYTHON%" goto run
set "PROJECT_PYTHON=D:\Final Project\.venv\Scripts\python.exe"
if exist "%PROJECT_PYTHON%" goto run
set "PROJECT_PYTHON=python"
:run
echo Code Origin - local research demo
echo Open http://127.0.0.1:8765 after the server starts.
echo Keep this window open. Press Ctrl+C to stop.
"%PROJECT_PYTHON%" "%~dp0webapp\server.py" --device cpu
pause
