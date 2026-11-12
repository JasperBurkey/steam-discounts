@echo off
rem Called by Task Scheduler every hour:  run_group.bat fast|daily|ccu
cd /d "%~dp0"
if not exist "data\logs" mkdir "data\logs"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" collect.py run --group %1 >> "data\logs\task_%1.log" 2>&1
