@echo off
rem Start the nodule analysis web app and open it in the browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8000
.venv\Scripts\python -m app.server %*
pause
