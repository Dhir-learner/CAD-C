@echo off
rem Start the CADC web app. The browser opens automatically once the server is ready.
rem Keep this window open while using the app; close it to stop the app.
cd /d "%~dp0"
.venv\Scripts\python -m app.server %*
pause
