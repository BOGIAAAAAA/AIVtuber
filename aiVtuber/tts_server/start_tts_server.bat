@echo off
rem Start the GPT-SoVITS api_v2 TTS server for aiVtuber (double-click friendly).
rem All options are handled by start_tts_server.ps1; arguments are passed through, e.g.
rem   start_tts_server.bat -GsvDir "D:\GPT-SoVITS-v2pro-20250604" -Port 9880
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_tts_server.ps1" %*
pause
