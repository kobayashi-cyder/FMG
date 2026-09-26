@echo off
setlocal EnableExtensions
cd /d "%~dp0"

:restart
echo [FMG watchdog] starting strict evaluation loop...
call RUN_FMG_EVAL_LOOP.cmd %*
set "rc=%errorlevel%"
echo [FMG watchdog] evaluator exited with code %rc%.
echo [FMG watchdog] restarting in 10 seconds. Close this window to stop.
timeout /t 10 /nobreak >nul
goto restart
