@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if not errorlevel 1 (
  py -3 fmg_image_generator_v23.py
  exit /b %errorlevel%
)

where python >nul 2>&1
if not errorlevel 1 (
  python fmg_image_generator_v23.py
  exit /b %errorlevel%
)

echo [ERROR] Python 3 was not found.
exit /b 1
