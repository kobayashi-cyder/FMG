@echo off
setlocal
cd /d "%~dp0"
python run_fmg_eval_loop.py --split train --forever --publish-all --git-push-every 32 %*
endlocal
