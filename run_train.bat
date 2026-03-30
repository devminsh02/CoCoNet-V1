@echo off
cd /d %~dp0
set RUN_NAME=%1
if "%RUN_NAME%"=="" set RUN_NAME=debug_ab_noamp
python .\main.py train --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name %RUN_NAME% --skip-prepare
pause
