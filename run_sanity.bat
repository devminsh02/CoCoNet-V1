@echo off
cd /d %~dp0
python .\main.py sanity-model --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml
pause
