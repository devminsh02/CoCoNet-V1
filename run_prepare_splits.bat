@echo off
cd /d %~dp0
python .\main.py prepare-splits --config .\configs\model_v1_resnet50.yaml
pause
