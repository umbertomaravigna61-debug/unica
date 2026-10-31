@echo off
title LAB YuE2 - banco dell attenzione della rifinitura (NAR)
cd /d <LAB>
set HF_HOME=<MODELS>\yue2\hf_cache
"<LAB_ROOT>\.venv\Scripts\python.exe" -u banco_attenzione_nar.py
echo ===== FINITO =====
pause
