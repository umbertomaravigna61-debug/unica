@echo off
title LAB YuE2 - piano lungo della #271 (3187 token) sulla #273
cd /d <LAB>
set HF_HOME=<MODELS>\yue2\hf_cache
set HF_HUB_OFFLINE=1
set PYTORCH_ALLOC_CONF=expandable_segments:True
"<LAB_ROOT>\.venv\Scripts\python.exe" -u prova_piano_lungo.py
echo ===== FINITO =====
pause
