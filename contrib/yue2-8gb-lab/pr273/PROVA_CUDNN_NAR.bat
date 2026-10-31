@echo off
title LAB YuE2 - rifinitura #273: efficient contro cuDNN (rock_en s11)
cd /d <LAB>
set HF_HOME=<MODELS>\yue2\hf_cache
set PYTORCH_ALLOC_CONF=expandable_segments:True
"<LAB_ROOT>\.venv\Scripts\python.exe" -u prova_cudnn_nar.py
echo ===== FINITO =====
pause
