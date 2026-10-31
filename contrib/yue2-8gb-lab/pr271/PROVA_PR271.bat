@echo off
title LAB YuE2 - prova della PR #271 (--low-vram) sulla 5060 8 GB
cd /d <LAB>
set HF_HOME=<MODELS>\yue2\hf_cache
set HF_HUB_OFFLINE=1
set LAB_PYTHON=<LAB_ROOT>\.venv\Scripts\python.exe
echo 1) scarico il codice YuE dopo la #271 (solo la prima volta, non installa niente)
"%LAB_PYTHON%" -u scarica_pr271.py
echo 2) prove: giro rapido (rock_en e pop_it, seed 11, tre modi)
"%LAB_PYTHON%" -u lancia_pr271.py --giro rapido
echo ===== FINITO =====
pause
