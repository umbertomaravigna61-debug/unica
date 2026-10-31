@echo off
title LAB YuE2 - prova della PR #273 (8daaa70) sulla 5060 8 GB
cd /d <LAB>
set LAB_PYTHON=<LAB_ROOT>\.venv\Scripts\python.exe
echo 1) scarico il ramo codex/issue-portability-fixes @ 8daaa70 (non installa niente)
"%LAB_PYTHON%" -u scarica_pr271.py 8daaa7002efdb54d5ec003e602f0521c0d5e97db yue_pr273
echo 2) prove: rock_en e pop_it, seed 11, --low-vram bf16 e fp8, nessun ritocco
"%LAB_PYTHON%" -u lancia_pr271.py --giro rapido --modi pr273,pr273_fp8
echo ===== FINITO =====
pause
