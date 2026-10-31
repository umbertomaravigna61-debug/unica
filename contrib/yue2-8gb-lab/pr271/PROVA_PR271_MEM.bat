@echo off
title LAB YuE2 - #271 bf16 + riserva 93%
cd /d <LAB>
"<LAB_ROOT>\.venv\Scripts\python.exe" -u lancia_pr271.py --giro rapido --modi pr271_mem
echo ===== FINITO =====
pause
