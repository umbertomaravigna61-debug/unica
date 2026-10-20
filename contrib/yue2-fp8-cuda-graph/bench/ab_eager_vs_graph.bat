@echo off
setlocal
REM ============================================================
REM  A/B same seed: YuE2 FP8 eager vs FP8 CUDA graph (cuDNN), UNICA runner.
REM  uso: ab_eager_vs_graph.bat [nome=balla_con_me] [seed=8] [passi=4]
REM  Produce: out\ab_<nome>_seed<seed>\eager\music.wav e ...\graph\music.wav + ab.log con le righe JSON.
REM  Durata indicativa su RTX 5060 8 GB: eager ~8-9 min, graph ~2-3 min.
REM ============================================================
set UNICA=C:\Users\Acer\Desktop\unica
set LAB=C:\Users\Acer\UNICA-Chronos-Lab\yue2
set NAME=%1
if "%NAME%"=="" set NAME=balla_con_me
set SEED=%2
if "%SEED%"=="" set SEED=8
set STEPS=%3
if "%STEPS%"=="" set STEPS=4
set AB=%LAB%\out\ab_%NAME%_seed%SEED%
set LOG=%AB%\ab.log
set PYTHONUNBUFFERED=1
mkdir "%AB%\eager" 2>nul
mkdir "%AB%\graph" 2>nul
cd /d "%UNICA%\app"
for /f "usebackq delims=" %%s in ("%LAB%\%NAME%.style.txt") do set STYLE=%%s
set TEE=".venv_yue2\Scripts\python.exe" -c "import sys; [ (sys.stdout.write(l), sys.stdout.flush(), open(r'%LOG%','a',encoding='utf-8').write(l)) for l in sys.stdin ]"

echo === [A] EAGER  %NAME% seed %SEED% passi %STEPS%  (--graph 0)
echo # ===== A: EAGER seed %SEED% ===== >> "%LOG%"
".venv_yue2\Scripts\python.exe" vendor\yue2_runner.py --style "%STYLE%" --lyrics-file "%LAB%\%NAME%.lyrics.txt" --seed %SEED% --nar-steps %STEPS% --max-seconds 240 --graph 0 ^
  --model-dir "%UNICA%\models\yue2\YuE2-3B" --vae-dir "%UNICA%\models\yue2\YuE2-Vae" --work "%AB%\eager" --out "%AB%\eager\music.wav" 2>&1 | %TEE%

echo.
echo === [B] GRAPH  %NAME% seed %SEED% passi %STEPS%  (--graph 1 --graph-attn cudnn)
echo # ===== B: GRAPH seed %SEED% ===== >> "%LOG%"
".venv_yue2\Scripts\python.exe" vendor\yue2_runner.py --style "%STYLE%" --lyrics-file "%LAB%\%NAME%.lyrics.txt" --seed %SEED% --nar-steps %STEPS% --max-seconds 240 --graph 1 --graph-attn cudnn ^
  --model-dir "%UNICA%\models\yue2\YuE2-3B" --vae-dir "%UNICA%\models\yue2\YuE2-Vae" --work "%AB%\graph" --out "%AB%\graph\music.wav" 2>&1 | %TEE%

echo.
echo === confronto token
".venv_yue2\Scripts\python.exe" "%LAB%\pubblicazione_ruibin\compare_tokens.py" "%AB%\eager\semantic.pkl" "%AB%\graph\semantic.pkl" 2>&1 | %TEE%
echo.
echo Fine. Log: %LOG%
pause
