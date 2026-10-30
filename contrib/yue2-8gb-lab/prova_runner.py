"""Ricetta UNICA + SCHEGGIA, provata dal programma vero di UNICA (app\\vendor\\yue2_runner.py), come quando si preme "Crea" (30/10/2026).

rock_en, seed 11, rifinitura 4 passi: lo stesso caso misurato oggi dal laboratorio con la ricetta normale
(partitura 60,7 s + canzone 112,1 s a 50,6 token/s, rifinitura 4 passi 95,1 s), cosi' i tempi si confrontano direttamente.
Il WAV va in lab_misure\\out_runner\\ per l'ascolto. Tutto a schermo.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = Path(os.environ.get("UNICA_VENDOR", str(HERE / "unica_recipe"))) / "yue2_runner.py"
PY = Path(os.environ.get("LAB_PYTHON", sys.executable))
song = sys.argv[1] if len(sys.argv) > 1 else "rock_en"
scheggia = sys.argv[2] if len(sys.argv) > 2 else "1"
style = (HERE / "canzoni" / f"{song}.style.txt").read_text(encoding="utf-8").strip()
out_dir = HERE / "out_runner"
out_dir.mkdir(exist_ok=True)
tag = f"{song}_s11_{'scheggia' if scheggia == '1' else 'ricetta'}_nar4"
env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
env.pop("YUE2_GRAPH", None)
cmd = [str(PY), "-u", str(RUNNER), "--style", style, "--lyrics-file", str(HERE / "canzoni" / f"{song}.lyrics.txt"),
       "--seed", "11", "--nar-steps", "4", "--scheggia", scheggia, "--work", str(out_dir / f"{tag}_work"), "--out", str(out_dir / f"{tag}.wav")]
print(f"UNICA yue2_runner | {song} | seed 11 | rifinitura 4 passi | scheggia {'ON' if scheggia == '1' else 'OFF'}", flush=True)
sys.exit(subprocess.call(cmd, env=env, cwd=str(RUNNER.parent)))
