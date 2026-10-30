"""Laboratorio YuE2: esegue tutta la matrice di prove, una alla volta, in processi separati.

Ripartibile: le prove gia' registrate in risultati.jsonl (riuscite o fallite) vengono saltate.
Tutto quello che succede viene stampato a schermo e copiato in laboratorio.log.

Uso:  python lancia_laboratorio.py [--giro rapido|completo] [--canzoni a,b] [--modi unica,ufficiale8,ufficiale]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
PY = Path(os.environ.get("LAB_PYTHON", sys.executable))
UNICA_VENDOR = Path(os.environ.get("UNICA_VENDOR", str(HERE / "unica_recipe")))
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
RESULTS = HERE / "risultati.jsonl"
LOG = HERE / "laboratorio.log"

CANZONI = ["pop_it", "rock_it", "ballad_en", "rock_en", "coro_it"]
SEEDS = {"rapido": [11], "completo": [11, 22]}
PASSI = {"unica": "2,4,32", "scheggia": "4", "ufficiale8": "32,4", "ufficiale8_mem": "32,4", "ufficiale": "32"}
NOMI = {"unica": "YuE di UNICA", "scheggia": "YuE di UNICA + SCHEGGIA", "ufficiale8": "YuE ufficiale (opzioni 8 GB)", "ufficiale8_mem": "YuE ufficiale + solo ritocco memoria", "ufficiale": "YuE ufficiale (default)"}
FASI = {"load": "caricamento", "plan": "partitura (plan)", "semantic": "canzone (semantic)"}


def say(msg=""):
    line = f"[{datetime.now():%H:%M:%S}] {msg}" if msg else ""
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fase_it(name):
    if name.startswith("nar_"):
        return f"rifinitura NAR {name[4:]} passi"
    if name.startswith("vae_"):
        return f"decodifica VAE ({name[4:]} passi)"
    return FASI.get(name, name)


def gpu_status():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        mem, temp = (float(x) for x in out.split(","))
        return mem, temp
    except Exception:
        return None, None


def cool_down(max_wait=120, target=60):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        mem, temp = gpu_status()
        if temp is None or temp <= target:
            break
        print(f"\r   attendo che la GPU si raffreddi: {temp:.0f} C   ", end="", flush=True)
        time.sleep(5)
    print("\r" + " " * 60 + "\r", end="")
    mem, temp = gpu_status()
    if mem is not None:
        say(f"   GPU prima della prova: {mem:.0f} MiB usati, {temp:.0f} C")
        if mem > 600:
            say("   ATTENZIONE: qualcos'altro sta usando la GPU (chiudi UNICA, browser con video, giochi).")


def done_keys():
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                keys.add((r["mode"], r["song"], r["seed"]))
            except Exception:
                pass
    return keys


def run_one(mode, song, seed, frac=None, steps=None, max_seconds=None):
    label = mode if frac is None else f"{mode}_f{frac:.2f}"
    if max_seconds is not None:
        label = f"{label}_t{int(max_seconds * 25)}"
    out_dir = HERE / "out" / song / f"seed{seed}" / label
    cmd = [str(PY), "-u", str(HERE / "misura_yue.py"), "--mode", mode, "--song", song, "--songs-dir", str(HERE / "canzoni"),
           "--seed", str(seed), "--steps", steps or PASSI[mode], "--out-dir", str(out_dir), "--unica-vendor", str(UNICA_VENDOR)]
    if frac is not None:
        cmd += ["--mem-fraction", str(frac)]
    if max_seconds is not None:
        cmd += ["--max-seconds", str(max_seconds)]
    env = dict(os.environ)
    env.pop("YUE2_GRAPH", None)
    env.update(HF_HOME=HF_HOME, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
               PYTHONIOENCODING="utf-8")
    result = None
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env, cwd=str(HERE))
    for raw in proc.stdout:
        raw = raw.rstrip()
        try:
            ev = json.loads(raw)
        except Exception:
            if raw.strip():
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write("      | " + raw + "\n")
                if any(k in raw for k in ("Error", "error", "Traceback", "WARNING", "Warning")):
                    print("      | " + raw[:200], flush=True)
            continue
        e = ev.get("event")
        if e == "fase":
            say(f"   -> {fase_it(ev['fase'])}...")
        elif e == "fine_fase":
            extra = []
            if "tokens" in ev:
                extra.append(f"{ev['tokens']} token")
            if "tok_per_s" in ev:
                extra.append(f"{ev['tok_per_s']} token/s")
            if "song_seconds" in ev:
                extra.append(f"brano {ev['song_seconds']} s")
            if ev.get("truncated"):
                extra.append("CHIUSA AL TETTO")
            if "peak_alloc_gib" in ev:
                extra.append(f"picco VRAM {ev['peak_alloc_gib']} GB")
            say(f"      ok in {ev['s']} s" + (" | " + ", ".join(extra) if extra else ""))
        elif e == "phase" and ev.get("message"):                 # messaggi della ricetta UNICA (grafo CUDA, prefill a blocchi)
            say(f"      [UNICA] {ev['message']}")
        elif e == "progresso":
            print(f"      ... {ev['token']} token ({ev['tok_s']} token/s)", flush=True)
        elif e == "errore":
            say(f"      ERRORE in {fase_it(ev['fase'])}: {'MEMORIA ESAURITA (OOM)' if ev.get('oom') else ''} {ev['error'][:200]}")
        elif e == "result":
            result = ev["result"]
    proc.wait()
    if result is None:
        result = {"mode": label, "song": song, "seed": seed, "success": False, "error": f"processo terminato senza risultato (codice {proc.returncode})",
                  "total_s": round(time.time() - t0, 1), "stages": {}}
    result["mode"] = label                        # etichetta distinta (es. unica_t9000) per non sovrascrivere le prove a 6000 token
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--giro", default="completo", choices=list(SEEDS))
    ap.add_argument("--canzoni", default=",".join(CANZONI))
    ap.add_argument("--modi", default="unica,ufficiale8,ufficiale8_mem,ufficiale")
    ap.add_argument("--soglie", default="", help="prova del tetto di memoria: es. 0.84,0.86,0.88,0.90,0.92 (ricetta UNICA, passi 2,4)")
    ap.add_argument("--lungo", type=float, default=0, help="prova del tetto lungo: secondi massimi (360 = 9000 token, il default di YuE ufficiale), ricetta UNICA, rock_en seed 11 e 22, passi 2,4")
    ap.add_argument("--room", nargs="?", const=str(HERE / "out"), default=None, help=argparse.SUPPRESS)   # nothing to see here
    a = ap.parse_args()
    if a.room:
        return subprocess.call([str(PY), str(HERE / "ruibin_room.py"), a.room])
    if a.soglie:
        return soglie(a)
    if a.lungo:
        return lungo(a)
    canzoni = [c for c in a.canzoni.split(",") if c]
    modi = [m for m in a.modi.split(",") if m]
    seeds = SEEDS[a.giro]

    plan = []
    for i, seed in enumerate(seeds):
        for song in canzoni:
            for mode in modi:
                if mode == "ufficiale" and i > 0:
                    continue                        # default ufficiale: una volta per canzone basta (su 8 GB ci aspettiamo OOM)
                plan.append((mode, song, seed))
    done = done_keys()
    todo = [x for x in plan if x not in done]

    say("=" * 70)
    say(f"LABORATORIO YuE2 - giro {a.giro}: {len(plan)} prove, {len(plan) - len(todo)} gia' fatte, {len(todo)} da fare")
    say(f"Python: {PY}")
    say(f"Ricetta UNICA da: {UNICA_VENDOR}")
    say("=" * 70)
    if not PY.exists():
        say("ERRORE: non trovo il Python del venv di YuE."); return 1
    for k, (mode, song, seed) in enumerate(todo, 1):
        say("")
        say(f"PROVA {k}/{len(todo)}: {song} | seed {seed} | {NOMI[mode]} | passi NAR {PASSI[mode]}")
        cool_down()
        r = run_one(mode, song, seed)
        esito = "RIUSCITA" if r.get("success") else ("PARZIALE (passi riusciti: %s)" % r.get("steps_ok") if r.get("partial") else "FALLITA")
        say(f"   => {esito} in {r.get('total_s')} s" + (f" | {r.get('error')}" if r.get("error") else ""))
    say("")
    say("Laboratorio finito. Ora: python riassunto.py  (tabella in riassunto.md e riassunto.csv)")
    return 0


def soglie(a):
    fracs = [float(x) for x in a.soglie.split(",") if x]
    song = a.canzoni.split(",")[0] if a.canzoni != ",".join(CANZONI) else "rock_en"
    seed = 11
    done = done_keys()
    say("=" * 70)
    say(f"PROVA DEL TETTO DI MEMORIA - ricetta UNICA, {song}, seed {seed}, passi NAR 2,4, tetti {fracs}")
    say("=" * 70)
    for f in fracs:
        if (f"unica_f{f:.2f}", song, seed) in done:
            say(f"tetto {f:.0%}: gia' fatto, salto"); continue
        say("")
        say(f"TETTO {f:.0%} ({f * 7.96:.2f} GiB su 7.96)")
        cool_down()
        r = run_one("unica", song, seed, frac=f, steps="2,4")
        esito = "RIUSCITA" if r.get("success") else ("PARZIALE (passi riusciti: %s)" % r.get("steps_ok") if r.get("partial") else "FALLITA")
        say(f"   => tetto {f:.0%}: {esito}")
    say("")
    say("Prova del tetto finita. Ora: python riassunto.py")
    return 0


def lungo(a):
    secs = a.lungo
    song = a.canzoni.split(",")[0] if a.canzoni != ",".join(CANZONI) else "rock_en"
    label = f"unica_t{int(secs * 25)}"
    done = done_keys()
    say("=" * 70)
    say(f"PROVA DEL TETTO LUNGO - ricetta UNICA, {song}, tetto {int(secs * 25)} token ({secs / 60:.0f} minuti), passi NAR 2,4")
    say("Domanda: con il tetto di YuE ufficiale (9000 token) la composizione sta ancora negli 8 GB?")
    say("=" * 70)
    for seed in (11, 22):
        if (label, song, seed) in done:
            say(f"seed {seed}: gia' fatto, salto"); continue
        say("")
        say(f"TETTO LUNGO: {song} | seed {seed} | {int(secs * 25)} token")
        cool_down()
        r = run_one("unica", song, seed, steps="2,4", max_seconds=secs)
        esito = "RIUSCITA" if r.get("success") else ("PARZIALE (passi riusciti: %s)" % r.get("steps_ok") if r.get("partial") else "FALLITA")
        say(f"   => seed {seed}: {esito} in {r.get('total_s')} s" + (f" | {r.get('error')}" if r.get("error") else ""))
    say("")
    say("Prova del tetto lungo finita.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
