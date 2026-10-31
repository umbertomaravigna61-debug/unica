"""Prova della PR #271 (YuE --low-vram) sulla RTX 5060 Laptop 8 GB, 31/10/2026.

Stesse canzoni, stessi seed e stessa misura del laboratorio del 30/10. Una prova per processo, tutto a schermo,
risultati in results/pr271.jsonl (ripartibile: le prove gia' registrate si saltano). Alla fine una tabella riassuntiva.

  giro rapido   : rock_en e pop_it, seed 11      (prima risposta in poco tempo)
  giro completo : le 5 canzoni, seed 11 e 22
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.environ.get("LAB_PYTHON", sys.executable)
RESULTS = HERE / "results" / "pr271.jsonl"
LOG = HERE / "results" / "pr271.log"
MODI = {"pr271": "#271 --low-vram, default (bf16, riserva 2 GiB)",
        "pr271_fp8": "#271 --low-vram + fp8, budget 8 (riserva 2 GiB)",
        "pr271_fp8_mem": "#271 --low-vram + fp8, riserva -> 93% (solo questo ritocco)",
        "pr271_mem": "#271 --low-vram, bf16, riserva -> 93% (solo questo ritocco)",
        "pr273": "#273 @ 8daaa70 --low-vram, default (bf16, riserva 2 GiB, nessun ritocco)",
        "pr273_fp8": "#273 @ 8daaa70 --low-vram + fp8, budget 8 (riserva 2 GiB, nessun ritocco)"}
PASSI = "32,4"
GIRI = {"rapido": (["rock_en", "pop_it"], [11]), "completo": (["pop_it", "rock_it", "ballad_en", "rock_en", "coro_it"], [11, 22])}


def say(m=""):
    line = f"[{datetime.now():%H:%M:%S}] {m}" if m else ""
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done():
    if not RESULTS.exists():
        return {}
    out = {}
    for l in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(l); out[(r["mode"], r["song"], r["seed"])] = r
        except Exception:
            pass
    return out


def summary():
    rs = done()
    say("")
    say("=== RIASSUNTO PR #271 sulla RTX 5060 Laptop 8 GB ===")
    say(f"{'canzone':10s} {'seed':>4s}  {'modo':16s} {'esito':38s} {'plan tok/s':>10s} {'sem tok/s':>9s} {'NAR32 s':>8s} {'NAR4 s':>7s}")
    for (mode, song, seed), r in sorted(rs.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        st = r.get("stages", {})
        fail = next((f"FALLITA a {k}: {'OOM' if v.get('oom') else v.get('error', '')[:24]}" for k, v in st.items() if "error" in v), None)
        esito = fail or ("ok" if r.get("success") else "parziale")
        g = lambda k, f: (str(st.get(k, {}).get(f, "")) if k in st else "")
        say(f"{song:10s} {seed:>4d}  {mode:16s} {esito:38s} {g('plan', 'tok_per_s'):>10s} {g('semantic', 'tok_per_s'):>9s} {g('nar_32', 's'):>8s} {g('nar_4', 's'):>7s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--giro", default="rapido", choices=list(GIRI))
    ap.add_argument("--modi", default=",".join(MODI))
    a = ap.parse_args()
    RESULTS.parent.mkdir(exist_ok=True)
    songs, seeds = GIRI[a.giro]
    modi = a.modi.split(",")
    gia = done()
    todo = [(m, s, sd) for s in songs for sd in seeds for m in modi if (m, s, sd) not in gia]
    say(f"PROVA PR #271 | giro {a.giro} | {len(todo)} prove da fare | python {PY}")
    for k, (mode, song, seed) in enumerate(todo, 1):
        say(f"--- prova {k}/{len(todo)}: {song} seed {seed} | {MODI[mode]} | passi NAR {PASSI}")
        out_dir = HERE / "out_pr271" / song / f"seed{seed}" / mode
        cmd = [PY, "-u", str(HERE / "misura_pr271.py"), "--mode", mode, "--song", song, "--songs-dir", str(HERE / "canzoni"),
               "--seed", str(seed), "--steps", PASSI, "--out-dir", str(out_dir), "--unica-vendor", r"<UNICA>\app\vendor"]
        t = time.perf_counter()
        env = dict(os.environ); env.pop("YUE2_GRAPH", None)
        env.update(HF_HOME=r"<MODELS>\yue2\hf_cache", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True", PYTHONIOENCODING="utf-8")    # come lancia_laboratorio.py del 30/10
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        result = None
        for line in p.stdout:
            line = line.rstrip()
            try:
                ev = json.loads(line)
            except ValueError:
                if line.strip():
                    print("   " + line[:200], flush=True)
                continue
            e = ev.get("event")
            if e == "result":
                result = ev["result"]
            elif e == "fine_fase":
                extra = " | ".join(f"{x}={ev[x]}" for x in ("tokens", "tok_per_s", "song_seconds", "peak_alloc_gib", "peak_reserved_gib") if x in ev)
                say(f"   fine {ev['fase']}: {ev.get('s')} s | {extra}")
            elif e == "errore":
                say(f"   ERRORE in {ev['fase']}: {ev.get('error', '')[:160]} {'(OOM)' if ev.get('oom') else ''}")
            elif e in ("fase", "progresso"):
                if e == "fase":
                    say(f"   inizio {ev['fase']}")
                else:
                    print(f"   ... {ev.get('token')} token, {ev.get('tok_s')} tok/s", flush=True)
        p.wait()
        if result is None:
            result = {"mode": mode, "song": song, "seed": seed, "success": False, "error": f"processo terminato senza risultato (codice {p.returncode})", "stages": {}}
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        say(f"   esito: {'OK' if result.get('success') else 'FALLITA'} in {time.perf_counter() - t:.0f} s")
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
