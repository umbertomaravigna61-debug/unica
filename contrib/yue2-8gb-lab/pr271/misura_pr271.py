"""PROVA DELLA PR #271 di YuE (--low-vram), 31/10/2026: copia di misura_yue.py con tre modi in piu' (pr271, pr271_fp8, pr271_fp8_mem)
che caricano il codice YuE di GitHub dopo la #271 (commit 8ed3f25, cartella yue_pr271/src) solo dentro questo processo.
  pr271          --low-vram con i default della CLI: bf16, memory_budget_gib=24 (riserva fissa 2 GiB)
  pr271_fp8      --low-vram + fp8, memory_budget_gib=8 (riserva fissa 2 GiB)
  pr271_fp8_mem  come pr271_fp8 + SOLO la riserva 2 GiB -> 93% della scheda

Laboratorio YuE2 - una prova misurata (una canzone, un seed, una configurazione).

Modi:
  unica      ricetta di UNICA 1.1, importata da app\\vendor\\yue2_runner.py senza modifiche:
             FP8 in memoria + prefill a strati + grafo CUDA FP8 (cudnn) + prefill del prefisso a blocchi di 512 token
  ufficiale8 yue2_infer 0.1.5 con le SUE opzioni per poca memoria: quantization=fp8, offload_ar=True, memory_budget_gib=8
             (con pesi FP8 la libreria spegne da sola il grafo CUDA -> eager)
  ufficiale8_mem  come ufficiale8 + il solo ritocco della memoria di UNICA (riserva 2 GiB -> 93% della scheda), nient'altro
  ufficiale  yue2_infer 0.1.5 con i default della CLI: quantization=none, offload_ar=False, memory_budget_gib=24

Fasi misurate: load, plan (partitura ABC), semantic (token della canzone), poi per ogni passo NAR richiesto: nar_N e vae_N.
Per ogni fase: secondi, picco VRAM torch (allocata e riservata), e dalla scheda (nvidia-smi ogni secondo): memoria usata,
temperatura, frequenza SM, potenza. Stampa righe JSON a schermo; l'ultima riga ("result") contiene tutto.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

TOKENS_PER_SECOND = 25.0
SAMPLE_RATE = 48000


def emit(**kv):
    print(json.dumps(kv, ensure_ascii=False), flush=True)


class GpuMonitor(threading.Thread):
    """nvidia-smi ogni secondo; ogni campione e' etichettato con la fase in corso."""
    FIELDS = "memory.used,temperature.gpu,clocks.sm,power.draw,utilization.gpu"

    def __init__(self, csv_path: Path):
        super().__init__(daemon=True)
        self.stage = "start"
        self.samples: list[tuple] = []
        self.stop = threading.Event()
        self.csv = open(csv_path, "w", encoding="utf-8")
        self.csv.write("t,stage,mem_used_mib,temp_c,sm_mhz,power_w,util_pct\n")
        self.t0 = time.perf_counter()

    def run(self):
        # UN SOLO processo nvidia-smi che campiona da se' ogni secondo (-lms): non si lancia un programma nuovo a ogni campione.
        # LAB_GPU_MON=0 spegne del tutto il monitoraggio (per verificare che non rallenti la GPU).
        if os.environ.get("LAB_GPU_MON", "1") == "0":
            return
        try:
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "1000"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            return
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            try:
                vals = [float(x) if x.strip() not in ("", "[N/A]") else float("nan") for x in line.strip().split(",")]
            except Exception:
                continue
            t = round(time.perf_counter() - self.t0, 1)
            self.samples.append((self.stage, *vals))
            self.csv.write(f"{t},{self.stage}," + ",".join(str(v) for v in vals) + "\n")
            self.csv.flush()

    def close(self):
        self.stop.set()
        p = getattr(self, "proc", None)
        if p is not None and p.poll() is None:
            p.terminate()

    def summary(self, stage):
        rows = [s for s in self.samples if s[0] == stage]
        if not rows:
            return {}
        col = lambda i: [r[i] for r in rows if r[i] == r[i]]
        mem, temp, sm, pw = col(1), col(2), col(3), col(4)
        return {"gpu_mem_used_max_mib": max(mem) if mem else None, "gpu_temp_max_c": max(temp) if temp else None,
                "gpu_sm_mhz_avg": round(sum(sm) / len(sm)) if sm else None, "gpu_sm_mhz_min": min(sm) if sm else None,
                "gpu_power_avg_w": round(sum(pw) / len(pw), 1) if pw else None, "samples": len(rows)}


def environment(torch):
    env = {"python": sys.version.split()[0], "os": platform.platform(), "cpu": platform.processor(),
           "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)}
    try:
        env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        env["driver"] = None
    try:
        import psutil
        env["ram_total_gib"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    import yue2
    lib = Path(yue2.__file__).parent
    h = hashlib.sha256()
    for f in sorted(lib.glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    env["yue2_dir"] = str(lib)
    env["yue2_sources_sha256"] = h.hexdigest()
    try:
        from importlib.metadata import version
        env["yue2_infer"] = version("yue2_infer")
    except Exception:
        env["yue2_infer"] = None
    return env


def is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["unica", "scheggia", "ufficiale8", "ufficiale8_mem", "ufficiale", "pr271", "pr271_fp8", "pr271_fp8_mem", "pr271_mem"])
    p.add_argument("--song", required=True, help="nome della canzone (file canzoni/<nome>.style.txt e .lyrics.txt)")
    p.add_argument("--songs-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", required=True, help="passi NAR separati da virgola, es. 2,4,32")
    p.add_argument("--max-seconds", type=float, default=240.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--unica-vendor", required=True, help="cartella app\\vendor di UNICA (per la ricetta unica)")
    p.add_argument("--mem-fraction", type=float, default=None,
                   help="solo modo unica: tetto di memoria al posto del 93%% della ricetta (prova della soglia minima)")
    a = p.parse_args()
    if a.mode.startswith("pr271"):          # codice YuE dopo la #271, solo in questo processo (la libreria installata resta quella)
        src = Path(__file__).resolve().parent / "yue_pr271" / "src"
        if not (src / "yue2" / "placement.py").exists():
            raise SystemExit(f"manca il codice della #271 in {src}: lanciare prima scarica_pr271.py")
        sys.path.insert(0, str(src))

    songs = Path(a.songs_dir)
    style = (songs / f"{a.song}.style.txt").read_text(encoding="utf-8").strip()
    lyrics = (songs / f"{a.song}.lyrics.txt").read_text(encoding="utf-8")
    steps = [int(x) for x in a.steps.split(",") if x.strip()]
    max_tokens = int(a.max_seconds * TOKENS_PER_SECOND)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode_label = a.mode if a.mem_fraction is None else f"{a.mode}_f{a.mem_fraction:.2f}"
    tag = f"{a.song}_s{a.seed}_{mode_label}"

    res = {"gpu_monitor": os.environ.get("LAB_GPU_MON", "1") != "0", "mode": mode_label, "mem_fraction": a.mem_fraction, "song": a.song, "seed": a.seed, "steps": steps, "max_tokens": max_tokens,
           "style": style, "lyrics_words": len(lyrics.split()), "stages": {}, "outputs": {}, "success": False}
    mon = GpuMonitor(out / f"{tag}_gpu.csv")
    mon.start()
    peak_rss = {"gib": 0.0}
    stop_rss = threading.Event()

    def rss_watch():
        import psutil
        proc = psutil.Process()
        while not stop_rss.wait(1.0):
            peak_rss["gib"] = max(peak_rss["gib"], proc.memory_info().rss / 2 ** 30)
    threading.Thread(target=rss_watch, daemon=True).start()

    torch = None
    start_alloc = {}

    def stage_begin(name):
        mon.stage = name
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_alloc[name] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)   # memoria gia' occupata a inizio fase
        emit(event="fase", fase=name)
        return time.perf_counter()

    def stage_end(name, t0, **extra):
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        st = {"s": round(time.perf_counter() - t0, 1)}
        if torch is not None and torch.cuda.is_available():
            st["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            st["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
            st["alloc_at_start_gib"] = start_alloc.get(name)
        st.update(extra)
        res["stages"][name] = st
        emit(event="fine_fase", fase=name, **st)

    def stage_fail(name, t0, e):
        st = {"s": round(time.perf_counter() - t0, 1), "error": f"{type(e).__name__}: {str(e)[:400]}", "oom": is_oom(e),
              "traceback": traceback.format_exc()[-2500:], "alloc_at_start_gib": start_alloc.get(name)}
        if torch is not None and torch.cuda.is_available():
            st["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            st["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        res["stages"][name] = st
        emit(event="errore", fase=name, error=st["error"], oom=st["oom"])

    t_all = time.perf_counter()
    try:
        import torch as _torch
        torch = _torch
        res["env"] = environment(torch)
        if a.mode not in ("unica", "scheggia") and os.environ.get("YUE2_GRAPH") == "1":
            raise RuntimeError("YUE2_GRAPH=1 attivo: il modo ufficiale deve girare senza (libreria originale)")
        from yue2 import YuE2Pipeline
        from yue2.protocol import Sampling

        model_src, vae_src = "m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"
        if a.mode in ("unica", "scheggia"):
            sys.path.insert(0, a.unica_vendor)
            import yue2_runner as R                              # la ricetta di UNICA, identica a quella installata
            res["unica_runner_sha256"] = hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest()
            _orig_frac = torch.cuda.set_per_process_memory_fraction
            park_ar = R._patch_for_8gb(torch)
            if a.mem_fraction is not None:                     # la ricetta fissa 0.93: qui lo sostituiamo per cercare il minimo
                torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(a.mem_fraction, device)
            from yue2 import nar as _nar
            kw = dict(quantization="fp8", memory_budget_gib=9, offload_ar=True)
        elif a.mode == "ufficiale8":
            kw = dict(quantization="fp8", memory_budget_gib=8, offload_ar=True)
        elif a.mode == "ufficiale8_mem":
            # libreria ufficiale + UN SOLO ritocco: la riserva fissa di 2 GiB diventa il 93% della scheda (come in UNICA).
            # Niente grafo CUDA FP8, niente prefill a strati o a blocchi: misura cosa porta la sola memoria.
            _orig_frac = torch.cuda.set_per_process_memory_fraction
            torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(0.93, device)
            kw = dict(quantization="fp8", memory_budget_gib=9, offload_ar=True)
        elif a.mode == "pr271":
            kw = dict(quantization="none", memory_budget_gib=24, low_vram=True)
        elif a.mode == "pr271_mem":                 # bf16 --low-vram + SOLO la riserva 2 GiB -> 93%
            _orig_frac = torch.cuda.set_per_process_memory_fraction
            torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(0.93, device)
            kw = dict(quantization="none", memory_budget_gib=24, low_vram=True)
        elif a.mode == "pr271_fp8":
            kw = dict(quantization="fp8", memory_budget_gib=8, low_vram=True)
        elif a.mode == "pr271_fp8_mem":
            _orig_frac = torch.cuda.set_per_process_memory_fraction
            torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(0.93, device)
            kw = dict(quantization="fp8", memory_budget_gib=9, low_vram=True)
        else:
            kw = dict(quantization="none", memory_budget_gib=24, offload_ar=False)
        res["pipeline_args"] = kw
        if a.mode.startswith("pr271"):
            import yue2 as _y
            res["yue2_code"] = {"source": "github multimodal-art-projection/YuE main @ 8ed3f25 (merge of PR #271)", "path": str(Path(_y.__file__).parent)}

        name = "load"
        t0 = stage_begin(name)
        try:
            pipe = YuE2Pipeline.from_pretrained(model_src, vae=vae_src, device="cuda", progress=False, local_files_only=True, **kw)
            if a.mode in ("unica", "scheggia"):
                R._enable_fp8_graph("cudnn")
                if a.mode == "scheggia":                      # scorciatoie algebriche: testa ridotta + proiezioni raccolte
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import scheggia
                    scheggia.enable()
                    emit(event="phase", phase="composing", message="SCHEGGIA on: head slice + fused FP8 projections")
            res["backend"] = getattr(pipe, "backend", None)
            stage_end(name, t0)
        except Exception as e:
            stage_fail(name, t0, e); raise

        name = "plan"
        t0 = stage_begin(name)
        try:
            plan = pipe.plan(style, lyrics, seed=a.seed, cot="full")
            dt = time.perf_counter() - t0
            abc_tokens = len(plan.abc_ids or [])
            (out / f"{tag}_plan.abc.txt").write_text(plan.abc or "", encoding="utf-8")
            pref = plan.prefix
            pref_len = max(len(x) for x in pref) if pref and isinstance(pref[0], (list, tuple)) else (len(pref) if pref else None)
            stage_end(name, t0, tokens=abc_tokens, prefix_tokens=pref_len, truncated=bool(getattr(plan, "truncated", False)),
                      tok_per_s=round(abc_tokens / max(dt, 1e-6), 1))
        except Exception as e:
            stage_fail(name, t0, e); raise

        name = "semantic"
        t0 = stage_begin(name)
        try:
            last = {"t": 0.0, "n": 0}

            def on_token(_phase, _tok):
                last["n"] += 1
                if time.perf_counter() - last["t"] > 5.0:
                    last["t"] = time.perf_counter()
                    emit(event="progresso", fase=name, token=last["n"], tetto=max_tokens,
                         tok_s=round(last["n"] / (time.perf_counter() - t0), 1))
            semantic = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=max_tokens), on_token=on_token)
            n = len(semantic.tokens)
            dt = time.perf_counter() - t0
            pickle.dump(semantic, open(out / f"{tag}_semantic.pkl", "wb"))
            stage_end(name, t0, tokens=n, song_seconds=round(n / TOKENS_PER_SECOND, 1), tok_per_s=round(n / dt, 1),
                      truncated=bool(getattr(semantic, "truncated", False)))
        except Exception as e:
            stage_fail(name, t0, e); raise

        ok_steps = []
        for n_steps in steps:
            name = f"nar_{n_steps}"
            t0 = stage_begin(name)
            try:
                if a.mode in ("unica", "scheggia"):
                    if a.mode == "scheggia":
                        scheggia.release(pipe._model)       # i tensori affiancati non devono restare sulla scheda in rifinitura
                    model = pipe._load_model(for_nar=True)
                    park_ar(model)
                    cfg = pipe.generation_config
                    latents = _nar.synthesize(model, semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed,
                                              steps=n_steps, context=cfg.context, offload_ar=True, query_chunk_size=256
                                              ).detach().float().cpu().numpy()
                    # nessun riferimento locale al modello: altrimenti resta sulla scheda e la rifinitura successiva
                    # ne carica un secondo (in UNICA ogni rifinitura e' un processo nuovo, qui sono in fila nello stesso)
                    model = None
                else:
                    pipe.generation_config = dataclasses.replace(pipe.generation_config, ode_steps=n_steps)
                    latents = pipe.synthesize(semantic)
                stage_end(name, t0)
            except Exception as e:
                stage_fail(name, t0, e)
                torch.cuda.empty_cache()
                continue
            name = f"vae_{n_steps}"
            t0 = stage_begin(name)
            try:
                if a.mode in ("unica", "scheggia"):
                    import gc
                    pipe._model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                audio = pipe.decode(latents)
                import soundfile as sf
                wav = out / f"{tag}_nar{n_steps}.wav"
                sf.write(str(wav), audio, SAMPLE_RATE, format="WAV", subtype="PCM_24")
                res_audio_s = round(len(audio) / SAMPLE_RATE, 2)
                res["outputs"][str(n_steps)] = str(wav)
                # il decoder VAE resta in cache sulla scheda: va liberato prima della rifinitura successiva,
                # altrimenti modello + VAE non stanno insieme in 8 GB (in UNICA ogni rifinitura e' un processo a se')
                import gc
                vae_obj = getattr(pipe, "_vae", None)
                if vae_obj is not None and hasattr(vae_obj, "to"):
                    vae_obj.to("cpu")
                pipe._vae = None
                del audio, latents
                gc.collect()
                torch.cuda.empty_cache()
                stage_end(name, t0, audio_seconds=res_audio_s)
                ok_steps.append(n_steps)
            except Exception as e:
                stage_fail(name, t0, e)
                torch.cuda.empty_cache()
        res["steps_ok"] = ok_steps
        res["success"] = len(ok_steps) == len(steps)
        res["partial"] = 0 < len(ok_steps) < len(steps)
        pipe.close()
    except Exception as e:
        if "error" not in res:
            res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            res["oom"] = is_oom(e)
    finally:
        mon.close()
        stop_rss.set()
        res["total_s"] = round(time.perf_counter() - t_all, 1)
        res["peak_rss_gib"] = round(peak_rss["gib"], 2)
        for k in list(res["stages"]):
            res["stages"][k]["gpu"] = mon.summary(k)
        emit(event="result", result=res)
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
