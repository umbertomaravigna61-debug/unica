"""Dove va il tempo di UN token della composizione di YuE2 (ricetta UNICA), 30/10/2026.

Il banco FP4/INT4 ha mostrato che le moltiplicazioni dei blocchi in FP8 avrebbero un tetto di ~250 token/s, ma la composizione va a ~50:
circa 4/5 del tempo di ogni token va altrove. Questo script lo misura sul modello vero, senza cambiare nulla del calcolo:

  passo del grafo CUDA   graph.step(): i 28 blocchi + attenzione sulla canzone + testa finale (184.704 voci), per i 2 rami (CFG)
  distribuzione          sampling.distribution(): maschera, repetition penalty, top-k, top-p (con ORDINAMENTO di tutte le 184.704 voci)
  estrazione             softmax + multinomial + .item() (la sincronizzazione CPU-GPU a ogni token) + resto del ciclo Python

Poi, sugli stessi logit veri, confronta la distribuzione ufficiale con una versione che ordina solo i 100 candidati del top-k
(stessa matematica: dopo il top-k tutte le altre voci sono gia' -inf). Controlla che i risultati coincidano e misura i tempi.

Canzone pop_it, seed 11, 1500 token di composizione (circa 1 minuto di musica). Tutto a schermo. Circa 3 minuti.
Lanciare SOLO a scheda libera.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = os.environ.get("UNICA_VENDOR", str(Path(__file__).resolve().parent / "unica_recipe"))
# HF_HOME: set it to your Hugging Face cache with m-a-p/YuE2-3B and m-a-p/YuE2-Vae
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
N_TOKENS = 1500


def say(s=""):
    print(s, flush=True)


def main():
    import torch
    sys.path.insert(0, VENDOR)
    import yue2_runner as R
    R._patch_for_8gb(torch)
    from yue2 import YuE2Pipeline
    from yue2.protocol import Sampling
    import yue2.sampling as S
    import yue2.cuda_graph as CG

    style = (HERE / "canzoni" / "pop_it.style.txt").read_text(encoding="utf-8").strip()
    lyrics = (HERE / "canzoni" / "pop_it.lyrics.txt").read_text(encoding="utf-8")
    say(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    say("carico YuE2 con la ricetta UNICA...")
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="fp8", memory_budget_gib=9, offload_ar=True)
    R._enable_fp8_graph("cudnn")
    say("partitura (plan), seed 11...")
    t0 = time.perf_counter()
    plan = pipe.plan(style, lyrics, seed=11, cot="full")
    say(f"   ok in {time.perf_counter() - t0:.1f} s, {len(plan.abc_ids or [])} token")

    # --- strumenti di misura: avvolgono le funzioni senza cambiarle -----------------------------------------------------------
    gen = S.generate_tokens                                   # la versione riscritta dalla ricetta (grafo con FP8)
    import inspect
    g = inspect.unwrap(gen).__globals__                       # generate_tokens e' avvolta da @torch.inference_mode(): servono i globali veri
    if "distribution" not in g:
        raise SystemExit("non trovo sampling.distribution nei globali di generate_tokens: " + ", ".join(sorted(k for k in g if not k.startswith("__"))[:40]))
    orig_dist = g["distribution"]
    orig_step = CG.GraphAR.step
    rec = {"step": [], "dist": [], "loop": []}
    samples = []                                              # logit veri da riusare nel confronto (alcuni passi)
    last = {"t": None}

    def sync():
        torch.cuda.synchronize()

    def timed_step(self, next_id):
        sync(); t = time.perf_counter()
        out = orig_step(self, next_id)
        sync(); rec["step"].append(time.perf_counter() - t)
        return out

    def timed_dist(logits, sampling, history, step, phase, legacy_off=False):
        now = time.perf_counter()
        if last["t"] is not None:
            rec["loop"].append(now - last["t"])               # da una distribuzione alla successiva = un token intero
        last["t"] = now
        if step in (50, 400, 800, 1200, 1450):
            samples.append((logits.detach().clone(), list(history), step))
        sync(); t = time.perf_counter()
        out = orig_dist(logits, sampling, history, step, phase, legacy_off)
        sync(); rec["dist"].append(time.perf_counter() - t)
        return out

    g["distribution"] = timed_dist
    CG.GraphAR.step = timed_step
    say(f"composizione di {N_TOKENS} token con le misure attive (le sincronizzazioni aggiunte rallentano un poco)...")
    t0 = time.perf_counter()
    sem = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=N_TOKENS))
    tot = time.perf_counter() - t0
    g["distribution"] = orig_dist
    CG.GraphAR.step = orig_step
    n = len(sem.tokens)

    def ms(v):
        v = v[20:] if len(v) > 40 else v                     # salta l'avvio
        return 1000 * sum(v) / max(1, len(v))
    step_ms, dist_ms, loop_ms = ms(rec["step"]), ms(rec["dist"]), ms(rec["loop"])
    rest_ms = loop_ms - step_ms - dist_ms
    say("")
    say(f"=== UN TOKEN (media su {n} token, {n / tot:.1f} token/s con le misure attive) ===")
    for nome, v in (("passo del grafo CUDA (blocchi + attenzione + testa, 2 rami)", step_ms),
                    ("distribuzione (maschera, penalty, top-k, top-p con ordinamento)", dist_ms),
                    ("estrazione + .item() + ciclo Python", rest_ms)):
        say(f"  {nome:66s} {v:7.2f} ms  ({100 * v / loop_ms:4.1f}%)")
    say(f"  {'TOTALE per token':66s} {loop_ms:7.2f} ms  -> {1000 / loop_ms:.1f} token/s")
    # andamento con la lunghezza: il passo del grafo cresce con la memoria della canzone (attenzione)
    st = rec["step"]
    if len(st) > 1200:
        say("  passo del grafo per tratto di canzone: " + ", ".join(
            f"token {a}-{b}: {1000 * sum(st[a:b]) / (b - a):.2f} ms" for a, b in ((20, 220), (600, 800), (1200, min(1400, len(st))))))

    # --- confronto: top-p ordinando solo i candidati del top-k ----------------------------------------------------------------
    def fast_distribution(logits, sampling, history, step, phase, legacy_off=False):
        from yue2.protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE
        scores = logits.clone() if legacy_off else logits.float().clone()
        end = ABC_END if phase == "abc" else MUSIC_END
        allowed = torch.full_like(scores, float("-inf"))
        if phase == "abc":
            allowed[..., :EOD] = 0
        else:
            allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
        allowed[..., end] = 0
        scores = scores + allowed
        if step < sampling.min_tokens:
            scores[..., end] = -torch.inf
        scores = S.window_penalty(scores, history[-sampling.penalty_window:], sampling.repetition_penalty)
        if sampling.temperature != 1:
            scores = scores / sampling.temperature
        values, indices = scores.topk(min(sampling.top_k, scores.shape[-1]))          # gia' ordinati, dal piu' alto
        if sampling.top_p < 1:
            p = values.softmax(-1)
            removed = p.cumsum(-1) - p > sampling.top_p
            removed[..., :3 if legacy_off else 1] = False
            values = values.masked_fill(removed, -torch.inf)
        return torch.full_like(scores, float("-inf")).scatter(-1, indices, values)

    say("")
    say("=== CONFRONTO sulla distribuzione: ufficiale (ordina 184.704 voci) contro solo i candidati del top-k ===")
    smp = Sampling()
    ok_all = True
    for logits, hist, stp in samples:
        a = orig_dist(logits, smp, hist, stp, "semantic")
        b = fast_distribution(logits, smp, hist, stp, "semantic")
        pa, pb = a.softmax(-1), b.softmax(-1)
        same_support = torch.equal(torch.isfinite(a), torch.isfinite(b))
        diff = (pa - pb).abs().max().item()
        ok = same_support and diff < 1e-6
        ok_all &= ok
        reps = 200
        sync(); t = time.perf_counter()
        for _ in range(reps):
            orig_dist(logits, smp, hist, stp, "semantic")
        sync(); ta = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(reps):
            fast_distribution(logits, smp, hist, stp, "semantic")
        sync(); tb = (time.perf_counter() - t) / reps
        say(f"  token {stp:5d}: stesse voci ammesse {same_support}, differenza max di probabilita' {diff:.1e} | "
            f"ufficiale {1000 * ta:.2f} ms, solo top-k {1000 * tb:.2f} ms")
    say(f"  ESITO: {'IDENTICI' if ok_all else 'DIVERSI: non usare la versione veloce'}")

    out = {"tokens": n, "tok_s_measured": round(n / tot, 1), "ms": {"graph_step": round(step_ms, 3), "distribution": round(dist_ms, 3),
           "sample_item_loop": round(rest_ms, 3), "total": round(loop_ms, 3)}, "fast_distribution_identical": ok_all}
    (HERE / "profilo_token.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    say("")
    say("Salvato in profilo_token.json")
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
