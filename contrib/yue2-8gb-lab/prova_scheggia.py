"""Prova della SCHEGGIA (scenario A): stessa canzone, stessi token, grafo originale contro grafo con le scorciatoie (30/10/2026).

Per evitare il caso (seed, arrotondamenti) i due grafi NON scelgono i token: si danno in pasto, uno dopo l'altro, i 400 token VERI
della canzone pop_it seed 11 composta oggi dal laboratorio ("teacher forcing"), con la guida CFG come in composizione.
A ogni passo si confrontano i logit delle voci ammesse (fine + 32.768 codici audio) e si misura il tempo del passo.

  RIFERIMENTO = ricetta UNICA di oggi (FP8 + grafo CUDA)        SCHEGGIA = stessa ricetta + testa ridotta + proiezioni raccolte

Esito atteso: stesso token piu' probabile praticamente sempre e divergenza KL molto piu' piccola di quella fra FP8 e originale bf16
misurata prima (0,0037): cioe' la scheggia cambia solo gli ultimi bit degli arrotondamenti, non la musica.
Tutto a schermo, circa 2 minuti. Solo a scheda libera.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = os.environ.get("UNICA_VENDOR", str(Path(__file__).resolve().parent / "unica_recipe"))
# HF_HOME: set it to your Hugging Face cache with m-a-p/YuE2-3B and m-a-p/YuE2-Vae
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CANZONE = HERE / "out" / "pop_it" / "seed11" / "unica" / "pop_it_s11_unica_semantic.pkl"
N = 400


def say(s=""):
    print(s, flush=True)


def main():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, VENDOR)
    sys.path.insert(0, str(HERE))
    import yue2_runner as R
    R._patch_for_8gb(torch)
    from yue2 import YuE2Pipeline
    from yue2.protocol import CODEC_OFFSET
    import yue2.cuda_graph as CG
    import yue2.pipeline as P
    import scheggia

    say(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="fp8", memory_budget_gib=9, offload_ar=True)
    R._enable_fp8_graph("cudnn")
    model = pipe._load_model()
    sem = pickle.load(open(CANZONE, "rb"))
    req = sem.plan.request
    prefixes = [list(sem.plan.prefix)]
    if req.guidance != 1:
        prefixes.append(P.negative_prefix(req, pipe.tokenizer, sem.plan.abc_ids))
    g = float(req.guidance)
    toks = [t + CODEC_OFFSET for t in sem.tokens[:N]]
    say(f"canzone: pop_it seed 11 | partitura {len(prefixes[0])} token | rami {len(prefixes)} (guida {g}) | passi {N}")

    def corsa(nome):
        graph = CG.GraphAR(model, prefixes, N + 2)
        graph.prefill()
        logs, times = [], []
        for t in toks:
            tok = torch.tensor([[t]], device="cuda")
            torch.cuda.synchronize(); a = time.perf_counter()
            out = graph.step(tok)
            torch.cuda.synchronize(); times.append(time.perf_counter() - a)
            cond = out[:1]
            lg = cond if len(prefixes) == 1 else out[1:] + g * (cond - out[1:])    # stessa formula CFG della libreria
            logs.append(lg[0, scheggia.HEAD_FIRST:scheggia.HEAD_LAST + 1].float().cpu())
        graph.close()
        torch.cuda.empty_cache()
        ms = 1000 * sum(times[20:]) / len(times[20:])
        say(f"  {nome:11s} passo del grafo {ms:6.2f} ms")
        return torch.stack(logs), ms

    def confronta(nome, a, b):
        top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        a100, b100 = a.topk(100, -1).indices, b.topk(100, -1).indices
        ov = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a100, b100)) / (100 * len(a100))
        kl = F.kl_div(b.log_softmax(-1), a.log_softmax(-1), log_target=True, reduction="batchmean").item()
        diff = (a - b).abs().max().item()
        say(f"  {nome:34s} top-1 {100 * top1:5.1f}% | top-100 {100 * ov:5.1f}% | KL {kl:.6f} | diff max {diff:.4f}")
        return kl

    # VERITA': il modello originale di Ruibin, pesi bf16 (quelli che l'FP8 ha sostituito), stessi token, un solo passaggio
    say("")
    say("calcolo l'ORIGINALE bf16 (pesi originali portati sulla scheda strato per strato)...")
    import torch.nn as nn
    originals = getattr(model, "_yue2_fp8_originals", {})

    class Streamed(nn.Module):
        def __init__(self, w):
            super().__init__(); self.w = w

        def forward(self, x):
            return F.linear(x, self.w.to(x.device, non_blocking=True))

    fp8_mods = {n: model.get_submodule(n) for n in originals}

    def put(name, mod):
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, mod)
    for n, m in fp8_mods.items():
        m.to("cpu"); put(n, Streamed(originals[n].weight.detach()))
    torch.cuda.empty_cache()
    with torch.inference_mode():
        out = model(torch.tensor([prefixes[0] + toks], device="cuda"), use_cache=False, logits_to_keep=len(toks))
        lg = out.logits[0]
        if len(prefixes) > 1:
            neg = model(torch.tensor([prefixes[1] + toks], device="cuda"), use_cache=False, logits_to_keep=len(toks)).logits[0]
            lg = neg + g * (lg - neg)
        truth = lg[:, scheggia.HEAD_FIRST:scheggia.HEAD_LAST + 1].float().cpu()
        del out, lg
    for n, m in fp8_mods.items():
        m.to("cuda"); put(n, m)
    torch.cuda.empty_cache()

    scheggia.enable()
    scheggia._STATE["phase"] = "semantic"
    say("")
    corse = {}
    for nome, head, fuse in (("riferimento A", False, False), ("riferimento B", False, False),
                             ("solo testa ridotta", True, False), ("solo proiezioni raccolte", False, True),
                             ("scheggia completa", True, True)):
        scheggia._STATE["head"], scheggia._STATE["fuse"] = head, fuse
        corse[nome] = corsa(nome)
    scheggia._STATE["phase"] = None
    ref = corse["riferimento A"][0]
    say("")
    say("=== CONFRONTO con il riferimento A, 400 passi della canzone vera (voci ammesse) ===")
    rumore = confronta("riferimento B (RUMORE DI FONDO)", ref, corse["riferimento B"][0])
    for nome in ("solo testa ridotta", "solo proiezioni raccolte", "scheggia completa"):
        confronta(nome, ref, corse[nome][0])
    say("")
    say("=== DISTANZA DALL'ORIGINALE bf16 di Ruibin (la verita'): chi e' piu' fedele? ===")
    for nome in ("riferimento A", "riferimento B", "solo testa ridotta", "solo proiezioni raccolte", "scheggia completa"):
        confronta(nome, truth, corse[nome][0])
    say("")
    t = {k: v[1] for k, v in corse.items()}
    say(f"=== VELOCITA' del passo: riferimento {t['riferimento A']:.2f} ms | solo testa {t['solo testa ridotta']:.2f} | "
        f"solo proiezioni {t['solo proiezioni raccolte']:.2f} | completa {t['scheggia completa']:.2f} ms ===")
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
