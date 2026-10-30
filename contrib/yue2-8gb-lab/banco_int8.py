"""int8 / int4 "solo pesi" al posto dell'FP8 di oggi: velocita' E qualita' (30/10/2026).

PARTE 1 - VELOCITA' (grafi CUDA, forme vere, 2 righe = composizione con la guida):
  per le 7 proiezioni di un blocco e per la testa finale: FP8 di oggi (con quantizzazione al volo), int8 solo pesi
  (torch._weight_int8pack_mm: un kernel, attivazioni in bf16), int4 solo pesi (tinygemm).

PARTE 2 - QUALITA' (prima di ascoltare): sul modello VERO e su una canzone VERA (pop_it, seed 11, composta oggi dal laboratorio)
  si calcolano i logit di 256 posizioni consecutive della canzone, a parita' di contesto, con:
     RIFERIMENTO   pesi originali bf16 della libreria (quelli che l'FP8 ha sostituito)
     FP8 oggi      la ricetta UNICA attuale
     INT8          proiezioni int8 per canale (testa bf16)        INT8+testa  anche la testa in int8
     INT4          proiezioni int4 a gruppi di 128 (testa bf16)   INT4+testa  anche la testa in int4
  e per ognuno, rispetto al riferimento: quante volte il token piu' probabile e' lo stesso (top-1), quanti dei 100 candidati
  coincidono (top-100), e la divergenza KL fra le distribuzioni (0 = identiche).
  Se INT8 si allontana dal riferimento MENO dell'FP8 di oggi, e' un miglioramento anche di qualita', non solo di velocita'.

Tutto a schermo, circa 3-4 minuti. Solo a scheda libera.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = os.environ.get("UNICA_VENDOR", str(Path(__file__).resolve().parent / "unica_recipe"))
# HF_HOME: set it to your Hugging Face cache with m-a-p/YuE2-3B and m-a-p/YuE2-Vae
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROWS = 2
CANZONE = HERE / "out" / "pop_it" / "seed11" / "unica" / "pop_it_s11_unica_semantic.pkl"
N_SONG = 768        # token di canzone dopo la partitura
N_EVAL = 256        # posizioni valutate (le ultime)


def say(s=""):
    print(s, flush=True)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    sys.path.insert(0, VENDOR)
    import yue2_runner as R
    R._patch_for_8gb(torch)
    from yue2 import YuE2Pipeline
    from yue2.protocol import CODEC_OFFSET
    import yue2.quantization as Q

    def gtime(fn, inner=20, outer=30):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(outer):
            g.replay()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / (inner * outer)

    def q_int8(w):                                   # per canale, simmetrico
        w = w.float()
        s = w.abs().amax(1).clamp_min(1e-8) / 127.0
        q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
        return q, s.to(torch.bfloat16)

    def dq_int4(w, group=128):                       # a gruppi di 128, asimmetrico: si restituisce la versione dequantizzata
        N, K = w.shape
        g = w.float().reshape(N, K // group, group)
        mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        sc = ((mx - mn) / 15.0).clamp_min(1e-8)
        q = ((g - mn) / sc).round().clamp(0, 15)
        return (q * sc + mn).reshape(N, K).to(torch.bfloat16)

    say(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="fp8", memory_budget_gib=9, offload_ar=True)
    model = pipe._load_model()
    originals = getattr(model, "_yue2_fp8_originals", {})
    say(f"proiezioni FP8 attive: {len(originals)} (pesi originali bf16 conservati sulla CPU dalla libreria)")
    has_i8 = hasattr(torch, "_weight_int8pack_mm")

    # ---------------- PARTE 1: velocita' ----------------
    say("")
    say("=== PARTE 1: VELOCITA' per token (2 righe), 7 proiezioni x 28 blocchi + testa ===")
    names0 = [n for n in originals if n.startswith("model.layers.0.")]
    t_fp8 = t_i8 = 0.0
    with torch.inference_mode():
        for n in names0:
            m = model.get_submodule(n)
            w = originals[n].weight.to("cuda")
            x = torch.randn(ROWS, m.in_features, device="cuda", dtype=torch.bfloat16)
            a = gtime(lambda: m(x))
            b = None
            if has_i8:
                try:
                    q, s = q_int8(w)
                    b = gtime(lambda: torch._weight_int8pack_mm(x, q, s))
                    del q, s
                except Exception as e:
                    say(f"  int8 non disponibile su CUDA: {type(e).__name__}: {str(e)[:150]}"); has_i8 = False
            del w
            t_fp8 += a; t_i8 += b or 0
            say(f"  {n.split('.')[-1]:9s} FP8 {1000 * a:6.1f} us | int8 " + (f"{1000 * b:6.1f} us" if b else "  n/d"))
        L = len(model.model.layers)
        head = model.lm_head
        h = torch.randn(ROWS, head.in_features, device="cuda", dtype=torch.bfloat16)
        th = gtime(lambda: head(h))
        th8 = None
        if has_i8:
            try:
                q, s = q_int8(head.weight.detach().cpu())          # sulla CPU: in float la testa pesa 1,5 GB
                q, s = q.to("cuda"), s.to("cuda")
                th8 = gtime(lambda: torch._weight_int8pack_mm(h, q, s))
                del q, s
            except Exception as e:
                say(f"  testa int8 non misurabile: {type(e).__name__}: {str(e)[:150]}")
        torch.cuda.empty_cache()
    say(f"  proiezioni per token: FP8 {t_fp8 * L:5.2f} ms" + (f" | int8 {t_i8 * L:5.2f} ms" if t_i8 else ""))
    say(f"  testa finale:         bf16 {th:5.2f} ms" + (f" | int8 {th8:5.2f} ms" if th8 else ""))
    if t_i8:
        risp = (t_fp8 - t_i8) * L + ((th - th8) if th8 else 0)
        say(f"  risparmio stimato con int8 (proiezioni + testa): {risp:4.2f} ms per token su ~16,6 -> circa "
            f"{1000 / (16.6 - risp):.0f} token/s (oggi ~60 a inizio canzone, ~50 di media)")

    # ---------------- PARTE 2: qualita' ----------------
    say("")
    say("=== PARTE 2: QUALITA' sui logit della canzone vera (pop_it, seed 11) ===")
    if not CANZONE.exists():
        say(f"  non trovo {CANZONE}: salto la parte 2"); return 0
    sem = pickle.load(open(CANZONE, "rb"))
    ids = list(sem.plan.prefix) + [t + CODEC_OFFSET for t in sem.tokens[:N_SONG]]
    say(f"  contesto: {len(sem.plan.prefix)} token di partitura + {N_SONG} di canzone; valutate le ultime {N_EVAL} posizioni")
    inp = torch.tensor([ids], device="cuda")

    class Streamed(nn.Module):                       # pesi bf16 tenuti sulla CPU, portati sulla scheda solo durante il calcolo
        def __init__(self, w_cpu):
            super().__init__()
            self.w = w_cpu
            self.out_features, self.in_features = w_cpu.shape

        def forward(self, x):
            return F.linear(x, self.w.to(x.device, non_blocking=True))

    class Int8(nn.Module):
        def __init__(self, w_cpu):
            super().__init__()
            q, s = q_int8(w_cpu.cpu())                             # quantizzazione sulla CPU: niente picchi sulla scheda
            self.q, self.s = q, s
            self.out_features, self.in_features = w_cpu.shape

        def forward(self, x):
            sh = x.shape
            rows = x.reshape(-1, self.in_features)
            q, s = self.q.to(x.device), self.s.to(x.device)
            if has_i8:
                out = torch._weight_int8pack_mm(rows, q, s)
            else:
                out = F.linear(rows, (q.float() * s.float()[:, None]).to(rows.dtype))
            return out.reshape(*sh[:-1], self.out_features)

    def parent_set(name, mod):
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, mod)

    fp8_mods = {n: model.get_submodule(n) for n in originals}
    head_bf16 = model.lm_head
    head_cpu = head_bf16.weight.detach().cpu()

    def build(kind):
        for n, lin in originals.items():
            w = lin.weight.detach()
            if kind == "fp8":
                parent_set(n, fp8_mods[n])
            elif kind == "ref":
                parent_set(n, Streamed(w))
            elif kind == "int8":
                parent_set(n, Int8(w))
            elif kind == "int4":
                parent_set(n, Streamed(dq_int4(w.cpu())))

    def set_head(kind):
        if kind == "bf16":
            model.lm_head = head_bf16
        elif kind == "int8":
            model.lm_head = Int8(head_cpu)
        elif kind == "int4":
            model.lm_head = Streamed(dq_int4(head_cpu))
        torch.cuda.empty_cache()

    def logits():
        with torch.inference_mode():
            out = model(inp, use_cache=False, logits_to_keep=N_EVAL)
            return out.logits[0].float().cpu()

    # per far posto, le proiezioni FP8 lasciano la scheda mentre si provano le altre
    for m in fp8_mods.values():
        m.to("cpu")
    torch.cuda.empty_cache()
    build("ref"); set_head("bf16"); ref = logits()
    results = {}
    for nome, kp, kh in (("FP8 oggi", "fp8", "bf16"), ("INT8", "int8", "bf16"), ("INT8+testa", "int8", "int8"),
                         ("INT4", "int4", "bf16"), ("INT4+testa", "int4", "int4")):
        if kp == "fp8":
            for m in fp8_mods.values():
                m.to("cuda")
        build(kp); set_head(kh)
        lg = logits()
        if kp == "fp8":
            for m in fp8_mods.values():
                m.to("cpu")
        torch.cuda.empty_cache()
        top1 = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
        r100, l100 = ref.topk(100, -1).indices, lg.topk(100, -1).indices
        ov = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(r100, l100)) / (100 * len(r100))
        kl = F.kl_div(lg.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
        results[nome] = (top1, ov, kl)
        say(f"  {nome:11s} top-1 uguale {100 * top1:5.1f}% | top-100 in comune {100 * ov:5.1f}% | KL {kl:.5f}")
    say("")
    f8 = results.get("FP8 oggi", (0, 0, 9))[2]
    for nome in ("INT8", "INT8+testa"):
        if nome in results:
            v = results[nome][2]
            say(f"  {nome}: {'PIU VICINO' if v < f8 else 'PIU LONTANO'} all'originale dell'FP8 di oggi (KL {v:.5f} contro {f8:.5f})")
    say("Questi numeri dicono quanto ci si allontana dal modello originale; l'ultima parola resta all'ascolto.")
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
