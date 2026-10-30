"""Dentro il passo del grafo, misurato pezzo per pezzo con i moduli veri di YuE2 (30/10/2026).

Il profilatore di PyTorch su questa scheda non funziona (CUPTI_ERROR_INVALID_DEVICE), quindi si misura in un altro modo,
come fa il grafo vero: ogni pezzo viene registrato in un grafo CUDA con i PESI VERI del modello e le forme di un token con la guida
(2 righe: positivo + negativo), poi si riproduce molte volte cronometrando la GPU.

  proiezioni FP8 intere      le 7 proiezioni di un blocco (q, k, v, o, gate, up, down) x 28 blocchi, come nel modello
    di cui: quantizzazione   solo la parte che converte al volo l'attivazione in FP8 (pad, float, abs, max, scala, conversione)
  testa finale bf16          lm_head 184.704 x 2048 in bf16
  resto (per differenza)     attenzione sulla canzone, norme, rotary, SiLU, somme: passo intero misurato (15,4 ms) meno i pezzi sopra

E per confronto, sulle stesse forme: le 7 proiezioni in int4 "solo pesi" (un solo kernel, pesi da 4 bit) e la testa in FP8 e int4.
Nessuna generazione di musica. Tutto a schermo, circa 1 minuto. Solo a scheda libera.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = os.environ.get("UNICA_VENDOR", str(Path(__file__).resolve().parent / "unica_recipe"))
# HF_HOME: set it to your Hugging Face cache with m-a-p/YuE2-3B and m-a-p/YuE2-Vae
os.environ.setdefault("HF_HUB_OFFLINE", "1")
PASSO_MS = 15.41          # passo del grafo misurato da PROFILA_TOKEN (2 rami, contesto ~1500)
ROWS = 2


def say(s=""):
    print(s, flush=True)


def main():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, VENDOR)
    import yue2_runner as R
    R._patch_for_8gb(torch)
    from yue2 import YuE2Pipeline
    import yue2.quantization as Q

    def gtime(fn, inner=20, outer=30):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
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
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / (inner * outer)          # ms per chiamata

    say(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="fp8", memory_budget_gib=9, offload_ar=True)
    model = pipe._load_model()
    layers = model.model.layers
    L = len(layers)
    names = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
             ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
    mods = [getattr(getattr(layers[0], a), b) for a, b in names]
    say(f"proiezioni del blocco 0: " + ", ".join(f"{b} {m.out_features}x{m.in_features} ({type(m).__name__})" for (a, b), m in zip(names, mods)))

    with torch.inference_mode():
        lin_ms, quant_ms, int4_ms = 0.0, 0.0, 0.0
        for (a, b), m in zip(names, mods):
            x = torch.randn(ROWS, m.in_features, device="cuda", dtype=torch.bfloat16)
            t_full = gtime(lambda: m(x))

            def quant_only():
                p = F.pad(x, (0, 0, 0, (-ROWS) % 16))
                return Q.quantize_tensor(p)
            t_q = gtime(quant_only)
            # int4 solo pesi, stesse forme (pesi casuali: conta solo il tempo)
            N, K = m.out_features, m.in_features
            wp = torch._convert_weight_to_int4pack(torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8), 8)
            sz = torch.ones(K // 128, N, 2, device="cuda", dtype=torch.bfloat16)
            t_i4 = gtime(lambda: torch._weight_int4pack_mm(x, wp, 128, sz))
            del wp, sz
            lin_ms += t_full; quant_ms += t_q; int4_ms += t_i4
            say(f"  {b:9s} FP8 intera {1000 * t_full:7.1f} us (di cui quantizzazione {1000 * t_q:6.1f} us) | int4 {1000 * t_i4:6.1f} us")
        lin_tok, quant_tok, int4_tok = lin_ms * L, quant_ms * L, int4_ms * L

        head = model.lm_head
        h = torch.randn(ROWS, head.in_features, device="cuda", dtype=torch.bfloat16)
        t_head = gtime(lambda: head(h))
        # testa in FP8 (stessa classe della libreria) e in int4, per confronto
        try:                                   # pesi FP8 casuali della stessa forma (contano solo i tempi): pad + quantizza + _scaled_mm
            w8 = torch.randint(0, 120, (head.out_features, head.in_features), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
            ws = torch.ones(1, device="cuda")

            def head8():
                a8, sa = Q.quantize_tensor(F.pad(h, (0, 0, 0, (-ROWS) % 16)))
                return torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=ws, out_dtype=torch.bfloat16)
            t_head8 = gtime(head8)
            del w8
        except Exception as e:
            t_head8 = None; say(f"  testa FP8 non misurabile: {type(e).__name__}: {e}")
        try:
            N, K = head.out_features, head.in_features
            wp = torch._convert_weight_to_int4pack(torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8), 8)
            sz = torch.ones(K // 128, N, 2, device="cuda", dtype=torch.bfloat16)
            t_head4 = gtime(lambda: torch._weight_int4pack_mm(h, wp, 128, sz))
            del wp, sz
        except Exception as e:
            t_head4 = None; say(f"  testa int4 non misurabile: {type(e).__name__}: {str(e)[:120]}")
        torch.cuda.empty_cache()

    rest = max(0.0, PASSO_MS - lin_tok - t_head)
    say("")
    say(f"=== UN TOKEN CON LA GUIDA (2 righe), passo del grafo misurato: {PASSO_MS:.2f} ms ===")
    for nome, v in (("proiezioni FP8 (7 x 28 blocchi)", lin_tok), ("   di cui: quantizzazione al volo delle attivazioni", quant_tok),
                    ("   di cui: moltiplicazioni FP8", lin_tok - quant_tok), ("testa finale bf16 (184.704 voci)", t_head),
                    ("resto: attenzione, norme, rotary, SiLU, somme (per differenza)", rest)):
        say(f"  {nome:64s} {v:6.2f} ms  ({100 * v / PASSO_MS:4.1f}%)")
    say("")
    say("=== COSA SI POTREBBE GUADAGNARE (stime sulle stesse forme) ===")
    say(f"  proiezioni in int4 solo pesi: {int4_tok:6.2f} ms invece di {lin_tok:6.2f} ms  -> risparmio {lin_tok - int4_tok:5.2f} ms")
    if t_head8 is not None:
        say(f"  testa in FP8:                 {t_head8:6.2f} ms invece di {t_head:6.2f} ms  -> risparmio {t_head - t_head8:5.2f} ms")
    if t_head4 is not None:
        say(f"  testa in int4:                {t_head4:6.2f} ms invece di {t_head:6.2f} ms  -> risparmio {t_head - t_head4:5.2f} ms")
    best = PASSO_MS - (lin_tok - int4_tok) - (t_head - min(x for x in (t_head, t_head8, t_head4) if x is not None))
    say(f"  passo stimato con int4 ovunque possibile: {best:5.2f} ms -> circa {1000 / (best + 1.2):.0f} token/s "
        f"(oggi {1000 / (PASSO_MS + 1.2):.0f}, contando ~1,2 ms di scelta del token)")
    say("")
    say("Attenzione: sono stime di velocita'. La qualita' con pesi a 4 bit va verificata ascoltando, prima di qualsiasi uso.")
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
