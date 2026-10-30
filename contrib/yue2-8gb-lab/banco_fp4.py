"""Banco di prova FP4 / INT4 per la composizione di YuE2 (30/10/2026).

Domanda: la composizione (un token alla volta) e' limitata dalla lettura dei pesi. Se i pesi pesano la meta' (4 bit invece di 8),
quanto piu' veloce va UNA moltiplicazione come quelle di YuE2, sulla RTX 5060 Laptop, con il PyTorch del venv?

Non tocca il modello e non genera musica: misura solo le moltiplicazioni di un blocco del modello, con le forme vere lette
dal config.json di YuE2-3B, in quattro modi:
  bf16   pesi a 16 bit (riferimento)
  fp8    pesi a 8 bit, torch._scaled_mm (come la ricetta UNICA di oggi)
  int4   pesi a 4 bit "solo pesi", kernel tinygemm di PyTorch (_weight_int4pack_mm): non serve hardware FP4
  fp4    FP4 nativo Blackwell (float4_e2m1fn_x2 + _scaled_mm a blocchi da 16): se questo PyTorch su Windows non lo supporta,
         lo dice e passa oltre
Ogni modo e' misurato a righe 1 e 2 (2 = composizione con guida CFG: positivo + negativo insieme).
Tutto a schermo. Durata: circa un minuto. Lanciare SOLO a scheda libera (laboratorio finito, UNICA chiusa).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import torch

HF = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
ITER = 300


def say(s=""):
    print(s, flush=True)


def config():
    hits = glob.glob(os.path.join(HF, "hub", "models--m-a-p--YuE2-3B", "snapshots", "*", "config.json"))
    if not hits:
        hits = glob.glob(os.path.join(HF, "**", "config.json"), recursive=True)
        hits = [h for h in hits if "YuE2-3B" in h or "yue2-3b" in h.lower()]
    if not hits:
        raise SystemExit(f"non trovo il config.json di YuE2-3B sotto {HF}")
    c = json.load(open(hits[0], encoding="utf-8"))
    c = c.get("text_config", c)
    h, i = c["hidden_size"], c["intermediate_size"]
    nh = c["num_attention_heads"]
    kv = c.get("num_key_value_heads", nh)
    hd = c.get("head_dim", h // nh)
    layers = c["num_hidden_layers"]
    vocab = c.get("vocab_size")
    say(f"config: {hits[0]}")
    say(f"YuE2-3B: hidden {h}, intermediate {i}, teste {nh} (kv {kv}, dim {hd}), strati {layers}, vocabolario {vocab}")
    # le moltiplicazioni di un blocco: q, k, v, o, gate, up, down  (N uscite, K ingressi)
    shapes = [("q", nh * hd, h), ("k", kv * hd, h), ("v", kv * hd, h), ("o", h, nh * hd), ("gate", i, h), ("up", i, h), ("down", h, i)]
    return shapes, layers


def timeit(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()                 # come in produzione: grafo CUDA, niente costo di lancio da Python
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()
        run = g.replay
        per = 10
    except Exception:
        run, per = fn, 1
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    n = max(1, ITER // per)
    t0.record()
    for _ in range(n):
        run()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / (n * per) / 1000.0   # secondi per chiamata


def make_bf16(N, K):
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    return lambda x: x @ w.t(), N * K * 2


def make_fp8(N, K):
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    sw = torch.tensor(1.0, device="cuda")
    sx = torch.tensor(1.0, device="cuda")

    def f(x):
        xq = x.to(torch.float8_e4m3fn)
        if xq.shape[0] % 16:                    # _scaled_mm vuole righe multiple di 16
            xq = torch.nn.functional.pad(xq.float(), (0, 0, 0, 16 - xq.shape[0] % 16)).to(torch.float8_e4m3fn)
        return torch._scaled_mm(xq, w.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    return f, N * K


def make_int4(N, K, group=128, tiles=8):
    q = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.int32)
    packed_u8 = ((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8)
    wp = torch._convert_weight_to_int4pack(packed_u8, tiles)
    sz = torch.ones(K // group, N, 2, device="cuda", dtype=torch.bfloat16)
    sz[..., 1] = 8.0
    return lambda x: torch._weight_int4pack_mm(x, wp, group, sz), N * K // 2 + sz.numel() * 2


def make_fp4(N, K):
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("questo PyTorch non ha il tipo float4_e2m1fn_x2")
    w = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sw = torch.ones(N, K // 16, device="cuda").to(torch.float8_e4m3fn)

    def f(x):
        M = 128                                  # i kernel FP4 a blocchi vogliono righe multiple di 128
        xa = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sx = torch.ones(M, K // 16, device="cuda").to(torch.float8_e4m3fn)
        return torch._scaled_mm(xa, w.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    return f, N * K // 2 + N * K // 16


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA non disponibile")
    used = torch.cuda.mem_get_info()
    say(f"GPU: {torch.cuda.get_device_name()} | capacita' {torch.cuda.get_device_capability()} | torch {torch.__version__} | "
        f"memoria libera {used[0] / 2**30:.1f} GiB")
    if used[0] < 6 * 2**30:
        say("ATTENZIONE: la scheda e' occupata da qualcos'altro; i numeri non saranno puliti.")
    shapes, layers = config()
    modi = [("bf16", make_bf16), ("fp8", make_fp8), ("int4", make_int4), ("fp4", make_fp4)]
    risultati = {}
    for M in (1, 2):
        say("")
        say(f"=== righe {M} ({'una sola sequenza' if M == 1 else 'con guida CFG: positivo + negativo'}) ===")
        for nome, mk in modi:
            tot_s, tot_b, err = 0.0, 0, None
            for lab, N, K in shapes:
                try:
                    f, nbytes = mk(N, K)
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    f(x)
                    torch.cuda.synchronize()
                    tot_s += timeit(lambda: f(x))
                    tot_b += nbytes
                except Exception as e:
                    err = f"{lab}: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    break
                finally:
                    torch.cuda.empty_cache()
            if err:
                say(f"  {nome:5s} NON DISPONIBILE  ({err})")
                continue
            gbs = tot_b / tot_s / 1e9
            tok = 1.0 / (tot_s * layers)
            risultati[(nome, M)] = tot_s
            rif = risultati.get(("fp8", M))
            conf = f" | rispetto a fp8: x{rif / tot_s:.2f}" if rif and nome != "fp8" else ""
            say(f"  {nome:5s} un blocco {tot_s * 1e6:8.1f} us | pesi letti a {gbs:6.1f} GB/s | "
                f"tetto teorico solo moltiplicazioni: {tok:6.1f} token/s{conf}")
    say("")
    say("Come leggerlo: 'tetto teorico' conta SOLO le moltiplicazioni dei pesi dei blocchi (niente attenzione, niente testa finale),")
    say("quindi e' piu' alto della velocita' vera (oggi ~50 token/s). Conta il RAPPORTO fra i modi: se int4/fp4 battono fp8 di x1,5-x2,")
    say("vale la pena provarli sul modello vero, e poi ascoltare se la qualita' regge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
