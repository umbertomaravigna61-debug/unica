"""Banco dell'attenzione della RIFINITURA (NAR prefill) di YuE2 sulla 5060 (31/10/2026).

Domanda: cosa fa esattamente il kernel "efficient" della #273 e perche' la nostra rifinitura a blocchi da 256 era ~6x piu' lenta?
Forme vere del modello (lette da config.json): q = [1, teste, L, 128], k/v = [1, teste_kv, L, 128], causale, bf16.
Lunghezze L = 4862 e 9435 (le due che Ruibin ha usato) + 7000.

Per ogni variante: kernel scelto/forzato, tempo medio (3 ripetizioni), memoria in piu' durante la chiamata,
differenza dal calcolo esatto in FP32 sulle ultime 256 righe (quelle che vedono tutta la sequenza).
Tutto a schermo e in results/banco_attenzione_nar.txt.
"""
import glob, json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "banco_attenzione_nar.txt"
_log = open(OUT, "w", encoding="utf-8")


def say(m=""):
    print(m, flush=True); _log.write(m + "\n"); _log.flush()


def model_shape():
    hf = os.environ.get("HF_HOME", r"<MODELS>\yue2\hf_cache")
    for c in glob.glob(os.path.join(hf, "hub", "models--m-a-p--YuE2-3B", "snapshots", "*", "config.json")):
        j = json.load(open(c))
        return j["num_attention_heads"], j["num_key_value_heads"], j.get("head_dim", 128), c
    return 16, 8, 128, "default di modeling_yue2.py (config.json non trovato)"


def run(fn, reps=3):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()                                  # riscaldamento
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        out = fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps, peak, out


def chunked(q, k, v, block, gqa):
    """Come nar.attention() con query_chunk_size=block (la nostra ricetta usa 256)."""
    L = q.shape[2]; outs = []
    for start in range(0, L, block):
        end = min(start + block, L)
        mask = None
        if start:
            mask = (torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None])
        outs.append(F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, :end], v[:, :, :end], attn_mask=mask,
                                                   is_causal=start == 0, enable_gqa=gqa))
    return torch.cat(outs, dim=2)


def main():
    H, KV, D, src = model_shape()
    say(f"GPU {torch.cuda.get_device_name()} | torch {torch.__version__} | FlashAttention compilata: {torch.backends.cuda.is_flash_attention_available()}")
    say(f"forma del modello: {H} teste query, {KV} teste chiavi/valori (GQA x{H // KV}), head_dim {D}  [{src}]")
    torch.manual_seed(0)
    for L in (4862, 7000, 9435):
        say(f"\n=== L = {L} token (rifinitura, prefill causale) ===")
        q = torch.randn(1, H, L, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KV, L, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, KV, L, D, device="cuda", dtype=torch.bfloat16)
        ke, ve = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        R = 256                                      # ultime 256 righe, riferimento esatto FP32
        qs = q[:, :, L - R:].float()
        ref = torch.softmax(qs @ ke.float().transpose(-1, -2) / D ** 0.5, -1) @ ve.float()   # le ultime righe vedono tutto
        variants = [
            ("main oggi: GQA nativo, kernel scelto da PyTorch", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), None),
            ("  GQA nativo, forzato EFFICIENT", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), [SDPBackend.EFFICIENT_ATTENTION]),
            ("  GQA nativo, forzato cuDNN", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), [SDPBackend.CUDNN_ATTENTION]),
            ("  GQA nativo, forzato MATH (quello che resta)", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), [SDPBackend.MATH]),
            ("#273: k/v duplicati, kernel scelto da PyTorch", lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True), None),
            ("  k/v duplicati, forzato EFFICIENT", lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True), [SDPBackend.EFFICIENT_ATTENTION]),
            ("  k/v duplicati, forzato cuDNN", lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True), [SDPBackend.CUDNN_ATTENTION]),
            ("NOSTRA ricetta: blocchi 256, GQA nativo", lambda: chunked(q, k, v, 256, True), None),
            ("  blocchi 256, k/v duplicati (efficient)", lambda: chunked(q, ke, ve, 256, False), None),
            ("  blocchi 1024, k/v duplicati (efficient)", lambda: chunked(q, ke, ve, 1024, False), None),
        ]
        for name, fn, backends in variants:
            try:
                if backends:
                    with sdpa_kernel(backends):
                        ms, peak, out = run(fn)
                else:
                    ms, peak, out = run(fn)
                o = out[:, :, L - R:].float()
                err = (o - ref).abs()
                cos = F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
                say(f"  {name:50s} {ms:8.1f} ms | memoria in piu' {peak:5.2f} GiB | errore medio {err.mean().item():.1e} max {err.max().item():.1e} | coseno {cos:.6f}")
                del out
            except Exception as ex:  # noqa: BLE001
                msg = str(ex).splitlines()[0][:110]
                say(f"  {name:50s} NON VA: {type(ex).__name__}: {msg}")
            torch.cuda.empty_cache()
        del q, k, v, ke, ve, ref, qs
        torch.cuda.empty_cache()
    say("\nNota: per strato. La rifinitura chiama questa attenzione una volta per strato nel prefill e poi a ogni passo ODE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
