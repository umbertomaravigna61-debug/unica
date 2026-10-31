"""Prova su canzone vera: rifinitura (NAR) della #273 con il suo kernel (efficient, K/V duplicati)
contro cuDNN forzato con GQA nativo (31/10/2026).

Stessi token semantici gia' generati dalla prova #273 (rock_en, seed 11, bf16 --low-vram), quindi la sola
cosa che cambia e' il kernel dell'attenzione nella rifinitura. Per ogni variante e numero di passi:
tempo della rifinitura, tempo del VAE, picco di memoria, quante chiamate di attenzione per kernel,
e confronto dell'audio (differenza massima, rapporto segnale/differenza in dB, correlazione)
con la wav originale della prova #273 e tra le due varianti.
Tutto a schermo e in results/prova_cudnn_nar.txt.
"""
import gc, os, pickle, sys, time, dataclasses
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "yue_pr273" / "src"))
SRC = HERE / "out_pr271" / "rock_en" / "seed11" / "pr273"
OUT = SRC.parent / "cudnn"
OUT.mkdir(exist_ok=True)
LOG = open(HERE / "results" / "prova_cudnn_nar.txt", "w", encoding="utf-8")
SR = 44100


def say(m=""):
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import yue2
from yue2 import YuE2Pipeline
from yue2 import nar as _nar

ORIG_SDPA = _nar.sdpa
COUNT = {}


def sdpa_cudnn(query, key, value, *, attn_mask=None, is_causal=False):
    grouped = query.shape[1] != key.shape[1]
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=grouped)
        COUNT["cudnn"] = COUNT.get("cudnn", 0) + 1
        return out
    except RuntimeError:
        COUNT["ripiego #273"] = COUNT.get("ripiego #273", 0) + 1
        return ORIG_SDPA(query, key, value, attn_mask=attn_mask, is_causal=is_causal)


def sdpa_273(query, key, value, *, attn_mask=None, is_causal=False):
    COUNT["#273 efficient"] = COUNT.get("#273 efficient", 0) + 1
    return ORIG_SDPA(query, key, value, attn_mask=attn_mask, is_causal=is_causal)


def compare(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    d = a - b
    snr = 10 * np.log10(np.sum(a ** 2) / max(np.sum(d ** 2), 1e-30))
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    return f"diff max {np.abs(d).max():.2e} | segnale/diff {snr:6.1f} dB | correlazione {corr:.6f}"


def main():
    say(f"yue2 da {Path(yue2.__file__).parent}")
    say(f"GPU {torch.cuda.get_device_name()} | torch {torch.__version__}")
    semantic = pickle.load(open(SRC / "rock_en_s11_pr273_semantic.pkl", "rb"))
    say(f"token semantici: {len(semantic.tokens)} (rock_en seed 11, dalla prova #273)")
    t = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="none", memory_budget_gib=24, low_vram=True)
    say(f"caricamento {time.perf_counter() - t:.1f} s")
    wavs = {}
    for steps in (32, 4):
        for name, fn in (("#273 efficient", sdpa_273), ("cuDNN GQA nativo", sdpa_cudnn)):
            _nar.sdpa = fn
            COUNT.clear()
            pipe.generation_config = dataclasses.replace(pipe.generation_config, ode_steps=steps)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            latents = pipe.synthesize(semantic)
            torch.cuda.synchronize()
            t_nar = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            t = time.perf_counter()
            audio = pipe.decode(latents)
            t_vae = time.perf_counter() - t
            tag = "efficient" if name.startswith("#") else "cudnn"
            wav = OUT / f"rock_en_s11_{tag}_nar{steps}.wav"
            sf.write(str(wav), audio, SR, format="WAV", subtype="PCM_24")
            wavs[(steps, tag)] = np.asarray(audio)
            vae = getattr(pipe, "_vae", None)
            if vae is not None and hasattr(vae, "to"):
                vae.to("cpu")
            pipe._vae = None
            del latents, audio
            gc.collect(); torch.cuda.empty_cache()
            say(f"\nNAR {steps} passi, {name}: rifinitura {t_nar:6.1f} s | VAE {t_vae:5.1f} s | picco {peak:4.2f} GiB | chiamate {dict(COUNT)}")
            ref, _ = sf.read(str(SRC / f"rock_en_s11_pr273_nar{steps}.wav"))
            say(f"   contro la wav originale della prova #273: {compare(ref, wavs[(steps, tag)])}")
        say(f"   cuDNN contro #273 efficient (stessa sessione):  {compare(wavs[(steps, 'efficient')], wavs[(steps, 'cudnn')])}")
    say(f"\nper confronto, quanto cambia l'audio tra 4 e 32 passi: {compare(wavs[(32, 'efficient')], wavs[(4, 'efficient')])}")
    say(f"wav in {OUT}")
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
