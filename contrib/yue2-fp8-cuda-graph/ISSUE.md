Title: yue2_infer 0.1.5: CUDA graph refused on the FP8 AR path — small patch, 4.5× faster composing on an 8 GB GPU (+ 2 Windows/FP8 findings)

Hi, following up on the exchange with Ruibin Yuan about UNICA's YuE2 integration.

**Problem.** `yue2_infer` 0.1.5 always falls back to the eager decode loop when the AR weights are FP8
(`GraphAR only supports unquantized standard AR Linear layers; use eager for FP8`). On consumer 8 GB cards FP8 is
the only way to run YuE2‑3B, so those users never get the CUDA graph.

**Change.** Two small hunks (patch attached / linked) let `GraphAR` accept the `FP8Linear` modules that the eager
path already uses; `generate_tokens` takes the graph path for FP8 when `YUE2_GRAPH=1`. Default behaviour unchanged.

**Measured** (RTX 5060 Laptop 8 GB, Windows 11, torch 2.10.0+cu128, YuE2‑3B rev `8f24312`, VAE rev `152733a`,
one song, single runs):

| | eager | FP8 CUDA graph (cuDNN attention) |
|---|---|---|
| Lab script, song stage (AR) | 4120 tok in 439.0 s — 9.4 tok/s | 4000 tok in 76.5 s — 52.3 tok/s (5.6×) |
| Lab script, plan stage | 1578 tok in 127.0 s — 12.4 tok/s | 1635 tok in 28.5 s — 57.4 tok/s (4.6×) |
| UNICA runner, plan+song end to end | 3648 tok in 477.9 s — 0.131 s/tok | 3047 tok in 87.1 s — 0.0286 s/tok (**4.58×**) |
| UNICA runner, same seed 8, back to back | 3648 tok in 648.4 s — 0.178 s/tok | 3997 tok in 120.3 s — 0.0301 s/tok (**5.9×**) |
| Peak VRAM, AR stage | 6.88 GB | 6.74 GB |
| Load / NAR 4 steps / VAE | 6.7 s / 42.0 s / 6.4 s | 5.8 s / 28.8 s / 5.3 s |

Full stage‑by‑stage tables, exact revisions, `pip freeze`, reproduction commands, before/after logs, the
patch and the benchmark scripts are here:
https://github.com/umbertomaravigna61-debug/unica/tree/main/contrib/yue2-fp8-cuda-graph

**Output / stability.** Five complete graph runs, no errors, audio normal. Each path is deterministic with a fixed
seed (eager seed 8 gave exactly 3648 tokens on two different days), but the two paths are not identical to each
other: in a same‑seed A/B they diverge at the second token (3648 vs 3997 tokens; 0.2 % position‑wise equal), as
expected from a different attention kernel under sampling. Listening verdict on that pair (same NAR steps) is in
the write‑up, §5d.

**Two independent findings while doing this:**
1. `attention_backend="auto"` selects FlashAttention on the official Windows torch 2.10.0+cu128 build (the
   `hasattr`/schema test passes) but the first capture fails with `USE_FLASH_ATTENTION was not enabled for build`.
   Forcing `cudnn` works. Log in the folder.
2. `nar._offload_ar` raises `generator raised StopIteration` with FP8 weights (it looks for a parameter; `FP8Linear`
   only has buffers). A version that checks `parameters()` and `buffers()` works.

Happy to turn this into a PR against whatever repository holds the `yue2_infer` source (the wheel on
Hugging Face does not point to one), and to run any additional measurement you find useful.

Umberto Maravigna (UNICA)
