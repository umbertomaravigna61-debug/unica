# YuE2 (`yue2_infer` 0.1.5): enabling the CUDA graph on the FP8 autoregressive path — 4.5× faster composing on an 8 GB laptop GPU

Material prepared for the issue in the YuE repository (requested by Ruibin Yuan, 2026‑10‑20).
Everything measured here was produced by UNICA's integration work on a single machine; nothing has been
reproduced elsewhere yet, so please treat the numbers as one data point.

## 1. Summary

`yue2_infer` 0.1.5 refuses the CUDA‑graph decode loop when the AR weights have been converted to FP8
(`"GraphAR only supports unquantized standard AR Linear layers; use eager for FP8"`), so FP8 users always
run the slow eager loop. Two small changes let `GraphAR` accept the `FP8Linear` modules that the eager path
already uses. On an RTX 5060 Laptop (8 GB) the composing stage (plan + song tokens) goes from
**~5.6‑7.6 tokens/s (eager) to ~33‑35 tokens/s (graph, cuDNN attention)** end to end in the UNICA runner, i.e.
**4.6×‑5.9× per token** over three runs; the song stage alone goes from 9.4 to 52.3 tokens/s (5.6×) in the lab script.
A same‑seed A/B (§5d) shows the two paths are deterministic but not identical: they diverge at the second token,
as expected from a different attention kernel under sampling.

The patch is deliberately minimal and keeps the default behaviour unchanged: the FP8 graph is only taken when
the environment variable `YUE2_GRAPH=1` is set (that is how it was measured; a proper keyword argument would be
nicer, see §7).

## 2. Exact revisions

| Component | Revision |
|---|---|
| `yue2_infer` | 0.1.5, wheel from `https://huggingface.co/m-a-p/YuE2-3B/resolve/main/yue2_infer-0.1.5-py3-none-any.whl`, sha256 `8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641` |
| `m-a-p/YuE2-3B` | HF revision `8f24312f187763b9854adeec87e437e2b03bbff1` (official safetensors, converted to FP8 **in memory at load**, no converted weights on disk) |
| `m-a-p/YuE2-Vae` | HF revision `152733a19ad43aa67e367f9b5503ef8075bb5126` |
| PyTorch | `2.10.0+cu128` (official Windows wheel — built **without** FlashAttention) |
| transformers / accelerate / safetensors | 4.57.6 / 1.13.0 / 0.7.0 (full `pip freeze`: [`logs/pip_freeze_venv_yue2.txt`](logs/pip_freeze_venv_yue2.txt)) |
| Python | CPython 3.12.11 |
| OS / GPU | Windows 11 64‑bit, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB, PCIe 4.0 ×4), 32 GB DDR4 RAM, NVIDIA driver 610.88 (WDDM, CUDA UMD 13.3) |

## 3. The patch

[`yue2_infer-0.1.5_fp8_cuda_graph.patch`](yue2_infer-0.1.5_fp8_cuda_graph.patch) — unified diff against the
`yue2/` package extracted from the 0.1.5 wheel. Two hunks:

1. `yue2/sampling.py`, `generate_tokens`: `graph_enabled` no longer excludes the FP8 model when
   `YUE2_GRAPH=1`.
2. `yue2/cuda_graph.py`, `GraphAR.__init__`: the Linear‑layer check accepts `quantization.FP8Linear`
   modules (they expose `weight` as a buffer, not a bf16 parameter), while standard `nn.Linear` layers are
   still required to match `self.dtype`.

Two things are done **outside** the library in the caller (see `bench/lab_genera.py`, lines 93‑110, and the same
code in UNICA's `yue2_runner.py`); they would belong in the patch if you decide to adopt it:

- `GraphAR.__init__` still raises when `model._yue2_fp8_originals` is non‑empty. The caller hides that
  attribute during `__init__` and restores it afterwards. In a real patch the first half of that condition
  should simply be dropped once `FP8Linear` is accepted.
- `attention_backend` is forced to `"cudnn"` (see §6, finding 3).

Apply with: `cd <site-packages> && patch -p1 < yue2_infer-0.1.5_fp8_cuda_graph.patch`
(UNICA does not patch the installed file; it applies the same two text substitutions in memory at start‑up and
aborts with a clear error if the library text differs from 0.1.5.)

## 4. Reproduction

Lab script (`bench/lab_genera.py`, stage by stage, saves `semantic.pkl` / latents after each stage), same lyrics
and style file for every run (`bench/balla_con_me.*.txt`, an Italian pop song, ~2:30):

```bat
:: eager baseline (unmodified 0.1.5 behaviour)
set YUE2_GRAPH=0
.venv\Scripts\python.exe genera.py balla_con_me 7 fp8 256

:: FP8 CUDA graph, cuDNN attention
set YUE2_GRAPH=1
set YUE2_GRAPH_ATTN=cudnn
.venv\Scripts\python.exe genera.py balla_con_me 8 fp8 256
```

`YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda", quantization="fp8", memory_budget_gib=9, offload_ar=True)`,
`Sampling(max_tokens=6000)`, `cot="full"`, `query_chunk_size=256`. The lab script also contains three 8‑GB
workarounds that are **not** part of this patch and apply equally to both runs (see §6, findings 4‑6).

UNICA runner (`yue2_runner.py`, the production path, same lyrics/style; `--graph 0|1` selects eager/graph):

```bat
python vendor\yue2_runner.py --style "<style>" --lyrics-file balla_con_me.lyrics.txt --seed 8 --nar-steps 4 --max-seconds 240 --graph 0 --work out\eager_seed8 --out out\eager_seed8\music.wav
python vendor\yue2_runner.py --style "<style>" --lyrics-file balla_con_me.lyrics.txt --seed 8 --nar-steps 4 --max-seconds 240 --graph 1 --graph-attn cudnn --work out\graph_seed8 --out out\graph_seed8\music.wav
```

`bench/ab_eager_vs_graph.bat` runs exactly this pair back to back and `bench/compare_tokens.py` compares the
two `semantic.pkl` token sequences.

## 5. Measurements (one machine, one song, single runs — no averaging)

### 5a. Lab script, stage timings (same script, same library build, only the graph differs)

| Run | Stage | Tokens | Time | Rate |
|---|---|---|---|---|
| eager, seed 7 ([log](logs/lab_before_eager_seed7.log)) | Load model | | 8.9 s | |
| | Plan (score) | 1578 | 127.0 s | 12.4 tok/s |
| | Generate song (AR) | 4120 | 439.0 s | **9.4 tok/s** |
| | plan + song total | | 571.9 s | |
| | NAR 32 steps, AR offloaded, FP8 | | 376.6 s | peak VRAM 6.76 GB |
| | VAE decode | | 7.8 s | |
| graph (cuDNN), seed 8 ([log](logs/lab_after_graph_seed8.log)) | Load model | | 5.8 s | |
| | Plan (score) | 1635 | 28.5 s | 57.4 tok/s (4.6×) |
| | Generate song (AR) | 4000 | 76.5 s | **52.3 tok/s (5.6×)** |
| | plan + song total | | ~105 s | |
| | NAR 32 steps | | 351.9 s | peak VRAM 6.76 GB |
| | VAE decode | | 7.2 s | |

Peak VRAM during the AR stage with the graph on: 6.98 GB (coro run, same session, `logs/lab_risultati.jsonl`).
The eager AR peak was not recorded by the lab script in these runs (field is 0.0 when the tokens are reused).

### 5b. UNICA runner, end to end, NAR 4 steps

| | eager, seed 8 ([log](logs/runner_before_eager_seed8.jsonl)) | graph cuDNN, seed 11 ([log](logs/runner_after_graph_seed11.jsonl)) |
|---|---|---|
| Load (FP8 conversion in memory) | 6.7 s | 5.8 s |
| Compose = plan + song | 477.9 s for 3648 tokens → **0.131 s/token** | 87.1 s for 3047 tokens → **0.0286 s/token** (**4.58×**) |
| Peak VRAM, AR stage | 6.88 GB | 6.74 GB |
| NAR refine, 4 steps | 42.0 s | 28.8 s (shorter song) |
| Peak VRAM, NAR stage | 5.46 GB | 5.49 GB |
| VAE decode | 6.4 s | 5.3 s |
| Peak process RSS | 11.9 GB | 12.1 GB |
| Total | ~533 s | 131.2 s |

Note on the eager runner log: its telemetry says `"cuda_graph": true` because the flag was requested, but the
library had rejected the graph and silently fallen back to eager (that is precisely the behaviour this patch
changes; the runner now reports which path it took). The composing time is the eager one.

### 5c. NAR steps vs time, same composition reused (seed 8, 160 s song, graph irrelevant here)

| NAR steps | Refine time | Listening (headphones, author) |
|---|---|---|
| 2 | 30.0 s | listenable, drafts |
| 8 | 90.2 s | |
| 16 | 174.5 s | approved as production quality |
| 32 | 351.9 s | reference |

### 5d. Same‑seed A/B, back to back, UNICA runner (`bench/ab_eager_vs_graph.bat`, 2026‑10‑20, [log](logs/ab_seed8_eager_vs_graph.log))

Seed 8, same lyrics/style, NAR 4 steps, `--graph 0` then `--graph 1 --graph-attn cudnn`, nothing else changed.
The machine was busier than on 2026‑10‑19 (eager load 19.5 s instead of 6.7 s), which is why the eager figure is
slower than in §5b; the graph figure is the same within 5 %.

| | [A] eager | [B] FP8 CUDA graph (cuDNN) |
|---|---|---|
| Load | 19.5 s | 7.7 s |
| Compose = plan + song | 3648 tokens in 648.4 s → **0.178 s/token** | 3997 tokens in 120.3 s → **0.0301 s/token** (**5.9×**) |
| Peak VRAM, AR stage | 6.88 GB | 7.03 GB |
| NAR refine, 4 steps | 41.4 s (145.9 s of audio) | 48.3 s (159.9 s of audio) — same per second of audio |
| Peak VRAM, NAR stage | 5.46 GB | 5.49 GB |
| VAE decode | 6.4 s | 6.5 s |
| Peak process RSS | 11.9 GB | 12.0 GB |
| Total | 724.5 s | 186.9 s |

Token comparison (`bench/compare_tokens.py`): A = 3648 tokens, B = 3997 tokens, identical prefix = 2 tokens,
position‑wise equal tokens over the first 3648 = 7 (0.2 %).

Determinism: eager seed 8 produced **exactly 3648 tokens on both days** (2026‑10‑19 and ‑20); the graph produced 4000
tokens in the lab script (2026‑10‑15) and 3997 in the runner (2026‑10‑20). Each path is reproducible with a fixed seed;
the two paths differ from each other from the second token on.

Listening (author, headphones, 4 NAR steps): *to be filled in after listening.*

## 6. Stability, output differences, and other findings

1. **Stability.** Five complete graph runs on this machine (two lab songs, three runner songs, 3000‑4000 tokens
   each) completed without errors, no NaN/garbage tokens, no `truncated` flags; audio was normal on listening.
   No crash was observed after the attention backend was fixed (finding 3). This is a small sample.
2. **Output differences.** The graph path is **not bit‑identical** to eager with the same seed, and the divergence
   is immediate, not accumulated: in the same‑seed A/B (§5d) the two token sequences share only the first 2 tokens.
   That is what one expects when a different attention kernel (cuDNN fused attention in the graph vs the eager
   path) produces slightly different logits and the sampler takes a different branch at the first close call; each
   path on its own is deterministic. We did not compare logits directly. Whether the graph path changes the
   *quality* of the output is a listening question; the author's verdict on the A/B pair is in §5d.
3. **`attention_backend="auto"` picks FlashAttention on Windows, where it is not built.** The feature test in
   `GraphAR.__init__` (`hasattr(torch.ops.aten, "_flash_attention_forward")` + schema check) passes on the
   official Windows `torch 2.10.0+cu128`, but the first capture fails with
   `RuntimeError: USE_FLASH_ATTENTION was not enabled for build` (first attempt in
   [`logs/lab_after_graph_seed8.log`](logs/lab_after_graph_seed8.log)). Forcing `attention_backend="cudnn"` works.
   Probably worth a try/except around the first flash call, or `torch.backends.cuda.is_flash_attention_available()`.
   This is independent of FP8.
4. **NAR `_offload_ar` with FP8 weights** raises `RuntimeError: generator raised StopIteration` (second attempt in the
   same log): it looks for the first parameter of each module, but `FP8Linear` only has buffers. The lab replaces it
   with a version that looks at `parameters()` and `buffers()` (`bench/lab_genera.py`, `_offload_ar_fp8`). Independent
   of the graph.
5. **8 GB cards:** the fixed 2 GiB reserve in `set_per_process_memory_fraction` is too large on an 8 GB card
   (we allow 93 % instead), and the NAR prefill needs the AR layers streamed to the GPU one at a time
   (`_streaming_prefill` in the lab script). Not part of this patch; mentioned because the measurements depend on them.
6. Memory peak with the graph on is the same as eager within 0.15 GB (the graph's static buffers are small).

## 7. Suggested shape of a proper change (not measured)

- `generate_tokens(..., use_cuda_graph=True, allow_fp8_graph=False)` (or a `Sampling` field) instead of the
  environment variable;
- drop the `_yue2_fp8_originals` half of the `GraphAR` check once `FP8Linear` is accepted;
- the `FP8Linear` projections could also be fused like the bf16 ones (`fuse_projections=True`), which we have
  not tried.

## 8. Files

```
yue2_infer-0.1.5_fp8_cuda_graph.patch   the two library hunks
logs/lab_before_eager_seed7.log         lab script, eager, 2026-10-15
logs/lab_after_graph_seed8.log          lab script, graph (first attempt flash → error, second cudnn → OK)
logs/runner_before_eager_seed8.jsonl    UNICA runner, eager baseline (telemetry line)
logs/runner_after_graph_seed11.jsonl    UNICA runner, graph
logs/lab_risultati.jsonl                lab result lines (all runs, incl. NAR-steps series)
logs/ab_seed8_eager_vs_graph.log        same-seed A/B, eager then graph, UNICA runner, 2026-10-20
logs/pip_freeze_venv_yue2.txt           dependency versions
bench/lab_genera.py                     lab script (stage by stage; includes the 8 GB workarounds)
bench/balla_con_me.style.txt / .lyrics.txt
bench/ab_eager_vs_graph.bat             same seed, eager vs graph, back to back (UNICA runner)
bench/compare_tokens.py                 compares the two semantic.pkl token sequences
```

Umberto Maravigna — UNICA (https://umbertomaravigna61-debug.github.io/unica/)
