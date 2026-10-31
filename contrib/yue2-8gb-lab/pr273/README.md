# YuE PR #273 on an 8 GB laptop GPU, and what the NAR attention kernel does

Test of `multimodal-art-projection/YuE` branch `codex/issue-portability-fixes` @ `8daaa70` (PR #273), run on 31 October 2026 on the same machine and with the same harness as `../pr271`: RTX 5060 Laptop (7.96 GiB), Windows 11, PyTorch 2.10.0+cu128 (built without FlashAttention), cuDNN 9.10.2.

The #273 code was loaded from source in each test process (`scarica_pr271.py 8daaa70… yue_pr273`; `misura_pr271.py` puts it first on `sys.path`). The installed package was not modified. Source SHA-256 of the tested `src/yue2/*.py`: `779c25f0de9a9e2c24b81632fd3d1187ec83cfc9440f465070d41a8e9981c9bb`.

## 1. Full songs: 4 of 4 complete

| song | mode | plan | semantic | NAR 32 | VAE 32 | NAR 4 | VAE 4 | total |
|---|---|---|---|---|---|---|---|---|
| rock_en s11 | `pr273` (bf16, `--low-vram`) | 2586 tok, 33.0 tok/s | 4815 tok, 25.5 tok/s, peak 5.19 GiB | 107.5 s, peak 5.28 GiB | 7.9 s | 18.7 s, peak 5.53 GiB | 7.3 s | 436.7 s |
| rock_en s11 | `pr273_fp8` (fp8, budget 8, `--low-vram`) | 2387 tok, 10.2 tok/s | 4634 tok, 8.1 tok/s | 101.9 s, peak 5.20 GiB | 7.1 s | 16.4 s | 6.9 s | 953.3 s |
| pop_it s11 | `pr273` | 1506 tok, 31.3 tok/s | 4073 tok, 27.3 tok/s, peak 5.03 GiB | 79.1 s, peak 5.01 GiB | 6.3 s | 14.0 s | 6.0 s | 311.6 s |
| pop_it s11 | `pr273_fp8` | 1525 tok, 7.4 tok/s | 3766 tok, 7.8 tok/s | 73.1 s, peak 4.94 GiB | 6.0 s | 12.4 s | 5.8 s | 796.7 s |

Caveat: the bf16 rock_en plan was 2586 tokens this time. In the #271 run the same seed produced 3187 tokens and the semantic prefill ran out of memory (744 MiB request, 5.96 GiB allowed). Long plans may still hit the AR prefill limit on 8 GB; these four runs did not exercise that case.

## 2. One attention call, kernel by kernel (`banco_attenzione_nar.py`)

Model shapes read from the YuE2-3B `config.json`: 16 query heads, 8 KV heads, head_dim 128. Causal prefill, bf16, random inputs; time averaged over 3 calls after one warm-up; "+GiB" is the extra memory during the call; error against an FP32 reference on the last 256 rows.

| variant | 4862 tok | 7000 tok | 9435 tok |
|---|---|---|---|
| `main`: native GQA, kernel chosen by PyTorch (= MATH) | 90.1 ms, +3.44 GiB | 193.1 ms, +7.02 GiB | 9150 ms, +12.63 GiB |
| native GQA, forced EFFICIENT | No available kernel | No available kernel | No available kernel |
| native GQA, forced cuDNN | 3.1 ms, +0.02 GiB | 5.9 ms, +0.03 GiB | 11.5 ms, +0.04 GiB |
| #273: K/V expanded, kernel chosen by PyTorch (= EFFICIENT) | 7.1 ms, +0.02 GiB | 16.7 ms, +0.03 GiB | 26.2 ms, +0.04 GiB |
| 256-row chunks, native GQA (our profile) | 52.2 ms, +0.30 GiB | 110.9 ms, +0.43 GiB | 208.0 ms, +0.57 GiB |
| 256-row chunks, K/V expanded | 13.3 ms | 22.6 ms | 42.3 ms |
| 1024-row chunks, K/V expanded | 9.7 ms | 19.6 ms | 35.6 ms |

The error against FP32 is the same for every variant to the sixth digit (cosine 0.98659 / 0.99047 / 0.99321): on a single call no kernel is less accurate than another. Full output in `banco_attenzione_nar.txt`.

- The memory-efficient kernel rejects `enable_gqa=True` in this build, so with grouped heads SDPA falls back to MATH, which materializes the full score matrix (12.6 GiB at 9435 tokens on a 7.96 GiB card).
- #273 expands K/V to matching heads, which makes the efficient kernel eligible.
- Chunking the queries bounds the memory but every chunk still goes through MATH: this is why the NAR of our profile (`query_chunk_size=256`) is several times slower than #273 on full songs.
- cuDNN accepts native GQA without expanding K/V, but SDPA does not select it by itself; it has to be requested with `sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`.

## 3. cuDNN on a real song (`prova_cudnn_nar.py`)

Same #273 pipeline (bf16, `--low-vram`), same semantic tokens as the rock_en `pr273` run above (4815 tokens, loaded from its pickle). Only `yue2.nar.sdpa` is swapped: once the #273 function, once cuDNN with native GQA.

| | NAR 32 | NAR 4 | peak | attention calls |
|---|---|---|---|---|
| #273 (efficient) | 106.8 s | 17.6 s | 5.26–5.51 GiB | 1820 / 252 |
| cuDNN, native GQA | **70.4 s (−34 %)** | **12.9 s (−27 %)** | 5.50–5.51 GiB | 1820 / 252, no fallback |

Audio comparison (24-bit WAV, 44.1 kHz):

| | signal / difference | correlation | max abs difference |
|---|---|---|---|
| #273 rerun vs the original `pr273` WAV, 32 steps | 128.9 dB | 1.000000 | 1.2e-7 |
| cuDNN vs #273, 32 steps | 30.3 dB | 0.999539 | 0.361 |
| cuDNN vs #273, 4 steps | 26.7 dB | 0.998943 | 0.517 |
| for scale: #273 4 steps vs 32 steps | 10.6 dB | 0.956880 | 0.889 |

The #273 path is reproducible bit for bit. cuDNN is not identical to it: the two kernels sum in a different order and the small differences accumulate through the ODE steps. Whether that is audible is a listening question, not answered here.

## Files

- `pr273.jsonl`: one record per full-song run (per-stage seconds, tokens, tok/s, peak memory, GPU telemetry); `pr273.log`: console log.
- `misura_pr271.py`, `lancia_pr271.py`, `scarica_pr271.py`, `PROVA_PR273.bat`: harness and entry point (same files as `../pr271`, with the `pr273` modes).
- `banco_attenzione_nar.py` + `.bat` → `banco_attenzione_nar.txt`.
- `prova_cudnn_nar.py` + `.bat` → `prova_cudnn_nar.txt`.

Local paths are replaced with `<LAB>`, `<LAB_ROOT>`, `<UNICA>`, `<MODELS>`.
