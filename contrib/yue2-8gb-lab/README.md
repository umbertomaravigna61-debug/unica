# YuE2-3B on an 8 GB laptop GPU: measurement lab and UNICA opt-in profile

Everything used for the numbers posted in issue #262, so the results can be reproduced and checked.
Measured on one machine only: RTX 5060 Laptop (7.96 GiB), Windows 11, driver 610.88, PyTorch 2.10.0+cu128, Python 3.12.11,
`yue2_infer` 0.1.5 (sources SHA-256 `7967f659a4574e2f8dea8a7aafd33780b5453f6f5be3a0d06ac827d26ec87f82`, recorded in every run).
Diffed against 0.1.6: `pipeline`, `nar`, `modeling_yue2`, `quantization`, `fast`, `protocol` are byte-identical; only the version
string and the CLI (help text, `verify`/`bench`/`eval` subcommands) differ, so the numbers apply to 0.1.6 as well.

## Setup

```
pip install yue2_infer==0.1.5          # torch 2.10 + CUDA 12.8 build
export HF_HOME=/path/to/hf_cache        # must contain m-a-p/YuE2-3B and m-a-p/YuE2-Vae (runs are offline)
```
Optional: `UNICA_VENDOR` (folder with `yue2_runner.py`, default `./unica_recipe`), `LAB_PYTHON` (interpreter for the
per-run subprocesses, default the current one). Close anything else that uses the GPU. On Windows, set the power plan to
high performance: power saving alone cut composition from 52 to 17 tok/s on this laptop.

## Files

| file | what it does |
|---|---|
| `unica_recipe/yue2_runner.py` | the UNICA opt-in profile, as shipped in UNICA (FP8 AR weights, layer-wise NAR prefill, memory fraction 0.93, FP8 CUDA graph with cuDNN attention, prefill in 512-token blocks). Patches are applied in memory; the installed library is not modified. |
| `unica_recipe/yue2_scheggia.py` | "scheggia" (Italian for *splinter*): two algebraic shortcuts in the decode graph, see below. On by default in the runner, `--scheggia 0` turns it off. |
| `misura_yue.py` | one measured run: per-stage seconds, tok/s, peak allocated/reserved memory, memory already allocated at stage start, per-second GPU telemetry, full tracebacks. Modes `ufficiale` (CLI defaults), `ufficiale8` (fp8 + offload_ar, budget 8), `ufficiale8_mem` (same + memory fraction 0.93 only), `unica`, `scheggia`. |
| `lancia_laboratorio.py` | runs the matrix one run per process, resumable (`results/risultati.jsonl`), waits for the GPU to cool below 60 °C. `--soglie 0.84,...` = memory-floor test, `--lungo 360` = 9000-token cap test. |
| `riassunto.py` | builds `riassunto.md` / `riassunto.csv` from `risultati.jsonl`. Failures are kept on purpose. |
| `profila_token.py` | where one decode token's time goes: graph step vs sampling vs `.item()`/loop (sync-instrumented). |
| `profila_pezzi.py` | inside the graph step, with the real modules: FP8 projections (activation quantization vs `_scaled_mm`), bf16 head, rest by difference. Each piece is timed in its own CUDA graph (the PyTorch profiler fails here with `CUPTI_ERROR_INVALID_DEVICE`). |
| `banco_fp4.py` | synthetic GEMV bench on YuE2 shapes: bf16 / FP8 / int4 weight-only (tinygemm) / native FP4. Same weights reused, so absolute numbers are L2-optimistic; ratios are the point. |
| `banco_int8.py` | int8 / int4 weight-only: speed on real shapes, and fidelity of the logits vs the bf16 originals on a real song. |
| `prova_scheggia.py` | teacher-forced check of the shortcuts: 400 real song tokens fed to the original graph and to the scheggia graph, both compared with the bf16 original; includes a run-to-run control. |
| `ruibin_room.py` | a small numpy-only mastering chain (see the P.S.) |
| `prova_runner.py` | end-to-end through `yue2_runner.py`, as the UNICA app calls it (`rock_en 1` / `rock_en 0`). |
| `canzoni/` | the five test songs (style + lyrics). |
| `results/` | raw records of every run (`risultati.jsonl`), the summary tables and the console log. |

## Typical commands

```
python lancia_laboratorio.py --giro completo                 # whole matrix, seeds 11 and 22
python lancia_laboratorio.py --soglie 0.84,0.86,0.88,0.90,0.92
python lancia_laboratorio.py --giro rapido --modi scheggia --canzoni rock_en
python profila_token.py ; python profila_pezzi.py ; python banco_int8.py ; python prova_scheggia.py
python prova_runner.py rock_en 1 ; python prova_runner.py rock_en 0
python riassunto.py
```

## The two shortcuts (scheggia)

1. **Head slice in the semantic phase.** `sampling.distribution` sets every logit outside `MUSIC_END` (151852) and the
   codec range (151853..184620) to -inf. Those 32,769 rows are contiguous in `lm_head`, so the decode graph multiplies only
   that slice (a view, no copy) and leaves the rest of the output at a finite -1e4 (not -inf: CFG computes
   `uncond + s * (cond - uncond)`, and -inf - -inf would give NaN). The mask then makes them -inf exactly as before.
   The ABC planning phase is untouched.
2. **Fused FP8 projections.** q|k|v and gate|up read the same input: their FP8 weights are concatenated once and the
   original modules become views of the fused tensor (no extra memory). One activation quantization and one `_scaled_mm`
   per group instead of one per projection (7 to 4 per layer); the per-projection weight scales are applied per column in
   FP32 before rounding. It is the FP8 counterpart of `GraphAR(fuse_projections=True)`, which only accepts bf16 weights.
   The fused tensors are released before NAR (`yue2_scheggia.release`).

Everything else in the decode step is the 0.1.5 `GraphAR._decode`, line by line.

P.S. This lab has one room that is not on the map. `python lancia_laboratorio.py --room` (or `--room <folder>`), headphones on.
