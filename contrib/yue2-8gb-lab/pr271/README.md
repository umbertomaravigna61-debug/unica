# YuE PR #271 (`--low-vram`) on an 8 GB laptop GPU

Test of `multimodal-art-projection/YuE` `main` @ `8ed3f25` (merge of PR #271), run on 31 October 2026 on the same machine and with the same harness as the rest of this lab: RTX 5060 Laptop (7.96 GiB), Windows 11, PyTorch 2.10.0+cu128.

The #271 code was loaded from source in each test process (`scarica_pr271.py` downloads `src/yue2` into `yue_pr271/`; `misura_pr271.py` puts it first on `sys.path`). The installed package was not modified. Source SHA-256 of the tested `src/yue2/*.py`: `673d8346d02d1c02cadb8b3b38235f05b737dcbd2c191499089fef87049f8488`.

## Configurations

| mode | pipeline arguments | extra |
|---|---|---|
| `pr271` | `low_vram=True`, `quantization="none"`, `memory_budget_gib=24` (CLI defaults + `--low-vram`) | none |
| `pr271_fp8` | `low_vram=True`, `quantization="fp8"`, `memory_budget_gib=8` | none |
| `pr271_fp8_mem` | as `pr271_fp8` | memory fraction 0.93 instead of the fixed 2 GiB reserve |
| `pr271_mem` | as `pr271` | memory fraction 0.93 instead of the fixed 2 GiB reserve |

Songs `rock_en` and `pop_it` (see `../canzoni`), seed 11, NAR steps 32 then 4.

## Results

| song | mode | plan | semantic | NAR (first call) |
|---|---|---|---|---|
| rock_en | pr271 | 3187 tok, 28.5 tok/s | OOM at prefill: 744 MiB requested, 5.96 GiB allowed | – |
| rock_en | pr271_mem | 2519 tok, 33.9 tok/s | 5374 tok, 25.5 tok/s | OOM: 4.01 GiB requested (7.40 allowed) |
| rock_en | pr271_fp8 | 3127 tok, 6.8 tok/s | 6000 tok (cap), 7.5 tok/s | OOM: 5.30 GiB requested |
| rock_en | pr271_fp8_mem | 3127 tok, 6.9 tok/s | 6000 tok (cap), 7.6 tok/s | OOM: 5.30 GiB requested (7.40 allowed) |
| pop_it | pr271 | 1231 tok, 30.9 tok/s | 3249 tok, 27.8 tok/s | OOM: 1.43 GiB requested |
| pop_it | pr271_mem | 1377 tok, 30.9 tok/s | 3656 tok, 27.4 tok/s | OOM: 1.77 GiB requested (0.5 GiB free) |
| pop_it | pr271_fp8 | 1373 tok, 6.3 tok/s | 3653 tok, 7.4 tok/s | OOM: 1.76 GiB requested |
| pop_it | pr271_fp8_mem | 1373 tok, 7.1 tok/s | 3653 tok, 7.5 tok/s | OOM: 1.76 GiB requested (0.6 GiB free) |

Every NAR failure has the same stack: `nar.py` `_prefill` → `attention()` → `sdpa`. `YuE2Pipeline.synthesize()` does not pass `query_chunk_size` to `nar.synthesize()`, so on CUDA `attention()` uses `block = len(q)` (the whole sequence in one block). Full traces are in `pr271.jsonl`.

## Files

- `pr271.jsonl`: one record per run: per-stage seconds, tokens, tok/s, peak allocated and reserved memory, GPU telemetry, full error traces.
- `pr271.log`: console log of the runs.
- `misura_pr271.py`: `misura_yue.py` plus the four modes above.
- `lancia_pr271.py`, `scarica_pr271.py`, `PROVA_PR271.bat`, `PROVA_PR271_MEM.bat`: launcher, source download, Windows entry points.

Local paths are replaced with `<LAB>`, `<LAB_ROOT>`, `<UNICA>`, `<MODELS>`.
