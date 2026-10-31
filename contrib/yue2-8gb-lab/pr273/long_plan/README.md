# Long plan from the #271 run, replayed on #273

The rock_en seed 11 plan that made the semantic prefill run out of memory under #271 (3187 ABC tokens, 744 MiB request with 5.96 GiB allowed), replayed on #273 @ `8daaa70` on 31 October 2026. Same machine and conditions as `../README.md`.

## Input

The #271 run saved the plan only as ABC text, not as token IDs. The text was given back to the pipeline as an external score (`SongRequest.abc`, the same path as `--abc-file`). Retokenized, it gives **3187 ABC tokens and a 3491-token full prefix**, the same lengths as the #271 run. The original IDs were not saved, so identity is by text and length, not by ID.

`plan/` is the output of `SymbolicPlan.save`: `plan.json`, `abc_tokens.npy`, `prefix.npy`, `score.abc` and `plan_manifest.json` (SHA-256 of each file). `SymbolicPlan.load("plan")` restores it without retokenizing.

## Conditions

bf16, `low_vram=True`, `memory_budget_gib=24` (CLI defaults + `--low-vram`, so the fixed 2 GiB reserve), semantic cap 6000 tokens, `PYTORCH_ALLOC_CONF=expandable_segments:True`, offline Hugging Face cache. Only the semantic stage was run.

Command, from the lab folder with the #273 sources in `yue_pr273/src` (see `../scarica_pr271.py`):

```
set HF_HOME=<MODELS>\yue2\hf_cache
set HF_HUB_OFFLINE=1
set PYTORCH_ALLOC_CONF=expandable_segments:True
python -u prova_piano_lungo.py
```

## Result: prefill passes, composition completes

| | |
|---|---|
| before composition | 6.88 GiB free of 7.96, nothing allocated |
| first token (prefill done, model placement included) | 23.4 s |
| after prefill | 5.08 GiB allocated, 5.29 GiB reserved, 1.54 GiB free, peak 5.23 GiB |
| composition | 5823 tokens (3:53 of music) in 259.5 s, 22.4 tok/s, ended by itself (not truncated) |

Under #271 the same plan length failed at this point. Full console output in `prova_piano_lungo.txt`, summary in `risultato.json`.
