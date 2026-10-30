# YuE2-3B on an 8 GB laptop GPU: official yue2_infer 0.1.5 vs UNICA recipe

## Environment

| | |
|---|---|
| gpu | NVIDIA GeForce RTX 5060 Laptop GPU |
| gpu_total_gib | 7.96 |
| driver | 610.88 |
| cuda_runtime | 12.8 |
| cudnn | 91002 |
| torch | 2.10.0+cu128 |
| python | 3.12.11 |
| os | Windows-11-10.0.26300-SP0 |
| cpu | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| ram_total_gib | 31.7 |
| yue2_infer | 0.1.5 |
| yue2_sources_sha256 | 7967f659a4574e2f8dea8a7aafd33780b5453f6f5be3a0d06ac827d26ec87f82 |

## Composition (plan + semantic)

| song | seed | config | load s | plan s | plan tok | plan tok/s | semantic s | semantic tok | semantic tok/s | song s | peak alloc GiB (plan / semantic) | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ballad_en | 11 | official defaults | 6.1 | 5.3 |  |  |  |  |  |  | 5.93 /  | FAIL at plan: OOM |
| ballad_en | 11 | official, fp8+offload_ar, budget 8 | 6.7 | 9.5 |  |  |  |  |  |  | 5.87 /  | FAIL at plan: OOM |
| ballad_en | 11 | official fp8+offload_ar + memory fraction 0.93 only | 6.0 | 193.8 | 1423 | 7.3 | 719.4 | 5444 | 7.6 | 217.8 | 5.99 / 6.81 | FAIL at nar_32: OOM |
| ballad_en | 11 | UNICA recipe | 6.3 | 36.4 | 1462 | 40.2 | 111.6 | 5300 | 47.5 | 212.0 | 5.97 / 6.49 | ok |
| ballad_en | 22 | official, fp8+offload_ar, budget 8 | 6.1 | 8.3 |  |  |  |  |  |  | 5.87 /  | FAIL at plan: OOM |
| ballad_en | 22 | official fp8+offload_ar + memory fraction 0.93 only | 6.2 | 157.8 | 1147 | 7.3 | 709.0 | 5497 | 7.8 | 219.9 | 5.98 / 6.63 | FAIL at nar_32: OOM |
| ballad_en | 22 | UNICA recipe | 6.1 | 30.8 | 1204 | 39.0 | 100.4 | 5418 | 53.9 | 216.7 | 5.97 / 6.43 | ok |
| coro_it | 11 | official defaults | 6.1 | 5.5 |  |  |  |  |  |  | 5.93 /  | FAIL at plan: OOM |
| coro_it | 11 | official, fp8+offload_ar, budget 8 | 6.4 | 8.3 |  |  |  |  |  |  | 5.89 /  | FAIL at plan: OOM |
| coro_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 6.0 | 213.7 | 1531 | 7.2 | 505.4 | 3597 | 7.1 | 143.9 | 6.02 / 7.02 | FAIL at nar_32: OOM |
| coro_it | 11 | UNICA recipe | 6.1 | 32.9 | 1309 | 39.8 | 62.3 | 3276 | 52.6 | 131.0 | 6.02 / 6.49 | ok |
| coro_it | 22 | official, fp8+offload_ar, budget 8 | 6.2 | 7.9 |  |  |  |  |  |  | 5.89 /  | FAIL at plan: OOM |
| coro_it | 22 | UNICA recipe | 6.2 | 31.2 | 1254 | 40.2 | 59.8 | 3189 | 53.3 | 127.6 | 6.02 / 6.49 | ok |
| pop_it | 11 | UNICA recipe + head slice + fused FP8 projections | 6.0 | 32.4 | 1415 | 43.7 | 57.5 | 3738 | 65.1 | 149.5 | 5.99 / 6.49 | ok |
| pop_it | 11 | official defaults | 6.1 | 5.8 |  |  |  |  |  |  | 5.93 /  | FAIL at plan: OOM |
| pop_it | 11 | official, fp8+offload_ar, budget 8 | 5.9 | 8.1 |  |  |  |  |  |  | 5.88 /  | FAIL at plan: OOM |
| pop_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 5.8 | 145.3 | 1373 | 9.4 | 452.6 | 3653 | 8.1 | 146.1 | 6.0 / 6.84 | FAIL at nar_32: OOM |
| pop_it | 11 | UNICA recipe | 5.7 | 28.7 | 1312 | 45.7 | 66.5 | 3493 | 52.5 | 139.7 | 5.99 / 6.49 | ok |
| pop_it | 22 | official, fp8+offload_ar, budget 8 | 6.3 | 8.1 |  |  |  |  |  |  | 5.88 /  | FAIL at plan: OOM |
| pop_it | 22 | official fp8+offload_ar + memory fraction 0.93 only | 5.7 | 104.3 | 1237 | 11.9 | 445.6 | 3372 | 7.6 | 134.9 | 5.99 / 6.73 | FAIL at nar_32: OOM |
| pop_it | 22 | UNICA recipe | 6.1 | 35.3 | 1458 | 41.2 | 66.2 | 3570 | 53.9 | 142.8 | 5.99 / 6.49 | ok |
| rock_en | 11 | UNICA recipe + head slice + fused FP8 projections | 6.2 | 59.9 | 3107 | 51.9 | 86.8 | 5288 | 60.9 | 211.5 | 5.97 / 6.79 | ok |
| rock_en | 11 | official defaults | 6.2 | 5.9 |  |  |  |  |  |  | 5.93 /  | FAIL at plan: OOM |
| rock_en | 11 | official, fp8+offload_ar, budget 8 | 6.3 | 7.7 |  |  |  |  |  |  | 5.87 /  | FAIL at plan: OOM |
| rock_en | 11 | official fp8+offload_ar + memory fraction 0.93 only | 6.3 | 385.6 | 3127 | 8.1 | 0.0 |  |  |  | 6.04 / 6.71 | FAIL at semantic: OOM |
| rock_en | 11 | UNICA recipe | 6.0 | 60.7 | 2875 | 47.4 | 112.1 | 5671 | 50.6 | 226.8 | 5.96 / 6.76 | ok |
| rock_en | 11 | UNICA recipe, memory cap 84% | 5.6 | 60.9 | 3133 | 51.5 | 0.1 |  |  |  | 5.96 / 6.55 | FAIL at semantic: OOM |
| rock_en | 11 | UNICA recipe, memory cap 86% | 6.1 | 58.8 | 2804 | 47.7 | 0.9 |  |  |  | 5.96 / 6.71 | FAIL at semantic: OOM |
| rock_en | 11 | UNICA recipe, memory cap 88% | 6.5 | 60.3 | 2804 | 46.5 | 111.1 | 5574 | 50.2 | 223.0 | 5.96 / 6.76 | partial: NAR ok for [2], FAIL at nar_4: OOM |
| rock_en | 11 | UNICA recipe, memory cap 90% | 6.4 | 70.7 | 3338 | 47.2 | 123.9 | 6000 | 48.4 | 240.0 | 5.96 / 6.86 | ok |
| rock_en | 11 | UNICA recipe, memory cap 92% | 6.8 | 66.5 | 3079 | 46.3 | 122.6 | 5993 | 48.9 | 239.7 | 5.96 / 6.78 | ok |
| rock_en | 11 | UNICA recipe, token cap 9000 (6 min) | 5.9 | 57.3 | 2978 | 52.0 | 125.5 | 5796 | 46.2 | 231.8 | 5.96 / 7.09 | ok |
| rock_en | 22 | official, fp8+offload_ar, budget 8 | 6.3 | 7.8 |  |  |  |  |  |  | 5.87 /  | FAIL at plan: OOM |
| rock_en | 22 | UNICA recipe | 6.1 | 42.5 | 1944 | 45.7 | 112.6 | 6000 | 53.3 | 240.0 | 5.96 / 6.56 | ok |
| rock_en | 22 | UNICA recipe, token cap 9000 (6 min) | 6.1 | 41.9 | 1852 | 44.2 | 116.2 | 5578 | 48.0 | 223.1 | 5.96 / 6.89 | ok |
| rock_it | 11 | official defaults | 6.8 | 6.1 |  |  |  |  |  |  | 5.93 /  | FAIL at plan: OOM |
| rock_it | 11 | official, fp8+offload_ar, budget 8 | 6.3 | 8.0 |  |  |  |  |  |  | 5.88 /  | FAIL at plan: OOM |
| rock_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 6.3 | 321.4 | 2318 | 7.2 | 0.0 |  |  |  | 6.02 / 7.01 | FAIL at semantic: OOM |
| rock_it | 11 | UNICA recipe | 6.2 | 48.9 | 2285 | 46.7 | 109.8 | 5522 | 50.3 | 220.9 | 5.98 / 6.66 | ok |
| rock_it | 22 | official, fp8+offload_ar, budget 8 | 6.1 | 8.2 |  |  |  |  |  |  | 5.88 /  | FAIL at plan: OOM |
| rock_it | 22 | official fp8+offload_ar + memory fraction 0.93 only | 6.2 | 291.3 | 2190 | 7.5 | 0.0 |  |  |  | 6.02 / 6.94 | FAIL at semantic: OOM |
| rock_it | 22 | UNICA recipe | 6.1 | 44.5 | 2060 | 46.3 | 96.5 | 4999 | 51.8 | 200.0 | 5.98 / 6.58 | ok |

## Acoustic synthesis (NAR) and VAE decoding

| song | seed | config | NAR steps | NAR s | NAR peak GiB | VAE s | VAE peak GiB | audio s | result |
|---|---|---|---|---|---|---|---|---|---|
| ballad_en | 11 | official fp8+offload_ar + memory fraction 0.93 only | 4 | 0.4 | 7.29 |  |  |  | FAIL |
| ballad_en | 11 | official fp8+offload_ar + memory fraction 0.93 only | 32 | 1.7 | 7.29 |  |  |  | FAIL |
| ballad_en | 11 | UNICA recipe | 2 | 39.5 | 5.49 | 9.9 | 2.76 | 212.0 | ok |
| ballad_en | 11 | UNICA recipe | 4 | 78.5 | 7.05 | 8.8 | 3.01 | 212.0 | ok |
| ballad_en | 11 | UNICA recipe | 32 | 516.4 | 7.05 | 8.4 | 3.01 | 212.0 | ok |
| ballad_en | 22 | official fp8+offload_ar + memory fraction 0.93 only | 4 | 0.1 | 7.27 |  |  |  | FAIL |
| ballad_en | 22 | official fp8+offload_ar + memory fraction 0.93 only | 32 | 1.7 | 7.33 |  |  |  | FAIL |
| ballad_en | 22 | UNICA recipe | 2 | 37.4 | 5.49 | 9.0 | 2.76 | 216.72 | ok |
| ballad_en | 22 | UNICA recipe | 4 | 75.1 | 7.05 | 7.7 | 3.01 | 216.72 | ok |
| ballad_en | 22 | UNICA recipe | 32 | 532.9 | 7.05 | 7.8 | 3.01 | 216.72 | ok |
| coro_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 4 | 0.1 | 7.24 |  |  |  | FAIL |
| coro_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 32 | 2.0 | 7.24 |  |  |  | FAIL |
| coro_it | 11 | UNICA recipe | 2 | 17.7 | 5.49 | 6.2 | 2.76 | 131.04 | ok |
| coro_it | 11 | UNICA recipe | 4 | 36.8 | 7.05 | 5.1 | 3.01 | 131.04 | ok |
| coro_it | 11 | UNICA recipe | 32 | 235.9 | 7.05 | 5.0 | 3.01 | 131.04 | ok |
| coro_it | 22 | UNICA recipe | 2 | 16.8 | 5.49 | 6.2 | 2.76 | 127.56 | ok |
| coro_it | 22 | UNICA recipe | 4 | 34.8 | 7.05 | 5.1 | 3.01 | 127.56 | ok |
| coro_it | 22 | UNICA recipe | 32 | 222.1 | 7.05 | 5.1 | 3.01 | 127.56 | ok |
| pop_it | 11 | UNICA recipe + head slice + fused FP8 projections | 4 | 39.9 | 5.6 | 7.1 | 3.64 | 149.52 | ok |
| pop_it | 11 | UNICA recipe + head slice + fused FP8 projections | 32 | 307.4 | 7.05 | 6.2 | 3.01 | 149.52 | ok |
| pop_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 4 | 0.1 | 7.21 |  |  |  | FAIL |
| pop_it | 11 | official fp8+offload_ar + memory fraction 0.93 only | 32 | 1.9 | 7.21 |  |  |  | FAIL |
| pop_it | 11 | UNICA recipe | 2 | 19.6 | 5.49 | 6.6 | 2.76 | 139.72 | ok |
| pop_it | 11 | UNICA recipe | 4 | 39.2 | 7.05 | 5.7 | 3.01 | 139.72 | ok |
| pop_it | 11 | UNICA recipe | 32 | 274.3 | 7.05 | 6.0 | 3.01 | 139.72 | ok |
| pop_it | 22 | official fp8+offload_ar + memory fraction 0.93 only | 4 | 0.0 | 7.18 |  |  |  | FAIL |
| pop_it | 22 | official fp8+offload_ar + memory fraction 0.93 only | 32 | 1.5 | 7.18 |  |  |  | FAIL |
| pop_it | 22 | UNICA recipe | 2 | 19.7 | 5.49 | 6.7 | 2.76 | 142.8 | ok |
| pop_it | 22 | UNICA recipe | 4 | 40.5 | 7.05 | 5.5 | 3.01 | 142.8 | ok |
| pop_it | 22 | UNICA recipe | 32 | 263.3 | 7.05 | 5.6 | 3.01 | 142.8 | ok |
| rock_en | 11 | UNICA recipe + head slice + fused FP8 projections | 4 | 92.4 | 6.28 | 9.6 | 3.64 | 211.52 | ok |
| rock_en | 11 | UNICA recipe + head slice + fused FP8 projections | 32 | 633.9 | 7.05 | 8.6 | 3.01 | 211.52 | ok |
| rock_en | 11 | UNICA recipe | 2 | 46.8 | 5.49 | 9.6 | 2.76 | 226.84 | ok |
| rock_en | 11 | UNICA recipe | 4 | 95.1 | 7.05 | 8.3 | 3.01 | 226.84 | ok |
| rock_en | 11 | UNICA recipe | 32 | 648.5 | 7.05 | 8.2 | 3.01 | 226.84 | ok |
| rock_en | 11 | UNICA recipe, memory cap 88% | 2 | 48.0 | 5.49 | 10.2 | 2.76 | 222.96 | ok |
| rock_en | 11 | UNICA recipe, memory cap 88% | 4 | 3.4 | 6.24 |  |  |  | FAIL |
| rock_en | 11 | UNICA recipe, memory cap 90% | 2 | 57.7 | 5.61 | 10.5 | 2.76 | 240.0 | ok |
| rock_en | 11 | UNICA recipe, memory cap 90% | 4 | 109.2 | 7.05 | 9.7 | 3.01 | 240.0 | ok |
| rock_en | 11 | UNICA recipe, memory cap 92% | 2 | 56.4 | 5.56 | 10.8 | 2.76 | 239.72 | ok |
| rock_en | 11 | UNICA recipe, memory cap 92% | 4 | 111.3 | 7.05 | 9.8 | 3.01 | 239.72 | ok |
| rock_en | 11 | UNICA recipe, token cap 9000 (6 min) | 2 | 55.5 | 5.49 | 11.3 | 2.76 | 231.84 | ok |
| rock_en | 11 | UNICA recipe, token cap 9000 (6 min) | 4 | 108.3 | 7.05 | 9.1 | 3.01 | 231.84 | ok |
| rock_en | 22 | UNICA recipe | 2 | 48.6 | 5.49 | 9.7 | 2.76 | 240.0 | ok |
| rock_en | 22 | UNICA recipe | 4 | 95.5 | 7.05 | 8.5 | 3.01 | 240.0 | ok |
| rock_en | 22 | UNICA recipe | 32 | 684.3 | 7.05 | 8.7 | 3.01 | 240.0 | ok |
| rock_en | 22 | UNICA recipe, token cap 9000 (6 min) | 2 | 47.0 | 5.49 | 10.1 | 2.76 | 223.12 | ok |
| rock_en | 22 | UNICA recipe, token cap 9000 (6 min) | 4 | 92.4 | 7.05 | 9.2 | 3.01 | 223.12 | ok |
| rock_it | 11 | UNICA recipe | 2 | 46.5 | 5.49 | 10.0 | 2.76 | 220.88 | ok |
| rock_it | 11 | UNICA recipe | 4 | 93.5 | 7.05 | 9.4 | 3.01 | 220.88 | ok |
| rock_it | 11 | UNICA recipe | 32 | 670.0 | 7.05 | 9.3 | 3.01 | 220.88 | ok |
| rock_it | 22 | UNICA recipe | 2 | 35.9 | 5.49 | 8.2 | 2.76 | 199.96 | ok |
| rock_it | 22 | UNICA recipe | 4 | 72.9 | 7.05 | 7.3 | 3.01 | 199.96 | ok |
| rock_it | 22 | UNICA recipe | 32 | 498.7 | 7.05 | 7.2 | 3.01 | 199.96 | ok |

Failures are kept in the table on purpose. Raw per-run records (including tracebacks and per-second GPU telemetry) are in risultati.jsonl and out/<song>/seed<N>/<config>/*_gpu.csv.
