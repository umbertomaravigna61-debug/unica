"""Riassunto del laboratorio YuE2: risultati.jsonl -> riassunto.md (tabelle in inglese, per l'issue) e riassunto.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODE_EN = {"unica": "UNICA recipe", "scheggia": "UNICA recipe + head slice + fused FP8 projections", "ufficiale8": "official, fp8+offload_ar, budget 8", "ufficiale8_mem": "official fp8+offload_ar + memory fraction 0.93 only", "ufficiale": "official defaults"}


def load():
    rows = {}
    p = HERE / "risultati.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows[(r["mode"], r["song"], r["seed"])] = r          # the last record wins
            except Exception:
                pass
    return list(rows.values())


def label(mode):
    if mode.startswith("unica_t"):
        return f"UNICA recipe, token cap {int(mode[7:])} ({int(mode[7:]) // 25 // 60} min)"
    if mode.startswith("unica_f"):
        return f"UNICA recipe, memory cap {float(mode[7:]):.0%}"
    return MODE_EN.get(mode, mode)


def g(st, name, key):
    v = st.get(name, {}).get(key)
    return "" if v is None else v


def first_failure(r):
    for name, st in r.get("stages", {}).items():
        if st.get("error"):
            return name, ("OOM" if st.get("oom") else st["error"][:80])
    return None, r.get("error", "")


def main():
    rows = sorted(load(), key=lambda r: (r["song"], r["seed"], r["mode"]))
    flat = []
    for r in rows:
        st = r.get("stages", {})
        steps = sorted({int(k.split("_")[1]) for k in st if k.startswith("nar_")})
        fail_stage, fail_msg = first_failure(r)
        base = {"song": r["song"], "seed": r["seed"], "config": label(r["mode"]),
                "load_s": g(st, "load", "s"), "plan_s": g(st, "plan", "s"), "plan_tokens": g(st, "plan", "tokens"),
                "plan_tok_s": g(st, "plan", "tok_per_s"), "plan_peak_gib": g(st, "plan", "peak_alloc_gib"),
                "semantic_s": g(st, "semantic", "s"), "semantic_tokens": g(st, "semantic", "tokens"),
                "semantic_tok_s": g(st, "semantic", "tok_per_s"), "song_s": g(st, "semantic", "song_seconds"),
                "truncated": g(st, "semantic", "truncated"), "semantic_peak_gib": g(st, "semantic", "peak_alloc_gib"),
                "peak_rss_gib": r.get("peak_rss_gib", ""), "failed_at": fail_stage or "", "failure": fail_msg if fail_stage else ""}
        if not steps:
            flat.append({**base, "nar_steps": "", "nar_s": "", "nar_peak_gib": "", "vae_s": "", "vae_peak_gib": "", "audio_s": ""})
        for n in steps:
            flat.append({**base, "nar_steps": n, "nar_s": g(st, f"nar_{n}", "s"), "nar_peak_gib": g(st, f"nar_{n}", "peak_alloc_gib"),
                         "vae_s": g(st, f"vae_{n}", "s"), "vae_peak_gib": g(st, f"vae_{n}", "peak_alloc_gib"),
                         "audio_s": g(st, f"vae_{n}", "audio_seconds")})
    if flat:
        with open(HERE / "riassunto.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader(); w.writerows(flat)

    md = ["# YuE2-3B on an 8 GB laptop GPU: official yue2_infer 0.1.5 vs UNICA recipe", ""]
    env = next((r.get("env") for r in rows if r.get("env")), None)
    if env:
        md += ["## Environment", "", "| | |", "|---|---|"]
        for k in ("gpu", "gpu_total_gib", "driver", "cuda_runtime", "cudnn", "torch", "python", "os", "cpu", "ram_total_gib",
                  "yue2_infer", "yue2_sources_sha256"):
            md.append(f"| {k} | {env.get(k, '')} |")
        md.append("")
    md += ["## Composition (plan + semantic)", "",
           "| song | seed | config | load s | plan s | plan tok | plan tok/s | semantic s | semantic tok | semantic tok/s | song s | peak alloc GiB (plan / semantic) | result |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = r.get("stages", {})
        fs, fm = first_failure(r)
        res = "ok" if not fs and r.get("success") else ("partial: NAR ok for %s, " % r.get("steps_ok") if r.get("partial") else "") + \
               (f"FAIL at {fs}: {fm}" if fs else ("FAIL: " + str(r.get("error", ""))[:60]))
        md.append(f"| {r['song']} | {r['seed']} | {label(r['mode'])} | {g(st,'load','s')} | {g(st,'plan','s')} | "
                  f"{g(st,'plan','tokens')} | {g(st,'plan','tok_per_s')} | {g(st,'semantic','s')} | {g(st,'semantic','tokens')} | "
                  f"{g(st,'semantic','tok_per_s')} | {g(st,'semantic','song_seconds')} | {g(st,'plan','peak_alloc_gib')} / "
                  f"{g(st,'semantic','peak_alloc_gib')} | {res} |")
    md += ["", "## Acoustic synthesis (NAR) and VAE decoding", "",
           "| song | seed | config | NAR steps | NAR s | NAR peak GiB | VAE s | VAE peak GiB | audio s | result |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for row in flat:
        if row["nar_steps"] == "":
            continue
        ok = "ok" if row["audio_s"] != "" else "FAIL"
        md.append(f"| {row['song']} | {row['seed']} | {row['config']} | {row['nar_steps']} | {row['nar_s']} | {row['nar_peak_gib']} | "
                  f"{row['vae_s']} | {row['vae_peak_gib']} | {row['audio_s']} | {ok} |")
    md += ["", "Failures are kept in the table on purpose. Raw per-run records (including tracebacks and per-second GPU telemetry) "
           "are in risultati.jsonl and out/<song>/seed<N>/<config>/*_gpu.csv.", ""]
    (HERE / "riassunto.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nScritti: {HERE / 'riassunto.md'} e {HERE / 'riassunto.csv'}")


if __name__ == "__main__":
    main()
