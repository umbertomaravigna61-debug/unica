"""Ultima prova chiesta da Ruibin (#262, 31/10/2026): il piano lungo che falliva con la #271, ripassato sulla #273.

Il piano di rock_en seed 11 della prova #271 (3187 token ABC) e' salvato solo come testo ABC
(out_pr271/rock_en/seed11/pr271/rock_en_s11_pr271_plan.abc.txt). Lo diamo alla #273 come partitura esterna
(SongRequest.abc, la stessa via di --abc-file), controlliamo che i token tornino 3187 e il prefisso 3491,
salviamo il piano con SymbolicPlan.save (JSON + NPY + manifest), poi lanciamo SOLO la composizione (semantic)
con le stesse condizioni della #271: bf16, --low-vram, budget 24 (riserva fissa di 2 GiB), tetto 6000 token.
Se va in errore di memoria: lunghezza del prefisso, prima allocazione fallita, memoria libera/allocata/riservata.
Tutto a schermo e in results/prova_piano_lungo.txt.
"""
import json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "yue_pr273" / "src"))
ABC = HERE / "out_pr271" / "rock_en" / "seed11" / "pr271" / "rock_en_s11_pr271_plan.abc.txt"
SONGS = HERE / "canzoni"
OUT = HERE / "out_pr271" / "rock_en" / "seed11" / "pr273_piano_lungo"
OUT.mkdir(exist_ok=True)
LOG = open(HERE / "results" / "prova_piano_lungo.txt", "w", encoding="utf-8")
GIB = 2 ** 30


def say(m=""):
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


import torch
import yue2
from yue2 import YuE2Pipeline
from yue2.protocol import Sampling


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    say(f"   memoria {tag}: libera {free / GIB:.2f} GiB su {total / GIB:.2f} | allocata {torch.cuda.memory_allocated() / GIB:.2f} | "
        f"riservata {torch.cuda.memory_reserved() / GIB:.2f} | picco allocata {torch.cuda.max_memory_allocated() / GIB:.2f}")


def main():
    say(f"yue2 da {Path(yue2.__file__).parent}")
    say(f"GPU {torch.cuda.get_device_name()} | torch {torch.__version__}")
    style = (SONGS / "rock_en.style.txt").read_text(encoding="utf-8").strip()
    lyrics = (SONGS / "rock_en.lyrics.txt").read_text(encoding="utf-8")
    abc = ABC.read_text(encoding="utf-8")
    t = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda", progress=False,
                                        local_files_only=True, quantization="none", memory_budget_gib=24, low_vram=True)
    say(f"caricamento {time.perf_counter() - t:.1f} s | argomenti: bf16, low_vram=True, memory_budget_gib=24 (CLI default + --low-vram)")
    plan = pipe.plan(style, lyrics, seed=11, cot="full", abc=abc)
    say(f"piano dal testo ABC salvato: {len(plan.abc_ids)} token ABC, prefisso completo {len(plan.prefix)} token "
        f"(attesi 3187 / 3491)")
    if len(plan.abc_ids) != 3187:
        say("ATTENZIONE: il testo ABC ritokenizzato non torna a 3187 token: non e' lo stesso input della #271")
    plan.save(OUT / "plan")
    say(f"piano salvato in {OUT / 'plan'} (plan.json, abc_tokens.npy, prefix.npy, score.abc, plan_manifest.json)")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    mem("prima della composizione")
    res = {"abc_tokens": len(plan.abc_ids), "prefix_tokens": len(plan.prefix), "code": "PR #273 @ 8daaa70"}
    t = time.perf_counter()
    try:
        n = {"n": 0, "t": 0.0}

        def on_token(_phase, _tok):
            n["n"] += 1
            if n["n"] == 1:
                say(f"   prefill superato: primo token dopo {time.perf_counter() - t:.1f} s")
                mem("dopo il prefill")
            if time.perf_counter() - n["t"] > 15:
                n["t"] = time.perf_counter()
                say(f"   {n['n']} token, {n['n'] / (time.perf_counter() - t):.1f} tok/s")
        sem = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=6000), on_token=on_token)
        dt = time.perf_counter() - t
        say(f"COMPOSIZIONE RIUSCITA: {len(sem.tokens)} token ({len(sem.tokens) / 25:.1f} s di musica) in {dt:.1f} s, "
            f"{len(sem.tokens) / dt:.1f} tok/s, troncata {sem.truncated}")
        mem("alla fine")
        res.update(ok=True, semantic_tokens=len(sem.tokens), seconds=round(dt, 1),
                   peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GIB, 2))
    except Exception as e:  # noqa: BLE001
        say(f"FALLITA dopo {time.perf_counter() - t:.1f} s: {type(e).__name__}: {str(e)[:600]}")
        mem("al momento dell'errore")
        tb = traceback.format_exc()
        say(tb[-2500:])
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:600]}", traceback=tb[-2500:])
    free, total = torch.cuda.mem_get_info()
    res.update(free_gib=round(free / GIB, 2), total_gib=round(total / GIB, 2), allocated_gib=round(torch.cuda.memory_allocated() / GIB, 2),
               reserved_gib=round(torch.cuda.memory_reserved() / GIB, 2), env={"torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    (OUT / "risultato.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    say(f"\nrisultato in {OUT / 'risultato.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
