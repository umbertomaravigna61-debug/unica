"""CLI runner for YuE2-3B inside UNICA, run under app/.venv_yue2 (Python 3.12, torch 2.10+cu128, yue2_infer 0.1.5).

Recipe = the one validated in UNICA-Chronos-Lab\\yue2 (2026-10-15, RTX 5060 Laptop 8 GB): AR weights FP8 (converted in memory
at load, nothing written to disk), streaming prefill of the synthesis (each AR layer goes to the GPU only while it works), CUDA
graph with cuDNN attention, NAR steps chosen by the user (2..32). Measured there: 2:40 song = ~105 s composing + 11 s per NAR step
+ 7 s decoding; peak VRAM 6.8 GB.

Stages (all in one process, each result saved so a later run can reuse it):
  compose   style + lyrics + seed -> plan + song tokens         -> <work>/semantic.pkl
  refine    song tokens + N NAR steps -> audio latents          -> <work>/latents_narN.pt
  decode    latents -> 48 kHz stereo WAV                        -> --out
`--semantic <file>` skips composing (same song, new refinement); `--latents <file>` skips refining too.

Prints JSON lines: {"event": "phase", ...} while running, then one {"success": ..., "path": ..., "semantic": ..., "telemetry": {...}} line.
Weights: --model-dir / --vae-dir (UNICA installs them under <root>/models/yue2/), or the Hugging Face ids resolved offline
from HF_HOME when the directories are missing (developer machines with the Lab cache).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MODEL_ID, VAE_ID = "m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"
TOKENS_PER_SECOND = 25.0            # Lab 2026-10-15: 3999 tokens = 160.0 s of music
SAMPLE_RATE = 48000


def emit(**kv):
    print(json.dumps(kv, ensure_ascii=False), flush=True)


def _patch_for_8gb(torch):
    """The Lab's three patches for an 8 GB card and an exFAT model drive (unchanged from genera.py)."""
    import huggingface_hub.file_download as _fd

    def _copy_instead_of_symlink(src, dst, new_blob=False):             # exFAT has no symlinks: copy into the cache instead
        src, dst = os.path.abspath(src), os.path.abspath(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return
        (shutil.move if new_blob else shutil.copyfile)(src, dst)
    _fd._create_symlink = _copy_instead_of_symlink

    _orig_frac = torch.cuda.set_per_process_memory_fraction               # the library keeps 2 GiB in reserve: let it use 93 % of the card
    torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(0.93, device)

    from yue2 import nar as _nar
    import torch.nn.functional as F  # noqa: F401  (kept: the Lab prefill relies on torch functional being importable)
    _AR_PARTS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")

    def _streaming_prefill(self):                                          # synthesis prefill layer by layer: AR layer on GPU only while it works
        backbone = self.model.model
        dev = self.device
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long)
        positions = torch.arange(self.ar_length, device=dev)[None]
        cos, sin = backbone.rotary_emb(positions)
        emb_dev = next(backbone.embed_tokens.parameters()).device
        x = backbone.embed_tokens(ids.to(emb_dev)).to(dev)
        for layer in backbone.layers:
            for part in _AR_PARTS:
                getattr(layer, part).to(dev)
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            cached = (k[0, :self.visible_length], v[0, :self.visible_length])
            if self.visible_length != self.ar_length:
                cached = tuple(t.clone() for t in cached)
            self.cache.append(cached)
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            for part in _AR_PARTS:
                getattr(layer, part).to("cpu")
            del q, k, v, h
        torch.cuda.empty_cache()
    _nar.CachedNAR._prefill = _streaming_prefill

    from contextlib import contextmanager

    @contextmanager
    def _offload_ar_fp8(model, enabled):                                   # like the original, but FP8 modules only have buffers (no parameters)
        modules = [model.model.embed_tokens, model.lm_head]
        for layer in model.model.layers:
            modules.extend((layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp))
        moved = []
        try:
            if enabled:
                for module in modules:
                    t = next(iter(list(module.parameters()) + list(module.buffers())), None)
                    if t is not None and t.device.type != "cpu":
                        module.to(device="cpu")
                        moved.append((module, t.device))
                torch.cuda.empty_cache()
            yield
        finally:
            for module, device in moved:
                module.to(device=device)
    _nar._offload_ar = _offload_ar_fp8

    def _park_ar_on_cpu(model):
        for layer in model.model.layers:
            for part in _AR_PARTS:
                getattr(layer, part).to("cpu")
        model.lm_head.to("cpu")
        torch.cuda.empty_cache()
    return _park_ar_on_cpu


def _enable_fp8_graph(attention_backend: str):
    """CUDA graph with FP8 AR weights: the Lab recipe (validated 2026-10-15; composing at ~34 tokens/s vs 7.6 eager on the RTX 5060).

    yue2_infer 0.1.5 refuses the graph when the AR weights are FP8 ("use eager for FP8"), in two places. The Lab changed those two lines
    in its own copy of the library; here the SAME two changes are applied in memory at start-up, so the installed library stays untouched.
    The library is pinned by hash (freeze_.venv_yue2.txt), so the source text is known: if it ever differs we stop with a clear error
    rather than silently falling back to the 4.5x slower eager path."""
    import inspect
    import textwrap
    import yue2.cuda_graph as _cg
    import yue2.sampling as _sm
    import yue2.pipeline as _pl

    def _rewrite(module, func, substitutions):
        src = textwrap.dedent(inspect.getsource(func))
        changed = False
        for old, new in substitutions:
            if src.count(old) == 0 and (src.count(new) == 1 or 'environ.get("YUE2_GRAPH")' in src):   # already applied (developer venv with the Lab's edited library)
                continue
            if src.count(old) != 1:
                raise RuntimeError(f"yue2 library text differs from the validated 0.1.5 ({module.__name__}.{func.__name__}): "
                                   "cannot enable the FP8 CUDA graph")
            src = src.replace(old, new)
            changed = True
        if not changed:
            return func
        ns = dict(module.__dict__)
        exec(compile(src, getattr(module, "__file__", module.__name__), "exec"), ns)
        return ns[func.__name__]

    # 1) sampling.generate_tokens: allow the graph when the weights are FP8 (same condition as the Lab: env YUE2_GRAPH=1)
    _sm.generate_tokens = _rewrite(_sm, _sm.generate_tokens, [(
        'graph_enabled = use_cuda_graph and device.type == "cuda" and not getattr(model, "_yue2_fp8_originals", {})',
        'graph_enabled = use_cuda_graph and device.type == "cuda" and (not getattr(model, "_yue2_fp8_originals", {}) '
        'or __import__("os").environ.get("YUE2_GRAPH") == "1")')])
    if getattr(_pl, "generate_tokens", None) is not None:                 # the pipeline imported the name directly: point it at the new one
        _pl.generate_tokens = _sm.generate_tokens
    # 2) cuda_graph.GraphAR.__init__: accept the FP8 linear modules (buffers, no bf16 weight)
    old_check = ('    if getattr(model, "_yue2_fp8_originals", None) or any(\n'                  # indentation after textwrap.dedent of the method
                 '            not isinstance(module, torch.nn.Linear) or module.weight.dtype != self.dtype for module in linears):')
    new_check = ('    from .quantization import FP8Linear as _FP8\n'
                 '    if any(not isinstance(module, (torch.nn.Linear, _FP8)) or '
                 '(isinstance(module, torch.nn.Linear) and module.weight.dtype != self.dtype) for module in linears):')
    lab_check = ('    from .quantization import FP8Linear as _FP8\n'                        # exact text of the Lab's edit (venv of UNICA-Chronos-Lab\yue2)
                 '    if getattr(model, "_yue2_fp8_originals", None) or any(\n'
                 '            not isinstance(module, (torch.nn.Linear, _FP8)) or (isinstance(module, torch.nn.Linear) and module.weight.dtype != self.dtype) for module in linears):')
    init_src = textwrap.dedent(inspect.getsource(_cg.GraphAR.__init__))
    if init_src.count(old_check) == 1:
        _cg.GraphAR.__init__ = _rewrite(_cg, _cg.GraphAR.__init__, [(old_check, new_check)])
    elif init_src.count(lab_check) == 1 or init_src.count(new_check) == 1:
        pass                                                                # already patched
    else:
        raise RuntimeError("yue2 library text differs from the validated 0.1.5 (yue2.cuda_graph.GraphAR.__init__): cannot enable the FP8 CUDA graph")
    _orig_init = _cg.GraphAR.__init__

    def _init(self, model, prefixes, max_tokens, **kw):
        kw.setdefault("attention_backend", attention_backend)              # official Windows torch has no FlashAttention: cuDNN (or sdpa)
        emit(event="phase", phase="composing", message=f"CUDA graph (FP8) on, attention backend {kw['attention_backend']}")
        saved = getattr(model, "_yue2_fp8_originals", None)                 # the Lab's library still checks this flag first: hide it during __init__
        object.__setattr__(model, "_yue2_fp8_originals", {})
        try:
            return _orig_init(self, model, prefixes, max_tokens, **kw)
        finally:
            object.__setattr__(model, "_yue2_fp8_originals", saved or {})
    _cg.GraphAR.__init__ = _init
    os.environ["YUE2_GRAPH"] = "1"

    # 3) GraphAR.prefill: read the prefix in blocks instead of one forward over the whole prefix (UNICA 1.1, measured 2026-10-21).
    #    With cot="full" the prefix carries the planned ABC score; a rock plan reaches ~4300 tokens and the single-shot causal attention
    #    (memory ~ length^2) runs out of the 8 GB card right there, before composing starts (pop plans are shorter and fit).
    #    The model's forward already builds the bottom-right causal mask from cache_position when past_len > 0, so block prefill is
    #    the same computation with memory ~ block x length. UNICA_YUE2_PREFILL_CHUNK=0 restores the single-shot prefill.
    import torch as _torch
    chunk = int(os.environ.get("UNICA_YUE2_PREFILL_CHUNK", "512") or 0)
    if chunk > 0:
        @_torch.inference_mode()
        def _prefill(self):
            if self.closed or self.ready:
                raise RuntimeError("prefill must be called exactly once on an open GraphAR")
            logits = []
            for branch, prefix in enumerate(self.prefixes):
                cache = _cg._PrefixCache(self.keys, self.values, branch)
                result = None
                for start in range(0, len(prefix), chunk):
                    block = _torch.tensor([prefix[start:start + chunk]], dtype=_torch.long, device=self.device)
                    result = self.model(block, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits.append(result.logits[:, -1])
            result = _torch.cat(logits, dim=0)
            if self.capture and self.max_tokens > 1:
                self._capture()
            self.ready = True
            return result
        _cg.GraphAR.prefill = _prefill
        emit(event="phase", phase="composing", message=f"prefill in blocks of {chunk} tokens")


def _resolve(model_dir: str | None, vae_dir: str | None):
    m = Path(model_dir) if model_dir else None
    v = Path(vae_dir) if vae_dir else None
    model = str(m) if m and m.is_dir() else MODEL_ID
    vae = str(v) if v and v.is_dir() else VAE_ID
    return model, vae


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--style", required=True, help="genre, voice/language, instruments, tempo, mood (YuE2 [Tags] block)")
    p.add_argument("--lyrics", default="", help="structured lyrics ([verse]/[chorus]...); empty = instrumental")
    p.add_argument("--lyrics-file", default=None, help="read the lyrics from this UTF-8 file instead of --lyrics")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=360.0, help="cap on the song length (tokens = seconds * 25); 360 = 9000 tokens, the official YuE cap, measured on 8 GB 2026-10-30")
    p.add_argument("--nar-steps", type=int, default=4, help="audio refinement steps, 2..32 (Lab: 2 is listenable, 16 approved in headphones)")
    p.add_argument("--quant", default="fp8", choices=["fp8", "none"])
    p.add_argument("--query-chunk", type=int, default=256)
    p.add_argument("--graph", type=int, default=1, help="1 = CUDA graph in FP8 (Lab recipe), 0 = eager")
    p.add_argument("--graph-attn", default="cudnn")
    p.add_argument("--scheggia", type=int, default=int(os.environ.get("UNICA_YUE2_SCHEGGIA", "1")),
                   help="1 = algebraic shortcuts in the composing graph (head slice + fused FP8 projections, Lab 2026-10-30: "
                        "-24%% per token, same distance from the bf16 original as the plain recipe); 0 = plain recipe")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--vae-dir", default=None)
    p.add_argument("--work", required=True, help="directory for semantic.pkl / latents (kept with the song for later refinement)")
    p.add_argument("--semantic", default=None, help="existing semantic.pkl: skip composing, refine the same song")
    p.add_argument("--latents", default=None, help="existing latents .pt: skip refining too, only decode")
    p.add_argument("--out", required=True, help="WAV output path")
    a = p.parse_args()

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little")
    nar_steps = max(1, min(64, int(a.nar_steps)))
    max_tokens = max(200, int(a.max_seconds * TOKENS_PER_SECOND))
    lyrics = Path(a.lyrics_file).read_text(encoding="utf-8") if a.lyrics_file else (a.lyrics or "")
    if not lyrics.strip():
        lyrics = "[instrumental]"
    tel = {"engine": "yue2-3b", "quantization": a.quant, "nar_steps": nar_steps, "max_tokens": max_tokens, "seed": seed,
           "query_chunk": a.query_chunk, "cuda_graph": bool(a.graph) and a.quant == "fp8"}
    peak = {"rss": 0.0}
    stop = threading.Event()

    def watch():
        import psutil
        proc = psutil.Process()
        while not stop.wait(1.0):
            peak["rss"] = max(peak["rss"], proc.memory_info().rss / 2 ** 30)
    threading.Thread(target=watch, daemon=True).start()

    try:
        t_all = time.perf_counter()
        import torch
        tel.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        park_ar = _patch_for_8gb(torch)
        from yue2 import YuE2Pipeline
        from yue2.protocol import Sampling
        from yue2 import nar as _nar

        model_src, vae_src = _resolve(a.model_dir, a.vae_dir)
        tel.update(model=model_src, vae=vae_src)
        emit(event="phase", phase="loading", message="Loading YuE2-3B (FP8 in memory)...")
        t0 = time.perf_counter()
        pipe = YuE2Pipeline.from_pretrained(model_src, vae=vae_src, device="cuda", quantization=a.quant, memory_budget_gib=9,
                                            offload_ar=True, progress=False, local_files_only=True)
        if tel["cuda_graph"]:
            _enable_fp8_graph(a.graph_attn)
            if a.scheggia:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import yue2_scheggia
                tel["scheggia"] = bool(yue2_scheggia.enable())
                if tel["scheggia"]:
                    emit(event="phase", phase="composing", message="SCHEGGIA on: head slice + fused FP8 projections")
        tel["load_s"] = round(time.perf_counter() - t0, 1)
        torch.cuda.reset_peak_memory_stats()

        # ---- 1. compose (plan + song tokens) ----
        sem_file = work / "semantic.pkl"
        if a.semantic:
            semantic = pickle.load(open(a.semantic, "rb"))
            if Path(a.semantic).resolve() != sem_file.resolve():
                shutil.copyfile(a.semantic, sem_file)
            tel["composed"] = "reused"
            emit(event="phase", phase="composing", message=f"Song tokens reused ({len(semantic.tokens)} tokens)")
        elif a.latents:
            semantic = None
            tel["composed"] = "skipped"
        else:
            emit(event="phase", phase="composing", message="Planning the score and composing the song...")
            t1 = time.perf_counter()
            request = pipe._request(a.style, lyrics, seed=seed, cot="full")
            plan = pipe.plan(request=request)
            last = {"t": 0.0}

            def on_token(_phase, _tok, _n=[0]):
                _n[0] += 1
                if time.perf_counter() - last["t"] > 1.0:
                    last["t"] = time.perf_counter()
                    emit(event="progress", phase="composing", done=_n[0], total=max_tokens)
            semantic = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=max_tokens), on_token=on_token)
            tel["compose_s"] = round(time.perf_counter() - t1, 1)
            tel["truncated"] = bool(getattr(semantic, "truncated", False))
            pickle.dump(semantic, open(sem_file, "wb"))
            emit(event="phase", phase="composing", message=f"Song composed: {len(semantic.tokens)} tokens in {tel['compose_s']} s")
        if semantic is not None:
            tel["tokens"] = len(semantic.tokens)
            tel["song_seconds_est"] = round(len(semantic.tokens) / TOKENS_PER_SECOND, 1)
        tel["peak_vram_ar_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

        # ---- 2. refine (NAR) ----
        lat_file = work / f"latents_nar{nar_steps}.pt"
        if a.latents:
            latents = torch.load(a.latents)
            tel["refined"] = "reused"
        else:
            emit(event="phase", phase="synthesizing", message=f"Refining audio ({nar_steps} steps)...")
            t2 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            if tel.get("scheggia"):
                yue2_scheggia.release(pipe._model)          # the fused tensors would otherwise stay on the card during refinement
            model = pipe._load_model(for_nar=True)
            park_ar(model)
            cfg = pipe.generation_config
            last = {"t": 0.0}

            def on_progress(done, total):
                if time.perf_counter() - last["t"] > 0.5 or done == total:
                    last["t"] = time.perf_counter()
                    emit(event="progress", phase="synthesizing", done=int(done), total=int(total))
            latents = _nar.synthesize(model, semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed, steps=nar_steps,
                                      context=cfg.context, offload_ar=True, query_chunk_size=a.query_chunk,
                                      on_progress=on_progress).detach().float().cpu()
            torch.save(latents, lat_file)
            tel["nar_s"] = round(time.perf_counter() - t2, 1)
            tel["peak_vram_nar_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            emit(event="phase", phase="synthesizing", message=f"Audio refined in {tel['nar_s']} s (peak VRAM {tel['peak_vram_nar_gb']} GB)")

        # ---- 3. decode ----
        emit(event="phase", phase="decoding", message="Decoding audio (VAE)...")
        t3 = time.perf_counter()
        pipe._model = None
        torch.cuda.empty_cache()
        audio = pipe.decode(latents.numpy())
        tel["vae_s"] = round(time.perf_counter() - t3, 1)
        import soundfile as sf
        tmp = out.with_suffix(out.suffix + ".part")
        sf.write(str(tmp), audio, SAMPLE_RATE, format="WAV", subtype="PCM_24")        # .part has no known extension: the format must be explicit
        os.replace(tmp, out)
        tel["audio_seconds"] = round(len(audio) / SAMPLE_RATE, 2)
        tel["sample_rate_hz"] = SAMPLE_RATE
        pipe.close()
        tel["total_s"] = round(time.perf_counter() - t_all, 1)
        tel["peak_rss_gb"] = round(peak["rss"], 2)
        emit(success=True, path=str(out), semantic=str(sem_file) if sem_file.exists() else None, latents=str(lat_file) if lat_file.exists() else None,
             telemetry=tel)
        return 0
    except Exception as e:                                                  # noqa: BLE001 - the parent reads the last JSON line
        import traceback
        tel["peak_rss_gb"] = round(peak["rss"], 2)
        emit(success=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:], telemetry=tel)
        return 1
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
