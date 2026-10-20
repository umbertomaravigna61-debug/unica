# UNICA Chronos Lab - YuE2 a tappe (piano -> canzone -> sintesi -> audio), con salvataggio dopo ogni fase.
# Uso: genera.py <nome> <seed> <none|fp8> <chunk_attenzione> [--da sintesi]
import sys, time, json, os, pickle, shutil, torch
from pathlib import Path
LAB = Path(__file__).resolve().parent
OUT = LAB / "out"; OUT.mkdir(exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "balla_con_me"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
quant = sys.argv[3] if len(sys.argv) > 3 else "fp8"
qchunk = int(sys.argv[4]) if len(sys.argv) > 4 else 256
max_tok = int(os.environ.get("YUE2_MAX_TOKENS", "6000"))
tag = f"{name}_seed{seed}"
sem_file = OUT / f"{tag}.semantic.pkl"          # la fase AR dipende solo da nome+seed
nar_steps = int(os.environ.get("YUE2_NAR_STEPS", "0")) or None
if nar_steps: tag = f"{tag}_nar{nar_steps}"      # latents/flac distinti per confronto
lat_file = OUT / f"{tag}.latents.pt"
print(f"[lab] {tag} quant={quant} max_tokens={max_tok} query_chunk={qchunk}", flush=True)

# --- exFAT (E:) non supporta i symlink: la cache HF copia i file invece di collegarli ---
import huggingface_hub.file_download as _fd
def _copy_instead_of_symlink(src, dst, new_blob=False):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): return
    (shutil.move if new_blob else shutil.copyfile)(src, dst)
_fd._create_symlink = _copy_instead_of_symlink
# --- 8 GB: la libreria riserva 2 GiB fissi. Lasciamo usare il 93% della scheda ---
_orig_frac = torch.cuda.set_per_process_memory_fraction
torch.cuda.set_per_process_memory_fraction = lambda frac, device=None: _orig_frac(0.93, device)

from yue2 import YuE2Pipeline
from yue2.protocol import Sampling
from yue2 import nar as _nar

# ---------- 8 GB: prefill della sintesi "a strati" (ogni strato AR va in GPU solo quando lavora) ----------
import torch.nn.functional as F
_AR_PARTS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
def _ar_modules(model):
    for layer in model.model.layers:
        for part in _AR_PARTS: yield getattr(layer, part)
def _park_ar_on_cpu(model):
    for m in _ar_modules(model): m.to("cpu")
    model.lm_head.to("cpu")
    torch.cuda.empty_cache()
def _streaming_prefill(self):
    backbone = self.model.model
    dev = self.device
    ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long)
    positions = torch.arange(self.ar_length, device=dev)[None]
    cos, sin = backbone.rotary_emb(positions)
    emb_dev = next(backbone.embed_tokens.parameters()).device
    x = backbone.embed_tokens(ids.to(emb_dev)).to(dev)
    for layer in backbone.layers:
        for part in _AR_PARTS: getattr(layer, part).to(dev)
        q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
        cached = (k[0, :self.visible_length], v[0, :self.visible_length])
        if self.visible_length != self.ar_length:
            cached = tuple(t.clone() for t in cached)
        self.cache.append(cached)
        h = self._attention(q[0], k[0], v[0], causal=True)
        x = x + layer.self_attn.o_proj(h.flatten(1)[None])
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        for part in _AR_PARTS: getattr(layer, part).to("cpu")
        del q, k, v, h
    torch.cuda.empty_cache()
_nar.CachedNAR._prefill = _streaming_prefill

from contextlib import contextmanager as _cm
@_cm
def _offload_ar_fp8(model, enabled):
    # Come l'originale, ma i moduli FP8 hanno solo buffer (niente parametri): guardiamo entrambi.
    modules = [model.model.embed_tokens, model.lm_head]
    for layer in model.model.layers:
        modules.extend((layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp))
    moved = []
    try:
        if enabled:
            for module in modules:
                t = next(iter(list(module.parameters()) + list(module.buffers())), None)
                if t is not None and t.device.type != "cpu":
                    module.to(device="cpu"); moved.append((module, t.device))
            torch.cuda.empty_cache()
        yield
    finally:
        for module, device in moved: module.to(device=device)
_nar._offload_ar = _offload_ar_fp8
style = open(LAB / f"{name}.style.txt", encoding="utf-8").read().strip()
lyrics = open(LAB / f"{name}.lyrics.txt", encoding="utf-8").read()
timing = {}

t0 = time.time()
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda", quantization=quant, memory_budget_gib=9, offload_ar=True)
try_graph = os.environ.get("YUE2_GRAPH", "0") == "1"
if try_graph and quant == "fp8":
    # Sperimentale: la libreria esclude il CUDA graph in FP8 per prudenza. Qui lo abilitiamo
    # (il percorso non fuso passa dagli stessi moduli FP8 gia' usati in modalita' eager).
    import yue2.cuda_graph as _cg
    _orig_init = _cg.GraphAR.__init__
    def _init(self, model, prefixes, max_tokens, **kw):
        # Su Windows il PyTorch ufficiale non ha FlashAttention compilata: forziamo cuDNN (o sdpa).
        kw.setdefault("attention_backend", os.environ.get("YUE2_GRAPH_ATTN", "cudnn"))
        print(f"[lab] GraphAR attention backend = {kw['attention_backend']}", flush=True)
        saved = getattr(model, "_yue2_fp8_originals", None)
        object.__setattr__(model, "_yue2_fp8_originals", {})
        try:
            _orig_init(self, model, prefixes, max_tokens, **kw)
        finally:
            object.__setattr__(model, "_yue2_fp8_originals", saved or {})
    _cg.GraphAR.__init__ = _init
    print("[lab] CUDA graph in FP8: ATTIVO (sperimentale)", flush=True)
timing["load_s"] = round(time.time() - t0, 1)
torch.cuda.reset_peak_memory_stats()

# ---------- fase 1+2: piano e canzone (token), salvati su disco ----------
if sem_file.exists():
    semantic = pickle.load(open(sem_file, "rb"))
    print(f"[lab] token della canzone ripresi da {sem_file.name} ({len(semantic.tokens)} token)", flush=True)
else:
    t1 = time.time()
    request = pipe._request(style, lyrics, seed=seed, cot="full")
    plan = pipe.plan(request=request)
    semantic = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=max_tok))
    timing["semantic_s"] = round(time.time() - t1, 1)
    pickle.dump(semantic, open(sem_file, "wb"))
    print(f"[lab] token salvati: {len(semantic.tokens)} in {timing['semantic_s']} s", flush=True)
timing["peak_vram_ar_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

# ---------- fase 3: sintesi (NAR) con attenzione a blocchi e offload del modello AR ----------
if lat_file.exists():
    latents = torch.load(lat_file)
    print(f"[lab] latenti ripresi da {lat_file.name}", flush=True)
else:
    t2 = time.time()
    torch.cuda.reset_peak_memory_stats()
    nar_bf16 = os.environ.get("YUE2_NAR_BF16", "0") == "1"   # 1 = riconverti i pesi AR a bf16 prima della sintesi (come gli autori; +2.5 GB)
    if pipe.quantization != "none" and nar_bf16:
        from yue2.quantization import restore_ar
        restore_ar(pipe._model)
    print(f"[lab] sintesi con pesi AR in {'bf16' if (nar_bf16 or pipe.quantization=='none') else 'FP8'}", flush=True)
    model = pipe._load_model(for_nar=True)
    _park_ar_on_cpu(model)
    print(f"[lab] flusso AR parcheggiato in RAM; VRAM residente {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    cfg = pipe.generation_config
    with pipe._status("Synthesizing audio", unit="steps") as status:
        report = lambda c, t: status.update(c, total=t)
        latents = _nar.synthesize(model, semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed,
                                  steps=nar_steps or cfg.ode_steps, context=cfg.context, offload_ar=True,
                                  query_chunk_size=qchunk, on_progress=report).detach().float().cpu()
    torch.save(latents, lat_file)
    timing["nar_s"] = round(time.time() - t2, 1)
    timing["peak_vram_nar_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    print(f"[lab] sintesi ok in {timing['nar_s']} s, picco {timing['peak_vram_nar_gb']} GB", flush=True)

# ---------- fase 4: decodifica audio ----------
t3 = time.time()
pipe._model = None; torch.cuda.empty_cache()
audio = pipe.decode(latents.numpy())
timing["vae_s"] = round(time.time() - t3, 1)
import soundfile as sf
out = OUT / f"{tag}.flac"
sf.write(str(out), audio, 48000, subtype="PCM_24")
res = dict(name=name, seed=seed, quant=quant, query_chunk=qchunk, gpu=torch.cuda.get_device_name(0),
           tokens=len(semantic.tokens), audio_s=round(len(audio)/48000, 1), out=str(out), **timing)
print(json.dumps(res, indent=2), flush=True)
with open(LAB / "risultati.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
pipe.close()
