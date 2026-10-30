"""SCHEGGIA - scorciatoie algebriche per la composizione di YuE2 (scenario A, 30/10/2026). Nessun kernel nuovo, nessuna libreria nuova.

Si applica DOPO la ricetta UNICA (yue2_runner._patch_for_8gb + _enable_fp8_graph) e cambia solo il passo del grafo CUDA (GraphAR._decode):

  1. TESTA RIDOTTA (regola: non calcolare cio' che viene buttato). Durante la composizione della canzone la distribuzione mette a -inf
     tutto tranne MUSIC_END (151852) e i 32.768 codici audio (151853..184620): righe CONTIGUE 151852..184620 della testa. Si moltiplicano
     solo quelle 32.769 righe (una vista, nessuna copia) invece di 184.704. Le altre posizioni del vettore restano a un valore finito molto
     negativo (-1e4) che la maschera della libreria porta comunque a -inf: finito e non -inf, perche' la guida CFG fa (cond - uncond) e
     -inf - -inf darebbe NaN. Solo nella fase "semantic": la partitura (abc) usa altre voci e resta com'era.
  2. PROIEZIONI RACCOLTE (regola: a fattor comune). q, k, v leggono lo stesso vettore, gate e up pure: i pesi FP8 vengono affiancati in UN
     tensore per gruppo, e le proiezioni originali diventano VISTE di quel tensore (nessuna memoria in piu'). L'attivazione si quantizza
     una volta per gruppo invece che una per proiezione (da 7 a 4 per blocco) e le moltiplicazioni passano da 7 a 4. Le scale dei pesi
     (una per proiezione) si applicano dopo, in FP32, colonna per colonna: stessa formula di FP8Linear.
o_proj e down_proj, le norme, l'attenzione e tutto il resto restano IDENTICI al codice della libreria (copiato riga per riga).

Uso:  import yue2_scheggia as S; S.enable()   (dopo _enable_fp8_graph)   S.release(model) prima della rifinitura   S.disable()
Variabile UNICA_SCHEGGIA=0: enable() non fa nulla.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

HEAD_FIRST, HEAD_LAST = 151852, 184620          # MUSIC_END .. ultimo codice audio (CODEC_OFFSET + CODEC_SIZE - 1)
FILLER = -1.0e4
_STATE = {"enabled": False, "phase": None, "orig": {}, "head": True, "fuse": True}


def _fused_fp8(x, weight, col_scale):
    """Come FP8Linear.forward, ma con piu' proiezioni affiancate: scala dei pesi per colonna, applicata in FP32 prima dell'arrotondamento."""
    import yue2.quantization as Q
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    count = rows.shape[0]
    padded = F.pad(rows, (0, 0, 0, (-count) % 16))
    activation, scale = Q.quantize_tensor(padded)
    one = col_scale.new_ones(1)
    out = torch._scaled_mm(activation, weight.t(), scale_a=scale, scale_b=one, out_dtype=torch.float32, use_fast_accum=False)
    if isinstance(out, tuple):
        out = out[0]
    return (out[:count] * col_scale).to(x.dtype).reshape(*shape[:-1], weight.shape[0])


def _fuse_layer(layer):
    """Affianca q|k|v e gate|up in due tensori FP8; le proiezioni originali diventano viste (stessi byte, nessuna copia in piu')."""
    import yue2.quantization as Q
    a, m = layer.self_attn, layer.mlp
    groups = ((a.q_proj, a.k_proj, a.v_proj), (m.gate_proj, m.up_proj))
    if not all(isinstance(p, Q.FP8Linear) and p.bias is None for g in groups for p in g):
        return False
    if getattr(layer, "_scheggia_src", None) == tuple(id(p) for g in groups for p in g):
        return True                                            # gia' fatto per questi stessi moduli
    fused = []
    for g in groups:
        w = torch.cat([p.weight.view(torch.uint8) for p in g], dim=0).view(torch.float8_e4m3fn)
        s = torch.cat([p.weight_scale.float().expand(p.out_features) for p in g]).contiguous()
        start = 0
        for p in g:                                            # le proiezioni originali puntano dentro il tensore affiancato
            p.weight = w[start:start + p.out_features]
            start += p.out_features
        fused.append((w, s))
    torch.cuda.empty_cache()
    layer._scheggia_qkv, layer._scheggia_gu = fused
    layer._scheggia_src = tuple(id(p) for g in groups for p in g)
    return True


def _unfuse_all(model):
    for layer in getattr(getattr(model, "model", None), "layers", []):
        for k in ("_scheggia_qkv", "_scheggia_gu", "_scheggia_src"):
            if hasattr(layer, k):
                delattr(layer, k)


def _decode(self):
    """GraphAR._decode della libreria (yue2_infer 0.1.5), copiato, con le due scorciatoie. Senza scheggia -> l'originale."""
    if not getattr(self, "_scheggia", False):
        return _STATE["orig"]["_decode"](self)
    from yue2.modeling_yue2 import _apply_rotary
    backbone = self.model.model
    cos, sin = backbone.rotary_emb(self.positions[:, None])
    x = backbone.embed_tokens(self.tokens)
    visible = None
    if self.attention_backend != "flash":
        visible = (self.key_positions[None, :] <= self.positions[:, None])[:, None, None, :]
    used_lengths = (self.positions + 1).to(torch.int32)
    config = self.model.config
    slots = self.positions[:, None, None, None].expand(self.branches, 1, config.num_key_value_heads, config.head_dim)
    sizes = (config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim,
             config.num_key_value_heads * config.head_dim)
    for layer, keys, values in zip(backbone.layers, self.keys, self.values):
        normalized = layer.input_layernorm(x)
        if self._scheggia_fuse:
            wqkv, sqkv = layer._scheggia_qkv
            q, k, v = _fused_fp8(normalized, wqkv, sqkv).split(sizes, dim=-1)
            q = layer.self_attn.q_norm(q.view(self.branches, 1, config.num_attention_heads, config.head_dim))
            k = layer.self_attn.k_norm(k.view(self.branches, 1, config.num_key_value_heads, config.head_dim))
            v = v.view(self.branches, 1, config.num_key_value_heads, config.head_dim)
            q = _apply_rotary(q, cos.unsqueeze(2), sin.unsqueeze(2))
            k = _apply_rotary(k, cos.unsqueeze(2), sin.unsqueeze(2))
        else:
            q, k, v = layer.self_attn.project_qkv(normalized, cos, sin)
        keys.scatter_(1, slots, k)
        values.scatter_(1, slots, v)
        if self.attention_backend == "flash":
            h = torch.ops.aten._flash_attention_forward(
                q[:, 0], keys.view(-1, config.num_key_value_heads, config.head_dim),
                values.view(-1, config.num_key_value_heads, config.head_dim),
                self.cu_q, self.cu_k, 1, self.capacity, 0.0, False, False, seqused_k=used_lengths)[0][:, None]
        elif self.attention_backend == "cudnn":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                                   attn_mask=visible, is_causal=False,
                                                   enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
        else:
            h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                               attn_mask=visible, is_causal=False,
                                               enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
        x = x + layer.self_attn.o_proj(h.reshape(self.branches, 1, -1))
        normalized = layer.post_attention_layernorm(x)
        if self._scheggia_fuse:
            wgu, sgu = layer._scheggia_gu
            gate, up = _fused_fp8(normalized, wgu, sgu).chunk(2, dim=-1)
            x = x + layer.mlp.down_proj(F.silu(gate) * up)
        else:
            x = x + layer.mlp(normalized)
    hidden = backbone.norm(x)
    if self._scheggia_head is not None:
        full = self._scheggia_full
        full[:, HEAD_FIRST:HEAD_LAST + 1].copy_(F.linear(hidden, self._scheggia_head)[:, 0])
        output = full
    else:
        output = self.model.lm_head(hidden)[:, 0]
    self.positions.add_(1)
    return output


def release(model):
    """Prima della rifinitura: libera i tensori affiancati (UNICA parcheggia i pesi AR sulla CPU, questi resterebbero sulla scheda)."""
    _unfuse_all(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enable(head=True, fuse=True):
    _STATE["head"], _STATE["fuse"] = bool(head), bool(fuse)
    """Attiva le scorciatoie per i GraphAR creati da qui in poi."""
    if os.environ.get("UNICA_SCHEGGIA", "1") == "0" or _STATE["enabled"]:
        return _STATE["enabled"]
    import inspect
    import yue2.cuda_graph as CG
    import yue2.sampling as S
    import yue2.pipeline as P
    import yue2.quantization as Q
    st = _STATE["orig"]
    st.update(_decode=CG.GraphAR._decode, init=CG.GraphAR.__init__, gen=S.generate_tokens, restore=Q.restore_ar)
    gen = st["gen"]

    def generate_tokens(model, prefix, sampling, seed, phase, *a, **kw):          # annota la fase per il GraphAR che verra' creato
        _STATE["phase"] = phase
        try:
            return gen(model, prefix, sampling, seed, phase, *a, **kw)
        finally:
            _STATE["phase"] = None
    S.generate_tokens = generate_tokens
    if getattr(P, "generate_tokens", None) is not None:
        P.generate_tokens = generate_tokens

    init = st["init"]

    def __init__(self, model, prefixes, max_tokens, **kw):
        init(self, model, prefixes, max_tokens, **kw)
        fuse_on, head_on = _STATE["fuse"], _STATE["head"]
        self._scheggia_fuse = bool(fuse_on) and all(_fuse_layer(layer) for layer in model.model.layers)
        self._scheggia_head = None
        if head_on and _STATE["phase"] == "semantic":
            self._scheggia_head = model.lm_head.weight[HEAD_FIRST:HEAD_LAST + 1]               # vista, nessuna copia
            self._scheggia_full = torch.full((self.branches, model.config.vocab_size), FILLER,
                                             device=self.device, dtype=self.dtype)
        self._scheggia = self._scheggia_fuse or self._scheggia_head is not None
    CG.GraphAR.__init__ = __init__
    CG.GraphAR._decode = _decode

    def restore_ar(model):                                                             # prima della rifinitura: via i tensori affiancati
        _unfuse_all(model)
        return st["restore"](model)
    Q.restore_ar = restore_ar
    if getattr(P, "restore_ar", None) is not None:
        P.restore_ar = restore_ar
    _STATE["enabled"] = True
    return True


def disable():
    if not _STATE["enabled"]:
        return
    import yue2.cuda_graph as CG
    import yue2.sampling as S
    import yue2.pipeline as P
    import yue2.quantization as Q
    st = _STATE["orig"]
    CG.GraphAR._decode, CG.GraphAR.__init__ = st["_decode"], st["init"]
    S.generate_tokens = st["gen"]
    if getattr(P, "generate_tokens", None) is not None:
        P.generate_tokens = st["gen"]
    Q.restore_ar = st["restore"]
    if getattr(P, "restore_ar", None) is not None:
        P.restore_ar = st["restore"]
    _STATE["enabled"] = False
