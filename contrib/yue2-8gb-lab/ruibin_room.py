"""Ruibin Room - a small mastering chain for YuE2 songs (numpy only).

    python ruibin_room.py <file.wav | folder> [--lufs -14] [--ceiling -1] [--width 1.25] [--mono-below 120]

Chain (offline, zero-phase filters in the frequency domain, no phase shift between bands):
  1. clean-up    high-pass at 28 Hz (sub-sonic rumble out)
  2. bass centre everything below 120 Hz goes mono: a solid bass in the middle of the stage
  3. wide stage  mid/side widening of the side signal above 300 Hz only (the mono sum is untouched)
  4. glue        gentle RMS compression, 2:1, slow attack/release, soft knee
  5. loudness    integrated loudness to -14 LUFS (ITU-R BS.1770-4, K-weighted, gated)
  6. ceiling     true-peak limiter at -1 dBTP (4x oversampled peaks, 5 ms look-ahead)
The original files are never touched: results go to a "ruibin_room" folder next to them, with a before/after report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BANNER = "Welcome to the Ruibin Room — mastered and powered by an old DJ from Catania."


# ---------------------------------------------------------------- audio I/O (soundfile if present, else the wave module)
def read_audio(path: Path):
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x, sr
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = np.frombuffer(w.readframes(n), dtype=np.uint8)
        if sw == 3:
            b = raw.reshape(-1, 3).astype(np.int32)
            v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
            v = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / float(1 << 23)
        elif sw == 2:
            v = np.frombuffer(raw.tobytes(), dtype="<i2").astype(np.float32) / 32768.0
        else:
            raise ValueError(f"unsupported sample width {sw}")
        return v.reshape(-1, ch), sr


def write_audio(path: Path, x: np.ndarray, sr: int):
    try:
        import soundfile as sf
        sf.write(str(path), x, sr, subtype="PCM_24")
    except ImportError:
        import wave
        v = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
        b = np.stack([(v & 0xFF), (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(x.shape[1]); w.setsampwidth(3); w.setframerate(sr)
            w.writeframes(b.tobytes())


# ---------------------------------------------------------------- frequency-domain tools
def _spectrum(x):
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(n + 4096)))            # padding: no circular wrap of filter tails
    return np.fft.rfft(x, n=nfft, axis=0), nfft


def _freqs(nfft, sr):
    return np.fft.rfftfreq(nfft, 1.0 / sr)


def bw_lowpass_zero_phase(f, fc, order=4):
    """|H|^2 of a Butterworth low-pass: the zero-phase (filtfilt-like) response. LP + HP sum to exactly 1."""
    return 1.0 / (1.0 + (f / fc) ** (2 * order))


def bw_highpass_zero_phase(f, fc, order=4):
    with np.errstate(divide="ignore"):
        r = np.where(f > 0, (fc / np.maximum(f, 1e-9)) ** (2 * order), np.inf)
    return 1.0 / (1.0 + r)


def biquad_response(b, a, f, sr):
    z = np.exp(-1j * 2 * np.pi * f / sr)
    return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)


# ---------------------------------------------------------------- loudness (ITU-R BS.1770-4) and true peak
def _k_weighting(f, sr):
    # BS.1770 pre-filter (high shelf) and RLB high-pass, coefficients derived for any sample rate (as in libebur128)
    import math
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    K = math.tan(math.pi * fc / sr); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    Q2, fc2 = 0.5003270373238773, 38.13547087602444
    K2 = math.tan(math.pi * fc2 / sr)
    a0 = 1 + K2 / Q2 + K2 * K2
    b2 = [1, -2, 1]
    a2 = [1, 2 * (K2 * K2 - 1) / a0, (1 - K2 / Q2 + K2 * K2) / a0]
    return biquad_response(b1, a1, f, sr) * biquad_response(b2, a2, f, sr)


def integrated_lufs(x, sr):
    X, nfft = _spectrum(x)
    y = np.fft.irfft(X * _k_weighting(_freqs(nfft, sr), sr)[:, None], n=nfft, axis=0)[: x.shape[0]]
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if y.shape[0] < block:
        return float("-inf")
    p = np.cumsum(np.concatenate([np.zeros((1, y.shape[1])), y.astype(np.float64) ** 2]), axis=0)
    starts = np.arange(0, y.shape[0] - block + 1, hop)
    z = (p[starts + block] - p[starts]) / block                 # mean square per channel per 400 ms block
    zl = z.sum(axis=1)                                            # L and R weights are 1.0
    lk = -0.691 + 10 * np.log10(np.maximum(zl, 1e-12))
    g = zl[lk > -70]
    if g.size == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = zl[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean())) if g2.size else float("-inf")


def oversampled_peaks(x, factor=4, chunk=1 << 16, margin=1024):
    """Per original sample and channel: the max |x| over the 4x band-limited interpolation (inter-sample peaks).
    Done in overlapping power-of-two blocks: fast, little memory, edges discarded."""
    n, ch = x.shape
    out = np.empty((n, ch), dtype=np.float32)
    for start in range(0, n, chunk):
        a, b = max(0, start - margin), min(n, start + chunk + margin)
        seg = x[a:b].astype(np.float64)
        nfft = 1 << int(np.ceil(np.log2(seg.shape[0])))
        X = np.fft.rfft(seg, n=nfft, axis=0)
        Y = np.zeros((nfft * factor // 2 + 1, ch), dtype=np.complex128)
        Y[: X.shape[0]] = X
        up = np.abs(np.fft.irfft(Y, n=nfft * factor, axis=0) * factor)
        k = min(chunk, n - start)
        i0 = (start - a) * factor
        out[start:start + k] = up[i0:i0 + k * factor].reshape(k, factor, ch).max(axis=1)
    return out


def true_peak_db(x):
    return float(20 * np.log10(max(float(oversampled_peaks(x).max()), 1e-9)))


def db(v):
    return 20 * np.log10(max(v, 1e-12))


# ---------------------------------------------------------------- the chain
def spectral_stage(x, sr, hp=28.0, mono_below=120.0, width=1.25, width_from=300.0):
    X, nfft = _spectrum(x.astype(np.float64))
    f = _freqs(nfft, sr)
    X = X * bw_highpass_zero_phase(f, hp, order=3)[:, None]                         # 1. clean-up
    L, R = X[:, 0], X[:, 1]
    M, S = (L + R) / 2, (L - R) / 2
    S = S * (1 - bw_lowpass_zero_phase(f, mono_below, order=4))                     # 2. bass centre: no side below 120 Hz
    S = S * (1 + (width - 1) * bw_highpass_zero_phase(f, width_from, order=2))      # 3. wide stage above 300 Hz
    y = np.fft.irfft(np.stack([M + S, M - S], axis=1), n=nfft, axis=0)[: x.shape[0]]
    return y.astype(np.float32)


def _smooth_gain(target_db, block_s, attack_s, release_s):
    a_att = np.exp(-block_s / attack_s); a_rel = np.exp(-block_s / release_s)
    g = np.empty_like(target_db); cur = 0.0
    for i, t in enumerate(target_db):                       # gain reduction (<= 0 dB): falling = attack, rising = release
        cur = a_att * cur + (1 - a_att) * t if t < cur else a_rel * cur + (1 - a_rel) * t
        g[i] = cur
    return g


def _to_samples(g_blocks, block, n):
    centres = np.arange(g_blocks.size) * block + block / 2
    return np.interp(np.arange(n), centres, g_blocks).astype(np.float32)


def glue_compressor(x, sr, ratio=2.0, threshold_db=-18.0, knee_db=6.0, attack_s=0.03, release_s=0.25):
    block = max(1, int(0.001 * sr))
    n = x.shape[0] // block * block
    lvl = np.sqrt(np.mean((x[:n] ** 2).reshape(-1, block, x.shape[1]).mean(axis=2), axis=1) + 1e-12)
    w = int(0.05 / 0.001)                                    # 50 ms RMS window, linked stereo
    lvl = np.sqrt(np.convolve(lvl ** 2, np.ones(w) / w, mode="same"))
    ldb = 20 * np.log10(lvl + 1e-12)
    over = ldb - threshold_db
    gr = np.where(over <= -knee_db / 2, 0.0,
         np.where(over >= knee_db / 2, -(over) * (1 - 1 / ratio),
                  -((over + knee_db / 2) ** 2) / (2 * knee_db) * (1 - 1 / ratio)))
    g = _smooth_gain(gr, 0.001, attack_s, release_s)
    gs = _to_samples(10 ** (g / 20), block, x.shape[0])
    return x * gs[:, None], float(gr.min())


def true_peak_limiter(x, sr, ceiling_db=-1.0, lookahead_s=0.005, release_s=0.08):
    ceiling = 10 ** (ceiling_db / 20)
    peaks = oversampled_peaks(x).max(axis=1)                 # linked stereo, inter-sample peaks (4x oversampled)
    block = max(1, int(0.0005 * sr))                          # 0.5 ms blocks
    n = int(np.ceil(peaks.size / block)) * block
    pb = np.pad(peaks, (0, n - peaks.size)).reshape(-1, block).max(axis=1)
    need = np.minimum(0.0, 20 * np.log10(ceiling / np.maximum(pb, 1e-9)))
    la = max(1, int(lookahead_s / 0.0005))
    pad = np.concatenate([need, np.zeros(la)])               # look-ahead: start reducing before the peak arrives
    ahead = np.lib.stride_tricks.sliding_window_view(pad, la + 1).min(axis=1)[: need.size]
    g = _smooth_gain(ahead, 0.0005, 0.0005, release_s)
    g = np.minimum(g, ahead)                                  # never above what the peak needs
    gs = _to_samples(10 ** (g / 20), block, x.shape[0])
    y = x * gs[:, None]
    return np.clip(y, -ceiling, ceiling), float(g.min())


def stereo_report(x, sr):
    X, nfft = _spectrum(x.astype(np.float64))
    f = _freqs(nfft, sr)
    lo = np.fft.irfft(X * bw_lowpass_zero_phase(f, 120.0)[:, None], n=nfft, axis=0)[: x.shape[0]]
    corr = lambda a: float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if a.std() > 0 else 1.0
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return {"stereo_correlation": round(corr(x), 3), "bass_correlation_below_120hz": round(corr(lo), 3),
            "crest_factor_db": round(db(peak) - db(rms), 2)}


def master(path: Path, out_dir: Path, lufs=-14.0, ceiling=-1.0, width=1.25, mono_below=120.0):
    x, sr = read_audio(path)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    before = {"lufs": round(integrated_lufs(x, sr), 2), "true_peak_dbtp": round(true_peak_db(x), 2), **stereo_report(x, sr)}
    y = spectral_stage(x, sr, mono_below=mono_below, width=width)
    y, comp_gr = glue_compressor(y, sr)
    lim_gr = 0.0
    for _ in range(3):                                        # gain to target, limit, re-check (limiting costs a little loudness)
        y = y * 10 ** ((lufs - integrated_lufs(y, sr)) / 20)
        y, lim_gr = true_peak_limiter(y, sr, ceiling_db=ceiling)
        if abs(integrated_lufs(y, sr) - lufs) < 0.2:
            break
    after = {"lufs": round(integrated_lufs(y, sr), 2), "true_peak_dbtp": round(true_peak_db(y), 2), **stereo_report(y, sr)}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_RuibinRoom.wav"
    write_audio(out, y, sr)
    rep = {"file": path.name, "output": out.name, "sample_rate": sr, "seconds": round(x.shape[0] / sr, 2),
           "settings": {"target_lufs": lufs, "ceiling_dbtp": ceiling, "width_above_300hz": width, "mono_below_hz": mono_below,
                        "highpass_hz": 28, "compressor": "2:1, -18 dB, 6 dB knee, 30/250 ms"},
           "max_compression_db": round(comp_gr, 2), "max_limiting_db": round(lim_gr, 2), "before": before, "after": after}
    return rep


def main():
    ap = argparse.ArgumentParser(description="Ruibin Room - mastering chain")
    ap.add_argument("target")
    ap.add_argument("--lufs", type=float, default=-14.0)
    ap.add_argument("--ceiling", type=float, default=-1.0)
    ap.add_argument("--width", type=float, default=1.25)
    ap.add_argument("--mono-below", type=float, default=120.0)
    a = ap.parse_args()
    print(BANNER, flush=True)
    t = Path(a.target)
    files = sorted(p for p in (t.rglob("*.wav") if t.is_dir() else [t]) if "RuibinRoom" not in p.stem and "ruibin_room" not in p.parts)
    reports = []
    for p in files:
        print(f"  mastering {p.name} ...", flush=True)
        r = master(p, p.parent / "ruibin_room", a.lufs, a.ceiling, a.width, a.mono_below)
        b, f = r["before"], r["after"]
        print(f"    LUFS {b['lufs']} -> {f['lufs']} | true peak {b['true_peak_dbtp']} -> {f['true_peak_dbtp']} dBTP | "
              f"bass corr {b['bass_correlation_below_120hz']} -> {f['bass_correlation_below_120hz']} | "
              f"stereo corr {b['stereo_correlation']} -> {f['stereo_correlation']} | comp {r['max_compression_db']} dB, "
              f"limit {r['max_limiting_db']} dB", flush=True)
        reports.append(r)
        (p.parent / "ruibin_room" / f"{p.stem}_RuibinRoom.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"Done: {len(reports)} file(s). Originals untouched.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
