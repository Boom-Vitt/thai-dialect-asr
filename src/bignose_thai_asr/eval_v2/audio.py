"""Audio I/O + deterministic long-audio chunking shared by ALL backends (so no model gets a different view)."""
from __future__ import annotations

import io
import os

import numpy as np

SR = 16000
CHUNK_MAX_S = 30.0     # Whisper's receptive field; applied to every backend for comparability
CHUNK_SEARCH_S = 5.0   # look for the quietest 25 ms frame in the last 5 s before the limit
# 172 (owner 2026-10-21): shorter-chunk system variant; both values land in the protocol hash, so outputs never mix
CHUNK_MAX_S = float(os.environ.get("EVAL_V2_CHUNK_MAX_S") or CHUNK_MAX_S)
CHUNK_SEARCH_S = float(os.environ.get("EVAL_V2_CHUNK_SEARCH_S") or CHUNK_SEARCH_S)
FRAME_S = 0.025


def to_mono16k(data: np.ndarray, sr: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        try:
            import soxr
            data = soxr.resample(data, sr, SR)
        except ImportError:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(data, dtype=np.float32)


def decode_bytes(b: bytes) -> np.ndarray:
    import soundfile as sf
    try:
        d, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    except Exception:  # e.g. mp3 on an old libsndfile
        import librosa
        d, sr = librosa.load(io.BytesIO(b), sr=None, mono=True)
    return to_mono16k(d, sr)


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    return to_mono16k(d, sr)


def split_long(audio: np.ndarray, max_s: float = CHUNK_MAX_S, search_s: float = CHUNK_SEARCH_S) -> list[np.ndarray]:
    """Cut audio longer than max_s at the lowest-energy frame inside [limit-search_s, limit]. Deterministic."""
    max_n, search_n, frame = int(max_s * SR), int(search_s * SR), int(FRAME_S * SR)
    if len(audio) <= max_n:
        return [audio]
    out, start = [], 0
    while len(audio) - start > max_n:
        lo, hi = start + max_n - search_n, start + max_n
        seg = audio[lo:hi]
        nfr = max(1, len(seg) // frame)
        rms = np.sqrt(np.mean(seg[:nfr * frame].reshape(nfr, frame) ** 2, axis=1) + 1e-12)
        cut = lo + int(np.argmin(rms)) * frame + frame // 2
        out.append(audio[start:cut])
        start = cut
    out.append(audio[start:])
    return [c for c in out if len(c) > 0]
