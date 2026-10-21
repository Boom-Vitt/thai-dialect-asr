"""Opt-in loop-guarded decoding. Retry gating is reference-free (never reads the transcript); reporting flags stay frozen."""
from __future__ import annotations

import gzip
import math
from dataclasses import asdict, dataclass

from . import hallucination


@dataclass(frozen=True)
class LoopGuard:
    no_repeat: int = 0
    loop_min_repeats: int = hallucination.LOOP_MIN_REPEATS
    loop_min_span: int = hallucination.LOOP_MIN_SPAN
    loop_min_repeats_unigram: int = hallucination.LOOP_MIN_REPEATS_UNIGRAM
    compression_max: float = hallucination.COMPRESSION_MAX
    max_cps: float = 25.0  # hyp chars per second of audio above this = blow-up (replaces the ref-length ratio)
    fallback_model: str | None = None
    seed: int = 20261020

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k == "fallback_model":
                continue
            if not math.isfinite(v) or v < 0 or (k not in ("no_repeat", "seed") and v == 0):
                raise ValueError(f"loop guard: invalid {k}={v}")

    def protocol(self):
        return {**asdict(self), "variant": "loop_guard", "reference_free_gating": True,
                "retry_order": ["beam5", "temperature_0.2", "temperature_0.4", "temperature_0.6", "fallback_model"],
                "num_beams": 5, "length_penalty": 1.0, "best_of": 5,
                "sampling": "top_k=0, top_p=1; undo temperature, highest mean token logprob through EOS per chunk",
                "compression": "detector char-zlib and raw UTF-8 gzip", "exhausted": "keep last candidate",
                "seed_scope": "seed + first 32 bits of SHA256(utterance id), reset per utterance",
                "retry_unit": "whole utterance, original chunks", "empty_triggers_retry": False}

    def flagged(self, hyp, cap, dur_s):
        # loop/compression from the frozen detector, fed the hyp as its own ref so its ref-length blow-up can't fire
        f = hallucination.flags(hyp, hyp, cap, loop_min_repeats=self.loop_min_repeats, loop_min_span=self.loop_min_span,
                                loop_min_repeats_unigram=self.loop_min_repeats_unigram, compression_max=self.compression_max)
        b = hyp.encode("utf-8")
        gzip_ratio = len(b) / len(gzip.compress(b, mtime=0)) if b else 0.0
        blow = len(hallucination.chars(hyp)) > self.max_cps * max(dur_s, 0.5)
        return bool(f["flag_loop"] or f["flag_cap"] or blow or gzip_ratio > self.compression_max)


def retry_utterance(be, chunks, hyp, cap, guard, fallback=None, sr=16000):
    """Return the first passing candidate; fallback-model greedy is the final, unconditional choice."""
    dur = sum(len(c) for c in chunks) / sr
    attempts = [{"path": "greedy", "flagged": guard.flagged(hyp, cap, dur)}]
    if not attempts[-1]["flagged"]:
        return hyp, cap, {"decode_path": "greedy", "decode_attempts": attempts}
    for temperature in (0.0, 0.2, 0.4, 0.6):
        path = "beam5" if not temperature else f"temperature_{temperature}"
        res = be.transcribe_retry(chunks, temperature=temperature, no_repeat_ngram_size=guard.no_repeat)
        hyp = " ".join(t for t, _ in res if t)
        cap = any(c for _, c in res)
        attempts.append({"path": path, "flagged": guard.flagged(hyp, cap, dur)})
        if not attempts[-1]["flagged"]:
            break
    else:
        if fallback is not None:
            res = fallback.transcribe(chunks)
            hyp = " ".join(t for t, _ in res if t)
            cap = any(c for _, c in res)
            attempts.append({"path": "fallback_model", "flagged": guard.flagged(hyp, cap, dur)})
    return hyp, cap, {"decode_path": attempts[-1]["path"], "decode_attempts": attempts}
