"""Cheap, model-agnostic hallucination / repetition-loop detectors (computed on the hypothesis only, plus the
reference length for the blow-up ratio). Flags are reported, never used to drop utterances from the main score."""
from __future__ import annotations

import zlib

from .textnorm import chars

# Calibrated on the REFERENCES of GS2-Typhoon (1000), FLEURS (1021), CV17 test (11042), TVSpeech (570):
# natural speech repeats a unit at most 6x ("จริงจริง จริงจริง", "ใน ใน ใน ใน"), max char-compression 1.69.
LOOP_MIN_REPEATS = 8          # same n-char unit (n>=2) back-to-back >= 8 times ...
LOOP_MIN_SPAN = 40            # ... or >= 5 times covering >= 40 chars
LOOP_MIN_REPEATS_UNIGRAM = 10
COMPRESSION_MAX = 2.4       # Whisper's own fallback threshold
LEN_RATIO_MAX = 1.5         # hyp chars / ref chars
LEN_ABS_MIN_EXTRA = 20      # ... and at least 20 extra chars


def max_consecutive_repeat(s: str, max_n: int = 30) -> tuple[int, int]:
    """Return (best_repeats, n) = longest run of an n-char unit repeated back-to-back, over n=1..max_n."""
    best, best_n = 1, 0
    L = len(s)
    for n in range(1, min(max_n, L // 2) + 1):
        i = 0
        while i + 2 * n <= L:
            k = 1
            while i + (k + 1) * n <= L and s[i + k * n:i + (k + 1) * n] == s[i:i + n]:
                k += 1
            if k > best or (k == best and n > best_n):
                best, best_n = k, n
            i += max(1, (k - 1) * n) if k > 1 else 1
    return best, best_n


def compression_ratio(s: str) -> float:
    # one byte per character (Thai block offset) so the multi-byte UTF-8 lead bytes of Thai don't inflate the ratio
    b = bytes((ord(c) - 0x0E00) & 0xFF if 0x0E00 <= ord(c) <= 0x0E7F else ord(c) & 0xFF for c in s)
    return len(b) / max(1, len(zlib.compress(b))) if b else 0.0


def flags(ref: str, hyp: str, hit_cap: bool = False, *, loop_min_repeats=LOOP_MIN_REPEATS,
          loop_min_span=LOOP_MIN_SPAN, loop_min_repeats_unigram=LOOP_MIN_REPEATS_UNIGRAM,
          compression_max=COMPRESSION_MAX, len_ratio_max=LEN_RATIO_MAX, len_abs_min_extra=LEN_ABS_MIN_EXTRA) -> dict:
    hc, rc = chars(hyp), chars(ref)
    rep, n = max_consecutive_repeat(hc)
    loop = ((n >= 2 and (rep >= loop_min_repeats or (rep >= 5 and rep * n >= loop_min_span)))
            or (n == 1 and rep >= loop_min_repeats_unigram))
    cr = compression_ratio(hc)
    blow = len(hc) > len_ratio_max * max(1, len(rc)) and len(hc) - len(rc) >= len_abs_min_extra
    empty = len(hc) == 0 and len(rc) > 0
    return {"rep_max": rep, "rep_unit": n, "compression": round(cr, 3), "flag_loop": bool(loop or cr > compression_max),
            "flag_blowup": bool(blow), "flag_empty": bool(empty), "flag_cap": bool(hit_cap),
            "flag_any": bool(loop or cr > compression_max or blow or empty or hit_cap)}


def summarize(rows) -> dict:
    n = len(rows) or 1
    out = {f"n_{k}": sum(1 for r in rows if r.get(k)) for k in ("flag_loop", "flag_blowup", "flag_empty", "flag_cap", "flag_any")}
    tot = sum(r.get("char_err", 0) for r in rows)
    bad = sum(r.get("char_err", 0) for r in rows if r.get("flag_any"))
    out["pct_utts_flagged"] = round(100.0 * out["n_flag_any"] / n, 2)
    out["pct_char_err_from_flagged"] = round(100.0 * bad / tot, 2) if tot else 0.0
    clean = [r for r in rows if not r.get("flag_any")]
    cn = sum(r.get("char_n", 0) for r in clean)
    out["CER_excl_flagged"] = round(100.0 * sum(r.get("char_err", 0) for r in clean) / cn, 3) if cn else None
    return out
