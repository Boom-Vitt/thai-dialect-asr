"""CER / WER for Thai. CER: edit distance over normalized chars with all whitespace removed.
WER: edit distance over word tokens obtained by tokenizing the whitespace-free normalized string
(newmm always; deepcut when pythainlp's deepcut engine is importable)."""
from __future__ import annotations

from functools import lru_cache

from .textnorm import chars

try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def edit_distance(a, b) -> int:
        return _Lev.distance(a, b)
except Exception:  # pragma: no cover - pure-python fallback
    def edit_distance(a, b) -> int:
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


@lru_cache(maxsize=None)
def tokenizer_available(engine: str) -> bool:
    try:
        from pythainlp.tokenize import word_tokenize
        word_tokenize("ทดสอบ", engine=engine)
        return True
    except Exception:
        return False


def tokenize(s: str, engine: str):
    from pythainlp.tokenize import word_tokenize
    return [w for w in word_tokenize(s, engine=engine) if w.strip()] if s else []


def score_pair(ref: str, hyp: str, *, yamok: str = "keep", engines=("newmm", "deepcut")) -> dict:
    """Per-utterance error counts. Keys: char_err/char_n, word_<engine>_err/_n, and the same with prefix
    'num_' where numbers are spelled out as Thai words in BOTH ref and hyp."""
    out = {}
    for tag, numbers in (("", "raw"), ("num_", "words")):
        rc, hc = chars(ref, numbers=numbers, yamok=yamok), chars(hyp, numbers=numbers, yamok=yamok)
        out[f"{tag}char_err"], out[f"{tag}char_n"] = edit_distance(rc, hc), len(rc)
        for eng in engines:
            if not tokenizer_available(eng):
                continue
            rw, hw = tokenize(rc, eng), tokenize(hc, eng)
            out[f"{tag}word_{eng}_err"], out[f"{tag}word_{eng}_n"] = edit_distance(rw, hw), len(rw)
    return out


def corpus_rates(rows) -> dict:
    """Aggregate per-utterance counts into corpus-level percentages (sum errors / sum ref units)."""
    sums = {}
    for r in rows:
        for k, v in r.items():
            if k.endswith("_err") or k.endswith("_n"):
                sums[k] = sums.get(k, 0) + v
    out = {}
    for k in sorted(sums):
        if k.endswith("_err"):
            base = k[:-4]
            n = sums.get(base + "_n", 0)
            out[base] = round(100.0 * sums[k] / n, 3) if n else None
    return out
