"""Thai normalization used identically for references and hypotheses of every model.

Steps (in order):
 0. drop Whisper special tokens and annotation tags ([ns] in TVSpeech, <unk>/<noise> in Kaldi corpora)
 1. Unicode NFC, strip zero-width chars, lowercase (Latin), compose decomposed SARA AM (ํ + า → ำ)
 2. Thai digits ๐-๙ → Arabic 0-9; drop thousands separators inside numbers (1,500 → 1500)
 3. [num mode only] spell numbers as Thai words (1500 → หนึ่งพันห้าร้อย, 3.5 → สามจุดห้า, 50% → ห้าสิบเปอร์เซ็นต์)
 4. ไม้ยมก policy: "keep" (default) → ๆ is kept as a character and glued to the preceding word
    ("ดี ๆ" == "ดีๆ"); "expand" → the preceding newmm word is repeated ("ดีๆ" → "ดีดี")
 5. pythainlp.util.normalize (fix vowel/tone-mark order, duplicate marks)
 6. punctuation → space (ฯ, ฯลฯ, quotes, brackets, …), collapse whitespace
CER is computed on the result with ALL whitespace removed, so reference segmentation style can't bias scores.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_ZW = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_PUNCT = re.compile(r"[\"'“”‘’`´.,!?;:()\[\]{}<>«»…\-–—_/\\|@#$^&*+=~ฯ]")
_WHISPER_TAGS = re.compile(r"<\|[^|>]*\|>")
_NOISE_TAGS = re.compile(r"\[(?:[a-z_]{1,12})\]|<(?:unk|noise|sil|ns)>", re.I)  # TVSpeech "[ns]", Kaldi "<unk>"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
YAMOK = "ๆ"


def _num_word(m: re.Match) -> str:
    from pythainlp.util import num_to_thaiword
    whole, _, frac = m.group(0).partition(".")
    out = num_to_thaiword(int(whole)) if whole else ""
    if frac:
        out += "จุด" + "".join(num_to_thaiword(int(d)) for d in frac)
    return out


def digits_to_words(text: str) -> str:
    text = text.replace("%", "เปอร์เซ็นต์")
    return re.sub(r"\d+(?:\.\d+)?", _num_word, text)


@lru_cache(maxsize=4096)
def _last_word(prefix: str) -> str:
    from pythainlp.tokenize import word_tokenize
    toks = [t for t in word_tokenize(prefix, engine="newmm") if t.strip()]
    return toks[-1] if toks else ""


def apply_yamok(text: str, mode: str = "keep") -> str:
    # glue ๆ to the preceding token in both modes (writers vary: "ดี ๆ", "ดีๆ", "ดี ๆ ")
    text = re.sub(r"\s*ๆ", YAMOK, text)
    if mode == "keep":
        return text
    if mode != "expand":
        raise ValueError(f"unknown yamok mode {mode!r}")
    out = ""
    for ch in text:
        if ch == YAMOK:
            out += _last_word(out.split(" ")[-1]) if out else ""
        else:
            out += ch
    return out


def normalize(text: str, *, numbers: str = "raw", yamok: str = "keep") -> str:
    """numbers: 'raw' (digits kept, Thai→Arabic) or 'words' (spelled out as Thai words)."""
    t = unicodedata.normalize("NFC", text or "")
    t = _WHISPER_TAGS.sub(" ", t)
    t = _NOISE_TAGS.sub(" ", t)
    t = _ZW.sub("", t).lower().replace("\u0e4d\u0e32", "\u0e33")
    t = t.translate(_THAI_DIGITS)
    t = _THOUSANDS.sub("", t)
    if numbers == "words":
        t = digits_to_words(t)
    elif numbers != "raw":
        raise ValueError(f"unknown numbers mode {numbers!r}")
    t = apply_yamok(t, yamok)
    try:
        from pythainlp.util import normalize as th_norm
        t = th_norm(t)
    except Exception:  # pythainlp missing -> still deterministic, just without mark reordering
        pass
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def chars(text: str, **kw) -> str:
    return re.sub(r"\s+", "", normalize(text, **kw))
