"""Campaign character-LM rescoring: AM sum + lm_w * LM + bonus * tokens + ctc_w * judge/char."""
from __future__ import annotations

from dataclasses import fields

from .eval_v2.loopguard import LoopGuard
from .eval_v2.textnorm import chars

MAX_S = 30.0
CTC_FLOOR = -1e4


def make_encoder(spec: str):
    """The campaign's char LM uses code points of eval_v2 chars(), including its normalization."""
    if spec != "char":
        raise ValueError("only the released char LM is supported")
    return (lambda t: [ord(c) for c in chars(t)]), None


def guard_from_flags(text: str):
    t = text.split()
    types = {f.name: f.type for f in fields(LoopGuard) if f.name != "fallback_model"}
    try:
        if len(t) % 2 or not all(k.startswith("--lg-") for k in t[::2]):
            raise ValueError("expected --lg-* value pairs")
        return LoopGuard(**{k[5:].replace("-", "_"):
                            (int if types[k[5:].replace("-", "_")] in (int, "int") else float)(v)
                            for k, v in zip(t[::2], t[1::2])})
    except (KeyError, ValueError, OverflowError) as e:
        raise SystemExit(f"bad loop-guard flags: {e}") from e


def add_lm(cands, lm, encode):
    cache = {}
    for c in cands:
        t = " ".join(c["text"].split())
        if t not in cache:
            cache[t] = lm.sentence_logp(encode(t)) if t else lm.logp([], lm.eos)
        c["lm_logp"] = cache[t]


def judge_score(c):
    lp = c.get("ctc_logp")
    if lp is None:
        return 0.0
    n = c.get("ctc_n")
    if n is None:
        n = sum(not ch.isspace() for ch in c["text"])
    if lp != lp or lp <= CTC_FLOOR or n <= 0:
        return CTC_FLOOR
    return max(lp / max(1, n), CTC_FLOOR)


def total(c: dict, p: dict) -> float:
    s = c["att_logp"] + p["lm_w"] * c["lm_logp"] + p["bonus"] * c["n_tokens"]
    if p["ctc_w"]:
        s += p["ctc_w"] * judge_score(c)
    return s


def pick(cands, p, fallback, duration):
    if duration > MAX_S:
        return fallback
    cs = [c for c in cands if not c.get("flagged")]
    return max(cs, key=lambda c: total(c, p))["text"] if cs else fallback
