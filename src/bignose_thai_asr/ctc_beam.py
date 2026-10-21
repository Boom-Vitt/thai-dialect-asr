"""CTC prefix beam search with char n-gram shallow fusion, and dev-only weight tuning with 2-fold out-of-fold outputs
(queue 048, docs/methods/w2vbert-dialect-ctc.md).

  score(prefix) = log p_CTC(prefix | x) + lm_w * log p_LM(chars) + bonus * |chars|

`logp` is a (T, V) array of natural-log CTC posteriors with blank = id 0. `id2tok` maps ids to tokens; "|" (HF CTC
tokenizers) or " " (team H's BDH vocab) is the word delimiter (a space in the output, no LM step, no bonus), and tokens
wrapped in <...> (<pad>, <unk>, <blank>, <sos/eos>) are never emitted. The LM is a
`ngram.KNTokenLM` over code points of whitespace-free text (team E's `char` LMs, scripts/build_ngram_lm.py --tokenizer char).
Pure Python + numpy: it runs on the pod's CPUs.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

NEG = -1e30
DELIM = ("|", " ")


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def greedy_ids(logp: np.ndarray) -> List[int]:
    """Collapsed argmax path (repeats merged, blanks dropped)."""
    out, prev = [], -1
    for t in np.asarray(logp).argmax(-1).tolist():
        if t != prev and t != 0:
            out.append(t)
        prev = t
    return out


def ids_to_text(ids: Sequence[int], id2tok: Sequence[str]) -> str:
    s = "".join(" " if id2tok[i] in DELIM else ("" if id2tok[i].startswith("<") else id2tok[i]) for i in ids)
    return " ".join(s.split())


def prefix_beam(logp: np.ndarray, id2tok: Sequence[str], lm=None, lm_w: float = 0.0, bonus: float = 0.0,
                beam: int = 8, prune: float = 8.0) -> List[int]:
    """Best collapsed id sequence. At each frame only tokens within `prune` nats of the frame's best are expanded."""
    logp = np.asarray(logp, dtype=np.float64)
    use_lm = lm is not None and lm_w != 0.0
    lmc = {}  # token id -> LM symbol (code point), None for the delimiter / specials

    def sym(c):
        if c not in lmc:
            t = id2tok[c]
            lmc[c] = None if (t in DELIM or t.startswith("<") or len(t) != 1) else ord(t)
        return lmc[c]

    # prefix (tuple of ids) -> [p_blank, p_nonblank, fusion score, LM history tuple]
    beams: Dict[tuple, list] = {(): [0.0, NEG, 0.0, ()]}
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = np.nonzero(row >= row.max() - prune)[0].tolist()
        nxt: Dict[tuple, list] = {}

        def get(p, base):
            e = nxt.get(p)
            if e is None:
                e = nxt[p] = [NEG, NEG, base[0], base[1]]
            return e

        for pre, (pb, pnb, fs, hist) in beams.items():
            tot = _lse(pb, pnb)
            last = pre[-1] if pre else None
            for c in cand:
                lp = row[c]
                if c == 0:
                    e = get(pre, (fs, hist)); e[0] = _lse(e[0], tot + lp)
                elif c == last:
                    e = get(pre, (fs, hist)); e[1] = _lse(e[1], pnb + lp)        # repeat without blank: same prefix
                    s = sym(c)                                                    # blank-separated repeat: new char
                    add = (lm_w * lm.logp(hist, s) if use_lm and s is not None else 0.0) + (bonus if s is not None else 0.0)
                    e2 = get(pre + (c,), (fs + add, hist + (s,) if s is not None else hist)); e2[1] = _lse(e2[1], pb + lp)
                else:
                    s = sym(c)
                    add = (lm_w * lm.logp(hist, s) if use_lm and s is not None else 0.0) + (bonus if s is not None else 0.0)
                    e2 = get(pre + (c,), (fs + add, hist + (s,) if s is not None else hist)); e2[1] = _lse(e2[1], tot + lp)
        beams = dict(sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][2]))[:beam])

    def final(kv):
        pb, pnb, fs, hist = kv[1]
        return _lse(pb, pnb) + fs + (lm_w * lm.logp(hist, lm.eos) if use_lm else 0.0)

    return list(max(beams.items(), key=final)[0])


def decode_utt(chunks: Sequence[np.ndarray], id2tok, lm=None, lm_w=0.0, bonus=0.0, beam=8, prune=8.0) -> str:
    """eval_v2 splits audio > 30 s into chunks; hyps are joined with a space, as the runner does."""
    return " ".join(x for x in (ids_to_text(prefix_beam(c, id2tok, lm, lm_w, bonus, beam, prune), id2tok) for c in chunks) if x)


def fold_of(uid: str, k: int = 2) -> int:
    return int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16) % k


def pick_params(err: Dict[tuple, np.ndarray], n: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple:
    """Lowest corpus CER over the utterances in `mask` (all if None); ties -> the earlier grid point (grid order is
    weakest fusion first)."""
    m = np.ones(len(n), bool) if mask is None else mask
    return min(err, key=lambda p: (err[p][m].sum() / max(n[m].sum(), 1), list(err).index(p)))


def out_of_fold(ids: Sequence[str], err: Dict[tuple, np.ndarray], n: np.ndarray, k: int = 2) -> dict:
    """Per fold f: params picked on the other folds; returns {"by_fold": {f: params}, "choice": [params per utt]}."""
    folds = np.array([fold_of(i, k) for i in ids])
    by = {f: pick_params(err, n, folds != f) for f in range(k)}
    return {"by_fold": by, "choice": [by[f] for f in folds.tolist()]}
