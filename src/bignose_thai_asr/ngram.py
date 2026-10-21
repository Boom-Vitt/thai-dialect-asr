"""Token-level n-gram LM for decode-time shallow fusion (Typhoon ASR Streaming, arXiv:2609.14991, Eq. 1).

  score(y) = log p_AM(y | x, y_<t) + alpha * log p_LM(y | y_<t)          (phrase boosting: not implemented)

The paper uses 4-grams over the recognizer's *sub-word* tokens (NeMo NGPU-LM), trained on "ordinary training
transcripts" (no test text), with alpha = 0.5 picked on a dev half of TVSpeech; fusion "only re-ranks candidates the
model already proposes". We mirror that: n = 4 over Whisper BPE ids, interpolated Kneser-Ney (single discount D),
re-ranking only the top-K acoustic-model tokens at each step.

Pure Python (no KenLM dependency) so it runs on any pod; the model is a pickled dict of counts.
"""
from __future__ import annotations

import gzip
import json
import math
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


class KNTokenLM:
    def __init__(self, order: int = 4, discount: float = 0.75, vocab_size: int = 51866, bos: int = -1, eos: int = -2):
        self.n = order
        self.D = discount
        self.V = vocab_size
        self.bos, self.eos = bos, eos
        # per order k (1..n): counts[k][(h..., w)] ; for k<n these are continuation counts N1+(. h w)
        self.counts: List[Dict[Tuple[int, ...], int]] = [dict() for _ in range(order + 1)]
        self.ctx_total: List[Dict[Tuple[int, ...], int]] = [dict() for _ in range(order + 1)]
        self.ctx_types: List[Dict[Tuple[int, ...], int]] = [dict() for _ in range(order + 1)]
        self.n_sent = 0
        self.n_tok = 0
        self._cache: Dict[Tuple[int, ...], float] = {}

    # ------------------------------------------------------------------ training
    def fit(self, sequences: Iterable[Sequence[int]]) -> "KNTokenLM":
        n = self.n
        full: Dict[Tuple[int, ...], int] = defaultdict(int)
        for seq in sequences:
            s = [self.bos] * (n - 1) + list(seq) + [self.eos]
            self.n_sent += 1
            self.n_tok += len(seq) + 1
            for i in range(n - 1, len(s)):
                full[tuple(s[i - n + 1:i + 1])] += 1
        self.counts[n] = dict(full)
        # continuation counts for lower orders: N1+(. g) for every lower-order gram g
        for k in range(n - 1, 0, -1):
            cont: Dict[Tuple[int, ...], int] = defaultdict(int)
            for g in self.counts[k + 1]:
                cont[g[1:]] += 1
            self.counts[k] = dict(cont)
        for k in range(1, n + 1):
            tot: Dict[Tuple[int, ...], int] = defaultdict(int)
            typ: Dict[Tuple[int, ...], int] = defaultdict(int)
            for g, c in self.counts[k].items():
                tot[g[:-1]] += c
                typ[g[:-1]] += 1
            self.ctx_total[k], self.ctx_types[k] = dict(tot), dict(typ)
        return self

    # ------------------------------------------------------------------ query
    def _prob(self, hist: Tuple[int, ...], w: int) -> float:
        k = len(hist) + 1
        if k == 1:
            tot = self.ctx_total[1].get((), 0)
            typ = self.ctx_types[1].get((), 0)
            c = self.counts[1].get((w,), 0)
            uni = 1.0 / self.V
            if tot == 0:
                return uni
            return max(c - self.D, 0) / tot + self.D * typ / tot * uni
        lower = self._prob(hist[1:], w)
        tot = self.ctx_total[k].get(hist, 0)
        if tot == 0:
            return lower
        c = self.counts[k].get(hist + (w,), 0)
        return max(c - self.D, 0) / tot + self.D * self.ctx_types[k][hist] / tot * lower

    def logp(self, history: Sequence[int], w: int) -> float:
        """Natural-log probability of token w after `history` (raw token ids; BOS padding added here)."""
        h = ([self.bos] * (self.n - 1) + list(history))[-(self.n - 1):] if self.n > 1 else []
        key = tuple(h) + (w,)
        v = self._cache.get(key)
        if v is None:
            v = math.log(max(self._prob(tuple(h), w), 1e-30))
            if len(self._cache) < 2_000_000:
                self._cache[key] = v
        return v

    def sentence_logp(self, seq: Sequence[int], with_eos: bool = True) -> float:
        s = list(seq)
        tot = sum(self.logp(s[:i], s[i]) for i in range(len(s)))
        if with_eos:
            tot += self.logp(s, self.eos)
        return tot

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        self._cache = {}
        with gzip.open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def save_json(self, path: str) -> None:
        """Pickle-free release format (gzip JSON); n-gram tuples are stored as lists."""
        d = {k: v for k, v in self.__dict__.items() if k != "_cache"}
        for k in ("counts", "ctx_total", "ctx_types"):
            d[k] = [[[list(g), c] for g, c in m.items()] for m in d[k]]
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(d, fh)

    @classmethod
    def load_json(cls, path: str) -> "KNTokenLM":
        obj = cls.__new__(cls)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        for k in ("counts", "ctx_total", "ctx_types"):
            d[k] = [{tuple(g): c for g, c in m} for m in d[k]]
        obj.__dict__.update(d)
        obj._cache = {}
        return obj

    @classmethod
    def load(cls, path: str) -> "KNTokenLM":
        obj = cls.__new__(cls)
        with gzip.open(path, "rb") as fh:
            obj.__dict__.update(pickle.load(fh))
        obj._cache = {}
        return obj
