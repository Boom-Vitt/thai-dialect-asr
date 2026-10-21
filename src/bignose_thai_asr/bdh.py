"""BigNose Dialect Hybrid (BDH): a pretrained acoustic encoder (Whisper-large-v3 encoder or w2v-BERT 2.0) with LoRA and a
SUPERB-style softmax layer weighting, a char CTC head, self-conditioned intermediate CTC, and a small char-level
transformer attention decoder trained from scratch, decoded by attention beam search with CTC prefix scoring
(hybrid CTC/attention) and optional char n-gram shallow fusion. Design: docs/methods/bignose-dialect-hybrid.md.

  loss  = (1 - lam) * CE_att + lam * CTC + mu * mean(InterCTC)
  score = (1 - beta) * log p_att(y) + beta * log p_ctc_prefix(y) + gamma * log p_lm(y) + bonus * |y|

Token ids: 0 = CTC blank, 1 = <sos/eos> (ESPnet convention), then the train-text characters (space included).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctc_beam import greedy_ids, ids_to_text  # noqa: F401 (ids_to_text: shared id -> text)

BLANK, EOS = 0, 1
SR = 16000
NEG = -1e10  # finite "log 0": keeps score differences free of inf - inf = nan


# ---------------------------------------------------------------- vocab
def build_vocab(texts, min_count=1):
    c = Counter(ch for t in texts for ch in " ".join(t.split()))
    return ["<blank>", "<sos/eos>"] + sorted(k for k, n in c.items() if n >= min_count)


def text_to_ids(text, stoi):
    return [stoi[c] for c in " ".join(text.split()) if c in stoi]  # unseen chars are dropped




# ---------------------------------------------------------------- encoder
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))  # zero B: starts as the pretrained layer
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def add_lora(root: nn.Module, targets, r: int, alpha: float) -> int:
    n = 0
    for mod in list(root.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r, alpha)); n += 1
    return n


def spec_augment(x, valid, time_dim, n_freq=2, max_f=20, max_t=40, t_per=400, rng=np.random):
    """In-place SpecAugment (Park et al. 2019) on a batch of features; masks are filled with the utterance mean.
    valid[i] = number of real feature frames of item i; time masks = 2 + valid/t_per, each <= max_t frames."""
    xt = x.transpose(1, 2) if time_dim == -1 else x  # -> (B, T, F) view
    for i, v in enumerate(int(n) for n in valid):
        m = xt[i, :v].mean()
        for _ in range(n_freq):
            w = rng.randint(0, max_f + 1); f0 = rng.randint(0, max(1, xt.shape[2] - w))
            xt[i, :v, f0:f0 + w] = m
        for _ in range(2 + v // t_per):
            w = rng.randint(0, min(max_t, max(1, v // 10)) + 1); t0 = rng.randint(0, max(1, v - w))
            xt[i, t0:t0 + w] = m
    return x


class Featurizer:
    """Waveforms -> encoder inputs. Kept apart from the encoder so DataLoader workers never copy the weights."""

    def __init__(self, kind, fe):
        self.kind, self.fe = kind, fe

    def __call__(self, wavs, specaug=False):
        if self.kind == "whisper":  # always padded to 30 s; 100 mel frames/s, encoder 50 frames/s
            x = self.fe(wavs, sampling_rate=SR, return_tensors="pt").input_features
            hop = self.fe.hop_length
            valid = torch.tensor([min(x.shape[-1], math.ceil(len(w) / hop)) for w in wavs])
            out = {"x": x, "lens": torch.tensor([min(x.shape[-1] // 2, math.ceil(len(w) / (2 * hop))) for w in wavs])}
        else:  # w2v-BERT: 80-dim fbank stacked x2 -> 50 frames/s, no further downsampling
            f = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
            x, valid = f.input_features, f.attention_mask.sum(-1)
            out = {"x": x, "mask": f.attention_mask, "lens": valid.clone()}
        if specaug:
            spec_augment(x, valid, -1 if self.kind == "whisper" else 1)
        return out


class EncoderWrap(nn.Module):
    """Uniform view of a HF speech encoder: forward(x, mask) -> tuple of all hidden states (embeddings + every layer)."""
    TARGETS = {"whisper": ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
               "w2vbert": ("linear_q", "linear_k", "linear_v", "linear_out", "intermediate_dense", "output_dense")}

    def __init__(self, kind, enc, fe):
        super().__init__()
        self.kind, self.enc, self.feat = kind, enc, Featurizer(kind, fe)
        self.layers = enc.layers if kind == "whisper" else enc.encoder.layers
        self.d = enc.config.d_model if kind == "whisper" else enc.config.hidden_size
        # layerdrop would skip hooked layers; w2v-BERT's internal masking is replaced by our SpecAugment
        for k in ("encoder_layerdrop", "layerdrop"):
            if hasattr(enc.config, k):
                setattr(enc.config, k, 0.0)
        enc.config.apply_spec_augment = False

    def forward(self, x, mask=None):
        if self.kind == "whisper":
            return self.enc(input_features=x, output_hidden_states=True).hidden_states
        return self.enc(input_features=x, attention_mask=mask, output_hidden_states=True).hidden_states


def make_encoder(kind, init, revision=None, dtype=torch.float32):
    if kind == "whisper":
        from transformers import WhisperFeatureExtractor, WhisperModel
        enc = WhisperModel.from_pretrained(init, revision=revision, dtype=dtype).encoder
        fe = WhisperFeatureExtractor.from_pretrained(init, revision=revision)
    elif kind == "w2vbert":
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        enc = Wav2Vec2BertModel.from_pretrained(init, revision=revision, dtype=dtype)
        fe = AutoFeatureExtractor.from_pretrained(init, revision=revision)
    else:
        raise ValueError(f"unknown encoder kind {kind!r}")
    return EncoderWrap(kind, enc, fe)


def init_fingerprint(init, revision=None):
    """Identity of the frozen base: hash of its SHA256SUMS (local dirs) or the HF revision."""
    s = Path(init) / "SHA256SUMS"
    return hashlib.sha256(s.read_bytes()).hexdigest() if s.exists() else f"rev:{revision}"


# ---------------------------------------------------------------- model
def sinusoid(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


def ctc_loss(logits, lens, ys):
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(lp, torch.cat(ys).to(logits.device), lens.to(logits.device),
                      torch.tensor([len(y) for y in ys], device=logits.device), blank=BLANK, zero_infinity=True)


class BDH(nn.Module):
    def __init__(self, encoder: EncoderWrap, vocab_size, d=512, dec_layers=6, heads=8, ff=2048, dropout=0.1,
                 inter_layers=(), n_dialects=0, lora_r=32, lora_alpha=64, max_len=1024):
        super().__init__()
        self.arch = dict(vocab_size=vocab_size, d=d, dec_layers=dec_layers, heads=heads, ff=ff, dropout=dropout,
                         inter_layers=list(inter_layers), n_dialects=n_dialects, lora_r=lora_r, lora_alpha=lora_alpha,
                         max_len=max_len)
        self.encoder, V = encoder, vocab_size
        encoder.requires_grad_(False)
        if lora_r:
            add_lora(encoder.enc, EncoderWrap.TARGETS[encoder.kind], lora_r, lora_alpha)
        self.layer_w = nn.Parameter(torch.zeros(len(encoder.layers) + 1))  # SUPERB weighted sum over all hidden states
        self.proj = nn.Sequential(nn.LayerNorm(encoder.d), nn.Linear(encoder.d, d))
        self.ctc_head = nn.Linear(d, V)
        self.inter_layers = tuple(inter_layers)
        if self.inter_layers:  # self-conditioned InterCTC: one shared head, posteriors fed back into the residual stream
            self.inter_norm, self.inter_head = nn.LayerNorm(encoder.d), nn.Linear(encoder.d, V)
            self.cond = nn.Linear(V, encoder.d)
            nn.init.zeros_(self.cond.weight); nn.init.zeros_(self.cond.bias)  # starts as the unmodified encoder
            for k in self.inter_layers:
                encoder.layers[k - 1].register_forward_hook(self._make_hook(k))
        self.dialect_emb = nn.Embedding(n_dialects, d) if n_dialects else None
        if self.dialect_emb is not None:
            nn.init.zeros_(self.dialect_emb.weight)
        self.embed = nn.Embedding(V, d)
        self.register_buffer("pos", sinusoid(max_len, d), persistent=False)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, V)
        self._inter = {}

    def _make_hook(self, k):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            logits = self.inter_head(self.inter_norm(h))
            self._inter[k] = logits
            h = h + self.cond(logits.float().softmax(-1)).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def encode(self, x, lens, mask=None, dialect=None):
        """-> memory (B, T, d), pad mask (B, T) True = padding, [inter-CTC logits (B, T, V)]."""
        self._inter = {}
        hs = self.encoder(x, mask)
        T = int(lens.max())
        w = self.layer_w.softmax(0)
        h = sum(wi * hi[:, :T].float() for wi, hi in zip(w, hs))
        mem = self.proj(h)
        if self.dialect_emb is not None and dialect is not None:
            mem = mem + self.dialect_emb(dialect.to(mem.device))[:, None]
        pad = torch.arange(T, device=mem.device)[None] >= lens.to(mem.device)[:, None]
        return mem, pad, [self._inter[k][:, :T] for k in self.inter_layers]

    def att_logits(self, mem, pad, ys_in):
        L = ys_in.size(1)
        x = self.embed(ys_in) * math.sqrt(self.arch["d"]) + self.pos[:L]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.decoder(x, mem, tgt_mask=causal, memory_key_padding_mask=pad)  # EOS-padded tails are causally invisible
        return self.out(h)

    def forward(self, x, lens, ys, mask=None, dialect=None, lam=0.3, mu=0.1, lsm=0.1):
        """ys: list of 1-D LongTensors (char ids, no sos/eos). Returns dict of losses."""
        mem, pad, inter = self.encode(x, lens, mask, dialect)
        l_ctc = ctc_loss(self.ctc_head(mem), lens, ys)
        l_inter = torch.stack([ctc_loss(z, lens, ys) for z in inter]).mean() if inter else mem.new_zeros(())
        dev = mem.device
        ys_in = nn.utils.rnn.pad_sequence([torch.cat([torch.tensor([EOS]), y]) for y in ys], True, EOS).to(dev)
        ys_out = nn.utils.rnn.pad_sequence([torch.cat([y, torch.tensor([EOS])]) for y in ys], True, -100).to(dev)
        logits = self.att_logits(mem, pad, ys_in)
        l_att = F.cross_entropy(logits.float().flatten(0, 1), ys_out.flatten(), ignore_index=-100, label_smoothing=lsm)
        loss = (1 - lam) * l_att + lam * l_ctc + mu * l_inter
        return {"loss": loss, "att": l_att.detach(), "ctc": l_ctc.detach(), "inter": l_inter.detach()}


# ---------------------------------------------------------------- decoding
def ctc_greedy(logp, lens):
    return [greedy_ids(row[:int(n)].float().cpu().numpy()) for row, n in zip(logp, lens)]


class CTCPrefixScorer:
    """Batched CTC prefix probabilities (Watanabe et al. 2017, Alg. 2), log domain. N running hypotheses; row n reads
    utterance utt[n] of logp (U, T, V). state = (r (T, N, 2) [non-blank, blank] forward variables, last token (N,))."""

    def __init__(self, logp, lens):
        self.x, self.lens = logp.float().transpose(0, 1), lens.to(logp.device)  # (T, U, V)

    def init_state(self, utt):
        T, N = self.x.shape[0], len(utt)
        r = torch.full((T, N, 2), NEG, device=self.x.device)
        r[:, :, 1] = self.x[:, utt, BLANK].cumsum(0)
        return r, torch.full((N,), -1, dtype=torch.long, device=self.x.device)

    def score(self, state, utt, cand, start=0):
        """cand (N, K) token ids (never blank) -> (log P(prefix + c ...) (N, K), new r (T, N, K, 2)).
        For c = EOS the value is log P(the prefix is the whole output). start = prefix length: a prefix of length
        start + 1 cannot end before frame `start`, so earlier frames stay log 0 and are skipped."""
        r, last = state
        T, (N, K) = self.x.shape[0], cand.shape
        x = self.x[:, utt[:, None].expand(N, K), cand]  # (T, N, K)
        xb = self.x[:, utt, BLANK][:, :, None]  # (T, N, 1)
        r_sum = torch.logaddexp(r[..., 0], r[..., 1])  # (T, N)
        # a repeated char needs a blank in between: only blank-ending paths of the prefix may continue
        phi = torch.where((cand == last[:, None])[None], r[..., 1][..., None], r_sum[..., None])  # (T, N, K)
        rn = torch.full((T, N, K, 2), NEG, device=x.device)
        rn[0, :, :, 0] = torch.where((last == -1)[:, None], x[0], torch.full_like(x[0], NEG))
        for t in range(max(1, start), T):
            rn[t, ..., 0] = torch.logaddexp(rn[t - 1, ..., 0], phi[t - 1]) + x[t]
            rn[t, ..., 1] = torch.logaddexp(rn[t - 1, ..., 0], rn[t - 1, ..., 1]) + xb[t]
        n_t = self.lens[utt]  # (N,)
        valid = (torch.arange(1, T, device=x.device)[:, None] < n_t[None])[..., None]  # (T-1, N, 1)
        terms = torch.where(valid, phi[:-1] + x[1:], torch.full_like(x[1:], NEG))
        psi = torch.logsumexp(torch.cat([rn[:1, ..., 0], terms]), 0)
        end = r_sum.gather(0, (n_t - 1)[None])[0]  # (N,)
        psi = torch.where(cand == EOS, end[:, None].expand(N, K), psi)
        return psi, rn


@torch.no_grad()
def joint_decode(model, mem, pad, ctc_logp, lens, beam=1, ctc_weight=0.3, lm=None, lm_weight=0.0, len_bonus=0.0,
                 itos=None, pre_beam=None):
    """Attention beam search with CTC prefix scoring, batched over utterances. Returns [(ids, capped)] per utterance.
    Hard length cap per utterance = min(frames, 2 * |greedy CTC| + 10): CTC already forbids > frames chars, and a
    hypothesis may not run far past what the alignment holds (no runaway loops)."""
    dev = mem.device
    U, V = mem.size(0), ctc_logp.size(-1)
    lens = lens.to(dev)
    caps = [min(int(n), 2 * len(g) + 10) for g, n in zip(ctc_greedy(ctc_logp, lens), lens)]
    B, K = beam, min(V - 1, pre_beam or max(5, int(1.5 * beam)))
    utt = torch.arange(U, device=dev).repeat_interleave(B)
    N = U * B
    scorer = CTCPrefixScorer(ctc_logp, lens) if ctc_weight > 0 else None
    r, last = scorer.init_state(utt) if scorer else (None, torch.full((N,), -1, dtype=torch.long, device=dev))
    ys = torch.full((N, 1), EOS, dtype=torch.long, device=dev)
    att, lms = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    alive = torch.zeros(N, dtype=torch.bool, device=dev); alive[::B] = True  # one start hypothesis per utterance
    ended = [[] for _ in range(U)]
    done = [False] * U
    mem_r, pad_r = mem[utt], pad[utt]
    lm_tok = (lambda i: None if itos[i] == " " else ord(itos[i])) if lm is not None else None
    for step in range(max(caps) + 1):
        logp = model.att_logits(mem_r, pad_r, ys)[:, -1].float().log_softmax(-1)
        logp[:, BLANK] = NEG
        a_top, cand = logp.topk(K, -1)  # pre-beam on the attention scores
        new_att = att[:, None] + a_top
        if scorer:
            psi, rn = scorer.score((r, last), utt, cand, start=ys.size(1) - 1)
        else:
            psi = torch.zeros_like(new_att)
        new_lm = lms[:, None].expand(N, K).clone()
        if lm is not None and lm_weight:
            for n in alive.nonzero().flatten().tolist():
                hist = [t for t in (lm_tok(i) for i in ys[n, 1:].tolist()) if t is not None]
                for k, c in enumerate(cand[n].tolist()):
                    w = lm.eos if c == EOS else lm_tok(c)
                    if w is not None:
                        new_lm[n, k] += lm.logp(hist, w)
        tot = (1 - ctc_weight) * new_att + ctc_weight * psi + lm_weight * new_lm + len_bonus * (ys.size(1))
        tot = torch.where(alive[:, None], tot, torch.full_like(tot, NEG))
        best, idx = tot.view(U, B * K).topk(B, -1)
        src = (idx // K + torch.arange(U, device=dev)[:, None] * B).flatten()
        kk = (idx % K).flatten()
        tok = cand[src, kk]
        ys = torch.cat([ys[src], tok[:, None]], 1)
        att, lms = new_att[src, kk], new_lm[src, kk]
        if scorer:
            r = rn[:, src, kk]
        last, alive = tok, best.flatten() > NEG / 2
        sc = best.flatten().tolist()
        for n in ((tok == EOS) & alive).nonzero().flatten().tolist():
            ended[n // B].append((sc[n], ys[n, 1:-1].tolist(), False))
        alive &= tok != EOS
        for u in range(U):
            if done[u]:
                continue
            rows = range(u * B, (u + 1) * B)
            run = [sc[n] for n in rows if alive[n]]
            if step >= caps[u]:  # hard cap: keep the running hyps, flagged
                ended[u] += [(sc[n], ys[n, 1:].tolist(), True) for n in rows if alive[n]]
                done[u] = True
            elif not run or (ended[u] and (len(ended[u]) >= B or (len_bonus <= 0 and max(run) < max(e[0] for e in ended[u])))):
                done[u] = True  # scores only decrease with length (bonus 0): nobody running can win any more
            if done[u]:
                alive[u * B:(u + 1) * B] = False
        if all(done):
            break
    out = []
    for e in ended:
        s, ids, capped = max(e, key=lambda z: z[0]) if e else (NEG, [], True)
        out.append((ids, capped))
    return out


@torch.no_grad()
def transcribe(model, wavs, itos, beam=1, ctc_weight=0.3, lm=None, lm_weight=0.0, len_bonus=0.0, dialect=None,
               device="cpu"):
    """Waveforms (16 kHz float32, <= 30 s) -> [(text, capped)]. dialect: int id (or None) for every utterance."""
    f = model.encoder.feat(wavs)
    mask = f["mask"].to(device) if "mask" in f else None
    dl = None if dialect is None else torch.full((len(wavs),), int(dialect), dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):  # cuda: bf16 frozen base
        mem, pad, _ = model.encode(f["x"].to(device), f["lens"], mask, dl)
    mem = mem.float()
    logp = model.ctc_head(mem).float().log_softmax(-1)
    res = joint_decode(model, mem, pad, logp, f["lens"], beam, ctc_weight, lm, lm_weight, len_bonus, itos)
    return [(ids_to_text(ids, itos), capped) for ids, capped in res]


# ---------------------------------------------------------------- save / load
def trainable_state(model):
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def save_bdh(model, cfg, out_dir, state=None):
    """bdh.safetensors = the trainable weights only (LoRA, layer weights, heads, decoder); the frozen base is
    referenced by cfg['encoder'] = {kind, init, revision, fingerprint} and checked at load."""
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sd = state if state is not None else trainable_state(model)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in sd.items()}, str(out / "bdh.safetensors"))
    (out / "bdh_config.json").write_text(json.dumps(dict(cfg, arch=model.arch), ensure_ascii=False, indent=1))


def load_bdh(d, device="cpu", encoder=None, base_dtype=torch.float32):
    from safetensors.torch import load_file
    d = Path(d)
    cfg = json.loads((d / "bdh_config.json").read_text())
    e = cfg["encoder"]
    if encoder is None:
        fp = init_fingerprint(e["init"], e.get("revision"))
        if e.get("fingerprint") and fp != e["fingerprint"]:
            raise RuntimeError(f"base encoder {e['init']} changed since training ({fp} != {e['fingerprint']})")
        encoder = make_encoder(e["kind"], e["init"], e.get("revision"), base_dtype)
    model = BDH(encoder, **cfg["arch"])
    sd = load_file(str(d / "bdh.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    want = set(trainable_state(model))
    if unexpected or want - set(sd):
        raise RuntimeError(f"bdh weights mismatch: unexpected {unexpected[:5]}, missing {sorted(want - set(sd))[:5]}")
    for n, p in model.named_parameters():  # trainable params stay fp32 next to a (maybe bf16) frozen base
        if n in want:
            p.data = p.data.float()
    return model.to(device).eval(), cfg
