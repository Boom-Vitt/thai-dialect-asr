"""Whisper N-best and EOS-inclusive AM scoring from the release campaign."""
from __future__ import annotations

import math
from typing import List, Sequence

import torch

from .eval_v2.audio import SR

MAX_TARGET = 448


def prompt_ids(tokenizer, language: str = "th", task: str = "transcribe") -> List[int]:
    t = tokenizer.convert_tokens_to_ids
    return [t("<|startoftranscript|>"), t(f"<|{language}|>"), t(f"<|{task}|>"), t("<|notimestamps|>")]


def token_cap(duration: float, max_new_tokens: int = 440) -> int:
    return max(1, min(max_new_tokens, MAX_TARGET - 4, int(math.ceil(25.0 * max(duration, 0.0))) + 16))


def suppress_mask(model, vocab: int, device):
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, int) else gc.eos_token_id[0]
    always = torch.zeros(vocab, dtype=torch.bool, device=device)
    for t in (getattr(gc, "suppress_tokens", None) or []):
        if t < vocab:
            always[t] = True
    always[eos + 1:] = True
    begin = always.clone()
    for t in (getattr(gc, "begin_suppress_tokens", None) or []):
        if t < vocab:
            begin[t] = True
    return always, begin


def strip_generated(seq: Sequence[int], eos: int) -> List[int]:
    out = []
    started = False
    for t in seq:
        t = int(t)
        if t == eos and started:
            break
        if t >= eos:
            continue
        started = True
        out.append(t)
    return out


@torch.inference_mode()
def beam_nbest(model, tokenizer, feats: torch.Tensor, beams: int, max_new_tokens: int,
               language: str = "th") -> List[List[int]]:
    # Whisper's generate wrapper repeats the best beam instead of returning distinct N-best sequences.
    from transformers.generation.utils import GenerationMixin
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, int) else gc.eos_token_id[0]
    always, begin = suppress_mask(model, model.proj_out.out_features, feats.device)
    prompt = torch.tensor([prompt_ids(tokenizer, language)] * feats.shape[0], device=feats.device)
    seqs = GenerationMixin.generate(model, input_features=feats, decoder_input_ids=prompt, num_beams=beams,
                                    num_return_sequences=beams, do_sample=False, max_new_tokens=max_new_tokens,
                                    length_penalty=1.0,
                                    suppress_tokens=always.nonzero().flatten().tolist(),
                                    begin_suppress_tokens=begin.nonzero().flatten().tolist())
    return [strip_generated(x[prompt.shape[1]:], eos) for x in seqs.tolist()]


@torch.inference_mode()
def teacher_forced_logp(model, tokenizer, enc: torch.Tensor, seqs: Sequence[Sequence[int]], enc_index: Sequence[int],
                        language: str = "th", chunk: int = 32) -> List[tuple[float, int]]:
    """Sum log p_AM with the campaign's suppression masks, including EOS; never length-normalized."""
    device = enc.device
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, int) else gc.eos_token_id[0]
    always, begin = suppress_mask(model, model.proj_out.out_features, device)
    prompt = prompt_ids(tokenizer, language)
    P = len(prompt)
    out = []
    for s in range(0, len(seqs), chunk):
        part = [list(x)[:MAX_TARGET - P - 1] for x in seqs[s:s + chunk]]
        idx = torch.tensor(list(enc_index[s:s + chunk]), device=device)
        L = max(len(x) for x in part) + 1
        inp = torch.full((len(part), P + L - 1), eos, device=device, dtype=torch.long)
        tgt = torch.full((len(part), L), -1, device=device, dtype=torch.long)
        for i, x in enumerate(part):
            row = (prompt + x)[:P + L - 1]
            inp[i, :len(row)] = torch.tensor(row, device=device)
            tgt[i, :len(x) + 1] = torch.tensor(x + [eos], device=device)
        h = model.model.decoder(input_ids=inp, encoder_hidden_states=enc[idx], use_cache=False,
                                return_dict=True).last_hidden_state[:, P - 1:]
        logits = model.proj_out(h).float()
        m = always[None, None, :].expand(logits.shape[0], logits.shape[1], -1).clone()
        m[:, 0] = begin
        lp = torch.log_softmax(logits.masked_fill(m, float("-inf")), dim=-1)
        g = lp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
        g = torch.where(tgt >= 0, g, torch.zeros_like(g))
        for i, x in enumerate(part):
            out.append((max(float(g[i].sum()), -1e4), len(x) + 1))
    return out


@torch.inference_mode()
def greedy_tokens(model, tokenizer, enc, cap):
    always, begin = suppress_mask(model, model.proj_out.out_features, enc.device)
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, int) else gc.eos_token_id[0]
    inp = torch.tensor([prompt_ids(tokenizer)], device=enc.device)
    cache, tokens = None, []
    for step in range(cap):
        o = model.model.decoder(input_ids=inp, encoder_hidden_states=enc, past_key_values=cache,
                                use_cache=True, output_attentions=False, return_dict=True)
        cache = o.past_key_values
        logits = model.proj_out(o.last_hidden_state[:, -1]).float()
        logits = logits.masked_fill((begin if step == 0 else always)[None, :], float("-inf"))
        token = int(logits[0].argmax())
        if token == eos:
            return tokens, False
        tokens.append(token)
        inp = torch.tensor([[token]], device=enc.device)
    return tokens, True


@torch.inference_mode()
def whisper_candidates(be, audio):
    f = be.proc.feature_extractor([audio], sampling_rate=SR, return_tensors="pt")
    feats = f.input_features.to(be.device, be.torch_dtype)
    enc = be.model.model.encoder(input_features=feats).last_hidden_state
    # shortcut: per-file beam cap; use manifest batching to reproduce batch-dependent cap flags.
    cap = token_cap(len(audio) / SR)
    toks = beam_nbest(be.model, be.proc.tokenizer, feats, 8, cap)
    greedy, hit = greedy_tokens(be.model, be.proc.tokenizer, enc, cap)
    caps = [len(t) >= cap for t in toks] + [hit]
    toks.append(greedy)
    scores = teacher_forced_logp(be.model, be.proc.tokenizer, enc, toks, [0] * len(toks))
    return [{"text": be.proc.tokenizer.decode(t, skip_special_tokens=True).strip(),
             "att_logp": lp, "n_tokens": n, "hit_cap": hit}
            for t, (lp, n), hit in zip(toks, scores, caps)]
