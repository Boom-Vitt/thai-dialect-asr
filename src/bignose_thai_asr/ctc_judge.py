"""Frozen Typhoon character-CTC judge used by the release campaign."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, d_in, d, n_layers, n_heads, ffn, n_sym):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.act = nn.GELU()
        layer = nn.TransformerEncoderLayer(d, n_heads, ffn, activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_sym)

    def forward(self, h, pad_mask=None):
        return self.out(self.ln(self.layers(self.act(self.proj(h)), src_key_padding_mask=pad_mask)))


class CTCJudge:
    def __init__(self, repo="typhoon-ai/typhoon-whisper-large-v3-ctc", device="cuda", dtype=torch.bfloat16):
        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor, WhisperModel
        if Path(repo).is_dir():
            fetch = lambda f: str(Path(repo) / f)  # noqa: E731
        else:
            from huggingface_hub import hf_hub_download
            # The frozen judge snapshot used by the campaign; its config pins the encoder separately.
            rev = "4b7b7b836fb98d58972a1a44cf2b8ced56564345" if repo == "typhoon-ai/typhoon-whisper-large-v3-ctc" else None
            fetch = lambda f: hf_hub_download(repo, f, revision=rev)  # noqa: E731
        cfg = json.loads(Path(fetch("config.json")).read_text(encoding="utf-8"))
        self.symbols = json.loads(Path(fetch("ctc_vocab.json")).read_text(encoding="utf-8"))["symbols"]
        self.sym2id = {s: i for i, s in enumerate(self.symbols)}
        sd = load_file(fetch("head.safetensors"))
        n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("layers.layers."))
        self.head = _Head(sd["proj.weight"].shape[1], sd["proj.weight"].shape[0], n_layers, int(cfg["n_heads"]),
                          sd["layers.layers.0.linear1.weight"].shape[0], len(self.symbols))
        self.head.load_state_dict(sd, strict=True)
        self.device = device
        self.head.to(device).eval().requires_grad_(False)
        rev = cfg.get("encoder_revision")
        self.fe = WhisperFeatureExtractor.from_pretrained(cfg["encoder_id"], revision=rev)
        self.encoder = WhisperModel.from_pretrained(cfg["encoder_id"], revision=rev,
                                                    torch_dtype=dtype).encoder.to(device).eval().requires_grad_(False)
        self.enc_dtype = dtype

    def text_to_ids(self, text):
        return [self.sym2id[c] for c in text if not c.isspace() and c in self.sym2id and c != " "]

    @torch.inference_mode()
    def log_probs(self, audios):
        fb = self.fe(list(audios), sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
        mel_len = fb.attention_mask.sum(-1)
        frames = (mel_len - 1) // 2 + 1
        h = self.encoder(input_features=fb.input_features.to(self.device, self.enc_dtype)).last_hidden_state
        T = int(frames.max())
        h = h[:, :T].float()
        pad = torch.arange(T, device=self.device)[None, :] >= frames.to(self.device)[:, None]
        lp = torch.log_softmax(self.head(h, pad_mask=pad), dim=-1)
        return lp, frames

    @torch.inference_mode()
    def score(self, lp, n_frames, texts):
        out = []
        x = lp[:n_frames]
        for t in texts:
            ids = self.text_to_ids(t)
            if not ids:
                out.append(float("-inf"))
                continue
            if len(ids) > n_frames:
                out.append(-1e4)
                continue
            loss = F.ctc_loss(x.unsqueeze(1), torch.tensor([ids], device=x.device), torch.tensor([n_frames]),
                              torch.tensor([len(ids)]), blank=0, reduction="sum", zero_infinity=False)
            out.append(-float(loss))
        return out
