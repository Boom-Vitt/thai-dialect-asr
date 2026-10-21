"""ASR backends behind one interface. Every backend gets the SAME 16 kHz float32 chunks (<= 30 s) and decodes
greedily; it returns (text, hit_token_cap) per chunk. Missing optional deps raise BackendUnavailable (skipped
with a clear message by the runner) instead of crashing the run.

Model spec on the CLI:  name=[backend:]hf_id_or_path[@revision]
    backend ∈ {whisper, w2v2, nemo, qwen3asr, espnet, bdh, meralion, omniasr}; omitted -> auto-detected from config.json / *.nemo
    file (meralion and omniasr must be named explicitly).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SR


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class ModelSpec:
    name: str
    model_id: str
    backend: str | None = None
    revision: str | None = None

    @classmethod
    def parse(cls, s: str) -> "ModelSpec":
        name, rest = s.split("=", 1) if "=" in s else (None, s)
        backend = None
        head, sep, tail = rest.partition(":")
        if sep and head in BACKENDS:
            backend, rest = head, tail
        rev = None
        if "@" in rest and not Path(rest).exists():
            rest, rev = rest.rsplit("@", 1)
        return cls(name or rest.rstrip("/").split("/")[-1], rest, backend, rev)


def _hf_meta(model_id: str, revision: str | None):
    """(resolved_revision_sha, file list). Local dirs -> content hash of config files."""
    p = Path(model_id)
    if p.exists():
        import hashlib
        h = hashlib.sha256()
        for f in sorted(p.glob("*")):
            if f.is_file() and f.suffix in (".json", ".nemo", ".safetensors", ".bin"):
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}".encode())
                if f.suffix == ".json":
                    h.update(f.read_bytes())
        return "local:" + h.hexdigest()[:16], [f.name for f in p.glob("*")]
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(model_id, revision=revision)
    except Exception as e:
        snap = _cached_snapshot(model_id, revision)
        if snap is not None:  # offline: a pinned revision already in the HF cache is still exactly that revision
            return snap.name, [str(f.relative_to(snap)) for f in snap.rglob("*") if f.is_file()]
        raise BackendUnavailable(f"{model_id}: cannot resolve on HF Hub ({type(e).__name__}: {str(e)[:150]})") from e
    return info.sha, [s.rfilename for s in info.siblings]


def _cached_snapshot(model_id: str, revision: str | None) -> Path | None:
    """The one cached snapshot dir whose commit sha starts with `revision` (pinned shas only; no branch names)."""
    if not revision or len(revision) < 7 or any(c not in "0123456789abcdef" for c in revision):
        return None
    from huggingface_hub.constants import HF_HUB_CACHE
    d = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    hits = sorted(d.glob(f"{revision}*")) if d.is_dir() else []
    return hits[0] if len(hits) == 1 else None


def detect_backend(model_id: str, files: list[str], revision: str | None) -> str:
    if any(f.endswith(".nemo") for f in files):
        return "nemo"
    try:
        if Path(model_id).exists():
            cfg = json.load(open(Path(model_id) / "config.json"))
        else:
            from huggingface_hub import hf_hub_download
            cfg = json.load(open(hf_hub_download(model_id, "config.json", revision=revision)))
    except Exception as e:
        raise BackendUnavailable(f"{model_id}: no config.json / .nemo to detect backend ({type(e).__name__})") from e
    mt = cfg.get("model_type", "")
    if mt == "whisper":
        return "whisper"
    if mt in ("wav2vec2", "wav2vec2-conformer", "wav2vec2-bert", "hubert", "wavlm", "data2vec-audio", "mms"):
        return "w2v2"
    if mt.startswith("qwen3_asr") or "Qwen3ASR" in "".join(cfg.get("architectures") or []):
        return "qwen3asr"
    raise BackendUnavailable(f"{model_id}: unsupported model_type {mt!r}")


class Backend:
    kind = "base"

    def __init__(self, spec: ModelSpec, sha: str, files, device: str, dtype_name: str, max_new_tokens: int):
        self.spec, self.sha, self.files, self.device, self.dtype_name = spec, sha, files, device, dtype_name
        self.max_new_tokens = max_new_tokens
        self.decode_cfg: dict = {}

    @property
    def torch_dtype(self):
        import torch
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.dtype_name]

    def transcribe(self, chunks: list[np.ndarray]) -> list[tuple[str, bool]]:
        raise NotImplementedError

    def close(self):
        import gc
        for k in list(self.__dict__):
            if k in ("model", "proc"):
                setattr(self, k, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class WhisperBackend(Backend):
    """Any WhisperForConditionalGeneration (openai, Typhoon, Pathumma, Thonburian, distil/turbo, merged ckpts)."""
    kind = "whisper"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as e:
            raise BackendUnavailable(f"whisper: transformers missing ({e})") from e
        mid, rev = self.spec.model_id, self.spec.revision or (None if self.sha.startswith("local:") else self.sha)
        self.proc = WhisperProcessor.from_pretrained(mid, revision=rev)
        self.model = WhisperForConditionalGeneration.from_pretrained(mid, revision=rev, dtype=self.torch_dtype).to(self.device).eval()
        # Whisper's decoder has max_target_positions (448) positions INCLUDING the 4 forced prompt tokens
        # (<|startoftranscript|><|th|><|transcribe|><|notimestamps|>) -> the real ceiling is 444 new tokens.
        self.cap = min(self.max_new_tokens, self.model.config.max_target_positions - 4)
        gc = self.model.generation_config
        for attr, val in (("no_repeat_ngram_size", 0), ("repetition_penalty", 1.0), ("length_penalty", 1.0),
                          ("num_beams", 1), ("do_sample", False)):
            if hasattr(gc, attr):
                setattr(gc, attr, val)
        gc.forced_decoder_ids = None
        gc.max_length = None  # max_new_tokens governs
        self.decode_cfg = {"num_beams": 1, "do_sample": False, "language": "th", "task": "transcribe",
                           "return_timestamps": False, "max_new_tokens": self.cap, "no_repeat_ngram_size": 0,
                           "repetition_penalty": 1.0, "temperature_fallback": False}

    def transcribe(self, chunks):
        import torch
        fe = self.proc.feature_extractor(chunks, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        feats = fe.input_features.to(self.device, self.torch_dtype)
        with torch.inference_mode():
            out = self.model.generate(feats, attention_mask=fe.attention_mask.to(self.device), language="th",
                                      task="transcribe", num_beams=1, do_sample=False, max_new_tokens=self.cap,
                                      return_timestamps=False)
        seqs = out.sequences if hasattr(out, "sequences") else out
        texts = self.proc.batch_decode(seqs, skip_special_tokens=True)
        tok = self.proc.tokenizer
        special = set(tok.all_special_ids)
        res = []
        for row, t in zip(seqs.tolist(), texts):
            n_text = sum(1 for x in row if x not in special and x != tok.pad_token_id)
            res.append((t.strip(), n_text >= self.cap - 1))
        return res

    def transcribe_retry(self, chunks, *, temperature, no_repeat_ngram_size):
        """Beam-5, or best-of-5 sampling per original <=30s chunk. Greedy stays untouched."""
        import torch
        results = []
        for chunk in chunks:
            fe = self.proc.feature_extractor([chunk], sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
            sample = temperature > 0
            with torch.inference_mode():
                out = self.model.generate(
                    fe.input_features.to(self.device, self.torch_dtype),
                    attention_mask=fe.attention_mask.to(self.device), language="th", task="transcribe",
                    max_new_tokens=self.cap, return_timestamps=False, num_beams=1 if sample else 5,
                    length_penalty=1.0, no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=1.0,
                    do_sample=sample, num_return_sequences=5 if sample else 1,
                    return_dict_in_generate=True, output_scores=sample,
                    compression_ratio_threshold=None, logprob_threshold=None, no_speech_threshold=None,
                    **({"temperature": temperature, "top_k": 0, "top_p": 1.0} if sample else {}))
            seqs = out.sequences
            best = 0
            if sample:
                # Undo sampling temperature before ranking, as Whisper's avg_logprob does.
                logits = tuple((s * temperature).float() for s in out.scores)
                scores = self.model.compute_transition_scores(seqs, logits, normalize_logits=True)
                generated = seqs[:, -scores.shape[1]:]
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                averages = []
                for row, logprobs in zip(generated.tolist(), scores):
                    end = next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
                    averages.append(logprobs[:end].mean().item() if end else float("-inf"))
                best = max(range(len(averages)), key=averages.__getitem__)
            tok = self.proc.tokenizer
            row = seqs[best].tolist()
            n_text = sum(x not in tok.all_special_ids and x != tok.pad_token_id for x in row)
            text = self.proc.batch_decode(seqs[best:best + 1], skip_special_tokens=True)[0].strip()
            results.append((text, n_text >= self.cap - 1))
        return results


class W2V2Backend(Backend):
    """CTC models (airesearch/wav2vec2-large-xlsr-53-th, …): greedy argmax CTC, no LM."""
    kind = "w2v2"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as e:
            raise BackendUnavailable(f"w2v2: transformers missing ({e})") from e
        mid, rev = self.spec.model_id, self.spec.revision or (None if self.sha.startswith("local:") else self.sha)
        try:
            self.proc = AutoProcessor.from_pretrained(mid, revision=rev)
        except Exception:
            from transformers import Wav2Vec2Processor
            self.proc = Wav2Vec2Processor.from_pretrained(mid, revision=rev)
        if hasattr(self.proc, "decoder"):  # ProcessorWithLM -> force plain greedy for comparability
            from transformers import Wav2Vec2Processor
            self.proc = Wav2Vec2Processor(feature_extractor=self.proc.feature_extractor, tokenizer=self.proc.tokenizer)
        dt = self.torch_dtype
        self.model = AutoModelForCTC.from_pretrained(mid, revision=rev, dtype=dt).to(self.device).eval()
        self.decode_cfg = {"ctc": "greedy_argmax", "lm": None}

    def transcribe(self, chunks):
        import torch
        fe = self.proc.feature_extractor(chunks, sampling_rate=SR, return_tensors="pt", padding=True,
                                         return_attention_mask=True)
        x = (fe.input_values if "input_values" in fe else fe.input_features).to(self.device, self.torch_dtype)  # w2v-BERT: features
        am = fe.get("attention_mask")
        with torch.inference_mode():
            logits = self.model(x, attention_mask=am.to(self.device) if am is not None else None).logits
        ids = logits.argmax(-1).cpu()
        # trim padded frames so batch padding can't emit tokens
        if am is not None:
            lens = self.model._get_feat_extract_output_lengths(am.sum(-1)).tolist()
            ids = [r[:int(n)] for r, n in zip(ids, lens)]
        texts = [self.proc.tokenizer.decode(r) for r in ids]
        return [(t.strip(), False) for t in texts]


class NemoBackend(Backend):
    """NeMo .nemo checkpoints (typhoon-ai/typhoon-asr-realtime, nectec/Pathumma-stt-th-fastconformer-rnnt-large)."""
    kind = "nemo"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except Exception as e:
            raise BackendUnavailable(f"nemo: `pip install nemo_toolkit[asr]` needed ({type(e).__name__}: {str(e)[:120]})") from e
        nemo_file = next((f for f in self.files if f.endswith(".nemo")), None)
        if nemo_file is None:
            raise BackendUnavailable(f"{self.spec.model_id}: no .nemo file")
        if Path(self.spec.model_id).exists():
            path = str(Path(self.spec.model_id) / nemo_file)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.spec.model_id, nemo_file, revision=self.sha)
        self.model = nemo_asr.models.ASRModel.restore_from(path, map_location=self.device).eval()
        if self.dtype_name != "fp32" and self.device == "cuda":
            self.model = self.model.half() if self.dtype_name == "fp16" else self.model.to(dtype=self.torch_dtype)
        strat = "greedy_batch"
        try:
            from omegaconf import open_dict
            dec = self.model.cfg.decoding
            with open_dict(dec):
                dec.strategy = strat
            self.model.change_decoding_strategy(dec)
        except Exception as e:  # CTC-only or older API: keep model default, record it
            strat = f"model-default ({type(e).__name__})"
        self.decode_cfg = {"strategy": strat, "file": nemo_file}
        self.tmp = tempfile.mkdtemp(prefix="evalv2_nemo_")

    def transcribe(self, chunks):
        import soundfile as sf
        paths = []
        for i, c in enumerate(chunks):
            p = os.path.join(self.tmp, f"{i}.wav")
            sf.write(p, c, SR, subtype="FLOAT")
            paths.append(p)
        out = self.model.transcribe(paths, batch_size=len(paths), verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        texts = [o.text if hasattr(o, "text") else (o[0] if isinstance(o, list) else o) for o in out]
        return [(str(t).strip(), False) for t in texts]


class Qwen3ASRBackend(Backend):
    """Qwen3-ASR (transformers >= 5.13: Qwen/Qwen3-ASR-*-hf) and Typhoon's LoRA-merged typhoon-asr-qwen-*-ctx.
    Base Qwen: language forced to Thai via processor.apply_transcription_request(language='th').
    typhoon-asr-qwen-*-ctx: system prompt 'Thai' + generation prompt, exactly as its model card (no bias list)."""
    kind = "qwen3asr"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError as e:
            import transformers
            raise BackendUnavailable(f"qwen3asr: needs transformers>=5.13 (have {transformers.__version__})") from e
        mid, rev = self.spec.model_id, self.spec.revision or (None if self.sha.startswith("local:") else self.sha)
        self.proc = AutoProcessor.from_pretrained(mid, revision=rev)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(mid, revision=rev, dtype=self.torch_dtype).to(self.device).eval()
        self.typhoon_ctx = "typhoon-asr-qwen" in mid.lower()
        self.decode_cfg = {"do_sample": False, "num_beams": 1, "max_new_tokens": self.max_new_tokens,
                           "prompt": "system='Thai' (typhoon ctx card)" if self.typhoon_ctx else "language=th prefill"}

    def transcribe(self, chunks):
        import torch
        if self.typhoon_ctx:
            convs = [[{"role": "system", "content": [{"type": "text", "text": "Thai"}]},
                      {"role": "user", "content": [{"type": "audio", "audio": c}]}] for c in chunks]
            inputs = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors="pt", padding=True)
        else:
            inputs = self.proc.apply_transcription_request(audio=list(chunks), language="th")
        inputs = inputs.to(self.device)
        for k, v in list(inputs.items()):
            if hasattr(v, "is_floating_point") and v.is_floating_point():
                inputs[k] = v.to(self.torch_dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.proc.decode(gen, return_format="transcription_only")
        pad = getattr(self.proc.tokenizer, "pad_token_id", None)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        caps = [int(sum(1 for x in r if x != pad and x not in eos)) >= self.max_new_tokens for r in gen.tolist()]
        return [(t.strip(), c) for t, c in zip(texts, caps)]


class EspnetBackend(Backend):
    """ESPnet2 ASR checkpoints laid out as <repo>/<subdir>/{config.yaml, *.pth} (SLSCU/thai-dialect_*_model:
    HuBERT encoder + transformer decoder, hybrid CTC/attention, char tokens). Needs espnet==202301 + fairseq (py3.10;
    see docs/north_south_plan.md). Decoding: beam_size=1 (greedy, same as the benchmark protocol), ctc_weight=0.3
    (the training ctc weight), no LM. NOTE: SLSCU's paper used beam 10 -> their published numbers use a stronger search."""
    kind = "espnet"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from espnet2.bin.asr_inference import Speech2Text
        except ImportError as e:
            raise BackendUnavailable(f"espnet: espnet2 missing ({e})") from e
        from huggingface_hub import snapshot_download
        rev = self.spec.revision or (None if self.sha.startswith("local:") else self.sha)
        root = Path(self.spec.model_id) if Path(self.spec.model_id).exists() else Path(snapshot_download(self.spec.model_id, revision=rev))
        cfg = next(root.rglob("config.yaml")); pth = next(root.rglob("*.pth"))
        os.makedirs("downloads/hubert_pretrained_models", exist_ok=True)  # config's hubert_dir_path is relative
        self.beam, self.ctc_weight = int(os.environ.get("ESPNET_BEAM", "1")), float(os.environ.get("ESPNET_CTC_WEIGHT", "0.3"))
        self.model = Speech2Text(asr_train_config=str(cfg), asr_model_file=str(pth), device=self.device,
                                 beam_size=self.beam, ctc_weight=self.ctc_weight, lm_weight=0.0, nbest=1)
        self.decode_cfg = {"beam_size": self.beam, "ctc_weight": self.ctc_weight, "lm": None, "config": str(cfg.relative_to(root))}

    def transcribe(self, chunks):
        import torch
        outs = []
        with torch.inference_mode():
            for c in chunks:
                r = self.model(torch.from_numpy(np.asarray(c, dtype=np.float32)))
                outs.append(((r[0][0] if r else "").strip(), False))
        return outs


class BDHBackend(Backend):
    """BigNose Dialect Hybrid dirs (scripts/train_bdh.py best/, docs/methods/bignose-dialect-hybrid.md): pretrained
    encoder + LoRA, char CTC + attention decoder, joint CTC/attention decoding with the CTC length cap. Env: BDH_BEAM
    (default 1 = greedy, the eval_v2 protocol; 10 = SLSCU paper-style search), BDH_CTC_WEIGHT (default: the training
    config's 0.3), optional char n-gram shallow fusion BDH_LM=<KNTokenLM .pkl.gz over eval_v2 chars> + BDH_LM_WEIGHT."""
    kind = "bdh"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import torch
        from ..bdh import load_bdh
        from ..ngram import KNTokenLM
        if not Path(self.spec.model_id).is_dir():
            raise BackendUnavailable(f"bdh: {self.spec.model_id} is not a local BDH dir")
        self.model, self.cfg = load_bdh(self.spec.model_id, self.device,
                                        base_dtype=self.torch_dtype if self.device == "cuda" else torch.float32)
        self.beam = int(os.environ.get("BDH_BEAM", "1"))
        self.ctc_weight = float(os.environ.get("BDH_CTC_WEIGHT", self.cfg["decode"]["ctc_weight"]))
        lm = os.environ.get("BDH_LM")
        self.lm, self.lm_weight = (KNTokenLM.load(lm), float(os.environ.get("BDH_LM_WEIGHT", "0.3"))) if lm else (None, 0.0)
        self.dialect = self.cfg["target_dialect_id"] if self.cfg["arch"]["n_dialects"] else None
        self.decode_cfg = {"beam": self.beam, "ctc_weight": self.ctc_weight, "lm": lm, "lm_weight": self.lm_weight,
                           "dialect": self.cfg.get("target_dialect"), "length_cap": "min(frames, 2*|greedy CTC|+10)"}

    def transcribe(self, chunks):
        from ..bdh import transcribe
        return transcribe(self.model, list(chunks), self.cfg["vocab"], self.beam, self.ctc_weight, self.lm,
                          self.lm_weight, dialect=self.dialect, device=self.device)


def _snapshot(spec: ModelSpec, sha: str) -> str:
    """Local dir of the model at the resolved revision (so trust_remote_code / checkpoint files are pinned too)."""
    if Path(spec.model_id).exists():
        return spec.model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(spec.model_id, revision=spec.revision or sha)


class MeralionBackend(Backend):
    """MERaLiON/MERaLiON-3-3B-ASR through its own package `meralion-3-asr` (installed --no-deps: its vLLM pin is only
    for the vLLM path). Uses the package's TransformersBackend unchanged = the card's documented decoding: fixed prompt
    'Please transcribe this speech.' (no language hint), greedy, max_new_tokens 512, no_repeat_ngram_size 6, eager
    attention, package batch size 4. Chunks are already <= 30 s, the package's own window, so it does not re-chunk."""
    kind = "meralion"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            from meralion_3_asr.backends.transformers_backend import TransformersBackend
            from meralion_3_asr.prompts import ASR_PROMPT, GENERATION_CONFIG
        except ImportError as e:
            raise BackendUnavailable(f"meralion: `pip install --no-deps meralion-3-asr` needed ({e})") from e
        self.model = TransformersBackend(_snapshot(self.spec, self.sha), device=self.device, dtype=self.torch_dtype)
        self.decode_cfg = {**GENERATION_CONFIG, "prompt": ASR_PROMPT, "batch_size": self.model.batch_size,
                           "attn_implementation": "eager", "package": "meralion-3-asr transformers backend"}

    def transcribe(self, chunks):
        texts = self.model.transcribe_chunks([np.asarray(c, dtype=np.float32) for c in chunks])
        return [(t.strip(), False) for t in texts]  # the package does not expose token counts -> cap unknown


class OmniASRBackend(Backend):
    """facebook/omniASR-LLM-* (fairseq2 wav2vec2-llama) through omnilingual_asr's ASRInferencePipeline with its default
    beam-search config. Language conditioning from env OMNIASR_LANG (default tha_Thai; nod_Thai = Northern Thai).
    The checkpoint and tokenizer are the HF repo files at the pinned revision, wired in through a generated fairseq2
    asset card that inherits the package card `<OMNIASR_CARD>` (default omniASR_LLM_7B) for model_family/arch."""
    kind = "omniasr"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        root = Path(_snapshot(self.spec, self.sha))
        ckpt = next(root.glob("*.pt"), None)
        tok = next(root.glob("*.model"), None)
        if ckpt is None or tok is None:
            raise BackendUnavailable(f"{self.spec.model_id}: need one *.pt checkpoint and one *.model tokenizer")
        self.lang = os.environ.get("OMNIASR_LANG", "tha_Thai")
        card = omni_card(os.environ.get("OMNIASR_CARD", "omniASR_LLM_7B"), ckpt, tok)  # before fairseq2 reads its asset dirs
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError as e:
            raise BackendUnavailable(f"omniasr: `pip install omnilingual-asr` needed ({e})") from e
        self.model = ASRInferencePipeline(model_card=card, device=self.device, dtype=self.torch_dtype)
        self.decode_cfg = {"pipeline": "omnilingual_asr ASRInferencePipeline", "beam_search": "package default",
                           "lang": self.lang, "card": card, "checkpoint": ckpt.name, "tokenizer": tok.name}

    def transcribe(self, chunks):
        audio = [{"waveform": np.asarray(c, dtype=np.float32), "sample_rate": SR} for c in chunks]
        texts = self.model.transcribe(audio, lang=[self.lang] * len(audio), batch_size=len(audio))
        return [(str(t).strip(), False) for t in texts]


def omni_card(base: str, ckpt: Path, tok: Path) -> str:
    """Write fairseq2 asset cards (into $FAIRSEQ2_USER_ASSET_DIR) that point `base` at local files; return the name."""
    d = Path(os.environ.setdefault("FAIRSEQ2_USER_ASSET_DIR", tempfile.mkdtemp(prefix="evalv2_fs2_")))
    d.mkdir(parents=True, exist_ok=True)
    name = f"{base}_evalv2"
    (d / f"{name}.yaml").write_text(
        f"name: {name}_tokenizer\ntokenizer_family: char_tokenizer\ntokenizer: file://{tok}\n\n---\n\n"
        f"name: {name}\nbase: {base}\ncheckpoint: file://{ckpt}\ntokenizer_ref: {name}_tokenizer\n", encoding="utf-8")
    return name


BACKENDS = {"whisper": WhisperBackend, "w2v2": W2V2Backend, "nemo": NemoBackend, "qwen3asr": Qwen3ASRBackend,
            "espnet": EspnetBackend, "bdh": BDHBackend, "meralion": MeralionBackend, "omniasr": OmniASRBackend}


def load_backend(spec: ModelSpec, device: str, dtype: str, max_new_tokens: int) -> Backend:
    sha, files = _hf_meta(spec.model_id, spec.revision)
    kind = spec.backend or detect_backend(spec.model_id, files, spec.revision or (None if sha.startswith("local:") else sha))
    spec.backend = kind
    if dtype == "auto":
        dtype = ("fp16" if kind in ("whisper", "w2v2", "nemo") else "bf16") if device == "cuda" else "fp32"
    try:
        return BACKENDS[kind](spec, sha, files, device, dtype, max_new_tokens)
    except BackendUnavailable:
        raise
    except ImportError as e:
        raise BackendUnavailable(f"{kind}: missing dependency ({e})") from e
