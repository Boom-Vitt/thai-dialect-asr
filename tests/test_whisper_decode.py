import math

import pytest

from bignose_thai_asr.eval_v2.loopguard import LoopGuard


FLAGS = ("--lg-max-cps 20 --lg-compression-max 2.2 --lg-loop-min-repeats 6 "
         "--lg-loop-min-span 32 --lg-loop-min-repeats-unigram 8 --lg-no-repeat 3")


def test_rescore_uses_sums_and_all_weights():
    from bignose_thai_asr.lm_rescore import total
    c = {"text": "ภาษาไทย", "att_logp": -12.0, "lm_logp": -10.0, "n_tokens": 4,
         "ctc_logp": -20.0, "ctc_n": 5}
    assert total(c, {"lm_w": 0.3, "bonus": 0.5, "ctc_w": 1.0}) == -17.0


@pytest.mark.parametrize("logp,n,want", [(-20.0, 5, -4.0), (-1e4, 5, -1e4),
                                         (-math.inf, 5, -1e4), (math.nan, 5, -1e4), (-2.0, 0, -1e4)])
def test_judge_floor_before_character_normalization(logp, n, want):
    from bignose_thai_asr.lm_rescore import total
    c = {"text": "ไทย", "att_logp": 0.0, "lm_logp": 0.0, "n_tokens": 1,
         "ctc_logp": logp, "ctc_n": n}
    assert total(c, {"lm_w": 0.0, "bonus": 0.0, "ctc_w": 1.0}) == want


def test_pick_excludes_flagged_even_with_best_score():
    from bignose_thai_asr.lm_rescore import pick
    cands = [{"text": text, "flagged": flagged, "att_logp": am, "lm_logp": 0.0, "n_tokens": 2}
             for text, flagged, am in [("วนซ้ำ", True, 0), ("ไทย", False, -20), ("ภาษา", False, -10)]]
    p = {"lm_w": 0.3, "bonus": 0.0, "ctc_w": 0.0}
    assert pick(cands, p, "สำรอง", 30.0) == "ภาษา"


@pytest.mark.parametrize("cands,duration", [([{"flagged": True}], 1.0),
                                          ([{"flagged": False}], 30.001)])
def test_pick_falls_back_for_all_flagged_or_long_audio(cands, duration):
    from bignose_thai_asr.lm_rescore import pick
    assert pick(cands, {}, "ผลจากลูปการ์ด", duration) == "ผลจากลูปการ์ด"


def test_char_encoder_matches_campaign_and_json_lm(tmp_path):
    from bignose_thai_asr.lm_rescore import add_lm, make_encoder
    from bignose_thai_asr.ngram import KNTokenLM
    encode, vocab_size = make_encoder("char")
    assert vocab_size is None
    assert encode(" ทํา\u200b ๑,๕๐๐ A! ดี ๆ ") == [ord(c) for c in "ทำ1500aดีๆ"]
    lm = KNTokenLM(order=2, vocab_size=20).fit([encode("ภาษาไทย")])
    path = tmp_path / "lm.json.gz"
    lm.save_json(path)
    restored = KNTokenLM.load_json(path)
    cands = [{"text": " ภาษา  ไทย "}, {"text": "ภาษา ไทย"}, {"text": ""}]
    add_lm(cands, restored, encode)
    assert cands[0]["lm_logp"] == cands[1]["lm_logp"] == lm.sentence_logp(encode("ภาษาไทย"))
    assert cands[2]["lm_logp"] == lm.logp([], lm.eos)


def test_loopguard_flags_file_detects_synthetic_text():
    from bignose_thai_asr.lm_rescore import guard_from_flags
    guard = guard_from_flags(FLAGS)
    assert guard == LoopGuard(max_cps=20, compression_max=2.2, loop_min_repeats=6,
                              loop_min_span=32, loop_min_repeats_unigram=8, no_repeat=3)
    assert not guard.flagged("วันนี้อากาศดี", False, 2.0)
    assert not guard.flagged("", False, 2.0)
    assert guard.flagged("กข" * 6, False, 10.0)
    assert guard.flagged("ก" * 8, False, 10.0)
    assert guard.flagged("วันนี้อากาศดี", True, 2.0)
    assert guard.flagged("abcdefghijklmnopqrstuvwxyz", False, 1.0)


@pytest.mark.parametrize("flags", ["--lg-no-repeat", "--unknown 3", "--lg-unknown 3", "--lg-seed nan",
                                 "--lg-max-cps 0", "--lg-fallback-model 1"])
def test_bad_flags_fail_clearly(flags):
    from bignose_thai_asr.lm_rescore import guard_from_flags
    with pytest.raises(SystemExit, match="loop-guard flags"):
        guard_from_flags(flags)


def test_teacher_forced_sum_includes_eos_and_suppression():
    import torch
    from types import SimpleNamespace
    from bignose_thai_asr.decoding import teacher_forced_logp
    tok = SimpleNamespace(convert_tokens_to_ids={"<|startoftranscript|>": 5, "<|th|>": 6,
                                                "<|transcribe|>": 7, "<|notimestamps|>": 8}.__getitem__)

    def decoder(input_ids, **kw):
        logits = torch.zeros(*input_ids.shape, 10)
        logits[..., 3] = 100
        logits[..., 5:] = 100
        return SimpleNamespace(last_hidden_state=logits)

    proj = torch.nn.Identity()
    proj.out_features = 10
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=4, suppress_tokens=[3],
                                                              begin_suppress_tokens=[0]),
                            model=SimpleNamespace(decoder=decoder), proj_out=proj)
    enc = torch.zeros(1, 2, 10)
    scores = teacher_forced_logp(model, tok, enc, [[1], [1, 2]], [0, 0])
    assert scores[0] == pytest.approx((-math.log(3) - math.log(4), 2))
    assert scores[1] == pytest.approx((-math.log(3) - 2 * math.log(4), 3))


def test_beam_nbest_bypasses_whisper_wrapper(monkeypatch):
    import torch
    from types import SimpleNamespace
    from transformers.generation.utils import GenerationMixin
    from bignose_thai_asr.decoding import beam_nbest
    tok = SimpleNamespace(convert_tokens_to_ids={"<|startoftranscript|>": 5, "<|th|>": 6,
                                                "<|transcribe|>": 7, "<|notimestamps|>": 8}.__getitem__)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=4, suppress_tokens=[3],
                                                              begin_suppress_tokens=[0]),
                            proj_out=SimpleNamespace(out_features=10))

    def generate(self, **kw):
        assert self is model
        assert kw["num_beams"] == kw["num_return_sequences"] == 2
        assert kw["decoder_input_ids"].tolist() == [[5, 6, 7, 8]]
        assert kw["suppress_tokens"] == [3, 5, 6, 7, 8, 9]
        assert kw["begin_suppress_tokens"] == [0, 3, 5, 6, 7, 8, 9]
        return torch.tensor([[5, 6, 7, 8, 1, 4, 4], [5, 6, 7, 8, 2, 1, 4]])

    monkeypatch.setattr(GenerationMixin, "generate", generate)
    assert beam_nbest(model, tok, torch.zeros(1, 2, 10), 2, 10) == [[1], [2, 1]]


def test_ctc_judge_scores_alignments_and_skips_unknown_chars():
    import torch
    from bignose_thai_asr.ctc_judge import CTCJudge
    judge = object.__new__(CTCJudge)
    judge.sym2id = {"<blank>": 0, "ก": 1}
    assert judge.text_to_ids("ก A \tก") == [1, 1]
    lp = torch.full((2, 2), math.log(0.5))
    scores = judge.score(lp, 2, ["ก", "A", "กกก"])
    assert scores[0] == pytest.approx(math.log(0.75))
    assert scores[1:] == [-math.inf, -1e4]


@pytest.mark.parametrize("mode", ["greedy", "loopguard", "retry", "rescore", "all_flagged", "long"])
def test_cli_modes_without_model_downloads(tmp_path, monkeypatch, capsys, mode):
    import hashlib
    import json
    import runpy
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    from bignose_thai_asr import ctc_judge, decoding
    from bignose_thai_asr.eval_v2 import audio, backends
    from bignose_thai_asr.ngram import KNTokenLM
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "loopguard-flags.txt").write_text(FLAGS, encoding="utf-8")
    (model_dir / "rescore.json").write_text(json.dumps({"lm_w": 0.3, "bonus": 0.0, "ctc_w": 1.0}))
    KNTokenLM(order=2, vocab_size=10).fit([[ord(c) for c in "ภาษา"]]).save_json(model_dir / "lm.json.gz")
    loaded, retries = [], []

    class Backend:
        def __init__(self, fallback):
            self.fallback = fallback

        def transcribe(self, chunks):
            text = "ไต้ฝุ่น" if self.fallback else "กข" * 6 if mode == "retry" else "ต้นฉบับ"
            return [(text, False)] * len(chunks)

        def transcribe_retry(self, chunks, *, temperature, no_repeat_ngram_size):
            assert torch.initial_seed() == 20261020 + int(hashlib.sha256(b"sample").hexdigest()[:8], 16)
            retries.append((temperature, no_repeat_ngram_size))
            return [("กข" * 6, False)] * len(chunks)

    def load(spec, device, dtype, cap):
        loaded.append(spec.model_id)
        assert device == "cpu" and dtype == "auto" and cap == 448
        return Backend(spec.model_id != str(model_dir))

    def candidates(be, waveform):
        assert len(waveform) == 32000
        return [{"text": "กข" * 6 if mode == "all_flagged" else "ภาษา", "att_logp": -1.0,
                 "n_tokens": 2, "hit_cap": False}]

    class Judge:
        def __init__(self, **kw):
            assert mode == "rescore"

        def log_probs(self, audios):
            return [None], [100]

        def score(self, lp, n, texts):
            return [-4.0] * len(texts)

        def text_to_ids(self, text):
            return list(text)

    monkeypatch.setattr(backends, "load_backend", load)
    monkeypatch.setattr(audio, "read_wav", lambda p: np.zeros(496000 if mode == "long" else 32000, dtype=np.float32))
    monkeypatch.setattr(decoding, "whisper_candidates", candidates)
    monkeypatch.setattr(ctc_judge, "CTCJudge", Judge)
    args = ["transcribe_whisper.py", "--model", str(model_dir), "--device", "cpu"]
    if mode in ("loopguard", "retry"):
        args += ["--loopguard", "loopguard-flags.txt"]
    elif mode != "greedy":
        args += ["--rescore", "rescore.json", "--lm", "lm.json.gz"]
    monkeypatch.setattr(sys, "argv", args + ["sample.wav"])
    cli = runpy.run_path(str(root / "scripts/transcribe_whisper.py"))
    rng = torch.get_rng_state()
    cli["main"]()
    assert capsys.readouterr().out == "sample.wav\t" + ("ภาษา" if mode == "rescore" else
                                                     "ต้นฉบับ ต้นฉบับ" if mode == "long" else
                                                     "ไต้ฝุ่น" if mode == "retry" else "ต้นฉบับ") + "\n"
    assert loaded == [str(model_dir)] + (["typhoon-ai/typhoon-whisper-large-v3"] if mode == "retry" else [])
    assert retries == ([(t, 3) for t in (0.0, 0.2, 0.4, 0.6)] if mode == "retry" else [])
    assert torch.equal(rng, torch.get_rng_state())


def test_resolve_prefers_local_files_then_the_model_repo(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    import huggingface_hub
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/transcribe_whisper.py"))
    local = tmp_path / "flags.txt"
    local.write_text(FLAGS)
    calls = []

    def download(repo, name):
        calls.append((repo, name))
        return "cached.json.gz"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert cli["resolve"](str(tmp_path), "flags.txt") == str(local)
    assert not calls
    assert cli["resolve"]("model/repo", "lm/korat.char6.json.gz") == "cached.json.gz"
    assert calls == [("model/repo", "lm/korat.char6.json.gz")]
