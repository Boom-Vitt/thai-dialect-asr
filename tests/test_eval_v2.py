"""CPU tests for the eval-v2 harness (no network, no model downloads).  python -m pytest tests/test_eval_v2.py -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bignose_thai_asr.eval_v2 import audio, hallucination, metrics, textnorm  # noqa: E402
from bignose_thai_asr.eval_v2.backends import ModelSpec  # noqa: E402
from bignose_thai_asr.eval_v2.protocol import protocol_dict, protocol_hash  # noqa: E402
from bignose_thai_asr.eval_v2.report import merge_run  # noqa: E402
from bignose_thai_asr.eval_v2.runner import shard_rows  # noqa: E402
from bignose_thai_asr.eval_v2.loopguard import LoopGuard, retry_utterance  # noqa: E402


# ---------------------------------------------------------------- normalization
def test_yamok_keep_is_spacing_invariant():
    assert textnorm.chars("ดี ๆ มาก") == textnorm.chars("ดีๆมาก") == "ดีๆมาก"


def test_yamok_expand():
    assert textnorm.chars("เด็กๆ เล่น", yamok="expand") == "เด็กเด็กเล่น"


def test_thai_digits_and_thousands():
    assert textnorm.normalize("๑,๕๐๐ บาท") == "1500 บาท"
    assert textnorm.chars("๑,๕๐๐", numbers="words") == textnorm.chars("1500", numbers="words") == "หนึ่งพันห้าร้อย"


def test_number_words_variant_equates_digits_and_words():
    assert textnorm.chars("ราคา 25 บาท", numbers="words") == textnorm.chars("ราคายี่สิบห้าบาท", numbers="words")
    assert textnorm.chars("ราคา 25 บาท") != textnorm.chars("ราคายี่สิบห้าบาท")  # raw keeps them different


def test_tags_punct_sara_am_zero_width():
    assert textnorm.chars("คือ [ns] จริง<|th|>!") == "คือจริง"
    assert textnorm.chars("คําว่า") == textnorm.chars("คำว่า")  # decomposed SARA AM (NeMo output)
    assert textnorm.chars("ก\u200bข") == "กข"
    assert textnorm.chars("Hello, World.") == "helloworld"


# ---------------------------------------------------------------- metrics
def test_cer_ignores_spaces_and_counts_errors():
    s = metrics.score_pair("สวัสดี ครับ", "สวัสดีครับ", engines=("newmm",))
    assert s["char_err"] == 0 and s["char_n"] == len("สวัสดีครับ")
    s = metrics.score_pair("กขค", "กxค", engines=())
    assert s["char_err"] == 1 and s["num_char_err"] == 1


def test_wer_newmm_present():
    s = metrics.score_pair("ฉันกินข้าว", "ฉันกินน้ำ", engines=("newmm",))
    assert s["word_newmm_n"] >= 2 and s["word_newmm_err"] >= 1


def test_corpus_rates_are_micro_averaged():
    rows = [{"char_err": 1, "char_n": 10}, {"char_err": 9, "char_n": 90}]
    assert metrics.corpus_rates(rows)["char"] == 10.0


def test_edit_distance_lists():
    assert metrics.edit_distance(["a", "b", "c"], ["a", "c"]) == 1


# ---------------------------------------------------------------- hallucination
@pytest.mark.parametrize("ref", ["เพราะฉะนั้นใน ใน ใน ในอนาคต", "คือจริงจริง จริงจริง อย่างเดียว", "นี่ นี่ นี่ นี่ ภาพนี้"])
def test_no_false_positive_on_natural_repeats(ref):
    assert not hallucination.flags(ref, ref)["flag_any"]


def test_loop_and_blowup_detected():
    f = hallucination.flags("สวัสดีครับ", "สวัสดีครับ" + "นะ" * 40)
    assert f["flag_loop"] and f["flag_blowup"] and f["flag_any"]
    f = hallucination.flags("สวัสดี", "", hit_cap=False)
    assert f["flag_empty"]
    assert hallucination.flags("ก", "ก", hit_cap=True)["flag_cap"]


def test_summary_excl_flagged():
    rows = [{"char_err": 0, "char_n": 10, "flag_any": False}, {"char_err": 50, "char_n": 10, "flag_any": True}]
    s = hallucination.summarize(rows)
    assert s["CER_excl_flagged"] == 0.0 and s["pct_char_err_from_flagged"] == 100.0 and s["n_flag_any"] == 1


# ---------------------------------------------------------------- audio chunking
def test_split_long_respects_limit_and_preserves_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(audio.SR * 75).astype(np.float32) * 0.1
    x[audio.SR * 27:audio.SR * 27 + 800] = 0.0  # a pause inside the search window
    chunks = audio.split_long(x)
    assert len(chunks) == 3
    assert all(len(c) <= audio.SR * audio.CHUNK_MAX_S for c in chunks)
    assert sum(len(c) for c in chunks) == len(x)
    assert abs(len(chunks[0]) - audio.SR * 27) < 1000  # cut lands in the pause
    assert audio.split_long(x[:audio.SR * 10])[0] is not None and len(audio.split_long(x[:audio.SR * 10])) == 1


# ---------------------------------------------------------------- spec / protocol / sharding
def test_model_spec_parse():
    s = ModelSpec.parse("typhoon=typhoon-ai/typhoon-whisper-large-v3@abc123")
    assert (s.name, s.model_id, s.revision, s.backend) == ("typhoon", "typhoon-ai/typhoon-whisper-large-v3", "abc123", None)
    s = ModelSpec.parse("rt=nemo:typhoon-ai/typhoon-asr-realtime")
    assert s.backend == "nemo" and s.model_id == "typhoon-ai/typhoon-asr-realtime"
    assert ModelSpec.parse("openai/whisper-tiny").name == "whisper-tiny"


def test_protocol_hash_stable_and_sensitive():
    a = protocol_hash(protocol_dict())
    assert a == protocol_hash(protocol_dict())
    assert a != protocol_hash(protocol_dict(yamok="expand"))
    assert a != protocol_hash(protocol_dict(batch_size=8))


def test_shards_partition_rows():
    rows = list(range(103))
    parts = [shard_rows(rows, i, 4) for i in range(4)]
    assert sorted(x for p in parts for x in p) == rows
    assert not set(parts[0]) & set(parts[1])


def _fake_shard(run_dir: Path, tag: str, ids, n_total, phash="p1"):
    (run_dir / "utts").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    base = f"ds__m.{tag}"
    with open(run_dir / "utts" / f"{base}.jsonl", "w") as fh:
        for i in ids:
            fh.write(json.dumps({"id": str(i), "char_err": 1, "char_n": 10, "flag_any": False}) + "\n")
    meta = {"dataset": "ds", "model": "m", "model_id": "x/m", "model_revision": "r", "backend": "whisper", "n_total": n_total,
            "n_shard": len(ids), "shard": tag, "limit": 0, "audio_seconds": 10.0, "gen_seconds": 1.0, "gpu": "cpu",
            "dtype": "fp32", "protocol_hash": phash, "dataset_source": "s", "dataset_revision": "d", "date": "2026-10-19"}
    (run_dir / "meta" / f"{base}.json").write_text(json.dumps(meta))


def test_merge_across_pods(tmp_path):
    a, b = tmp_path / "podA", tmp_path / "podB"
    _fake_shard(a, "s0of2", [0, 2, 4], 6)
    _fake_shard(b, "s1of2", [1, 3, 5], 6)
    rows, problems = merge_run([a, b], tmp_path / "out")
    assert not problems and rows[0]["n"] == 6 and rows[0]["complete"] and rows[0]["CER_%"] == 10.0
    assert rows[0]["RTF"] == 0.1
    assert (tmp_path / "out" / "leaderboard.md").exists() and (tmp_path / "out" / "summary.csv").exists()


def test_merge_refuses_missing_shard_and_mixed_protocol(tmp_path):
    a = tmp_path / "a"
    _fake_shard(a, "s0of2", [0, 2, 4], 6)
    rows, problems = merge_run([a], tmp_path / "o1")
    assert rows == [] and "3/6" in problems[0]
    b = tmp_path / "b"
    _fake_shard(b, "s1of2", [1, 3, 5], 6, phash="p2")
    rows, problems = merge_run([a, b], tmp_path / "o2")
    assert rows == [] and "mixed protocol" in problems[0]


# ---------------------------------------------------------------- loop-guard (CPU, no weights)
GOOD = ("สวัสดีครับ", False)
LOOP = ("นะ" * 40, False)


class TinyBackend:
    kind, device, dtype_name, sha = "whisper", "cpu", "fp32", "fake-revision"

    def __init__(self, greedy=GOOD, retries=()):
        self.greedy, self.retries = greedy, iter(retries)
        self.calls, self.decode_cfg = [], {"num_beams": 1}

    def transcribe(self, chunks):
        self.calls.append(("greedy", len(chunks)))
        return [self.greedy] * len(chunks)

    def transcribe_retry(self, chunks, *, temperature, no_repeat_ngram_size):
        self.calls.append((temperature, no_repeat_ngram_size, len(chunks)))
        return [next(self.retries)] * len(chunks)

    def close(self):
        pass


def test_loopguard_unflagged_untouched():
    be, fb = TinyBackend(), TinyBackend()
    hyp, cap, path = retry_utterance(be, [np.zeros(16000)], *GOOD, LoopGuard(), fb)
    assert (hyp, cap) == GOOD and not be.calls and not fb.calls
    assert path == {"decode_path": "greedy", "decode_attempts": [{"path": "greedy", "flagged": False}]}


@pytest.mark.parametrize("passing", range(4))
def test_loopguard_fallback_order_and_stops_at_first_clean(passing):
    be, fb = TinyBackend(retries=[LOOP] * passing + [GOOD]), TinyBackend()
    hyp, cap, path = retry_utterance(be, [np.zeros(16000)], *LOOP, LoopGuard(no_repeat=3), fb)
    assert (hyp, cap) == GOOD and not fb.calls
    assert be.calls == [(t, 3, 1) for t in (0.0, 0.2, 0.4, 0.6)[:passing + 1]]
    assert path["decode_path"] == ("beam5", "temperature_0.2", "temperature_0.4", "temperature_0.6")[passing]
    assert all(a["flagged"] for a in path["decode_attempts"][:-1])
    assert not path["decode_attempts"][-1]["flagged"]


@pytest.mark.parametrize("use_fallback", [False, True])
def test_loopguard_exhaustion_and_fallback_model_last(use_fallback):
    be, fb = TinyBackend(retries=[LOOP] * 4), TinyBackend()
    hyp, cap, path = retry_utterance(be, [np.zeros(16000)], *LOOP, LoopGuard(), fb if use_fallback else None)
    assert len(be.calls) == 4
    assert (hyp, cap) == (GOOD if use_fallback else LOOP)
    assert fb.calls == ([("greedy", 1)] if use_fallback else [])
    assert path["decode_path"] == ("fallback_model" if use_fallback else "temperature_0.6")


def test_loopguard_final_model_output_kept_even_if_flagged():
    be, fb = TinyBackend(retries=[LOOP] * 4), TinyBackend(greedy=LOOP)
    assert retry_utterance(be, [np.zeros(16000)], *LOOP, LoopGuard(), fb)[2]["decode_path"] == "fallback_model"


def test_loopguard_thresholds_cap_blowup_gzip_and_frozen_reporting():
    guard = LoopGuard()
    assert guard.flagged(GOOD[0], True, 10.0)
    hyp = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert not hallucination.flags("a", hyp)["flag_loop"]
    assert guard.flagged(hyp, False, 1.0)  # 52 chars in 1 s > 25 chars/s
    assert not LoopGuard(max_cps=1000).flagged(hyp, False, 1.0)
    assert hallucination.flags("a", hyp)["flag_blowup"]  # reporting thresholds stay frozen
    # UTF-8 byte compression is independent of the detector's one-byte character compression.
    import gzip
    s = "ภาษาไทยเป็นภาษาที่มีเสียงวรรณยุกต์และตัวอักษรที่แตกต่างกัน" * 2
    ratio = len(s.encode()) / len(gzip.compress(s.encode(), mtime=0))
    assert ratio > hallucination.flags(s, s)["compression"]
    threshold = (ratio + hallucination.flags(s, s)["compression"]) / 2
    assert LoopGuard(compression_max=threshold, loop_min_repeats=100, loop_min_span=10000).flagged(s, False, 1000.0)


def test_loopguard_protocol_preserves_baseline_hash_and_labels_all_options():
    from dataclasses import fields, replace
    assert protocol_hash(protocol_dict()) == "f1f03f053b40"
    assert protocol_dict(loop_guard=None) == protocol_dict()
    guard = LoopGuard()
    p = protocol_dict(loop_guard=guard)
    assert p["decoding"]["strategy"] == "greedy+loop_guard"
    assert p["decoding"]["loop_guard"]["reference_free_gating"]
    for f in fields(guard):
        val = getattr(guard, f.name)
        changed = replace(guard, **{f.name: "central=whisper:x/y@rev" if val is None else val + 1})
        assert protocol_hash(protocol_dict(loop_guard=changed)) != protocol_hash(p)


@pytest.mark.parametrize("kw", [{"no_repeat": -1}, {"max_cps": 0}, {"compression_max": float("nan")}])
def test_loopguard_invalid_thresholds(kw):
    with pytest.raises(ValueError):
        LoopGuard(**kw)


def test_loopguard_runner_meta_paths_and_whole_utterance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bignose_thai_asr.eval_v2 import runner
    info = SimpleNamespace(name="ds", source="fake", revision="r", license="test", split="dev", notes="")
    rows = [{"id": "clean", "audio": "clean", "text": GOOD[0], "duration": 1},
            {"id": "long", "audio": "long", "text": GOOD[0] + " " + GOOD[0], "duration": 60}]
    monkeypatch.setattr(runner, "materialize", lambda *a, **k: (info, rows))
    monkeypatch.setattr(runner, "read_wav", lambda p: np.zeros(16 if p == "clean" else audio.SR * 60, dtype=np.float32))
    monkeypatch.setattr(runner, "tokenizer_available", lambda e: False)
    loaded = []

    class MixedBackend(TinyBackend):
        def transcribe(self, chunks):
            self.calls.append(("greedy", len(chunks)))
            return [GOOD if len(c) == 16 else LOOP for c in chunks]

    def load(spec, *a):
        be = MixedBackend(retries=[GOOD]) if spec.model_id == "expert" else TinyBackend()
        be.spec = spec
        loaded.append(be)
        return be

    monkeypatch.setattr(runner, "load_backend", load)
    baseline, basehash = runner.run_eval(["m=expert"], ["ds"], out=tmp_path, run="base", device="cpu")
    guard = LoopGuard(fallback_model="central=whisper:central")
    run, phash = runner.run_eval(["m=expert"], ["ds"], out=tmp_path, run="lg", device="cpu", loop_guard=guard)
    meta = json.loads(next((run / "meta").glob("*.json")).read_text())
    per = [json.loads(l) for l in next((run / "utts").glob("*.jsonl")).read_text().splitlines()]
    baseper = [json.loads(l) for l in next((baseline / "utts").glob("*.jsonl")).read_text().splitlines()]
    assert basehash == "f1f03f053b40" and phash != basehash
    assert per[0] == {**baseper[0], "decode_path": "greedy",
                      "decode_attempts": [{"path": "greedy", "flagged": False}]}
    assert per[1]["decode_path"] == "beam5" and per[1]["n_chunks"] == 3
    assert loaded[1].calls[-1] == (0.0, 0, 3) and not loaded[2].calls
    assert meta["model"] == "m+lg" and meta["decode_cfg"]["loop_guard"] == guard.protocol()
    assert meta["decoding_variant"] == guard.protocol()
    assert meta["fallback_model"]["model_revision"] == "fake-revision"
    with pytest.raises(ValueError, match="different loop-guard protocol"):
        runner.run_eval(["m=expert"], ["ds"], out=tmp_path, run="lg", device="cpu", loop_guard=LoopGuard())


def test_whisper_retry_beam_and_best_of_mean_logprob_through_eos():
    import torch
    from types import SimpleNamespace
    from bignose_thai_asr.eval_v2.backends import WhisperBackend
    be = object.__new__(WhisperBackend)
    be.device, be.dtype_name, be.cap = "cpu", "fp32", 4
    calls = []
    sequences = torch.tensor([[1, 2, 3, 4, 10, 0, 0, 0], [1, 2, 3, 4, 11, 12, 13, 0],
                              [1, 2, 3, 4, 14, 0, 0, 0], [1, 2, 3, 4, 15, 0, 0, 0],
                              [1, 2, 3, 4, 16, 0, 0, 0]])

    def generate(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(sequences=sequences if kw["do_sample"] else sequences[:1],
                               scores=(torch.tensor([[5.0]]),))

    def transitions(seqs, scores, normalize_logits):
        assert normalize_logits
        assert scores[0].item() == 1.0  # sampling temperature .2 has been undone
        return torch.tensor([[-1., -1., 0, 0], [-.6, -.6, -.6, -.6], [-2., -2., 0, 0],
                             [-.8, -.8, 0, 0], [-.9, -.9, 0, 0]])

    be.model = SimpleNamespace(generate=generate, compute_transition_scores=transitions,
                               generation_config=SimpleNamespace(eos_token_id=0))
    be.proc = SimpleNamespace(
        feature_extractor=lambda *a, **kw: SimpleNamespace(input_features=torch.zeros(1, 2, 8),
                                                           attention_mask=torch.ones(1, 8)),
        tokenizer=SimpleNamespace(all_special_ids=[0, 1, 2, 3, 4], pad_token_id=0),
        batch_decode=lambda seqs, **kw: [" ".join(str(t) for t in row.tolist() if t > 4) for row in seqs])
    assert be.transcribe_retry([np.zeros(16)], temperature=0, no_repeat_ngram_size=3) == [("10", False)]
    assert calls[-1]["num_beams"] == 5 and calls[-1]["length_penalty"] == 1.0
    assert calls[-1]["no_repeat_ngram_size"] == 3 and not calls[-1]["do_sample"]
    assert be.transcribe_retry([np.zeros(16)], temperature=.2, no_repeat_ngram_size=0) == [("11 12 13", True)]
    assert calls[-1]["num_return_sequences"] == 5 and calls[-1]["num_beams"] == 1
    assert calls[-1]["temperature"] == .2 and calls[-1]["do_sample"]


def test_loopguard_cli_carries_options(tmp_path, monkeypatch):
    import runpy
    from bignose_thai_asr.eval_v2 import runner
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/eval_v2.py"))
    captured = {}

    def run(*a, **kw):
        captured.update(kw)
        return tmp_path, "hash"

    monkeypatch.setattr(runner, "run_eval", run)
    args = ["eval_v2.py", "run", "--run", "test", "--models", "m=x", "--datasets", "fleurs_dev", "--no-merge"]
    monkeypatch.setattr(sys, "argv", args)
    cli["main"]()
    assert captured["loop_guard"] is None
    monkeypatch.setattr(sys, "argv", args + ["--loop-guard", "--fallback-model", "base=whisper:x/y",
                                            "--lg-no-repeat", "3", "--lg-max-cps", "30"])
    cli["main"]()
    assert captured["loop_guard"] == LoopGuard(no_repeat=3, max_cps=30, fallback_model="base=whisper:x/y")
    monkeypatch.setattr(sys, "argv", args + ["--lg-no-repeat", "3"])
    with pytest.raises(SystemExit):
        cli["main"]()


def test_merge_refuses_mixed_fallback_revisions(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path, tag, ids, rev in ((a, "s0of2", [0], "r1"), (b, "s1of2", [1], "r2")):
        _fake_shard(path, tag, ids, 2)
        meta = next((path / "meta").glob("*.json"))
        doc = json.loads(meta.read_text())
        doc["fallback_model"] = {"model_id": "central", "model_revision": rev}
        meta.write_text(json.dumps(doc))
    rows, problems = merge_run([a, b], tmp_path / "out")
    assert not rows and "mixed fallback model" in problems[0]


def test_loop_guard_gating_never_reads_reference():
    """Same hyp/audio must give the same retry decision whatever the reference is (no test leakage)."""
    g = LoopGuard()
    long_hyp = "วันนี้อากาศดีมากเราจึงออกไปเดินเล่นที่สวนสาธารณะใกล้บ้านกับเพื่อนสนิท"  # no repeats
    assert g.flagged(long_hyp, False, 0.5) is True       # ~60 chars in 0.5 s -> blow-up by chars/sec
    assert g.flagged(long_hyp, False, 10.0) is False     # same text over 10 s is plausible speech


def test_manifest_spec_never_reuses_builtin_cache(tmp_path):
    # a stale built-in "fleurs" (test) cache must not be served for manifest:fleurs=<dev>
    import json
    import soundfile as sf
    import numpy as np
    from bignose_thai_asr.eval_v2 import datasets as ds
    cache = tmp_path / "cache"
    (cache / "fleurs").mkdir(parents=True)
    (cache / "fleurs" / "manifest.jsonl").write_text(json.dumps({"id": "TEST", "audio": "x.wav", "text": "test"}) + "\n")
    (cache / "fleurs" / "info.json").write_text(json.dumps(ds.DatasetInfo("fleurs", "hf", "s", "l", "test").__dict__))
    wav = tmp_path / "a.wav"; sf.write(wav, np.zeros(1600, dtype="float32"), 16000)
    dev = tmp_path / "dev.jsonl"; dev.write_text(json.dumps({"id": "DEV", "audio": str(wav), "text": "dev"}) + "\n")
    _, rows = ds.materialize(f"manifest:fleurs={dev}", {"cache": str(cache)})
    assert [r["id"] for r in rows] == ["DEV"]


def test_hf_meta_falls_back_to_pinned_cached_snapshot(tmp_path, monkeypatch):
    # offline pod: a pinned sha already in the HF cache resolves to itself; branch names never do
    import huggingface_hub
    from bignose_thai_asr.eval_v2 import backends
    sha = "748e8a418697d5920e62bced10590072dce90da5"
    snap = tmp_path / "models--org--m" / "snapshots" / sha
    snap.mkdir(parents=True); (snap / "config.json").write_text("{}")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", boom)
    assert backends._hf_meta("org/m", sha[:12]) == (sha, ["config.json"])
    import pytest
    with pytest.raises(backends.BackendUnavailable):
        backends._hf_meta("org/m", "main")
