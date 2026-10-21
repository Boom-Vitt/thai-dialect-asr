"""models × datasets evaluation loop with deterministic sharding.

Outputs (under <out>/<run>/):
  utts/<dataset>__<model>.s<i>of<n>.jsonl   one row per utterance (ref, hyp, counts, hallucination flags)
  meta/<dataset>__<model>.s<i>of<n>.json    model id + resolved revision sha, dataset source + revision, n, date,
                                             protocol + hash, device, timing (RTF)
  skipped.jsonl                              datasets/backends that could not run, with the reason
  summary.{csv,json}, leaderboard.md        (written by report.merge_run; automatically when n_shards == 1)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

from . import hallucination
from .audio import SR, read_wav, split_long
from .backends import BackendUnavailable, ModelSpec, load_backend
from .datasets import DatasetUnavailable, materialize
from .metrics import score_pair, tokenizer_available
from .loopguard import retry_utterance
from .protocol import protocol_dict, protocol_hash


def shard_rows(rows, shard: int, n_shards: int):
    return [r for i, r in enumerate(rows) if i % n_shards == shard]


def _now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _log_skip(run_dir: Path, kind: str, what: str, reason: str):
    with open(run_dir / "skipped.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "what": what, "reason": reason, "time": _now()}, ensure_ascii=False) + "\n")
    print(f"[skip] {kind} {what}: {reason}", flush=True)


def run_eval(models, datasets, *, out: Path, run: str, shard=(0, 1), batch_size=16, max_new_tokens=448,
             device=None, dtype="auto", limit=0, yamok="keep", dataset_opts=None, overwrite=False, loop_guard=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(out) / run
    (run_dir / "utts").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    proto = protocol_dict(yamok=yamok, batch_size=batch_size, max_new_tokens=max_new_tokens, loop_guard=loop_guard)
    phash = protocol_hash(proto)
    (run_dir / f"protocol.{phash}.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")
    si, sn = shard
    tag = f"s{si}of{sn}"
    engines = tuple(e for e in ("newmm", "deepcut") if tokenizer_available(e))
    if "deepcut" not in engines:
        _log_skip(run_dir, "metric", "WER_deepcut", "pythainlp deepcut engine unavailable (pip install onnxruntime)")

    # 1) datasets first (cheap to fail), so a broken loader never wastes GPU time on model loading
    dsets = []
    for spec in datasets:
        try:
            info, rows = materialize(spec, dataset_opts or {}, limit=limit)
            mine = shard_rows(rows, si, sn)
            dsets.append((info, rows, mine))
            print(f"[data] {info.name}: {len(rows)} utts total, {len(mine)} in shard {tag}, "
                  f"{sum(r['duration'] for r in rows)/3600:.3f} h", flush=True)
        except DatasetUnavailable as e:
            _log_skip(run_dir, "dataset", spec, str(e))
        except Exception as e:
            _log_skip(run_dir, "dataset", spec, f"{type(e).__name__}: {str(e)[:300]}")

    # 2) one model at a time over all datasets (load weights once)
    for mspec_s in models:
        spec = ModelSpec.parse(mspec_s)
        if loop_guard is not None:
            spec.name += "+lg"
            for info, _, _ in dsets:
                existing = run_dir / "meta" / f"{info.name}__{spec.name}.{tag}.json"
                if existing.exists() and not overwrite and json.loads(existing.read_text())["protocol_hash"] != phash:
                    raise ValueError(f"{existing}: different loop-guard protocol; use a new --run or --overwrite")
        todo = [d for d in dsets if overwrite or not (run_dir / "meta" / f"{d[0].name}__{spec.name}.{tag}.json").exists()]
        if not todo:
            print(f"[done] {spec.name}: all datasets already present for {tag}", flush=True)
            continue
        be = fallback = None
        try:
            t0 = time.perf_counter()
            be = load_backend(spec, device, dtype, max_new_tokens)
            if loop_guard is not None:
                if be.kind != "whisper":
                    raise ValueError("--loop-guard requires a Whisper primary backend")
                be.decode_cfg = {**be.decode_cfg, "temperature_fallback": [0.2, 0.4, 0.6],
                                 "loop_guard": loop_guard.protocol()}
                if loop_guard.fallback_model:
                    fallback = load_backend(ModelSpec.parse(loop_guard.fallback_model), device, dtype, max_new_tokens)
            load_s = time.perf_counter() - t0
        except BackendUnavailable as e:
            if be is not None:
                be.close()
            _log_skip(run_dir, "model", f"{spec.name}={spec.model_id}", str(e))
            continue
        except Exception as e:
            if be is not None:
                be.close()
            _log_skip(run_dir, "model", f"{spec.name}={spec.model_id}", f"{type(e).__name__}: {str(e)[:300]}")
            continue
        print(f"[model] {spec.name} backend={be.kind} rev={be.sha} dtype={be.dtype_name} load={load_s:.1f}s", flush=True)
        warmed = False
        for info, rows, mine in todo:
            try:
                res = _eval_one(be, info, rows, mine, batch_size, yamok, engines, warm=not warmed,
                                loop_guard=loop_guard, fallback=fallback)
                warmed = True
            except Exception as e:
                _log_skip(run_dir, "model×dataset", f"{spec.name}×{info.name}",
                          f"{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-800:]}")
                continue
            per, timing = res
            base = f"{info.name}__{spec.name}.{tag}"
            with open(run_dir / "utts" / f"{base}.jsonl", "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in per)
            meta = {"run": run, "dataset": info.name, "dataset_source": info.source, "dataset_revision": info.revision,
                    "dataset_license": info.license, "dataset_split": info.split, "dataset_notes": info.notes,
                    "n_total": len(rows), "n_shard": len(mine), "shard": tag, "limit": limit,
                    "model": spec.name, "model_id": spec.model_id, "model_revision": be.sha, "backend": be.kind,
                    "dtype": be.dtype_name, "decode_cfg": be.decode_cfg, "protocol_hash": phash,
                    "wer_engines": list(engines), "device": device,
                    "gpu": (torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu"),
                    "date": _now(), **timing}
            if loop_guard is not None:
                meta["decoding_variant"] = loop_guard.protocol()
                if fallback is not None:
                    meta["fallback_model"] = {"model_id": fallback.spec.model_id, "model_revision": fallback.sha,
                                              "backend": fallback.kind, "dtype": fallback.dtype_name,
                                              "decode_cfg": fallback.decode_cfg}
            (run_dir / "meta" / f"{base}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            hs = hallucination.summarize(per)
            cer = 100 * sum(p["char_err"] for p in per) / max(1, sum(p["char_n"] for p in per))
            print(f"[result] {info.name} × {spec.name} [{tag}] n={len(per)} CER={cer:.2f}% "
                  f"flagged={hs['n_flag_any']} RTF={timing['RTF']}", flush=True)
        be.close()
        if fallback is not None:
            fallback.close()
    return run_dir, phash


def _eval_one(be, info, rows, mine, batch_size, yamok, engines, warm, loop_guard=None, fallback=None):
    import torch
    # expand to chunks, decode in duration-sorted batches (same order for every model)
    chunks, owner = [], []
    utt_chunks = [[] for _ in mine]
    for ui, r in enumerate(mine):
        for c in split_long(read_wav(r["audio"])):
            chunks.append(c)
            owner.append(ui)
            utt_chunks[ui].append(c)
    order = sorted(range(len(chunks)), key=lambda i: -len(chunks[i]))
    if warm and order:
        be.transcribe([chunks[i] for i in order[:batch_size]])  # warm-up, not timed
    out = [None] * len(chunks)
    sync = (lambda: torch.cuda.synchronize()) if be.device == "cuda" else (lambda: None)
    gen = 0.0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        sync()
        t0 = time.perf_counter()
        res = be.transcribe([chunks[i] for i in idx])
        sync()
        gen += time.perf_counter() - t0
        for i, r in zip(idx, res):
            out[i] = r
    hyps = [[] for _ in mine]
    caps = [False] * len(mine)
    nchunks = [0] * len(mine)
    for i, (t, cap) in enumerate(out):
        hyps[owner[i]].append(t)
        caps[owner[i]] |= cap
        nchunks[owner[i]] += 1
    per = []
    for r, h, cap, nc, original_chunks in zip(mine, hyps, caps, nchunks, utt_chunks):
        hyp = " ".join(x for x in h if x)
        decoding = {}
        if loop_guard is not None:
            sync()
            t0 = time.perf_counter()
            devices = [torch.cuda.current_device()] if be.device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                seed = loop_guard.seed + int(hashlib.sha256(str(r["id"]).encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                hyp, cap, decoding = retry_utterance(be, original_chunks, hyp, cap, loop_guard, fallback)
            sync()
            gen += time.perf_counter() - t0
        s = score_pair(r["text"], hyp, yamok=yamok, engines=engines)
        per.append({"id": r["id"], "duration": r["duration"], "n_chunks": nc, "ref": r["text"], "hyp": hyp, **s,
                    **hallucination.flags(r["text"], hyp, cap), **decoding})
    audio_s = sum(r["duration"] for r in mine)
    return per, {"audio_seconds": round(audio_s, 2), "gen_seconds": round(gen, 2),
                 "RTF": round(gen / audio_s, 5) if audio_s else None, "n_chunks": len(chunks),
                 "n_long_utts": sum(1 for n in nchunks if n > 1)}
