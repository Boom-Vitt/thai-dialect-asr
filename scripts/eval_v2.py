#!/usr/bin/env python3
"""bignose-thai-asr eval v2 — fair Thai ASR benchmark (models × datasets, greedy, identical normalization).

  # everything on one GPU
  python scripts/eval_v2.py run --run v2-main \
      --models typhoon=typhoon-ai/typhoon-whisper-large-v3 thonburian=biodatlab/whisper-th-large-v3-combined \
               w2v2=airesearch/wav2vec2-large-xlsr-53-th qwen3=Qwen/Qwen3-ASR-1.7B-hf \
      --datasets fleurs tvspeech gs2_typhoon cv

  # the same, split across 4 pods/GPUs (each runs 1/4 of every test set), then merge
  python scripts/eval_v2.py run --run v2-main --shard 0/4 ...   (pod 0) …  --shard 3/4 (pod 3)
  python scripts/eval_v2.py merge results/v2/v2-main [more dirs copied from other pods]

  python scripts/eval_v2.py prepare --datasets fleurs tvspeech cv     # pre-download/cache audio only
  python scripts/eval_v2.py list                                      # known datasets / backends
Model spec: name=[whisper:|w2v2:|nemo:|qwen3asr:]hf_id_or_local_path[@revision]   (backend auto-detected)
"""
import argparse
import json
import os
import socket
import sys
from pathlib import Path

# the GPU pod lost internet egress on 2026-10-20 (~14:30 UTC): use the local HF cache instead of failing every hub model.
# Must run before huggingface_hub is imported, because it reads HF_HUB_OFFLINE at import time.
if "HF_HUB_OFFLINE" not in os.environ:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("[offline] huggingface.co unreachable -> HF_HUB_OFFLINE=1 (pinned revisions served from the local cache)", flush=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bignose_thai_asr.eval_v2.backends import BACKENDS  # noqa: E402
from bignose_thai_asr.eval_v2.datasets import LOADERS, DatasetUnavailable, default_cache, materialize  # noqa: E402


def _dopts(a):
    return {"cache": str(a.cache), "cv_source": a.cv_source, "cv_split_tsv": a.cv_split_tsv, "gowajee_dir": a.gowajee_dir,
            "elderly_dir": a.elderly_dir, "gs2_audio_dir": a.gs2_audio_dir, "allow_download": a.allow_download}


def _shard(s):
    i, n = (int(x) for x in s.split("/"))
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError("--shard must be i/n with 0 <= i < n")
    return i, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("--datasets", nargs="+", required=True, help=f"{sorted(LOADERS)} or manifest:<name>=<path.jsonl>")
        p.add_argument("--cache", type=Path, default=default_cache(), help="audio cache (env EVAL_V2_CACHE)")
        p.add_argument("--limit", type=int, default=0, help="first N utts per dataset (SMOKE TEST ONLY)")
        p.add_argument("--cv-source", default="hf:fsicoli/common_voice_17_0",
                       help="local:/path/cv-corpus-XX/th (Mozilla Data Collective) or hf:<mirror>")
        p.add_argument("--cv-split-tsv", default=None, help="override CV test list (tsv with path, sentence)")
        p.add_argument("--gowajee-dir", default=None)
        p.add_argument("--elderly-dir", default=None)
        p.add_argument("--gs2-audio-dir", default=None, help="local dir of GigaSpeech2 th test audio (<audio_id>.wav)")
        p.add_argument("--allow-download", action="store_true", help="allow big non-HF downloads (Gowajee gdrive 1.5GB, Elderly 5GB)")

    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", required=True)
    data_args(r)
    r.add_argument("--run", required=True, help="run name -> results/v2/<run>/")
    r.add_argument("--out", type=Path, default=ROOT / "results" / "v2")
    r.add_argument("--shard", type=_shard, default=(0, 1), help="i/n: this process evaluates utterances with index %% n == i")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=448)
    r.add_argument("--device", default=None)
    r.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    r.add_argument("--yamok", default="keep", choices=["keep", "expand"])
    r.add_argument("--overwrite", action="store_true", help="re-run model×dataset pairs that already have outputs")
    r.add_argument("--no-merge", action="store_true")
    r.add_argument("--loop-guard", action="store_true", help="opt-in Whisper retry variant (+lg); tune on dev only")
    r.add_argument("--fallback-model", help="model spec: final greedy fallback after all retries fail")
    r.add_argument("--lg-no-repeat", type=int, default=0, help="retry no_repeat_ngram_size")
    r.add_argument("--lg-loop-min-repeats", type=int, default=8)
    r.add_argument("--lg-loop-min-span", type=int, default=40)
    r.add_argument("--lg-loop-min-repeats-unigram", type=int, default=10)
    r.add_argument("--lg-compression-max", type=float, default=2.4, help="char-zlib and UTF-8 gzip threshold")
    r.add_argument("--lg-max-cps", type=float, default=25.0, help="hyp chars per second of audio that counts as a blow-up (reference-free)")
    r.add_argument("--lg-seed", type=int, default=20261020)

    p = sub.add_parser("prepare")
    data_args(p)

    m = sub.add_parser("merge")
    m.add_argument("run_dirs", nargs="+", type=Path)
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--allow-partial", action="store_true")

    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "list":
        print(json.dumps({"datasets": sorted(LOADERS) + ["manifest:<name>=<path>"], "backends": sorted(BACKENDS)}, indent=2))
    elif a.cmd == "prepare":
        bad = 0
        for d in a.datasets:
            try:
                info, rows = materialize(d, _dopts(a), limit=a.limit)
                print(json.dumps({"dataset": info.name, "n": len(rows), "hours": round(sum(x["duration"] for x in rows) / 3600, 3),
                                  "max_s": max(x["duration"] for x in rows), "revision": info.revision, "source": info.source},
                                 ensure_ascii=False))
            except DatasetUnavailable as e:
                bad += 1
                print(json.dumps({"dataset": d, "unavailable": str(e)}, ensure_ascii=False))
            except Exception as e:
                bad += 1
                print(json.dumps({"dataset": d, "error": f"{type(e).__name__}: {str(e)[:300]}"}, ensure_ascii=False))
        sys.exit(1 if bad == len(a.datasets) else 0)
    elif a.cmd == "run":
        from bignose_thai_asr.eval_v2.loopguard import LoopGuard
        from bignose_thai_asr.eval_v2.report import merge_run
        from bignose_thai_asr.eval_v2.runner import run_eval
        if not a.loop_guard and (a.fallback_model or any(x.startswith("--lg-") for x in sys.argv[1:])):
            ap.error("--fallback-model and --lg-* require --loop-guard")
        guard = None
        if a.loop_guard:
            try:
                guard = LoopGuard(no_repeat=a.lg_no_repeat, loop_min_repeats=a.lg_loop_min_repeats,
                                  loop_min_span=a.lg_loop_min_span, loop_min_repeats_unigram=a.lg_loop_min_repeats_unigram,
                                  compression_max=a.lg_compression_max, max_cps=a.lg_max_cps,
                                  fallback_model=a.fallback_model, seed=a.lg_seed)
            except ValueError as e:
                ap.error(str(e))
        run_dir, phash = run_eval(a.models, a.datasets, out=a.out, run=a.run, shard=a.shard, batch_size=a.batch_size,
                                  max_new_tokens=a.max_new_tokens, device=a.device, dtype=a.dtype, limit=a.limit,
                                  yamok=a.yamok, dataset_opts=_dopts(a), overwrite=a.overwrite, loop_guard=guard)
        print(f"[protocol] {phash}  outputs: {run_dir}")
        if a.shard[1] == 1 and not a.no_merge:
            rows, problems = merge_run([run_dir])
            print(f"[merge] {len(rows)} rows -> {run_dir}/leaderboard.md", *problems, sep="\n")
        else:
            print("[merge] sharded run: copy every pod's results/v2/<run>/ here, then: scripts/eval_v2.py merge <dirs>")
    elif a.cmd == "merge":
        from bignose_thai_asr.eval_v2.report import merge_run
        rows, problems = merge_run(a.run_dirs, a.out, allow_partial=a.allow_partial)
        print(f"[merge] {len(rows)} rows -> {(a.out or a.run_dirs[0])}/leaderboard.md")
        for p in problems:
            print("  !", p)


if __name__ == "__main__":
    main()
