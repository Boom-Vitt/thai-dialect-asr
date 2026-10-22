#!/usr/bin/env python3
"""ถอดเสียงด้วย Whisper ที่ปล่อยบน Hugging Face (greedy, +lg หรือ Korat full system)

  python scripts/transcribe_whisper.py --model <hf-repo-or-dir> audio.wav [more.wav ...]
  python scripts/transcribe_whisper.py --model <hf-repo-or-dir> --loopguard loopguard-flags.txt audio.wav
  python scripts/transcribe_whisper.py --model <hf-repo-or-dir> --rescore rescore.json --lm lm/korat.char6.json.gz audio.wav

เสียงยาวใช้ chunker ของ eval_v2; --rescore ใช้ loopguard-flags.txt จาก repo โมเดลหากไม่ระบุ --loopguard"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bignose_thai_asr.eval_v2.audio import SR, read_wav, split_long  # noqa: E402
from bignose_thai_asr.eval_v2.backends import ModelSpec, load_backend  # noqa: E402
from bignose_thai_asr.eval_v2.loopguard import retry_utterance  # noqa: E402
from bignose_thai_asr.lm_rescore import MAX_S, add_lm, guard_from_flags, make_encoder, pick  # noqa: E402
from bignose_thai_asr.ngram import KNTokenLM  # noqa: E402


def resolve(model, name):
    p = Path(model) / name
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id หรือโฟลเดอร์โมเดล")
    ap.add_argument("--loopguard", help="ไฟล์ --lg-* flags เป็น path หรือชื่อไฟล์ใน repo โมเดล")
    ap.add_argument("--rescore", help="rescore.json (lm_w, bonus, ctc_w) เป็น path หรือชื่อไฟล์ใน repo โมเดล")
    ap.add_argument("--lm", help="char n-gram LM (.json.gz) เป็น path หรือชื่อไฟล์ใน repo โมเดล")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    if bool(a.rescore) != bool(a.lm):
        ap.error("--rescore and --lm must be supplied together")

    def asset(name):
        return name if Path(name).exists() else resolve(a.model, name)

    guard = None
    if a.loopguard or a.rescore:
        guard = guard_from_flags(Path(asset(a.loopguard or "loopguard-flags.txt")).read_text(encoding="utf-8"))
    if a.rescore:
        from bignose_thai_asr.ctc_judge import CTCJudge
        from bignose_thai_asr.decoding import whisper_candidates
        params = json.loads(Path(asset(a.rescore)).read_text(encoding="utf-8"))
        if not isinstance(params, dict) or any(type(params.get(k)) not in (int, float) or not math.isfinite(params[k])
                                              for k in ("lm_w", "bonus", "ctc_w")):
            ap.error("--rescore requires finite numeric lm_w, bonus and ctc_w")
        lm = KNTokenLM.load_json(asset(a.lm))
        encode, _ = make_encoder("char")
    be = load_backend(ModelSpec("whisper", a.model, "whisper"), a.device, "auto", 448)
    fallback, judge = None, None

    for path in a.audio:
        audio = read_wav(path)
        chunks = split_long(audio)
        res = be.transcribe(chunks)
        hyp, cap = " ".join(t for t, _ in res if t), any(c for _, c in res)
        duration = len(audio) / SR
        if guard is not None:
            if fallback is None and guard.flagged(hyp, cap, duration):
                fallback = load_backend(ModelSpec("typhoon", "typhoon-ai/typhoon-whisper-large-v3", "whisper"),
                                        a.device, "auto", 448)
            devices = [torch.cuda.current_device()] if a.device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                seed = guard.seed + int(hashlib.sha256(Path(path).stem.encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                hyp, _, _ = retry_utterance(be, chunks, hyp, cap, guard, fallback)
        if a.rescore and duration <= MAX_S:
            cands = whisper_candidates(be, audio)
            for c in cands:
                c["flagged"] = guard.flagged(c["text"], c["hit_cap"], duration)
            if any(not c["flagged"] for c in cands):
                add_lm(cands, lm, encode)
                if params["ctc_w"]:
                    if judge is None:
                        judge = CTCJudge(device=a.device, dtype=torch.bfloat16 if a.device == "cuda" else torch.float32)
                    lp, frames = judge.log_probs([audio])
                    texts = sorted({c["text"] for c in cands})
                    scores = dict(zip(texts, judge.score(lp[0], int(frames[0]), texts)))
                    for c in cands:
                        c["ctc_logp"] = scores[c["text"]]
                        c["ctc_n"] = len(judge.text_to_ids(c["text"]))
            hyp = pick(cands, params, hyp, duration)
        print(f"{path}\t{hyp}", flush=True)


if __name__ == "__main__":
    main()
