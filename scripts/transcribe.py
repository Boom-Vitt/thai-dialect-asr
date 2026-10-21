#!/usr/bin/env python3
"""ถอดเสียงด้วยโมเดล w2v-BERT 2.0 CTC สำเนียงไทยที่ปล่อยบน Hugging Face (greedy หรือ CTC prefix beam + char LM)

  python scripts/transcribe.py --model <hf-repo-or-dir> audio.wav [more.wav ...]
  python scripts/transcribe.py --model <hf-repo-or-dir> --lm lm/char6.json.gz --lm-w 0.3 --bonus 1.0 audio.wav

เสียงยาวเกิน 30 วินาทีจะถูกตัดด้วย chunker เดียวกับ eval_v2 แล้วต่อข้อความด้วยช่องว่าง"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertForCTC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bignose_thai_asr.ctc_beam import decode_utt, greedy_ids, ids_to_text  # noqa: E402
from bignose_thai_asr.eval_v2.audio import SR, read_wav, split_long  # noqa: E402
from bignose_thai_asr.ngram import KNTokenLM  # noqa: E402


def resolve(model, name):
    p = Path(model) / name
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id หรือโฟลเดอร์โมเดล")
    ap.add_argument("--lm", help="char n-gram LM (.json.gz) เป็น path หรือชื่อไฟล์ใน repo โมเดล")
    ap.add_argument("--lm-w", type=float, default=0.3)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()

    fe = AutoFeatureExtractor.from_pretrained(a.model)
    model = Wav2Vec2BertForCTC.from_pretrained(a.model).to(a.device).eval()
    vocab = json.load(open(resolve(a.model, "vocab.json"), encoding="utf-8"))
    id2tok = [None] * len(vocab)
    for t, i in vocab.items():
        id2tok[i] = t
    lm = KNTokenLM.load_json(a.lm if Path(a.lm).exists() else resolve(a.model, a.lm)) if a.lm else None

    for path in a.audio:
        logps = []
        for chunk in split_long(read_wav(path)):
            f = fe(chunk, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                lg = model(f.input_features.to(a.device), attention_mask=f.attention_mask.to(a.device)).logits[0]
            logps.append(lg.float().log_softmax(-1).cpu().numpy())
        if lm is None:
            text = " ".join(x for x in (ids_to_text(greedy_ids(lp), id2tok) for lp in logps) if x)
        else:
            text = decode_utt(logps, id2tok, lm, a.lm_w, a.bonus, a.beam)
        print(f"{path}\t{text}", flush=True)


if __name__ == "__main__":
    main()
