"""BoomBigNose Lab: Thai dialect ASR demo (w2v-BERT 2.0 CTC + char n-gram LM).

  pip install -e . "gradio==5.50.0" && python demo/app.py
"""
import functools
import json

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, Wav2Vec2BertForCTC

from bignose_thai_asr.ctc_beam import decode_utt, greedy_ids, ids_to_text
from bignose_thai_asr.eval_v2.audio import SR, split_long, to_mono16k
from bignose_thai_asr.ngram import KNTokenLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_S = 60  # shortcut: longer audio is cut to keep the demo responsive; use scripts/transcribe.py for long files

SYSTEMS = {
    "มลายูปัตตานี · w2v-BERT + LM (CER 12.71)": ("BoomBigNoseLab/w2v-bert-thai-pattani-ctc", "lm/pattani.char6.json.gz", 0.3),
    "มลายูปัตตานี · greedy (CER 14.57)": ("BoomBigNoseLab/w2v-bert-thai-pattani-ctc", None, 0.0),
    "คำเมือง · greedy (CER 6.79)": ("BoomBigNoseLab/w2v-bert-thai-khummuang-ctc", None, 0.0),
}


@functools.lru_cache(maxsize=None)
def load_model(repo):
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = Wav2Vec2BertForCTC.from_pretrained(repo).to(DEVICE).eval()
    vocab = json.load(open(hf_hub_download(repo, "vocab.json"), encoding="utf-8"))
    id2tok = [None] * len(vocab)
    for t, i in vocab.items():
        id2tok[i] = t
    return fe, model, id2tok


@functools.lru_cache(maxsize=None)
def load_lm(repo, name):
    return KNTokenLM.load_json(hf_hub_download(repo, name))


def transcribe(audio, system):
    if audio is None:
        raise gr.Error("อัดเสียงหรืออัปโหลดไฟล์ก่อนครับ")
    sr, data = audio
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    wav = to_mono16k(data, sr)[: MAX_S * SR]
    if wav.size < SR // 10:
        raise gr.Error("เสียงสั้นเกินไป")
    repo, lm_name, lm_w = SYSTEMS[system]
    fe, model, id2tok = load_model(repo)
    logps = []
    for chunk in split_long(wav):
        f = fe(chunk, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            lg = model(f.input_features.to(DEVICE), attention_mask=f.attention_mask.to(DEVICE)).logits[0]
        logps.append(lg.float().log_softmax(-1).cpu().numpy())
    if lm_name is None:
        return " ".join(x for x in (ids_to_text(greedy_ids(lp), id2tok) for lp in logps) if x)
    return decode_utt(logps, id2tok, load_lm(repo, lm_name), lm_w, 1.0, 8)


CSS = """
.gradio-container {max-width: 960px !important; margin: auto}
#hero {text-align: center; padding: 8px 0 4px}
#hero h1 {font-size: 2.1rem; margin-bottom: .2rem}
#hero p {opacity: .8}
footer {display: none !important}
"""

THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="sky", font=[gr.themes.GoogleFont("IBM Plex Sans Thai"), "sans-serif"])

with gr.Blocks(title="BoomBigNose Lab · Thai Dialect ASR", theme=THEME, css=CSS) as demo:
    gr.HTML(
        """<div id="hero"><h1>🐼 BoomBigNose Lab</h1>
        <p>ถอดเสียงภาษาไทยถิ่น: มลายูปัตตานีและคำเมือง · w2v-BERT 2.0 CTC + character n-gram LM</p></div>"""
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="เสียง (สูงสุด 60 วินาที)")
            system = gr.Dropdown(list(SYSTEMS), value=list(SYSTEMS)[0], label="โมเดล")
            btn = gr.Button("ถอดเสียง", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="ข้อความ", lines=8, show_copy_button=True)
    btn.click(transcribe, [audio, system], out)
    gr.Markdown(
        """CER วัดบนชุดทดสอบ SLSCU ด้วย protocol eval_v2 (ยิ่งต่ำยิ่งดี) · ครั้งแรกต้องดาวน์โหลดโมเดลราว 1.2 GB
        · ผลถอดมลายูปัตตานีเป็นอักษรไทยแบบที่ SLSCU ใช้ · ใช้เพื่อการวิจัยและไม่ใช่เชิงพาณิชย์ (CC BY-NC-SA 4.0)

[GitHub](https://github.com/Boom-Vitt/thai-dialect-asr) · [โมเดลทั้งหมด](https://huggingface.co/BoomBigNoseLab)"""
    )

demo.queue(default_concurrency_limit=1).launch()
