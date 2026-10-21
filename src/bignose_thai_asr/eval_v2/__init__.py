"""bignose-thai-asr eval v2: one fair, versioned benchmark harness for every Thai ASR model family.

Modules
- textnorm : Thai text normalization (NFC, ไม้ยมก policy, Thai/Arabic digits, optional number→words)
- metrics  : CER (no-space chars), WER (newmm / deepcut), corpus aggregation
- hallucination : repetition-loop / length-blow-up / token-cap detectors
- audio    : resampling + deterministic long-audio chunking (same chunker for all backends)
- datasets : loaders for FLEURS, GigaSpeech2-Typhoon, TVSpeech, Common Voice, Gowajee, Thai Elderly, manifests
- backends : Whisper (HF), wav2vec2-CTC (HF), NeMo, Qwen3-ASR (HF) with graceful skip
- protocol : the frozen protocol dict + its hash
- runner   : models × datasets loop, sharding, per-utterance outputs
- report   : merge shards, summary csv/json, markdown leaderboard
"""
PROTOCOL_VERSION = "2.0.0"
