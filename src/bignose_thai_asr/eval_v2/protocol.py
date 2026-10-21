"""The frozen evaluation protocol. Anything that can change a score is in here; its hash is stamped on every
output row, and merge refuses to mix rows with different hashes."""
from __future__ import annotations

import hashlib
import json

from . import PROTOCOL_VERSION, audio, hallucination


def protocol_dict(*, yamok: str = "keep", batch_size: int = 16, max_new_tokens: int = 448, loop_guard=None) -> dict:
    p = {
        "version": PROTOCOL_VERSION,
        "decoding": {"strategy": "greedy", "num_beams": 1, "do_sample": False, "language": "th",
                     "max_new_tokens_requested": max_new_tokens,
                     "whisper_effective_cap": "min(requested, max_target_positions-4)=444",
                     "batch_size": batch_size, "batch_order": "chunks sorted by duration (desc)"},
        "chunking": {"max_s": audio.CHUNK_MAX_S, "search_s": audio.CHUNK_SEARCH_S, "frame_s": audio.FRAME_S,
                     "join": "space"},
        "normalization": {"drop_tags": "[ns]-style latin tags, <unk>/<noise>", "nfc": True, "lowercase": True, "thai_digits_to_arabic": True, "drop_thousands_sep": True,
                          "yamok": yamok, "pythainlp_normalize": True, "punct_to_space": True,
                          "num_variant": "digits spelled as Thai words (pythainlp.num_to_thaiword) + %→เปอร์เซ็นต์"},
        "metrics": {"CER": "char edit distance, all whitespace removed, corpus-level (sum err / sum ref chars)",
                    "WER": "token edit distance on whitespace-free text tokenized with newmm (+deepcut if available)"},
        "hallucination": {"loop_min_repeats": hallucination.LOOP_MIN_REPEATS,
                          "loop_min_repeats_unigram": hallucination.LOOP_MIN_REPEATS_UNIGRAM,
                          "loop_min_span": hallucination.LOOP_MIN_SPAN,
                          "compression_max": hallucination.COMPRESSION_MAX, "len_ratio_max": hallucination.LEN_RATIO_MAX,
                          "len_abs_min_extra": hallucination.LEN_ABS_MIN_EXTRA},
    }
    if loop_guard is not None:
        p["decoding"]["strategy"] = "greedy+loop_guard"
        p["decoding"]["loop_guard"] = loop_guard.protocol()
    return p


def protocol_hash(p: dict) -> str:
    # batch_size IS part of the hash: padded batching can shift fp16 outputs slightly, so all shards/models of
    # one leaderboard must use the same batch size.
    return hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
