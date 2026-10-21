# Thai Dialect ASR: Same-Protocol Benchmarking of Central and Regional Thai Speech Recognition

*Draft v0, 2026-10-22. Work in progress; numbers trace to `results/scoreboard.csv`.*

## Abstract

We re-measure open Thai ASR systems and our own under one shared, hashed protocol (eval_v2) on eight test sets:
- **Central Thai:** Common Voice 17, GigaSpeech2-Typhoon, FLEURS and TVSpeech.
- **Regional Thai:** Isan, Korat, Kham Mueang (Northern) and Pattani Malay.

The protocol fixes:
- text normalization;
- greedy decoding with a shared 30 s chunker;
- corpus-level CER with whitespace removed;
- a paired bootstrap for significance (2,000 resamples).

On regional Thai, our w2v-BERT 2.0 CTC systems with character n-gram LMs **win decisively** on Pattani Malay and Kham Mueang. Decisive means ≥ 10 % relative CER reduction with the CI95 entirely below zero.
- **Pattani Malay:** 12.71 vs 65.05.
- **Kham Mueang:** 6.20 vs 7.77.

Korat ties the best competitor (13.45 vs 13.38).

On Central Thai:
- **TVSpeech:** a task-arithmetic Whisper merge wins (5.34 vs 5.75, significant).
- **Common Voice 17 and GigaSpeech2:** we lose (2.18 vs 2.03; 4.75 vs 4.52).

We also report:
- six cheap Central-Thai interventions that did not transfer to test;
- the misrouting cost of a single routed system across all sets.

## 1. Introduction

Thai ASR results are usually reported on Central-Thai benchmarks, each with its own normalization and decoding settings, so the numbers are not comparable. General-purpose models also fail badly on regional Thai. We contribute:
1. an open evaluation protocol with eight test sets, under which we re-measure competitor systems with pinned revisions;
2. regional-Thai acoustic models that beat every open Thai model on Pattani Malay and Kham Mueang;
3. a record of negative results on Central Thai;
4. an audit of data hygiene and of our own selection mistakes.

## 2. Related work

TODO: Thai Whisper fine-tunes (Typhoon, Pathumma, Thonburian), Pathumma-stt FastConformer-RNNT, LLM-ASR (typhoon-asr, Qwen3-ASR), the SLSCU Thai Dialect Corpus and its ESPnet baselines, w2v-BERT 2.0, model merging / WiSE-FT, CTC prefix beam search with shallow fusion.

## 3. Benchmarks and protocol

| Set | n | Domain | Selection dev |
|---|---:|---|---|
| CV17 (th) | 11,042 | read, crowd-sourced | cv17_dev |
| GigaSpeech2-Typhoon | 1,000 | YouTube, spontaneous | gs2_dev |
| FLEURS th_th | 1,021 | read | fleurs_dev |
| TVSpeech | 570 | broadcast | no dev split; bridge dev |
| Isan (Typhoon) | 500 | Isan | isan_dev |
| SLSCU Korat | 1,080 | Korat | dev_korat (from train) |
| SLSCU Kham Mueang | 805 | Northern Thai | dev_khummuang (from train) |
| SLSCU Pattani | 834 | Pattani Malay | dev_pattani_malay (from train) |

The SLSCU test sets are the released dev split, because SLSCU publishes no test split.

The eval_v2 protocol:
- **Normalization:** the same for reference and hypothesis.
- **CER:** whitespace-free character edit distance, summed over the corpus.
- **Decoding:** greedy, `language=th`.
- **Long audio:** one 30 s chunker for every backend.
- **Style-invariant variant:** numbers spelled out and ๆ expanded on both sides; reported next to raw CER, never instead of it.
- **Significance:** paired bootstrap, 2,000 resamples.

Competitors, at pinned revisions:
- typhoon-whisper-large-v3
- Pathumma-whisper-th-large-v3
- Pathumma-stt-th-fastconformer-rnnt-large
- typhoon-asr-qwen-1.7b-ctx
- Qwen3-ASR-1.7B
- Thonburian-whisper-large-v3

The SLSCU ESPnet baselines select their checkpoints on the same split that we use as test. We therefore list them as contaminated references and do not count them as competitors.

## 4. Systems

- **Pattani Malay:**
  - Model: w2v-BERT 2.0 character CTC, trained jointly on the three SLSCU dialects (deduplicated, about 110 h), with an auxiliary dialect-ID head.
  - Checkpoint: selected on Pattani dev.
  - Decoding: CTC prefix beam search (beam 8) with shallow fusion of a character 6-gram Kneser–Ney LM built from Pattani train text. Weights were chosen on dev by 2-fold out-of-fold estimates: lm_w = 0.3, bonus = 1.0.
- **Kham Mueang:**
  - Model: a short continued fine-tune of the joint model on Kham Mueang only, with speed perturbation. The final model averages the 3 best dev checkpoints.
  - Decoding: the same LM fusion, with lm_w = 0.2.
- **Korat:** a Whisper expert (Typhoon init, joint CTC loss), decoded to beam-8 N-best and rescored with a Korat character 6-gram LM and a CTC judge.
- **Central Thai:**
  - t60 is a task-arithmetic merge of Typhoon-whisper (0.60), Pathumma-whisper (0.25) and Thonburian (0.15), weighted on FLEURS dev.
  - cv17-091 continues t60 on 35 h of CV17 train.
- **One routed system:** a 5-class logistic-regression router on pooled t60 encoder features. Class biases and a fallback threshold are calibrated on dev.

## 5. Results

See the scoreboard in the README and `results/scoreboard.csv`.

The single routed system:
- keeps the decisive wins on Isan and Pattani;
- loses 0.8 CER on Kham Mueang and about 3 CER on Korat to misrouting, with routing accuracy of 0.89–0.93.

Classifier variants and w2v-BERT encoder features did not improve routing for Kham Mueang.

## 6. Negative results (Central Thai)

| Attempt | Outcome |
|---|---|
| Continued fine-tuning on ~95 h of Central data (LoRA r32) | Dev worse on every set after step 300; stopped |
| GS2 LoRA on t60 | Best checkpoint = step 0 |
| LM N-best rescoring (CV, TVSpeech) | Lost on test (039) / no significant dev gain (110) |
| WiSE-FT soup of a central fine-tune with t60 | Dev GS2 gain (10.54 → 9.73) did not transfer (test GS2 4.78 vs 4.79; TVSpeech 5.75 vs 5.34) |
| CTC re-scoring head on the CV generalist | Head dev CER 3.69 → 3.57 at one quarter of training, far above its gate; stopped |
| Combination of our own systems | No dev gain |
| LLM generative error correction | Excluded: its test parameters were selected by dataset name, so it is not one deployable system |
| Shorter chunks on Korat (8–15 s) | Dev worse (12.51 → 16.83) |

Our reading is that the competitors' Central-Thai lead comes from training data: GS2 train for Typhoon, and large-scale transducer training for Pathumma-stt. Post-hoc methods do not close that gap.

## 7. Data hygiene and disclosures

- **Deduplication:** all training text was deduplicated against every test and dev reference, both exact and near (Jaccard ≥ 0.8). The SLSCU tests show no exact, near, PCM or id overlap with training data. Speaker overlap cannot be checked.
- **Pending claims:** dialect2v2-interp (used for Isan routed 6.14 and FLEURS style-invariant 5.30) had its interpolation weight chosen with a rule on TVSpeech test CER. Those two results are marked *pending* until the weight is re-selected on dev only.
- **Kham Mueang LM not released:** the LM text includes research-only Bible text, so the LM is not redistributed. The released Kham Mueang model reports greedy CER (6.79).

## 8. Conclusion

A dialect-specific self-supervised CTC model with dialect character LMs beats every open Thai model on Pattani Malay and Kham Mueang by a wide margin. Central Thai remains data-bound. A single routed system pays a measurable misrouting cost.
