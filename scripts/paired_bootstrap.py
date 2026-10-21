#!/usr/bin/env python3
"""Paired bootstrap significance test for corpus CER (CAMPAIGN.md goal rule).

Pairs two eval_v2 utts jsonl files by utterance id, resamples utterances with
replacement (2,000 times by default) and reports the 95% CI of
CER(ours) - CER(theirs), where CER = sum(char_err) / sum(char_n) over the sample.
A win is significant only if the whole CI is below 0.

Usage: paired_bootstrap.py OURS.jsonl THEIRS.jsonl [--n 2000] [--seed 0] [--json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bignose_thai_asr.bootstrap import paired_bootstrap  # noqa: E402,F401  (re-exported for importers)


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = (r["char_err"], r["char_n"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = paired_bootstrap(load(a.ours), load(a.theirs), a.n, a.seed)
    if a.json:
        print(json.dumps(r))
    else:
        print(f"n={r['n_common']} ours={r['cer_ours']:.2f} theirs={r['cer_theirs']:.2f} "
              f"delta={r['delta']:+.2f} CI95=[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] "
              f"sig_win={r['significant_win']} sig_loss={r['significant_loss']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
