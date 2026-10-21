"""Re-score existing eval_v2 per-utt files under the style-invariant variant of the same protocol: numbers spelled
out as Thai words AND ๆ expanded, in BOTH ref and hyp, for every system alike (owner decision 2026-10-21: report
raw and style-invariant CER side by side). Paired bootstrap of every system vs --ref-system on both metrics.
  usage: python scripts/analysis/style_invariant.py --ref-system pathumma name=utts.jsonl [name=utts.jsonl ...]"""
import argparse, json
from bignose_thai_asr.bootstrap import paired_bootstrap
from bignose_thai_asr.eval_v2.metrics import edit_distance
from bignose_thai_asr.eval_v2.textnorm import chars


def si_counts(ref, hyp):
    rc, hc = chars(ref, numbers="words", yamok="expand"), chars(hyp, numbers="words", yamok="expand")
    return edit_distance(rc, hc), len(rc)


def load(path):
    raw, si = {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        raw[r["id"]] = (r["char_err"], r["char_n"])
        si[r["id"]] = si_counts(r["ref"], r["hyp"])
    return raw, si


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-system", required=True)
    ap.add_argument("systems", nargs="+")
    a = ap.parse_args()
    data = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in a.systems}
    ref = data[a.ref_system]
    print(f"| system | n | raw CER | style-inv CER | raw Δ vs {a.ref_system} [CI95] | style-inv Δ [CI95] |\n|---|---|---|---|---|---|")
    for name, (raw, si) in data.items():
        cer = lambda d: 100 * sum(e for e, _ in d.values()) / sum(n for _, n in d.values())
        cells = []
        for i in (0, 1):
            if name == a.ref_system:
                cells.append("—")
                continue
            b = paired_bootstrap((raw, si)[i], ref[i])
            cells.append(f"{b['delta']:+.2f} [{b['ci95'][0]:+.2f},{b['ci95'][1]:+.2f}]")
        print(f"| {name} | {len(raw)} | {cer(raw):.2f} | {cer(si):.2f} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
