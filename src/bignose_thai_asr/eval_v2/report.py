"""Merge per-shard outputs (from one or many pods) into summary.csv / summary.json / leaderboard.md.
Scores are always recomputed from the per-utterance rows (never averaged across shards)."""
from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from . import hallucination
from .metrics import corpus_rates

MAIN_COLS = ["char", "num_char", "word_newmm", "num_word_newmm", "word_deepcut", "num_word_deepcut"]
COL_NAMES = {"char": "CER", "num_char": "CER_num", "word_newmm": "WER_newmm", "num_word_newmm": "WER_newmm_num",
             "word_deepcut": "WER_deepcut", "num_word_deepcut": "WER_deepcut_num"}



# Human-readable dataset labels for leaderboards (dataset keys stay unchanged for scripts).
# Team-lead rule: the SLSCU 'pattani' set is ALWAYS "Pattani Malay (Thai script)", never "Southern Thai".
DISPLAY_NAMES = {
    "slscu_khummuang": "slscu_khummuang — Northern Thai / Kham Mueang (SLSCU dev split)",
    "slscu_korat": "slscu_korat — Korat Thai (SLSCU dev split)",
    "slscu_pattani": "slscu_pattani — Pattani Malay (Thai script) (SLSCU dev split)",
}

def _collect(run_dirs):
    metas = []
    for d in run_dirs:
        for m in sorted(Path(d).glob("meta/*.json")):
            j = json.loads(m.read_text(encoding="utf-8"))
            j["_utts"] = str(Path(d) / "utts" / (m.stem + ".jsonl"))
            metas.append(j)
    return metas


def merge_run(run_dirs, out_dir=None, allow_partial=False):
    run_dirs = [Path(d) for d in run_dirs]
    out_dir = Path(out_dir or run_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for m in _collect(run_dirs):
        groups[(m["dataset"], m["model"])].append(m)
    rows, problems = [], []
    for (ds, model), ms in sorted(groups.items()):
        hashes = {m["protocol_hash"] for m in ms}
        revs = {m["model_revision"] for m in ms}
        fallback_revs = {json.dumps(m.get("fallback_model"), sort_keys=True) for m in ms}
        if len(hashes) > 1 or len(revs) > 1:
            problems.append(f"{ds}×{model}: mixed protocol hashes {hashes} or model revisions {revs} — NOT merged")
            continue
        if len(fallback_revs) > 1:
            problems.append(f"{ds}×{model}: mixed fallback model revisions/configs — NOT merged")
            continue
        seen, per = set(), []
        for m in ms:
            for l in open(m["_utts"], encoding="utf-8"):
                r = json.loads(l)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    per.append(r)
        n_total = ms[0]["n_total"]
        complete = len(per) == n_total
        if not complete:
            problems.append(f"{ds}×{model}: {len(per)}/{n_total} utterances present (missing shards?)")
            if not allow_partial:
                continue
        rates = corpus_rates(per)
        hs = hallucination.summarize(per)
        audio = sum(m["audio_seconds"] for m in ms)
        gen = sum(m["gen_seconds"] for m in ms)
        gpus = sorted({m["gpu"] for m in ms})
        m0 = ms[0]
        row = {"dataset": ds, "model": model, "model_id": m0["model_id"], "model_revision": m0["model_revision"],
               "backend": m0["backend"], "n": len(per), "n_total": n_total, "complete": complete,
               "limit": m0.get("limit", 0), "audio_hours": round(audio / 3600, 3),
               **{COL_NAMES[k] + "_%": rates.get(k) for k in MAIN_COLS},
               "CER_excl_flagged_%": hs["CER_excl_flagged"], "n_flag_loop": hs["n_flag_loop"],
               "n_flag_blowup": hs["n_flag_blowup"], "n_flag_empty": hs["n_flag_empty"], "n_flag_cap": hs["n_flag_cap"],
               "pct_utts_flagged": hs["pct_utts_flagged"], "pct_char_err_from_flagged": hs["pct_char_err_from_flagged"],
               "RTF": round(gen / audio, 5) if audio else None, "gpu": "+".join(gpus), "dtype": m0["dtype"],
               "protocol_hash": m0["protocol_hash"], "dataset_source": m0["dataset_source"],
               "dataset_revision": m0["dataset_revision"], "date": max(m["date"] for m in ms),
               "shards": ",".join(sorted(m["shard"] for m in ms))}
        rows.append(row)
    skipped = []
    for d in run_dirs:
        f = d / "skipped.jsonl"
        if f.exists():
            skipped += [json.loads(l) for l in open(f, encoding="utf-8")]
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "problems": problems, "skipped": skipped},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "leaderboard.md").write_text(leaderboard_md(rows, problems, skipped), encoding="utf-8")
    return rows, problems


def _f(x):
    return "—" if x is None else f"{x:.2f}"


def leaderboard_md(rows, problems, skipped) -> str:
    hashes = sorted({r["protocol_hash"] for r in rows})
    smoke = any(r.get("limit") for r in rows)
    L = ["# bignose-thai-asr — eval v2 leaderboard", "",
         f"_generated {dt.datetime.now().astimezone().isoformat(timespec='minutes')} · protocol hash(es): "
         f"{', '.join('`'+h+'`' for h in hashes) or '—'} · see docs/eval-protocol.md_", ""]
    if smoke:
        L += ["> ⚠️ **SMOKE TEST** — run with `--limit`; numbers are NOT benchmark results.", ""]
    if len(hashes) > 1:
        L += ["> ⚠️ more than one protocol hash in this table — rows are not comparable across hashes.", ""]
    ds_list = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})
    if rows:
        L += ["## CER % (main metric, lower is better; **bold** = best per dataset)", "",
              "| model | " + " | ".join(DISPLAY_NAMES.get(d, d) for d in ds_list) + " |", "|---|" + "---|" * len(ds_list)]
        best = {d: min((r["CER_%"] for r in rows if r["dataset"] == d and r["CER_%"] is not None), default=None) for d in ds_list}
        for m in models:
            cells = []
            for d in ds_list:
                r = next((x for x in rows if x["model"] == m and x["dataset"] == d), None)
                if r is None:
                    cells.append("—")
                else:
                    v = _f(r["CER_%"]) + ("" if r["complete"] else "*")
                    cells.append(f"**{v}**" if r["CER_%"] == best[d] else v)
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", "`*` = incomplete (missing shards)", ""]
    for d in ds_list:
        rs = sorted((r for r in rows if r["dataset"] == d), key=lambda r: (r["CER_%"] is None, r["CER_%"]))
        r0 = rs[0]
        L += [f"## {d}" + (f" — {DISPLAY_NAMES[d]}" if d in DISPLAY_NAMES else ""), "", f"n = {r0['n_total']} utts · {r0['audio_hours']} h · source `{r0['dataset_source']}` "
              f"@ `{str(r0['dataset_revision'])[:12]}`", "",
              "| # | model | CER | CER_num | WER_newmm | WER_newmm_num | WER_deepcut | CER excl. flagged | loops | blow-ups | empty | token-cap | % err from flagged | RTF | revision |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rs, 1):
            L.append(f"| {i} | {r['model']} | {_f(r['CER_%'])} | {_f(r['CER_num_%'])} | {_f(r['WER_newmm_%'])} | "
                     f"{_f(r['WER_newmm_num_%'])} | {_f(r['WER_deepcut_%'])} | {_f(r['CER_excl_flagged_%'])} | "
                     f"{r['n_flag_loop']} | {r['n_flag_blowup']} | {r['n_flag_empty']} | {r['n_flag_cap']} | "
                     f"{r['pct_char_err_from_flagged']:.1f} | {r['RTF'] if r['RTF'] is not None else '—'} | "
                     f"`{r['model_id']}@{str(r['model_revision'])[:10]}` |")
        L.append("")
    if problems:
        L += ["## Merge problems", ""] + [f"- {p}" for p in problems] + [""]
    if skipped:
        L += ["## Skipped (could not run)", ""]
        seen = set()
        for s in skipped:
            k = (s["kind"], s["what"], s["reason"].split("\n")[0])
            if k not in seen:
                seen.add(k)
                L.append(f"- **{s['kind']}** `{s['what']}` — {k[2]}")
        L.append("")
    return "\n".join(L)
