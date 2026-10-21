"""Paired bootstrap of corpus CER (CAMPAIGN.md goal rule); CLI: scripts/paired_bootstrap.py."""
import numpy as np


def paired_bootstrap(ours, theirs, n=2000, seed=0):
    ids = sorted(set(ours) & set(theirs))
    if not ids:
        raise ValueError("no common utterance ids")
    e1 = np.array([ours[i][0] for i in ids], dtype=float)
    e2 = np.array([theirs[i][0] for i in ids], dtype=float)
    nc = np.array([ours[i][1] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n, len(ids)))
    den = nc[idx].sum(1)
    diff = 100 * (e1[idx].sum(1) - e2[idx].sum(1)) / den
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "n_common": len(ids), "n_ours": len(ours), "n_theirs": len(theirs),
        "cer_ours": 100 * e1.sum() / nc.sum(), "cer_theirs": 100 * e2.sum() / nc.sum(),
        "delta": 100 * (e1.sum() - e2.sum()) / nc.sum(),
        "ci95": [float(lo), float(hi)], "resamples": n, "seed": seed,
        "significant_win": bool(hi < 0), "significant_loss": bool(lo > 0),
    }
