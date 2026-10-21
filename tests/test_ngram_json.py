import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bignose_thai_asr.ngram import KNTokenLM  # noqa: E402


def test_json_roundtrip_keeps_probabilities(tmp_path):
    lm = KNTokenLM(order=3, vocab_size=10).fit([[1, 2, 3], [2, 3, 4], [1, 2, 4]])
    p = tmp_path / "lm.json.gz"
    lm.save_json(str(p))
    lm2 = KNTokenLM.load_json(str(p))
    for hist, w in [((1, 2), 3), ((2,), 4), ((), 1), ((5, 6), 2)]:
        assert abs(lm.logp(hist, w) - lm2.logp(hist, w)) < 1e-12
