import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "paired_bootstrap.py"
_spec = importlib.util.spec_from_file_location("paired_bootstrap", _p)
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)


def test_clear_win_is_significant():
    ours = {f"u{i}": (1, 20) for i in range(300)}
    theirs = {f"u{i}": (5, 20) for i in range(300)}
    r = pb.paired_bootstrap(ours, theirs, n=500)
    assert abs(r["delta"] + 20.0) < 1e-9
    assert r["significant_win"] and not r["significant_loss"]


def test_identical_systems_not_significant():
    ours = {f"u{i}": (i % 4, 20) for i in range(200)}
    r = pb.paired_bootstrap(ours, dict(ours), n=500)
    assert r["delta"] == 0 and not r["significant_win"] and not r["significant_loss"]


def test_pairs_only_common_ids():
    ours = {"a": (0, 10), "b": (10, 10)}
    theirs = {"a": (10, 10), "c": (0, 10)}
    r = pb.paired_bootstrap(ours, theirs, n=100)
    assert r["n_common"] == 1 and r["cer_ours"] == 0 and r["cer_theirs"] == 100
