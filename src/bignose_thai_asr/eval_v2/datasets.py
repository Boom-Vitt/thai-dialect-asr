"""Benchmark test-set loaders. Each loader materializes 16 kHz mono PCM16 wavs + a manifest.jsonl in a cache dir:
    {"id", "audio" (abs path), "text", "duration", ...extra}
so every model on every pod reads byte-identical audio. A loader raises DatasetUnavailable with a human-readable
reason (gated / missing local path / missing dependency) instead of crashing the whole run.

Dataset names (CLI): fleurs, fleurs_dev, tvspeech, gs2_typhoon, cv, cv17_dev (DEV), gowajee, thai_elderly, isan_typhoon, isan_typhoon_thairef, central_bridge_dev, central_bridge_dev_t60ref, slscu_khummuang, slscu_korat, slscu_pattani, and manifest:<name>=<path>.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .audio import SR, decode_bytes, read_wav, to_mono16k


class DatasetUnavailable(RuntimeError):
    pass


@dataclass
class DatasetInfo:
    name: str
    source: str            # e.g. "hf:google/fleurs@<sha>:parquet-data/th_th/test" or "local:/path"
    revision: str | None
    license: str
    split: str
    notes: str = ""
    extra: dict = field(default_factory=dict)


def default_cache() -> Path:
    return Path(os.environ.get("EVAL_V2_CACHE", str(Path.home() / ".cache" / "eval_v2")))


def _hf_sha(repo: str, repo_type: str = "dataset") -> str | None:
    from huggingface_hub import HfApi
    try:
        return HfApi().dataset_info(repo).sha if repo_type == "dataset" else HfApi().model_info(repo).sha
    except Exception as e:  # gated/offline -> caller decides
        name = type(e).__name__
        if "Gated" in name:
            raise DatasetUnavailable(f"{repo}: gated — the HF account behind HF_TOKEN must click 'Agree' on "
                                     f"https://huggingface.co/datasets/{repo} first ({name})") from e
        raise DatasetUnavailable(f"{repo}: cannot reach repo ({name}: {str(e)[:160]})") from e


def _check_access(repo: str, path: str, revision: str | None = None):
    """Gated repos expose metadata to everyone; probe the actual file so we fail fast with instructions."""
    from huggingface_hub import HfApi
    try:
        list(HfApi().get_paths_info(repo, [path], repo_type="dataset", revision=revision))
    except Exception as e:
        if "Gated" in type(e).__name__ or "403" in str(e):
            raise DatasetUnavailable(f"{repo}: GATED — the HF account behind HF_TOKEN has not been granted access. "
                                     f"Open https://huggingface.co/datasets/{repo} while logged in as that account, "
                                     f"accept the terms, then re-run (or pass a local audio dir)") from e
        raise DatasetUnavailable(f"{repo}/{path}: {type(e).__name__}: {str(e)[:160]}") from e


def _hf_open(repo: str, path: str, revision: str | None = None):
    from huggingface_hub import HfFileSystem
    rev = f"@{revision}" if revision else ""
    return HfFileSystem().open(f"datasets/{repo}{rev}/{path}", "rb", block_size=8 * 2**20)


def _hf_download(repo: str, path: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "403" in str(e):
            raise DatasetUnavailable(f"{repo}/{path}: gated / no access for this token ({name})") from e
        raise DatasetUnavailable(f"{repo}/{path}: download failed ({name}: {str(e)[:160]})") from e


def _iter_parquet(repo, path, revision, limit) -> Iterator[dict]:
    """Stream rows; with a limit only the needed row groups are fetched (HfFileSystem range reads)."""
    import pyarrow.parquet as pq
    src = _hf_open(repo, path, revision) if limit else _hf_download(repo, path, revision)
    pf = pq.ParquetFile(src)
    n = 0
    for b in pf.iter_batches(batch_size=16):
        for r in b.to_pylist():
            yield r
            n += 1
            if limit and n >= limit:
                return


# ----------------------------------------------------------------------------------------------- loaders
# Each loader: (opts, limit) -> (DatasetInfo, iterator of (id, np.ndarray|bytes|path, text, extra))

def _fleurs(split: str):
    def load(opts, limit):
        repo = "google/fleurs"
        sha = _hf_sha(repo)
        fname = {"test": "test", "validation": "validation", "dev": "validation"}[split]
        path = f"parquet-data/th_th/{fname}-00000-of-00001.parquet"
        info = DatasetInfo(f"fleurs{'' if split == 'test' else '_dev'}", f"hf:{repo}:{path}", sha, "CC-BY-4.0", split,
                           "text field = 'transcription' (FLEURS normalized); raw_transcription kept in extra. "
                           "All rows kept incl. repeated sentences read by different speakers.")

        def it():
            for r in _iter_parquet(repo, path, sha, limit):
                yield (f"fleurs_{fname}_{r['id']}_{Path(r['path'] or r['audio']['path']).stem}", r["audio"]["bytes"],
                       r["transcription"], {"raw_transcription": r["raw_transcription"], "gender": r["gender"]})
        return info, it()
    return load


def _tvspeech(opts, limit):
    repo, path = "typhoon-ai/TVSpeech", "data/test-00000-of-00001.parquet"
    sha = _hf_sha(repo)
    info = DatasetInfo("tvspeech", f"hf:{repo}:{path}", sha, "CC-BY-4.0 (per README)", "test",
                       "570 in-the-wild YouTube utts; long references (avg ~280 chars) -> token cap + chunking matter.")

    def it():
        for r in _iter_parquet(repo, path, sha, limit):
            yield r["audio_id"], r["audio"]["bytes"], r["sentence"], {}
    return info, it()


def _gs2_typhoon(opts, limit):
    """typhoon-ai/gigaspeech2-typhoon = 1,000 human-verified transcripts (CC-BY-4.0, text only). Audio comes from the
    GATED speechcolab/gigaspeech2 data/th/test.tar.gz (non-commercial terms) or --gs2-audio-dir (<audio_id>.wav)."""
    import pyarrow.parquet as pq
    trepo, tpath = "typhoon-ai/gigaspeech2-typhoon", "data/test-00000-of-00001.parquet"
    tsha = _hf_sha(trepo)
    want = {r["audio_id"]: r["sentence"] for r in pq.read_table(_hf_download(trepo, tpath, tsha)).to_pylist()}
    order = sorted(want)
    local = opts.get("gs2_audio_dir")
    if local:
        files = {p.stem: p for p in Path(local).rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".opus")}
        missing = [a for a in order if a not in files]
        info = DatasetInfo("gs2_typhoon", f"hf:{trepo}:{tpath} + local:{local}", tsha, "transcripts CC-BY-4.0; audio GigaSpeech2 terms",
                           "test", f"{len(missing)} of {len(order)} audio ids missing locally" if missing else "")

        def it_local():
            n = 0
            for a in order:
                if a in files:
                    yield a, str(files[a]), want[a], {}
                    n += 1
                    if limit and n >= limit:
                        return
        return info, it_local()
    arepo = "speechcolab/gigaspeech2"
    asha = _hf_sha(arepo)
    _check_access(arepo, "data/th/test.tar.gz", asha)  # raises DatasetUnavailable(gated) with instructions
    info = DatasetInfo("gs2_typhoon", f"hf:{trepo}:{tpath} + hf:{arepo}:data/th/test.tar.gz", f"{tsha}+{asha}",
                       "transcripts CC-BY-4.0; audio GigaSpeech2 (non-commercial)", "test",
                       "Typhoon's main benchmark; audio ids not found in the tar are reported.")

    def it():
        n = 0
        fh = _hf_open(arepo, "data/th/test.tar.gz", asha) if limit else open(_hf_download(arepo, "data/th/test.tar.gz", asha), "rb")
        with tarfile.open(fileobj=fh, mode="r|gz") as tf:
            for m in tf:
                stem = Path(m.name).stem
                if m.isfile() and stem in want:
                    yield stem, tf.extractfile(m).read(), want[stem], {}
                    n += 1
                    if limit and n >= limit:
                        return
    return info, it()


def _cv(opts, limit, split="test"):
    """Common Voice th test (split="test", dataset `cv`) or dev (split="dev", dataset `cv17_dev`, DEV only).
    Mozilla removed CV from HF (Oct 2025); official source is Mozilla Data Collective.
    --cv-source local:/path/to/cv-corpus-XX/th  (MDC extract: test.tsv + clips/*.mp3)   [preferred, cite version]
    --cv-source hf:fsicoli/common_voice_17_0 | hf:fsicoli/common_voice_19_0           [third-party CC0 mirror]
    --cv-split-tsv path  -> override the test list (e.g. ekapolc/Thai_commonvoice_split test csv/tsv with 'path' col)."""
    src = opts.get("cv_source") or "hf:fsicoli/common_voice_17_0"
    name = "cv" if split == "test" else f"cv17_{split}"
    split_override = opts.get("cv_split_tsv") if split == "test" else None

    def read_tsv(fp):
        rdr = csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [r for r in rdr]

    if src.startswith("local:"):
        root = Path(src[6:])
        tsv = Path(split_override) if split_override else root / f"{split}.tsv"
        if not tsv.exists():
            raise DatasetUnavailable(f"cv: {tsv} not found (download Thai CV from https://datacollective.mozillafoundation.org "
                                     "and pass --cv-source local:<.../th>)")
        rows = read_tsv(open(tsv, encoding="utf-8"))
        info = DatasetInfo(name, f"local:{root}", _file_sha(tsv), "CC0-1.0", split, f"{split} list = {tsv}")

        def it_local():
            for i, r in enumerate(rows):
                if limit and i >= limit:
                    return
                p = root / "clips" / r["path"]
                if not p.exists():
                    continue
                yield Path(r["path"]).stem, p.read_bytes(), r["sentence"], {"client_id": r.get("client_id", "")[:12]}
        return info, it_local()
    if not src.startswith("hf:"):
        raise DatasetUnavailable(f"cv: unknown --cv-source {src!r}")
    repo = src[3:]
    sha = _hf_sha(repo)
    if split_override:
        rows = read_tsv(open(split_override, encoding="utf-8"))
        tsv_src = split_override
    else:
        tsv_src = f"transcript/th/{split}.tsv"
        rows = read_tsv(open(_hf_download(repo, tsv_src, sha), encoding="utf-8"))
    want = {r["path"]: r["sentence"] for r in rows}
    ver = repo.rsplit("_", 2)[-2] if "common_voice_" in repo else "?"
    tar = f"audio/th/{split}/th_{split}_0.tar"
    info = DatasetInfo(name, f"hf:{repo}:{tar}", sha, "CC0-1.0", split,
                       f"THIRD-PARTY MIRROR of Common Voice {ver} (official CV is on Mozilla Data Collective); "
                       f"list={tsv_src}; official split has speaker/sentence overlap issues — "
                       "for publication prefer MDC + ekapolc split", {"n_listed": len(want)})

    def it():
        n = 0
        fh = _hf_open(repo, tar, sha) if limit else open(_hf_download(repo, tar, sha), "rb")
        with tarfile.open(fileobj=fh, mode="r|") as tf:
            for m in tf:
                base = Path(m.name).name
                if m.isfile() and base in want:
                    yield Path(base).stem, tf.extractfile(m).read(), want[base], {}
                    n += 1
                    if limit and n >= limit:
                        return
    return info, it()


GOWAJEE_GDRIVE_ID = "1soriRMMuZI5w5RZOjAnbpocBZxT6i1-l"  # as listed by SEACrowd/gowajee (v0.9.3 zip)
GOWAJEE_GDRIVE = f"https://drive.google.com/file/d/{GOWAJEE_GDRIVE_ID}/view"


def _gowajee(opts, limit):
    """Gowajee v0.9.x test split (Kaldi format: text, wav.scp). SEACrowd/gowajee is only a loader script that pulls a
    Google-Drive zip; we use --gowajee-dir <dir containing test/text> or download with gdown (--allow-download)."""
    d = opts.get("gowajee_dir")
    cache = Path(opts["cache"]) / "_raw" / "gowajee"
    if not d:
        cand = list(cache.glob("**/test/text"))
        if not cand and opts.get("allow_download"):
            try:
                import gdown
            except ImportError as e:
                raise DatasetUnavailable("gowajee: pip install gdown (or pass --gowajee-dir)") from e
            cache.mkdir(parents=True, exist_ok=True)
            z = cache / "gowajee_v0-9-3.zip"
            if not z.exists():
                try:
                    try:
                        ok = gdown.download(id=GOWAJEE_GDRIVE_ID, output=str(z), quiet=True)
                    except TypeError:  # older gdown
                        ok = gdown.download(GOWAJEE_GDRIVE, str(z), fuzzy=True, quiet=True)
                except Exception as e:
                    z.unlink(missing_ok=True)
                    raise DatasetUnavailable(f"gowajee: Google Drive download failed ({type(e).__name__}: {str(e)[:200]}); "
                                             "pass --gowajee-dir") from e
                if not ok or not z.exists():
                    raise DatasetUnavailable("gowajee: Google Drive download failed (quota/permission); pass --gowajee-dir")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(cache)
            z.unlink()  # keep only the extracted corpus
            cand = list(cache.glob("**/test/text"))
        if not cand:
            raise DatasetUnavailable("gowajee: no local copy. Pass --gowajee-dir <.../v0.9.2> or --allow-download "
                                     f"(gdown {GOWAJEE_GDRIVE}, ~1.5 GB, MIT)")
        d = str(cand[0].parent.parent)
    root = Path(d)
    tdir = root / "test"
    texts = dict(l.rstrip("\n").split(" ", 1) for l in open(tdir / "text", encoding="utf-8") if " " in l)
    wavs = dict(l.strip().split(" ", 1) for l in open(tdir / "wav.scp", encoding="utf-8") if l.strip())
    ids = sorted(set(texts) & set(wavs))
    info = DatasetInfo("gowajee", f"local:{root}", _file_sha(tdir / "text"), "MIT", "test",
                       "Gowajee corpus test split (smart-home commands, Chula).")

    def it():
        for i, u in enumerate(ids):
            if limit and i >= limit:
                return
            p = Path(wavs[u])
            p = p if p.is_absolute() else root / p
            yield u, str(p), texts[u], {}
    return info, it()


ELDERLY_URLS = [f"https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/download/v1.0.0/Dataset.zip.00{i}"
                for i in (1, 2, 3)]


def _elderly(opts, limit):
    """Thai Elderly Speech v1.0.0 (VISAI x DataWow, CC-BY-SA-4.0, 19,200 clips / ~17 h, ~5.07 GB zip).
    There is NO official test split (SEACrowd exposes only 'train'). bignose split: every clip whose
    sha1(filename) % 10 == 0 (~10 %, deterministic). Both subsets (Healthcare, Smarthome) included.
    WARNING: Thonburian/Pathumma trained on this corpus -> their numbers here are likely contaminated."""
    d = opts.get("elderly_dir")
    raw = Path(opts["cache"]) / "_raw" / "thai_elderly"
    if not d:
        cand = list(raw.glob("**/speaker_demography.json"))
        if not cand and opts.get("allow_download"):
            import urllib.request
            raw.mkdir(parents=True, exist_ok=True)
            z = raw / "thai_elderly_speech.zip"
            if not z.exists():
                with open(z, "wb") as out:
                    for u in ELDERLY_URLS:
                        with urllib.request.urlopen(u) as r:
                            while chunk := r.read(16 * 2**20):
                                out.write(chunk)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
            z.unlink()
            cand = list(raw.glob("**/speaker_demography.json"))
        if not cand:
            raise DatasetUnavailable("thai_elderly: no local copy. Pass --elderly-dir <.../Dataset> or --allow-download "
                                     "(3 GitHub release parts, 5.07 GB total)")
        d = str(cand[0].parent)
    root = Path(d)
    items = []
    for sub in ("Healthcare", "Smarthome"):
        tj = root / sub / "transcription.json"
        if not tj.exists():
            continue
        for a in json.load(open(tj, encoding="utf-8")):
            fn = a["filename"]
            if int(hashlib.sha1(fn.encode()).hexdigest(), 16) % 10 == 0:
                items.append((f"{sub.lower()}_{fn}", str(root / sub / "Record" / f"{fn}.wav"), a["transcript"],
                              {"speaker_id": a.get("speaker_id")}))
    items.sort()
    info = DatasetInfo("thai_elderly", f"local:{root}", _file_sha(root / "speaker_demography.json"), "CC-BY-SA-4.0",
                       "bignose-hash10", _elderly.__doc__.split("\n", 1)[1].strip().replace("\n   ", ""))
    return info, iter(items[:limit] if limit else items)


ISAN_REPO, ISAN_REV = "typhoon-ai/thai-dialect-isan-dataset", "dfc7e495b3e6"  # pinned (HF sha prefix, checked 2026-10-19)
ISAN_FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]


def isan_refs(raw: str) -> tuple[str, str]:
    """Typhoon Isan `raw` uses [isan word]<standard-thai word>. Returns (isan_spelling, thai_spelling) derived from raw.
    The dataset's own isan_spelling/thai_spelling columns are corrupted on rows with >1 annotated span (every span is
    replaced by the FIRST span's value; 200/500 test rows), so they are NOT used."""
    import re
    isan = re.sub(r"\[([^\]]*)\]<[^>]*>", r"\1", raw or "")
    thai = re.sub(r"\[[^\]]*\]<([^>]*)>", r"\1", raw or "")
    return re.sub(r"\s+", " ", isan).strip(), re.sub(r"\s+", " ", thai).strip()


def _isan_typhoon(ref: str):
    """Typhoon Isan test split (500 utts, 114 speakers, 10 provinces, ~1.29 h) = the held-out set Typhoon reports
    Isan CER on (arXiv 2601.13044 Table 7). ref='isan' (PRIMARY: Isan spelling, what Typhoon-Isan models are trained to
    output) or 'thai' (Standard-Thai spelling of the same utterance; what Central-Thai models tend to output).
    Caveats: all 114 test speakers also occur in the train split (not speaker-disjoint); 1 row has an empty reference
    (kept, so the set is exactly the published 500)."""
    def load(opts, limit):
        sha = _hf_sha(ISAN_REPO)
        info = DatasetInfo("isan_typhoon" + ("" if ref == "isan" else "_thairef"), f"hf:{ISAN_REPO}:" + "+".join(ISAN_FILES),
                           sha, "HF tag apache-2.0; card body says CC-BY-4.0 (conflict, see docs/isan_plan.md)", "test",
                           f"reference = {ref} spelling derived from `raw`; " + _isan_typhoon.__doc__.split("\n", 1)[1].strip()[:400])

        def it():
            n = 0
            for f in ISAN_FILES:
                for r in _iter_parquet(ISAN_REPO, f, sha, 0):
                    isan, thai = isan_refs(r["raw"])
                    a = r["audio"]
                    arr = to_mono16k(np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"]))
                    yield (r["id"], arr, isan if ref == "isan" else thai,
                           {"speaker": r["name"].split(";")[2], "province": r["province"], "raw": r["raw"]})
                    n += 1
                    if limit and n >= limit:
                        return
        return info, it()
    return load


SLSCU_REPO, SLSCU_REV = "n-order/Thai-dialect-corpus", "56b7c7d7e29b"  # pinned (HF sha prefix, checked 2026-10-19)


def _slscu(dialect: str):
    """SLSCU Thai-dialect corpus (Suwanbandit et al., Interspeech 2023) via the HF mirror n-order/Thai-dialect-corpus,
    split `<dialect>_test`. Reference = `sentence` (dialect spelling in Thai script, space-tokenized; the scorer strips
    spaces). `thai_sentence` is a Standard-Thai TRANSLATION, not a transcription -> kept in extra only, never scored.
    CAVEATS (see docs/north_south_plan.md): (1) the mirror's *_test sizes (khummuang 805, korat 1,080, pattani 834) match
    the paper's DEV split (806 / 1,080 / 834), not its withheld TEST split (3,003 / 3,852 / 2,793) on which the SLSCU
    model-card CERs (5.35 / 8.08 / 18.17) were measured -> numbers are not directly comparable. (2) Every row of these
    splits is byte-identical to a row inside CMKL/Porjai-Thai-voice-dataset-<dialect> 'train' -> anything trained on the
    CMKL mirrors without filtering is contaminated. (3) slscu_pattani = "Pattani Malay (Thai script)" — Patani Malay written in Thai script; it is
    NOT Southern Thai (Pak Tai) and must never be labelled so. (4) No speaker ids in the release; paper states Train/Dev/Test speakers do not overlap."""
    def load(opts, limit):
        path = f"data/{dialect}_test-00000-of-00001.parquet"
        sha = _hf_sha(SLSCU_REPO) or SLSCU_REV
        lic = ("upstream README: CC-BY-SA-4.0 (github.com/SLSCU/thai-dialect-corpus; repo LICENSE file = MIT; "
               "HF mirror has no license tag; CMKL mirror tags " + ("CC-BY-NC-SA-4.0" if dialect != "korat" else "CC-BY-SA-4.0") + ")")
        info = DatasetInfo(f"slscu_{dialect}", f"hf:{SLSCU_REPO}:{path}", sha, lic, f"{dialect}_test (= paper Dev)",
                           _slscu.__doc__.split("\n", 1)[1].strip()[:900])

        def it():
            for r in _iter_parquet(SLSCU_REPO, path, sha, limit):
                yield (r["utterance"], r["audio"]["bytes"], r["sentence"],
                       {"thai_translation": r["thai_sentence"], "domain": r["dialect_type"], "dialect": dialect})
        return info, it()
    return load


def _central_bridge_dev(ref: str):
    """[isan team] Central-Thai "bridge dev" (TVSpeech has no train/dev split): 326 spontaneous Central-Thai YouTube chunks
    (1.26 h, 13 videos, 2-20 s) from social-video mining round 2 where t60 and typhoon-whisper-large-v3 agree (chunk CER <= 0.15).
    INTERNAL ONLY (YouTube audio, not redistributable). References are MACHINE PSEUDO-LABELS: ref='typhoon' (default,
    typhoon-whisper-large-v3 hypothesis) or 't60'. Use for DEV / checkpoint selection and central-drift tracking only (deltas
    vs. base), never as a test claim. Path: $CENTRAL_BRIDGE_DEV_DIR (required). See
    results/central_bridge_dev.md."""
    def load(opts, limit):
        import os
        d = Path(os.environ.get("CENTRAL_BRIDGE_DEV_DIR", "central_bridge_dev"))
        p = d / "manifest.jsonl"
        if not p.exists():
            raise DatasetUnavailable(f"central_bridge_dev: {p} not found (set CENTRAL_BRIDGE_DEV_DIR)")
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        info = DatasetInfo("central_bridge_dev" + ("" if ref == "typhoon" else "_t60ref"), f"manifest:{p}", _file_sha(p),
                           "internal only (YouTube); pseudo-labels", "dev", _central_bridge_dev.__doc__.split("\n", 1)[1].strip()[:600])

        def it():
            for r in (rows[:limit] if limit else rows):
                yield r["id"], str(d / "wav" / Path(r["audio"]).name), (r["text"] if ref == "typhoon" else r["text_t60"]), {"vid": r["vid"]}
        return info, it()
    return load


def _manifest(name, path):
    def load(opts, limit):
        p = Path(path)
        if not p.exists():
            raise DatasetUnavailable(f"manifest {name}: {p} not found")
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        info = DatasetInfo(name, f"manifest:{p}", _file_sha(p), "see source", "as-given")

        def it():
            for i, r in enumerate(rows[:limit] if limit else rows):
                yield r.get("id") or Path(r["audio"]).stem, r["audio"], r["text"], {}
        return info, it()
    return load


def _file_sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(2**20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()[:16]


LOADERS: dict[str, Callable] = {
    "fleurs": _fleurs("test"), "fleurs_dev": _fleurs("validation"), "tvspeech": _tvspeech,
    "gs2_typhoon": _gs2_typhoon, "cv": _cv, "cv17_dev": lambda o, l: _cv(o, l, "dev"), "gowajee": _gowajee, "thai_elderly": _elderly,
    "isan_typhoon": _isan_typhoon("isan"), "isan_typhoon_thairef": _isan_typhoon("thai"),
    "central_bridge_dev": _central_bridge_dev("typhoon"), "central_bridge_dev_t60ref": _central_bridge_dev("t60"),
    "slscu_khummuang": _slscu("khummuang"), "slscu_korat": _slscu("korat"), "slscu_pattani": _slscu("pattani"),
}


def resolve(spec: str) -> tuple[str, Callable]:
    if spec.startswith("manifest:"):
        name, path = spec[9:].split("=", 1)
        return name, _manifest(name, path)
    if spec not in LOADERS:
        raise DatasetUnavailable(f"unknown dataset {spec!r}; known: {sorted(LOADERS)} or manifest:<name>=<path>")
    return spec, LOADERS[spec]


def _store(uid, audio, text, extra, wdir: Path) -> dict:
    import soundfile as sf
    uid = str(uid).replace("/", "_")
    if isinstance(audio, (bytes, bytearray)):
        a = decode_bytes(audio)
    elif isinstance(audio, np.ndarray):
        a = to_mono16k(audio, SR)
    elif Path(audio).suffix.lower() in (".wav", ".flac"):
        a = read_wav(audio)
    else:
        a = decode_bytes(Path(audio).read_bytes())
    p = wdir / f"{uid}.wav"
    if not p.exists():
        sf.write(p, a, SR, subtype="PCM_16")
    return {"id": uid, "audio": str(p.resolve()), "text": text, "duration": round(len(a) / SR, 3), **extra}


def materialize(spec: str, opts: dict, limit: int = 0) -> tuple[DatasetInfo, list[dict]]:
    """Return (info, rows). Writes wavs + manifest once into <cache>/<name>/ and reuses them afterwards."""
    name, loader = resolve(spec)
    cache = Path(opts.get("cache") or default_cache())
    opts = {**opts, "cache": str(cache)}
    ddir = cache / name
    if spec.startswith("manifest:"):
        # key manifest specs by content: a dev manifest named like a built-in test set (e.g. "fleurs") must never reuse its cache
        src = Path(spec[9:].split("=", 1)[1])
        ddir = cache / f"manifest-{name}-{_file_sha(src)[7:] if src.exists() else 'missing'}"
    man = ddir / (f"manifest.limit{limit}.jsonl" if limit else "manifest.jsonl")
    meta = ddir / (f"info.limit{limit}.json" if limit else "info.json")
    if man.exists() and meta.exists():
        info = DatasetInfo(**json.load(open(meta, encoding="utf-8")))
        return info, [json.loads(l) for l in open(man, encoding="utf-8")]
    info, it = loader(opts, limit)
    wdir = ddir / "wav"
    wdir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for uid, audio, text, extra in it:
            rows.append(_store(uid, audio, text, extra, wdir))
    except DatasetUnavailable:
        raise
    except Exception as e:
        if "Gated" in type(e).__name__ or "403" in str(e) or "401" in str(e):
            raise DatasetUnavailable(f"{name}: access denied while reading ({type(e).__name__})") from e
        raise
    if not rows:
        raise DatasetUnavailable(f"{name}: loader produced 0 utterances ({info.notes})")
    rows.sort(key=lambda r: r["id"])
    tmp = man.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp.replace(man)
    meta.write_text(json.dumps(info.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return info, rows
