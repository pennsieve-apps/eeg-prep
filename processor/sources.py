"""
Finding recordings under INPUT_DIR and opening them: BIDS-EEG (via MNE-BIDS) and plain
EDF/EDF+/BDF, BrainVision, EEGLAB, FIF, Nihon Kohden and Natus/Nicolet files (via MNE).

Every reader is wrapped in a `Source` with the same small interface: channel names, the
types the file itself records, sampling rate, length, annotations and
`get(picks, start, stop)` returning µV. Nothing is loaded whole: recordings are read in windows.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from processor import bids as B
from processor.channels import MNE_TO_BIDS

log = logging.getLogger("eeg-prep")

# extension (lower case) -> format; `.eeg` is Nihon Kohden only next to its .21e/.pnt files
# (a BrainVision recording's .eeg is read through its .vhdr)
EXTENSIONS = {".edf": "edf", ".bdf": "bdf", ".vhdr": "brainvision", ".set": "eeglab", ".fif": "fif",
              ".e": "natus", ".eeg": "nihon-kohden"}
SOURCE_FORMAT = {"edf": "edf", "bdf": "bdf", "brainvision": "brainvision", "eeglab": "eeglab", "fif": "fif",
                 "natus": "natus", "nihon-kohden": "nihon-kohden"}


def walk(root: Path) -> list[Path]:
    """Every file under root, following symlinks (INPUT_DIR entries are symlinks), hidden ones skipped."""
    files: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                files.append(p)
    return files


@dataclass
class Candidate:
    path: Path
    fmt: str
    bids: bool
    entities: dict[str, str]
    recording_id: str
    subject: str
    session: str
    task: str
    run: str
    bids_root: Path | None = None
    origin: str = ""              # path relative to INPUT_DIR


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "x"


# File-name stems that name no patient (exporters' defaults): the subject comes from the folder.
GENERIC_STEMS = {"eeg", "edf", "bdf", "rec", "recording", "record", "file", "data", "export", "exported",
                 "signal", "signals", "raw", "session", "ses", "run", "seg", "segment", "part", "x", "trace",
                 "study", "test", "untitled", "eegdata", "eegrecording"}


def describe(path: Path, fmt: str, folder_subject: bool = False) -> Candidate:
    """Identity of one recording: BIDS entities when the file is in a BIDS layout, else from its name.

    Loose files: `chb01_03.edf` -> subject `chb01`, run `03`, id `sub-chb01_task-eeg_run-03`;
    `chb02_16+.edf` (CHB-MIT's continuation of hour 16) -> `sub-chb02_task-eeg_acq-plus_run-16`;
    a name without a trailing number -> `sub-<name>_task-eeg`. With `folder_subject` (a generic stem
    such as `EEG_01`, or stems that collide across folders) the subject is the parent folder's name:
    `P1/EEG_01.edf` -> `sub-P1_task-eeg_run-01`.
    """
    ents = B.parse_entities(path.name)
    parent = path.parent
    if ents:
        is_bids = parent.name in ("eeg", "ieeg") and (
            parent.parent.name.startswith("sub-") or parent.parent.parent.name.startswith("sub-"))
        stem = B.stem_of(path)
        sub = ents["sub"]
        root = None
        if is_bids:
            for p in path.parents:
                if p.name == f"sub-{sub}":
                    root = p.parent
                    break
        return Candidate(path=path, fmt=fmt, bids=is_bids, entities=ents, recording_id=stem, subject=sub,
                         session=ents.get("ses", ""), task=ents.get("task", ""), run=ents.get("run", ""),
                         bids_root=root)
    stem = path.name.split(".")[0]
    acq = ""
    if stem.endswith("+"):                      # CHB-MIT: chb02_16+.edf continues hour 16 after a gap
        stem, acq = stem.rstrip("+"), "plus"
    prefix, run = stem, ""
    m = re.match(r"^(.*?)[\s_-]+0*(\d{1,4})$", stem)          # name_03, name-3, name 03
    if m and _alnum(m.group(1)) != "x":
        prefix, run = m.group(1), f"{int(m.group(2)):02d}"
    else:
        m = re.match(r"^([A-Za-z]+)0*(\d{1,4})$", stem)        # EEG01: a generic word and a number
        if m and m.group(1).lower() in GENERIC_STEMS:
            prefix, run = m.group(1), f"{int(m.group(2)):02d}"
    generic = prefix.strip(" _-").lower() in GENERIC_STEMS
    sub = _alnum(path.parent.name) if (folder_subject or generic) else _alnum(prefix)
    rid = f"sub-{sub}_task-eeg" + (f"_acq-{acq}" if acq else "") + (f"_run-{run}" if run else "")
    return Candidate(path=path, fmt=fmt, bids=False, entities={}, recording_id=rid, subject=sub, session="",
                     task="eeg", run=run)


def _format_of(p: Path, names_lower: set[str]) -> str | None:
    name = p.name.lower()
    if name.endswith(".fif.gz"):
        return "fif"
    ext = p.suffix.lower()
    fmt = EXTENSIONS.get(ext)
    if fmt == "nihon-kohden":
        stem = str(p.with_suffix("")).lower()
        if not any(f"{stem}{e}" in names_lower for e in (".21e", ".pnt", ".log")):
            return None    # BrainVision's binary .eeg, read through the .vhdr
    if fmt == "fif" and re.search(r"-\d+\.fif(\.gz)?$", name) and "_split-" not in name:
        return None        # continuation of a split FIF (raw-1.fif); MNE reads it through the first part
    return fmt


def find_recordings(input_dir: Path, pattern: str) -> tuple[list[Candidate], list[Path], list[str]]:
    """(recordings matching `pattern`, every file seen, notes)."""
    files = walk(input_dir)
    names_lower = {str(p).lower() for p in files}
    found: list[Candidate] = []
    for p in files:
        fmt = _format_of(p, names_lower)
        if fmt is None:
            continue
        if fmt == "eeglab" and not p.with_suffix(".fdt").exists() and p.stat().st_size < 1024:
            continue
        found.append(describe(p, fmt))
    notes: list[str] = []
    # loose files whose ids collide across folders (P1/EEG_01.edf, P2/EEG_01.edf): subject from the folder
    by_id: dict[str, list[int]] = {}
    for k, c in enumerate(found):
        if not c.bids:
            by_id.setdefault(c.recording_id, []).append(k)
    for rid, ks in by_id.items():
        if len(ks) > 1 and len({found[k].path.parent for k in ks}) == len(ks):
            for k in ks:
                found[k] = describe(found[k].path, found[k].fmt, folder_subject=True)
            notes.append(f"{len(ks)} files would all be {rid}; their subjects are taken from their folders")
    for c in found:
        c.origin = str(c.path.relative_to(input_dir))
    # a BIDS dataset's derivatives/ folder holds processed copies (ds003555 keeps HFO clips there)
    outside = [c for c in found if "derivatives" not in Path(c.origin).parts]
    if outside and len(outside) < len(found):
        notes.append(f"{len(found) - len(outside)} file(s) under derivatives/ folders skipped (processed copies)")
        found = outside
    pats = [s.strip() for s in (pattern or "*").split(",") if s.strip()] or ["*"]
    sel = [c for c in found if any(fnmatch.fnmatch(c.recording_id, p) or fnmatch.fnmatch(c.origin, p)
                                   or fnmatch.fnmatch(c.path.name, p) for p in pats)]
    if len(sel) < len(found):
        notes.append(f"RECORDINGS={pattern!r} selected {len(sel)} of {len(found)} recording(s)")
    sel.sort(key=lambda c: (c.subject, c.session, c.recording_id))
    return sel, files, notes


@dataclass
class Source:
    ch_names: list[str]
    file_types: dict[str, str]            # BIDS type per channel as the reader reports it
    types_trusted: bool
    sfreq: float
    n_times: int
    annotations: list[tuple[float, float, str]] = field(default_factory=list)
    line_freq: float | None = None
    meas_date: object = None
    reader: str = ""
    raw: object = None
    file_bads: list[str] = field(default_factory=list)
    subject_info: dict = field(default_factory=dict)
    highpass: float | None = None
    lowpass: float | None = None
    scale: np.ndarray | None = None       # per channel: MNE's values (V) -> µV; 1 where the file has no unit
    unitless: list[str] = field(default_factory=list)

    def get(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        """float64 µV, shape (len(picks), stop - start)."""
        x = self.raw.get_data(picks=picks, start=start, stop=stop)
        sc = self.scale[picks][:, None] if self.scale is not None else 1e6
        return np.nan_to_num(x * sc, nan=0.0, posinf=0.0, neginf=0.0)

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:  # noqa: BLE001
            pass


def _from_raw(raw, reader: str, trusted: bool) -> Source:
    ann = [(float(a["onset"] - raw.first_time if raw.annotations.orig_time is not None else a["onset"]),
            float(a["duration"]), str(a["description"])) for a in raw.annotations]
    # MNE pads EDF files whose last data record is incomplete and marks the pad BAD_ACQ_SKIP
    n_times = raw.n_times
    sf = raw.info["sfreq"]
    for on, du, de in ann:
        if de.upper() == "BAD_ACQ_SKIP" and abs(on + du - raw.n_times / sf) < 1.0 / sf + 1e-6:
            n_times = min(n_times, int(round(on * sf)))
    lf = raw.info.get("line_freq")
    types = {n: MNE_TO_BIDS.get(t, t.upper()) for n, t in zip(raw.ch_names, raw.get_channel_types())}
    si = raw.info.get("subject_info") or {}
    return Source(ch_names=list(raw.ch_names), file_types=types, types_trusted=trusted, sfreq=float(sf),
                  n_times=int(n_times), annotations=ann, line_freq=float(lf) if lf else None,
                  meas_date=raw.info.get("meas_date"), reader=reader, raw=raw, file_bads=list(raw.info["bads"]),
                  subject_info=dict(si), highpass=raw.info.get("highpass"), lowpass=raw.info.get("lowpass"))


def edf_dimensions(path: Path) -> list[tuple[str, str]] | None:
    """(label, physical dimension) of every signal in an EDF/BDF header, annotation signals left out."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(256)
            ns = int(head[252:256].decode("ascii", "replace").strip())
            body = fh.read(256 * ns)
    except (OSError, ValueError):
        return None
    labels = [body[16 * i:16 * i + 16].decode("latin-1").strip() for i in range(ns)]
    o = ns * (16 + 80)
    dims = [body[o + 8 * i:o + 8 * i + 8].decode("latin-1").strip() for i in range(ns)]
    return [(lb, d) for lb, d in zip(labels, dims) if lb not in ("EDF Annotations", "BDF Annotations")]


def apply_units(src: "Source", path: Path, fmt: str) -> list[str]:
    """EDF/BDF signals with a blank physical dimension: MNE takes them as volts; their numbers are µV."""
    if fmt not in ("edf", "bdf"):
        return []
    dims = edf_dimensions(path)
    if not dims or len(dims) != len(src.ch_names):
        return []
    scale = np.full(len(src.ch_names), 1e6)
    for i, (_, d) in enumerate(dims):
        if not d:
            scale[i] = 1.0
            src.unitless.append(src.ch_names[i])
    src.scale = scale
    if src.unitless:
        return [f"{len(src.unitless)} channel(s) have no physical dimension in the EDF header; their values are "
                "taken as µV (MNE would read them as volts)"]
    return []


def _read_mne(path: Path, fmt: str):
    import mne
    kw = {"preload": False, "verbose": "error"}
    if fmt == "edf":
        return mne.io.read_raw_edf(path, infer_types=False, **kw)
    if fmt == "bdf":
        return mne.io.read_raw_bdf(path, infer_types=False, **kw)
    if fmt == "brainvision":
        return mne.io.read_raw_brainvision(path, **kw)
    if fmt == "eeglab":
        return mne.io.read_raw_eeglab(path, **kw)
    if fmt == "fif":
        return mne.io.read_raw_fif(path, **kw)
    if fmt == "nihon-kohden":
        return mne.io.read_raw_nihon(path, **kw)
    if fmt == "natus":
        return mne.io.read_raw_nicolet(path, ch_type="eeg", **kw)
    raise ValueError(f"unsupported format {fmt}")


def open_source(c: Candidate) -> tuple[Source, list[str]]:
    """Open one recording; BIDS ones through MNE-BIDS (types and bads from channels.tsv), else directly."""
    notes: list[str] = []
    has_channels_tsv = c.bids and B.sidecar(c.path, "channels.tsv") is not None
    if c.bids and c.bids_root is not None and c.fmt in ("edf", "bdf", "brainvision", "eeglab", "fif"):
        try:
            import mne_bids
            bp = mne_bids.get_bids_path_from_fname(c.path)
            bp.update(root=c.bids_root)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne_bids.read_raw_bids(bp, verbose="error")
            src = _from_raw(raw, "mne-bids", has_channels_tsv or c.fmt == "fif")
            notes.extend(apply_units(src, c.path, c.fmt))
            return src, notes
        except Exception as exc:  # noqa: BLE001 - fall back to the plain reader
            notes.append(f"MNE-BIDS could not read it ({type(exc).__name__}: {str(exc)[:200]}); read with MNE directly")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = _read_mne(c.path, c.fmt)
    src = _from_raw(raw, f"mne-{c.fmt}", c.fmt == "fif")
    notes.extend(apply_units(src, c.path, c.fmt))
    return src, notes
