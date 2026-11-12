"""
BIDS sidecars of a source recording: _channels.tsv, _events.tsv (plain BIDS `trial_type` rows
and the HED-SCORE rows of SzCORE-style datasets: `eventType` sz, sz_foc_ia, ...), _eeg.json,
_electrodes.tsv, and participants.tsv, found next to the recording or up the BIDS inheritance chain.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("eeg-prep")

NA = {"n/a", "na", "nan", "none", ""}


def parse_entities(name: str) -> dict[str, str]:
    """BIDS entities of a file name: {'sub': 'PN06', 'ses': '01', ...} (empty when not BIDS)."""
    stem = name.split(".")[0]
    ents: dict[str, str] = {}
    for p in stem.split("_"):
        if "-" in p:
            k, v = p.split("-", 1)
            if re.fullmatch(r"[a-z]+", k) and v:
                ents[k] = v
    return ents if "sub" in ents else {}


def stem_of(path: Path) -> str:
    """The entity chain before the suffix: sub-X_ses-Y_task-Z_run-01 for sub-X_ses-Y_task-Z_run-01_eeg.edf."""
    base = path.name.split(".")[0]
    if "_" in base and base.rsplit("_", 1)[1] in ("eeg", "ieeg", "meg"):
        base = base.rsplit("_", 1)[0]
    return base


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]


def read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("%s: unreadable JSON (%s)", path, exc)
        return {}


def sidecar(rec_path: Path, suffix_ext: str) -> Path | None:
    """
    The sidecar `<stem>_<suffix_ext>` of a recording, or one up the inheritance chain (fewer
    entities, same directory or a parent up to the subject folder).
    """
    stem = stem_of(rec_path)
    d = rec_path.parent
    exact = d / f"{stem}_{suffix_ext}"
    if exact.is_file():
        return exact
    ents = parse_entities(rec_path.name)
    if not ents:
        return None
    candidates: list[Path] = []
    for folder in (d, d.parent, d.parent.parent, d.parent.parent.parent):
        if not folder.is_dir():
            continue
        for f in folder.glob(f"*_{suffix_ext}"):
            fe = parse_entities(f.name)
            if fe and all(ents.get(k) == v for k, v in fe.items() if k not in ("space",)):
                candidates.append(f)
        if candidates:
            break
        if folder.name.startswith("sub-"):
            break
    if not candidates:
        return None
    candidates.sort(key=lambda p: -len(parse_entities(p.name)))
    return candidates[0]


@dataclass
class ChannelRow:
    name: str
    type: str = ""
    status: str = ""
    status_description: str = ""
    low_cutoff: float | None = None
    high_cutoff: float | None = None


def num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def read_channels(path: Path | None) -> dict[str, ChannelRow]:
    if path is None:
        return {}
    out = {}
    for r in read_tsv(path):
        name = r.get("name", "")
        if not name:
            continue
        desc = r.get("status_description", "")
        out[name] = ChannelRow(name=name, type=r.get("type", "").upper(), status=r.get("status", "").lower(),
                               status_description="" if desc.lower() in NA else desc,
                               low_cutoff=num(r.get("low_cutoff")), high_cutoff=num(r.get("high_cutoff")))
    return out


@dataclass
class Event:
    onset: float
    duration: float | None
    trial_type: str            # as written in the source
    source: str                # file it came from
    channels: list[str] = field(default_factory=list)
    hed: str = ""              # HED-SCORE eventType (sz_foc_ia, ...)
    date_time: str = ""        # SzCORE dateTime column (the recording's start)


def read_events(path: Path | None) -> list[Event]:
    """events.tsv rows: `trial_type` (or `value`) for plain BIDS, `eventType` for HED-SCORE (SzCORE)."""
    if path is None:
        return []
    out = []
    for r in read_tsv(path):
        onset = num(r.get("onset"))
        if onset is None:
            continue
        dur = num(r.get("duration"))
        hed = r.get("eventType", "") or r.get("event_type", "")
        tt = r.get("trial_type") or hed or r.get("value") or ""
        chans = [c.strip() for c in re.split(r"[,;|]", r.get("channels", "")) if c.strip() and c.strip().lower() not in NA]
        out.append(Event(onset, dur, "" if tt.lower() in NA else tt, path.name, chans,
                         "" if hed.lower() in NA else hed, r.get("dateTime", "")))
    return out


def read_participants(root: Path | None) -> dict[str, dict[str, str]]:
    """participants.tsv of a BIDS root: {subject label: {age, sex, group, ...}}."""
    if root is None:
        return {}
    p = root / "participants.tsv"
    if not p.is_file():
        return {}
    out = {}
    for r in read_tsv(p):
        pid = r.get("participant_id", "")
        if pid:
            out[pid[4:] if pid.startswith("sub-") else pid] = r
    return out


def read_electrodes(path: Path | None) -> dict[str, tuple[float, float, float]]:
    """Measured electrode positions of a BIDS-EEG source (name -> xyz in its own units)."""
    if path is None:
        return {}
    out = {}
    for r in read_tsv(path):
        n = r.get("name", "")
        try:
            out[n] = (float(r["x"]), float(r["y"]), float(r["z"]))
        except (KeyError, ValueError):
            continue
    return out


def source_filters(sidecar_json: dict) -> tuple[float | None, float | None]:
    """(high-pass, low-pass) cutoffs in Hz found in a BIDS sidecar's SoftwareFilters/HardwareFilters."""
    hp, lp = None, None

    def walk(obj, path: str) -> None:
        nonlocal hp, lp
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path} {k}".lower())
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if "notch" in path:
                return
            if re.search(r"high[ -]?pass|highcut|low[ _-]?cutoff", path):
                hp = max(hp or 0.0, float(obj))
            elif re.search(r"low[ -]?pass|lowcut|high[ _-]?cutoff", path):
                lp = min(lp or float("inf"), float(obj))
    for key in ("SoftwareFilters", "HardwareFilters"):
        walk(sidecar_json.get(key), "")
    return hp, lp
