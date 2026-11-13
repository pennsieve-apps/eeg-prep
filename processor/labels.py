"""
Labels on input: seizures and other annotated states, from

- BIDS events.tsv (plain `trial_type` rows and HED-SCORE `eventType` rows),
- EDF+/BrainVision/... annotations in the file (text matching seizure|sz|onset|ictal),
- CHB-MIT `<subject>-summary.txt` files next to the EDFs,
- an optional labels.csv: subject, recording_id (empty = every recording of the subject),
  onset_s, duration_s (one event per row), trial_type (default seizure), plus recording-level
  rows (no onset_s) whose other columns (lateralisation, outcome, ...) go into recordings.csv.

Everything is mapped onto the trial_type vocabulary of conventions.md.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("eeg-prep")

NA = ("", "n/a", "na", "nan", "none")
KNOWN_COLUMNS = {"subject", "recording_id", "onset_s", "duration_s", "trial_type", "channels"}

_NOT_SEIZURE = re.compile(r"(inter|pre|post|peri|non)[ _-]?(ictal|seizure)|no[ _-]?seizure|bckg|background|"
                          r"seizure[ _-]?free", re.IGNORECASE)
_SEIZURE = re.compile(r"(^|[^a-z])(seizure|sz|ictal|crisi|onset|eeg[ _-]?onset|ueo)([^a-z]|$)|^sz|^seizure",
                      re.IGNORECASE)
_SEIZURE_END = re.compile(r"(seizure|sz|ictal)[ _-]*(end|offset|stop)|^(end|offset)[ _-]*(of[ _-]*)?(seizure|sz)",
                          re.IGNORECASE)

# other annotated states -> conventions.md trial_type
STATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"eyes?[ _-]*(open|opened|eo)\b|\bEO\b|augen[ _-]*auf", re.I), "eyes_open"),
    (re.compile(r"eyes?[ _-]*(closed?|ec)\b|\bEC\b|augen[ _-]*zu", re.I), "eyes_closed"),
    (re.compile(r"photic|\bIPS\b|\bPS\b|strobe|flash", re.I), "photic"),
    (re.compile(r"hyperv|\bHV\b", re.I), "hyperventilation"),
    (re.compile(r"lights?[ _-]*off", re.I), "lights_off"),
    (re.compile(r"lights?[ _-]*on", re.I), "lights_on"),
    (re.compile(r"\b(asleep|sleep|sleeping|drowsy|drowsiness|stage[ _-]*[1234n]|n[123]|rem)\b", re.I), "sleep"),
    (re.compile(r"\b(awake|wake|wakeful|arousal)\b", re.I), "wake"),
    (re.compile(r"artifact|artefact|^bad|movement|chewing|muscle|electrode[ _-]*(pop|off)|bckg_arti", re.I),
     "artifact"),
]


def is_seizure(text: str) -> bool:
    t = text.strip()
    if not t or _NOT_SEIZURE.search(t) or _SEIZURE_END.search(t):
        return False
    return bool(_SEIZURE.search(t))


def map_trial_type(text: str) -> str | None:
    """conventions.md trial_type for an annotation text, or None when it is something else."""
    if is_seizure(text):
        return "seizure"
    t = text.strip().lower()
    if t in ("seizure", "artifact", "bad_segment", "eyes_open", "eyes_closed", "photic", "hyperventilation",
             "sleep", "wake", "lights_off", "lights_on"):
        return t
    for rx, tt in STATE_PATTERNS:
        if rx.search(text):
            return tt
    return None


@dataclass
class Label:
    onset: float
    duration: float | None
    trial_type: str
    source: str                   # annotation | labels.csv
    detail: str                   # where exactly (file and line / annotation text)
    channels: list[str] = field(default_factory=list)
    hed: str = ""


def seizures_from_annotations(annotations: list[tuple[float, float, str]]) -> list[Label]:
    """Seizure and state annotations of a file; a seizure without duration gets its end from a later
    'seizure end' annotation when there is one."""
    out: list[Label] = []
    ends = sorted(on for on, _, de in annotations if _SEIZURE_END.search(de))
    for on, du, de in annotations:
        if de.upper().startswith("BAD_ACQ_SKIP") or de.upper() == "EDGE" or de.upper().startswith("BAD boundary"):
            continue
        tt = map_trial_type(de)
        if tt is None:
            continue
        dur: float | None = du if du and du > 0 else None
        if tt == "seizure" and dur is None:
            later = [e for e in ends if e > on]
            if later:
                dur = later[0] - on
        out.append(Label(on, dur, tt, "annotation", f"file annotation '{de}'"))
    return out


# ---- CHB-MIT summary files ------------------------------------------------------------------------

def read_chbmit_summary(path: Path) -> dict[str, dict]:
    """
    `chbNN-summary.txt` -> {edf file name: {"start": "hh:mm:ss" or None, "seizures": [(start_s, end_s)]}}.
    Handles both "Seizure Start Time: N seconds" and "Seizure 1 Start Time: N seconds".
    """
    out: dict[str, dict] = {}
    cur = None
    pending_start = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s*(\S+)", line, re.I)
            if m:
                cur = out.setdefault(m.group(1), {"start": None, "seizures": []})
                pending_start = None
                continue
            if cur is None:
                continue
            m = re.match(r"File Start Time:\s*([\d:]+)", line, re.I)
            if m:
                cur["start"] = m.group(1)
                continue
            m = re.match(r"Seizure\s*\d*\s*Start Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if m:
                pending_start = float(m.group(1))
                continue
            m = re.match(r"Seizure\s*\d*\s*End Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if m and pending_start is not None:
                cur["seizures"].append((pending_start, float(m.group(1))))
                pending_start = None
    return out


def find_chbmit_summaries(files: list[Path]) -> dict[Path, dict[str, dict]]:
    """Every *-summary.txt that parses as a CHB-MIT summary, by folder."""
    out: dict[Path, dict[str, dict]] = {}
    for p in files:
        if p.name.lower().endswith("-summary.txt"):
            try:
                parsed = read_chbmit_summary(p)
            except OSError:
                continue
            if parsed:
                out.setdefault(p.parent, {}).update(parsed)
    return out


# ---- labels.csv -----------------------------------------------------------------------------------

def norm_subject(s: str) -> str:
    s = s.strip()
    return s[4:] if s.lower().startswith("sub-") else s


def norm_recording(r: str) -> str:
    """A recording id from what a user may write: the stem, a file name, or a path."""
    r = r.strip().replace("\\", "/").split("/")[-1]
    for ext in (".edf", ".bdf", ".vhdr", ".set", ".fif", ".fif.gz", ".e", ".eeg"):
        if r.lower().endswith(ext):
            r = r[: -len(ext)]
    if r.endswith("_eeg"):
        r = r[:-4]
    return r


@dataclass
class LabelRow:
    subject: str
    recording: str
    onset_s: float | None
    duration_s: float | None
    trial_type: str
    channels: list[str]
    line: int
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class LabelSet:
    path: Path | None = None
    rows: list[LabelRow] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)

    def for_recording(self, subject: str, recording_ids: set[str]) -> list[LabelRow]:
        sub = norm_subject(subject)
        out = []
        for r in self.rows:
            if r.subject and norm_subject(r.subject) != sub:
                continue
            if r.recording and r.recording not in recording_ids:
                continue
            if not r.subject and not r.recording:
                continue
            out.append(r)
        return out


def _f(v: str) -> float | None:
    if v.strip().lower() in NA:
        return None
    return float(v)


def read_labels(path: Path | None) -> LabelSet:
    if path is None:
        return LabelSet()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "subject" not in header and "recording_id" not in header:
            raise ValueError(f"{path.name} needs a subject or recording_id column (columns: {', '.join(header)})")
        rows = []
        for i, raw in enumerate(reader, start=2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if k is not None}
            try:
                onset = _f(r.get("onset_s", ""))
                dur = _f(r.get("duration_s", ""))
            except ValueError:
                log.warning("%s line %d: onset_s/duration_s not a number; row ignored", path.name, i)
                continue
            tt = (r.get("trial_type") or "seizure").strip().lower()
            chans = [c.strip() for c in re.split(r"[,;|]", r.get("channels", "")) if c.strip()]
            extra = {} if onset is not None else {k: v for k, v in r.items()
                                                  if k and k not in KNOWN_COLUMNS and v.lower() not in NA}
            rows.append(LabelRow(subject=r.get("subject", ""), recording=norm_recording(r.get("recording_id", "")),
                                 onset_s=onset, duration_s=dur, trial_type=tt, channels=chans, line=i, extra=extra))
    filled = {k for r in rows for k in r.extra}
    return LabelSet(path=path, rows=rows, extra_columns=[h for h in header if h in filled])


def resolve(rows: list[LabelRow], file_name: str) -> tuple[list[Label], dict[str, str]]:
    """labels.csv rows of one recording -> (events, recording-level columns)."""
    events: list[Label] = []
    extra: dict[str, str] = {}
    for r in rows:
        if r.onset_s is None:
            for k, v in r.extra.items():
                if k in extra and extra[k] != v:
                    log.warning("%s line %d: two %s values for this recording (%r, %r); using the first",
                                file_name, r.line, k, extra[k], v)
                else:
                    extra[k] = v
            continue
        tt = map_trial_type(r.trial_type) or r.trial_type
        events.append(Label(r.onset_s, r.duration_s, tt, "labels.csv", f"{file_name} line {r.line}", r.channels))
    return events, extra
