"""
Scalp EEG channel names: which channels are EEG electrodes (10-20 / 10-10), which are bipolar
pairs of electrodes, which are cardiac, muscle, eye, trigger or bookkeeping channels, and the
canonical spelling MNE uses (`Fp1`, `T7` for `T3`, `P7` for `T5`; `EEG Fp1-Ref` -> `Fp1`).

Template positions come from MNE's 10-05 montage (`colin27_1005`, the montage MNE called
`standard_1005` before 1.13), in MNE head coordinates (metres).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

TEMPLATE = "colin27_1005"          # MNE >= 1.13 name of standard_1005
TEMPLATE_SPACE = "standard_1005"   # what conventions.md calls it (electrodes.tsv `space`)

# Old 10-20 names -> the 10-10 names MNE uses (American Clinical Neurophysiology Society 2006).
OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
# Scalp positions recorded in clinical EEG that the 10-05 template has no position for.
EXTRA_SCALP = {"T1": "T1", "T2": "T2", "SP1": "Sp1", "SP2": "Sp2", "PG1": "Pg1", "PG2": "Pg2"}

# BIDS channels.tsv `type` values (upper case) that are not EEG and are dropped.
NON_EEG_TYPES = {
    "ECG": "ecg", "EKG": "ecg", "EMG": "emg", "EOG": "eog", "VEOG": "eog", "HEOG": "eog",
    "TRIG": "trigger", "STIM": "trigger", "MISC": "misc", "RESP": "misc", "SYSCLOCK": "misc",
    "PPG": "misc", "GSR": "misc", "TEMP": "misc", "AUDIO": "misc", "PD": "misc", "EYEGAZE": "eog",
    "PUPIL": "eog", "REF": "misc", "ADC": "misc", "DAC": "misc", "OTHER": "misc", "BIO": "misc",
    "SYST": "misc", "SEEG": "ieeg", "ECOG": "ieeg", "DBS": "ieeg", "MEG": "misc", "MEGMAG": "misc",
    "MEGGRAD": "misc", "HEARTRATE": "misc",
}
MNE_TO_BIDS = {
    "eeg": "EEG", "ecg": "ECG", "emg": "EMG", "eog": "EOG", "stim": "TRIG", "misc": "MISC", "resp": "RESP",
    "bio": "BIO", "syst": "SYST", "temperature": "TEMP", "gsr": "GSR", "seeg": "SEEG", "ecog": "ECOG",
    "dbs": "DBS",
}

# Names of channels that are never EEG (checked on the whole name and on each part of a pair).
_DROP_PATTERNS: list[tuple[str, str]] = [
    (r"(EKG|ECG)\w*", "ecg"),
    (r"(EMG|CHIN|LEG|LLEG|RLEG|LAT|RAT|SUBMENTAL)\w*", "emg"),
    (r"(EOG|HEOG|VEOG|LOC|ROC|LUE|RAE|RLE|LLE|E1|E2)\w*", "eog"),
    (r"(DC|TRIG|TRIGGER|STATUS|EVENT|EVENTS|MARKER|PHOTIC|IBI|BURSTS|SUPPR|PATIENT ?EVENT|STI|STIM|MARK|VNS)\s*\d*",
     "trigger"),
    (r"(EDF )?ANNOTATIONS?", "misc"),
    (r"(SPO2|SAO2|OSAT|PULSE|PLETH|AIRFLOW|FLOW|SNORE|ETCO2|NIBP|PR|HR|RESP|RR|THOR|THORAX|ABD|ABDOMEN|CO2|BP|TEMP|"
     r"POS|POSITION|LIGHT|SOUND|MIC|ACC[XYZ]?|GYRO[XYZ]?|X|Y|Z|NASAL|ORAL|CANNULA|CPAP)\s*\d*", "misc"),
    (r"(EMPTY|NONE|N/?A|UNUSED|NOT ?USED|BLANK|NC|EXTRA|SPARE|OFF|-+|\.+|_+)\s*\d*", "empty"),
    (r"(REF|GND|GROUND|COM|VREF|CREF|REFERENCE)\s*\d*", "misc"),
]
_DROP_RES = [(re.compile(r"^(?:" + p + r")$", re.IGNORECASE), t) for p, t in _DROP_PATTERNS]
_PREFIX_RE = re.compile(r"^(EEG|POL|SCALP)[\s_:-]*(?=[A-Za-z])", re.IGNORECASE)
_REF_TOKENS = ("REF", "AVG", "AV", "AVE", "AVERAGE", "AR", "CAR", "LE", "REF1", "REF2", "CREF")
_REF_RE = re.compile(r"^(.+?)[\s_]*-?[\s_]*(" + "|".join(_REF_TOKENS) + r")$", re.IGNORECASE)
_DUP_RE = re.compile(r"^(.+)-(\d+)$")   # MNE renames duplicate channel names to NAME-0, NAME-1


@lru_cache(maxsize=1)
def _template() -> tuple[dict[str, str], dict[str, tuple[float, float, float]]]:
    """({upper name: MNE spelling}, {MNE spelling: head xyz in m}) for the 10-05 template."""
    import warnings

    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mont = mne.channels.make_standard_montage(TEMPLATE)
        except ValueError:          # MNE < 1.13 has no colin27_1005: the same positions as standard_1005
            mont = mne.channels.make_standard_montage(TEMPLATE_SPACE)
        info = mne.create_info(mont.ch_names, 1000.0, "eeg")
        info.set_montage(mont, on_missing="ignore")
    pos = {}
    for ch in info["chs"]:
        xyz = tuple(float(v) for v in ch["loc"][:3])
        pos[ch["ch_name"]] = xyz
    return {n.upper(): n for n in mont.ch_names}, pos


def canonical_electrode(token: str) -> str | None:
    """The MNE spelling of a 10-20/10-10 electrode name ('FP1' -> 'Fp1', 'T3' -> 'T7'), or None."""
    u = token.strip().upper().replace(" ", "").replace("_", "")
    if not u:
        return None
    if u in OLD_TO_NEW:
        return OLD_TO_NEW[u]
    std, _ = _template()
    if u in std:
        return std[u]
    return EXTRA_SCALP.get(u)


def position(name: str) -> tuple[float, float, float] | None:
    """Template head coordinates (m) of an electrode name in canonical spelling, or None."""
    return _template()[1].get(name)


def strip_prefix(name: str) -> str:
    return _PREFIX_RE.sub("", name.strip()).strip()


@dataclass
class Parsed:
    original: str
    kind: str                      # electrode | pair | other
    a: str | None = None           # canonical electrode (first electrode of a pair)
    b: str | None = None           # second electrode of a pair
    ref: str | None = None         # reference token written in the name ('Avg', 'Ref', ...)
    dup: int | None = None         # MNE's duplicate suffix (T8-P8-1 -> 1)


def parse(name: str) -> Parsed:
    """What a channel name says: an electrode (maybe with a reference suffix), a pair, or other."""
    n = strip_prefix(name)
    e = canonical_electrode(n)
    if e:
        return Parsed(name, "electrode", a=e)
    m = _REF_RE.match(n)
    if m:
        e = canonical_electrode(m.group(1).rstrip("-_ "))
        if e:
            return Parsed(name, "electrode", a=e, ref=m.group(2))
    dup = None
    md = _DUP_RE.match(n)
    if md and md.group(1).count("-") == 1:
        n, dup = md.group(1), int(md.group(2))
    if "-" in n:
        left, _, right = n.partition("-")
        a, b = canonical_electrode(left), canonical_electrode(right)
        if a and b:
            return Parsed(name, "pair", a=a, b=b, dup=dup)
    return Parsed(name, "other", dup=dup)


def drop_type_by_name(name: str) -> str | None:
    """The non-EEG type a name implies (ecg, emg, eog, trigger, misc, empty), or None."""
    raw = name.strip()
    if not raw or re.fullmatch(r"[-_. ]*\d*", raw):
        return "empty"
    n = strip_prefix(raw)
    parts = [raw, n] + [p for p in re.split(r"[-\s]+", n) if p]
    for p in parts:
        for rx, t in _DROP_RES:
            if rx.match(p) and not canonical_electrode(p):
                return t
    return None


def hemisphere(electrode: str) -> str:
    """L (odd number), R (even), Z (midline 'z'), '' when unknown."""
    if electrode.lower().endswith("z"):
        return "Z"
    m = re.search(r"(\d+)$", electrode)
    if not m:
        return ""
    return "L" if int(m.group(1)) % 2 else "R"


def region(electrode: str) -> str:
    """frontal | temporal | central | parietal | occipital | midline (scalp region of a 10-20/10-10 name)."""
    e = electrode
    if e.lower().endswith("z"):
        return "midline"
    u = e.upper()
    if u in ("P7", "P8", "P9", "P10", "A1", "A2", "M1", "M2", "T1", "T2", "SP1", "SP2", "PG1", "PG2"):
        return "temporal"
    for prefix, reg in (("FP", "frontal"), ("AF", "frontal"), ("FT", "temporal"), ("FC", "central"),
                        ("TP", "temporal"), ("CP", "central"), ("PO", "parietal"), ("F", "frontal"),
                        ("T", "temporal"), ("C", "central"), ("P", "parietal"), ("O", "occipital"),
                        ("I", "occipital"), ("N", "frontal")):
        if u.startswith(prefix):
            return reg
    return ""


def pair_hemisphere(a: str, b: str) -> str:
    ha, hb = hemisphere(a), hemisphere(b)
    if ha == hb:
        return ha
    if ha == "Z":
        return hb
    if hb == "Z":
        return ha
    return ""   # across hemispheres (FT9-FT10)


def pair_region(a: str, b: str) -> str:
    ra, rb = region(a), region(b)
    if "temporal" in (ra, rb):
        return "temporal"
    if ra == rb:
        return ra
    if ra == "midline":
        return rb
    return ra


@dataclass
class ChannelClass:
    index: int
    original: str
    keep: bool
    name: str = ""                 # output name ('Fp1', 'Fp1-F7')
    kind: str = ""                 # electrode | pair
    contacts: list[str] = field(default_factory=list)
    dropped_type: str = ""         # ecg, emg, eog, trigger, misc, empty, duplicate, ...
    reason: str = ""
    source_type: str = ""


def classify(names: list[str], source_types: dict[str, str], types_trusted: bool) -> tuple[list[ChannelClass], str, str]:
    """
    Decide per channel whether it is EEG and kept, and how the recording is montaged.

    Returns (classes, layout, recorded_reference): layout is `referential` or `bipolar`;
    recorded_reference is the reference the names carry ('Avg', 'Ref', 'A1', ...) or ''.
    `source_types` are BIDS types (channels.tsv, or MNE's types for FIF); when
    `types_trusted` is False (EDF, BrainVision, ...) every channel arrives as EEG and the
    name decides.
    """
    parsed = [parse(n) for n in names]
    out: list[ChannelClass] = []
    for i, (n, p) in enumerate(zip(names, parsed)):
        st = (source_types.get(n) or "").upper()
        if types_trusted and st in NON_EEG_TYPES:
            out.append(ChannelClass(i, n, False, dropped_type=NON_EEG_TYPES[st], reason=f"type {st}", source_type=st))
            continue
        if p.kind == "other":
            t = drop_type_by_name(n)
            if t:
                out.append(ChannelClass(i, n, False, dropped_type=t, reason=f"name '{n}' looks like {t}",
                                        source_type=st))
            elif types_trusted and st == "EEG":
                nm = strip_prefix(n) or n
                out.append(ChannelClass(i, n, True, name=nm, kind="electrode", contacts=[nm],
                                        reason="type EEG; not a 10-20/10-10 name (no template position)",
                                        source_type=st))
            else:
                out.append(ChannelClass(i, n, False, dropped_type="misc",
                                        reason=f"'{n}' is not a 10-20/10-10 electrode name", source_type=st))
            continue
        if p.kind == "electrode":
            out.append(ChannelClass(i, n, True, name=p.a or n, kind="electrode", contacts=[p.a or n],
                                    reason="10-20/10-10 electrode", source_type=st))
        else:
            out.append(ChannelClass(i, n, True, name=f"{p.a}-{p.b}", kind="pair", contacts=[p.a or "", p.b or ""],
                                    reason="bipolar pair", source_type=st))

    kept = [c for c in out if c.keep]
    pairs = [c for c in kept if c.kind == "pair"]
    recorded_ref = ""
    layout = "referential"
    if kept and len(pairs) >= 0.5 * len(kept):
        seconds = [c.contacts[1] for c in pairs]
        common = max(set(seconds), key=seconds.count)
        if seconds.count(common) >= 0.9 * len(pairs):
            # every pair shares its second electrode: referential to that electrode (C3-A1, C4-A1, ...)
            recorded_ref = common
            for c in pairs:
                c.kind, c.name, c.contacts = "electrode", c.contacts[0], [c.contacts[0]]
                c.reason = f"electrode referenced to {common}"
        else:
            layout = "bipolar"
            for c in kept:
                if c.kind == "electrode":
                    c.keep, c.dropped_type = False, "referential"
                    c.reason = "single-electrode channel in a bipolar recording"
    else:
        for c in pairs:   # a few pairs in a referential recording (e.g. a bipolar EKG lead that parsed as EEG)
            c.keep, c.dropped_type, c.reason = False, "bipolar", "bipolar pair in a referential recording"
        refs = [parsed[c.index].ref for c in kept if c.kind == "electrode" and parsed[c.index].ref]
        if refs:
            recorded_ref = max(set(refs), key=refs.count)

    # duplicates: the same output name twice (CHB-MIT's T8-P8), or a pair and its inverse (P7-T7 = -(T7-P7))
    seen: dict[str, ChannelClass] = {}
    for c in out:
        if not c.keep:
            continue
        if c.name in seen:
            c.keep, c.dropped_type = False, "duplicate"
            c.reason = f"duplicate of '{seen[c.name].original}'"
            continue
        if c.kind == "pair":
            inv = f"{c.contacts[1]}-{c.contacts[0]}"
            if inv in seen:
                c.keep, c.dropped_type = False, "duplicate"
                c.reason = f"inverse of '{seen[inv].original}' (same derivation, opposite sign)"
                continue
        seen[c.name] = c
    return out, layout, recorded_ref
