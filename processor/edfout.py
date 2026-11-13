"""
EDF+ output written incrementally (data record by data record from a disk-backed array, so a
24 h recording never has to fit in memory), with a physical range per channel, EDF+
annotations, and the read-back check with MNE.

Each channel's physical range is symmetric, ±L with L = 1.1 × robust max |x| (at least 1 µV),
where robust max = min(max |x|, 3 × the 99.99th percentile of |x|): a recording without
extreme outliers is stored without any clipping, while a single saturated artefact far above
the rest of the channel cannot coarsen its quantisation step (the rule of processor-ieeg-prep's
edfout.py). Digital range ±32767, so 0 µV is stored exactly and the step is 2L / 65534.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIGITAL = 32767
ROBUST_PCT = 99.99
ROBUST_FACTOR = 3.0
MARGIN = 1.1
MIN_RANGE_UV = 1.0
ANN_LABEL = "EDF Annotations"


def _field(value: str, width: int) -> bytes:
    b = value.encode("latin-1", "replace")[:width]
    return b + b" " * (width - len(b))


def _num8(x: float) -> str:
    """A number in at most 8 characters (EDF header field)."""
    if float(x).is_integer() and abs(x) < 1e8:
        return str(int(x))
    for d in range(7, -1, -1):
        s = f"{x:.{d}f}"
        if len(s) <= 8:
            return s
    raise ValueError(f"{x} does not fit 8 characters")


def symmetric_limit(lim: float) -> float:
    """The smallest L >= lim whose 8-character form, and that of -L, are exact, so ±L is symmetric."""
    lim = float(lim)
    for d in range(6, -1, -1):
        scale = 10 ** d
        cand = math.ceil(lim * scale - 1e-9) / scale
        s = f"{cand:.{d}f}"
        if len(s) + 1 <= 8:                   # room for the minus sign of -L
            return float(s)
    return float(math.ceil(lim))


@dataclass
class Range:
    limit: float        # L, µV, as stored in the header
    step: float         # quantisation step, µV
    clipped: int        # samples clipped to ±L
    max_abs: float      # max |x| before clipping


def channel_range(x: np.ndarray) -> Range:
    ax = np.abs(np.asarray(x, dtype=np.float32))
    ax = np.where(np.isfinite(ax), ax, 0.0)
    mx = float(ax.max()) if ax.size else 0.0
    p = float(np.percentile(ax, ROBUST_PCT)) if ax.size else 0.0
    robust = min(mx, ROBUST_FACTOR * p) if p > 0 else mx
    lim = symmetric_limit(max(MARGIN * robust, MIN_RANGE_UV))
    clipped = int(np.count_nonzero(ax > lim))
    return Range(limit=lim, step=2 * lim / (2 * DIGITAL), clipped=clipped, max_abs=mx)


def record_duration(sfreq: float) -> tuple[float, int]:
    """(data record duration in s, samples per record) such that sfreq × duration is a whole number."""
    for d in (1.0, 0.5, 0.25, 0.2, 0.1, 2.0, 4.0, 5.0, 10.0):
        spr = sfreq * d
        if abs(spr - round(spr)) < 1e-9 and round(spr) >= 1:
            return d, int(round(spr))
    return 1.0, int(round(sfreq))


def _onset(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return ("+" if x >= 0 else "") + (s or "0")


def _tal(onset: float, duration: float | None, text: str) -> bytes:
    text = text.replace("\x14", " ").replace("\x15", " ").replace("\x00", " ")
    b = _onset(onset).encode()
    if duration is not None and duration > 0:
        b += b"\x15" + _onset(duration).lstrip("+").encode()
    return b + b"\x14" + text.encode("utf-8")[:120] + b"\x14\x00"


@dataclass
class Signal:
    label: str
    limit: float
    prefiltering: str = ""
    transducer: str = ""


def write_edf(path: Path, data: np.ndarray, sfreq: float, signals: list[Signal],
              annotations: list[tuple[float, float | None, str]], start: dt.datetime | None,
              record_s: float, chunk_records: int = 60, progress=None) -> int:
    """
    Write `data` (channels × samples, µV; a numpy memmap is fine) as EDF+C. Samples beyond the
    last whole data record are dropped. Returns the number of data records.
    """
    n_ch, n_samp = data.shape
    spr = int(round(sfreq * record_s))
    n_rec = n_samp // spr
    if n_rec < 1:
        raise ValueError("recording shorter than one EDF data record")
    tals = [_tal(on, du, tx) for on, du, tx in sorted(annotations, key=lambda a: a[0]) if on >= 0]
    tk_max = len(_tal(n_rec * record_s, None, "")) + 2
    max_tal = max((len(t) for t in tals), default=0)
    total = sum(len(t) for t in tals)
    cap = max_tal + math.ceil(total / n_rec)          # event bytes one record can hold
    ann_bytes = tk_max + cap
    ann_bytes += ann_bytes % 2
    ann_samples = ann_bytes // 2
    # assign event TALs to records greedily (EDF+ onsets are absolute, any record may carry them)
    per_record: list[list[bytes]] = [[] for _ in range(n_rec)]
    r, used = 0, 0
    for t in tals:
        if used + len(t) > cap:
            r, used = r + 1, 0
        per_record[r].append(t)
        used += len(t)

    labels = [s.label for s in signals] + [ANN_LABEL]
    ns = len(labels)
    if start is not None and 1985 <= start.year <= 2084:
        date_s, time_s = start.strftime("%d.%m.%y"), start.strftime("%H.%M.%S")
        rec_id = "Startdate " + start.strftime("%d-%b-%Y").upper() + " X X X"
    else:
        date_s, time_s = "01.01.85", (start.strftime("%H.%M.%S") if start is not None else "00.00.00")
        rec_id = "Startdate X X X X"
    hdr = bytearray()
    hdr += _field("0", 8) + _field("X X X X", 80) + _field(rec_id, 80) + _field(date_s, 8) + _field(time_s, 8)
    hdr += _field(str(256 * (ns + 1)), 8) + _field("EDF+C", 44) + _field(str(n_rec), 8)
    hdr += _field(_num8(record_s), 8) + _field(str(ns), 4)
    hdr += b"".join(_field(lb, 16) for lb in labels)
    hdr += b"".join(_field(s.transducer, 80) for s in signals) + _field("", 80)
    hdr += b"".join(_field("uV", 8) for _ in signals) + _field("", 8)
    hdr += b"".join(_field(_num8(-s.limit), 8) for s in signals) + _field("-1", 8)
    hdr += b"".join(_field(_num8(s.limit), 8) for s in signals) + _field("1", 8)
    hdr += b"".join(_field(str(-DIGITAL), 8) for _ in signals) + _field("-32768", 8)
    hdr += b"".join(_field(str(DIGITAL), 8) for _ in signals) + _field("32767", 8)
    hdr += b"".join(_field(s.prefiltering, 80) for s in signals) + _field("", 80)
    hdr += b"".join(_field(str(spr), 8) for _ in signals) + _field(str(ann_samples), 8)
    hdr += b"".join(_field("", 32) for _ in range(ns))
    assert len(hdr) == 256 * (ns + 1)

    gains = np.array([DIGITAL / s.limit for s in signals], dtype=np.float64)[:, None, None]
    sig_bytes = n_ch * spr * 2
    rec_bytes = sig_bytes + ann_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        for r0 in range(0, n_rec, chunk_records):
            r1 = min(r0 + chunk_records, n_rec)
            k = r1 - r0
            x = np.asarray(data[:, r0 * spr:r1 * spr], dtype=np.float64).reshape(n_ch, k, spr)
            dig = np.clip(np.rint(np.nan_to_num(x) * gains), -DIGITAL, DIGITAL).astype("<i2")
            block = np.zeros((k, rec_bytes), dtype=np.uint8)
            block[:, :sig_bytes] = dig.transpose(1, 0, 2).reshape(k, -1).view(np.uint8)
            for j in range(k):
                rec = r0 + j
                ann = _tal(rec * record_s, None, "") + b"".join(per_record[rec])
                block[j, sig_bytes:sig_bytes + len(ann)] = np.frombuffer(ann, dtype=np.uint8)
            fh.write(block.tobytes())
            if progress is not None:
                progress(r1, n_rec)
    return n_rec


def verify(path: Path, checks: list[tuple[int, int, np.ndarray]], names: list[str], steps: list[float],
           n_annotations: int | None = None) -> dict:
    """
    Read the EDF back with MNE and compare windows of it with the values that were written.
    `checks`: (channel index, start sample, expected µV after clipping). Returns the largest
    error in µV and as a fraction of that channel's quantisation step (≤ 0.5 when correct).
    """
    import warnings

    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    if raw.ch_names[: len(names)] != [n[:16] for n in names]:
        raise ValueError("channel names read back differ from those written")
    worst_uv, worst_frac, worst_ch = 0.0, 0.0, ""
    by_window: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for ch, start, exp in checks:
        by_window.setdefault((start, start + len(exp)), []).append((ch, exp))
    for (start, stop), items in by_window.items():
        picks = [c for c, _ in items]
        got = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
        for (ch, exp), g in zip(items, got):
            err = float(np.max(np.abs(g - exp))) if len(exp) else 0.0
            frac = err / steps[ch] if steps[ch] > 0 else 0.0
            if frac > worst_frac:
                worst_uv, worst_frac, worst_ch = err, frac, names[ch]
    out = {"max_abs_error_uv": worst_uv, "max_error_in_steps": worst_frac, "worst_channel": worst_ch,
           "windows_checked": len(by_window), "channel_windows_checked": len(checks)}
    if n_annotations is not None:
        out["annotations_written"] = n_annotations
        out["annotations_read"] = len(raw.annotations)
    raw.close()
    return out
