"""
Bad segments of the output: stretches a later step should not analyse because the recording is
technically unreadable there, not because the brain activity is large. Found window by window on
the good channels:

- amplitude (technical): |x| above AMPLITUDE_UV on at least AMPLITUDE_CHANNELS (a fraction) of the
  good channels at the same time (movement, a global artifact), or above EXTREME_UV on any good
  channel (amplifier saturation, an electrode pop), padded by 0.5 s. A spike, a sharp wave or a
  focal seizure discharge on a few channels is not a bad segment;
- flat: MNE's `annotate_amplitude` with `flat` (every sample-to-sample change of a good
  channel below FLAT_UV for at least FLAT_S seconds: a disconnected or saturated amplifier);
- muscle: MNE's `annotate_muscle_zscore` (band-passed 70+ Hz envelope, z-scored per channel within
  the window, summed over channels; above MUSCLE_Z), when the sampling rate leaves room for it.

Annotated seizures are never bad segments (prep.py cuts them out: they are events of their own).
Intervals closer than 1 s are merged.
"""
from __future__ import annotations

import warnings

import numpy as np

PAD_S = 0.5
MERGE_GAP_S = 1.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def muscle_band(sfreq: float) -> tuple[float, float] | None:
    hi = min(140.0, 0.45 * sfreq)
    lo = 70.0
    return (lo, hi) if hi - lo >= 20.0 else None


AMPLITUDE = "amplitude"
EXTREME = "extreme amplitude"
FLAT = "flat"
MUSCLE = "muscle"


def find(x: np.ndarray, names: list[str], sfreq: float, t0: float, amplitude_uv: float, flat_uv: float,
         flat_s: float, muscle_z: float, amplitude_channels: float = 0.5,
         extreme_uv: float = 0.0) -> list[tuple[float, float, str]]:
    """Bad intervals (onset_s, end_s, reason) in a window that starts at t0 seconds."""
    import mne
    out: list[tuple[float, float, str]] = []
    if x.shape[0] == 0 or x.shape[1] < 2:
        return out
    ax = np.abs(x)
    if amplitude_uv > 0:
        need = max(1, int(np.ceil(amplitude_channels * x.shape[0])))
        mask = np.count_nonzero(ax > amplitude_uv, axis=0) >= need
        for a, b in _runs(mask):
            out.append((t0 + a / sfreq - PAD_S, t0 + b / sfreq + PAD_S, AMPLITUDE))
    if extreme_uv > 0:
        for a, b in _runs(np.any(ax > extreme_uv, axis=0)):
            out.append((t0 + a / sfreq - PAD_S, t0 + b / sfreq + PAD_S, EXTREME))
    need_mne = (flat_uv > 0 and flat_s > 0) or (muscle_z > 0 and muscle_band(sfreq))
    if not need_mne:
        return out
    info = mne.create_info(names, sfreq, "eeg")
    raw = mne.io.RawArray(x * 1e-6, info, verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if flat_uv > 0 and flat_s > 0 and x.shape[1] > flat_s * sfreq:
            ann, _ = mne.preprocessing.annotate_amplitude(raw, flat=flat_uv * 1e-6, bad_percent=100.0,
                                                          min_duration=flat_s, verbose="error")
            for a in ann:
                out.append((t0 + a["onset"], t0 + a["onset"] + a["duration"], FLAT))
        band = muscle_band(sfreq)
        if muscle_z > 0 and band and x.shape[1] > 10 * sfreq:
            ann, _ = mne.preprocessing.annotate_muscle_zscore(raw, threshold=muscle_z, ch_type="eeg",
                                                              min_length_good=0.2, filter_freq=band,
                                                              verbose="error")
            for a in ann:
                out.append((t0 + a["onset"], t0 + a["onset"] + a["duration"], MUSCLE))
    return out


def merge(intervals: list[tuple[float, float, str]], duration: float) -> list[tuple[float, float, str]]:
    """Clip to the recording and merge intervals closer than MERGE_GAP_S (reasons joined)."""
    ivs = sorted((max(0.0, a), min(duration, b), r) for a, b, r in intervals if b > 0 and a < duration)
    out: list[list] = []
    for a, b, r in ivs:
        if out and a <= out[-1][1] + MERGE_GAP_S:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].add(r)
        else:
            out.append([a, b, {r}])
    return [(a, b, "; ".join(sorted(r))) for a, b, r in out if b > a]


def without(intervals: list[tuple[float, float, str]], keep_out: list[tuple[float, float]]) -> list[tuple[float, float, str]]:
    """The intervals with every `keep_out` interval (annotated seizures) cut out of them."""
    out = []
    for a, b, r in intervals:
        parts = [(a, b)]
        for s0, s1 in keep_out:
            parts = [p for x0, x1 in parts for p in ((x0, min(x1, s0)), (max(x0, s1), x1)) if p[1] > p[0]]
        out.extend((x0, x1, r) for x0, x1 in parts)
    return out


def seconds_by_reason(intervals: list[tuple[float, float, str]], duration: float) -> dict[str, float]:
    """Seconds covered by each criterion on its own (overlaps between criteria counted in each)."""
    out = {}
    for reason in sorted({r for _, _, r in intervals}):
        out[reason] = round(sum(b - a for a, b, _ in merge([i for i in intervals if i[2] == reason], duration)), 1)
    return out
