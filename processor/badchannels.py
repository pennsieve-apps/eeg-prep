"""
Bad channels.

Referential recordings: pyprep's NoisyChannels (Bigdely-Shamlo et al. 2015; Appelhoff et al.,
PyPREP) on up to three QC segments spread over the recording, on notch-filtered data referenced
to the per-sample median of the channels (a robust first estimate of the reference, as PREP
does before computing its robust average reference): bad by NaN, flat, deviation, high-frequency
noise, correlation, SNR, dropout and RANSAC (when positions exist). A channel is bad when it is
bad in most segments. Channels without a template position are checked with the robust metrics
below.

Bipolar recordings (and channels without positions): flat, line noise, and robust z of
log line length / amplitude across channels, as in processor-ieeg-prep.
"""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from processor import dsp

log = logging.getLogger("eeg-prep")

# pyprep's default flags a channel whose maximal correlation with any other channel is below 0.4 in
# more than 1% of 1-s windows: made for short lab recordings; in clinical EEG transient artifacts on
# single channels are common (and handled as bad segments), so a channel must fail in 20% of windows.
CORRELATION_FRAC_BAD = 0.2

# Low correlation with the other channels and poor RANSAC prediction also flag channels that carry focal
# activity (temporal muscle, frequent spikes, a focal slowing): on scalp EEG of epilepsy that is signal,
# not a broken electrode. Alone they make a channel bad only when they fail in every QC segment (>= 2).
WEAK_CRITERIA = {"bad_by_correlation", "bad_by_ransac"}

PYPREP_REASONS = {
    "bad_by_nan": "NaN values", "bad_by_flat": "flat", "bad_by_deviation": "deviation (robust amplitude z > 5)",
    "bad_by_hf_noise": "high-frequency noise (z > 5)", "bad_by_correlation": "low correlation with other channels (in > 20% of 1-s windows)",
    "bad_by_SNR": "low SNR (HF noise and low correlation)", "bad_by_dropout": "dropouts (intermittently flat)",
    "bad_by_ransac": "poorly predicted by neighbours (RANSAC)",
}


@dataclass
class BadResult:
    reasons: dict[str, list[str]] = field(default_factory=dict)    # channel -> reasons (empty = good)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    suspect: dict[str, list[str]] = field(default_factory=dict)  # flagged, not counted
    method: str = ""
    notes: list[str] = field(default_factory=list)


def pyprep_bads(segments: list[np.ndarray], names: list[str], sfreq: float, seed: int, ransac: bool,
                montage_info) -> BadResult:
    """
    segments: (channels, samples) µV arrays, already notch-filtered and median-referenced.
    montage_info: an mne.Info with the channels' positions (names in the same order).
    """
    import mne
    from pyprep.find_noisy_channels import NoisyChannels

    res = BadResult(method="pyprep NoisyChannels" + (" + RANSAC" if ransac else ""))
    tech: dict[str, int] = defaultdict(int)
    weak: dict[str, int] = defaultdict(int)
    why: dict[str, set[str]] = defaultdict(set)
    used = 0
    for k, seg in enumerate(segments):
        raw = mne.io.RawArray(seg * 1e-6, montage_info.copy(), verbose="error")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nd = NoisyChannels(raw, do_detrend=True, random_state=seed + k, matlab_strict=False, ransac=ransac)
            nd.find_bad_by_deviation()
            nd.find_bad_by_hfnoise()
            nd.find_bad_by_correlation(frac_bad=CORRELATION_FRAC_BAD)
            nd.find_bad_by_SNR()
            if ransac:
                try:
                    nd.find_bad_by_ransac(channel_wise=False)
                except Exception as exc:  # noqa: BLE001 - RANSAC refuses too few good channels
                    res.notes.append(f"RANSAC skipped on QC segment {k + 1} ({type(exc).__name__}: {exc})")
        bads = nd.get_bads(as_dict=True)
        used += 1
        t_set, w_set = set(), set()
        for key, chans in bads.items():
            if key not in PYPREP_REASONS:
                continue
            for ch in chans:
                why[ch].add(PYPREP_REASONS[key])
                (w_set if key in WEAK_CRITERIA else t_set).add(ch)
        for ch in t_set:
            tech[ch] += 1
        for ch in w_set - t_set:
            weak[ch] += 1
    need = used // 2 + 1
    for n in names:
        n_any = tech[n] + weak[n]
        if tech[n] >= need or (used >= 2 and weak[n] == used):
            res.reasons[n] = sorted(why[n]) + ([f"in {n_any} of {used} QC segments"] if used > 1 else [])
        else:
            res.reasons[n] = []
            if n_any:
                res.metrics.setdefault(n, {})["pyprep_segments_flagged"] = n_any
                res.suspect[n] = sorted(why[n]) + [f"in {n_any} of {used} QC segments; not counted as bad"]
    return res


def simple_bads(segment: np.ndarray, names: list[str], sfreq: float, line_freq: float, flat_uv: float,
                bad_z: float, line_noise_max: float, against_median: bool) -> BadResult:
    """Flat, line noise and robust-z outliers of line length / amplitude (bipolar or unpositioned channels)."""
    res = BadResult(method="robust z of line length and amplitude, flat, line noise")
    qcf = dsp.Filter(dsp.plan_filters(sfreq, line_freq, 1.0 if sfreq > 20 else 0.0, 0.0))
    raw_m = [dsp.metrics(segment[j], sfreq, line_freq, qcf) for j in range(len(names))]
    mets = raw_m
    live = [j for j, m in enumerate(raw_m) if m.sd_raw >= flat_uv and m.zero_diff_fraction <= 0.5]
    if against_median and len(live) >= 3:
        med = np.median(segment[live], axis=0)
        mets = []
        for j in range(len(names)):
            m = dsp.metrics(segment[j] - med, sfreq, line_freq, qcf)
            m.sd_raw, m.zero_diff_fraction, m.line_noise_ratio = (raw_m[j].sd_raw, raw_m[j].zero_diff_fraction,
                                                                   raw_m[j].line_noise_ratio)
            mets.append(m)
    flat = np.array([m.sd_raw < flat_uv or m.zero_diff_fraction > 0.5 for m in mets])
    idx = np.array([j for j in range(len(names)) if not flat[j]], dtype=int)
    ll_z = np.full(len(names), np.nan)
    amp_z = np.full(len(names), np.nan)
    if len(idx) >= 5:
        ll_z[idx] = dsp.robust_z(np.log10([max(mets[j].line_length, 1e-9) for j in idx]), dsp.MIN_LOG_SPREAD)
        amp_z[idx] = dsp.robust_z(np.log10([max(mets[j].amplitude, 1e-9) for j in idx]), dsp.MIN_LOG_SPREAD)
    for j, n in enumerate(names):
        m = mets[j]
        rs = []
        if flat[j]:
            rs.append(f"flat (sd {m.sd_raw:.2g} µV, {100 * m.zero_diff_fraction:.0f}% repeated samples)")
        if m.line_noise_ratio > line_noise_max:
            rs.append(f"line noise ({100 * m.line_noise_ratio:.0f}% of 1-100 Hz power at {line_freq:g} Hz and harmonics)")
        if np.isfinite(ll_z[j]) and ll_z[j] > bad_z:
            rs.append(f"high line length (robust z {ll_z[j]:.1f})")
        if np.isfinite(amp_z[j]) and amp_z[j] > bad_z:
            rs.append(f"high amplitude (robust z {amp_z[j]:.1f})")
        res.reasons[n] = rs
        res.metrics[n] = {"robust_sd_uv": m.amplitude, "line_noise_ratio": m.line_noise_ratio,
                          "line_length_z": float(ll_z[j]), "amplitude_z": float(amp_z[j])}
    return res
