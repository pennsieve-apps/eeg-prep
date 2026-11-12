"""
Signal processing on windows of a recording: zero-phase IIR filters (notch at the line frequency
and its harmonics, high-pass, optional low-pass) applied to padded windows so the result matches
filtering the whole recording, optional resampling, 50/60 Hz detection, and the robust
per-channel metrics used for bad channels of bipolar recordings.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal

NOTCH_Q = 30.0          # notch quality factor: -3 dB width = f0 / Q (1.7 Hz at 50 Hz)
HIGHPASS_ORDER = 4      # Butterworth; applied forwards and backwards (zero phase, order 8 effective)
LOWPASS_ORDER = 4


@dataclass
class FilterPlan:
    sfreq: float
    line_freq: float
    notch: list[float]
    highpass: float
    lowpass: float

    def describe(self) -> dict:
        return {
            "notch_hz": self.notch, "notch": f"IIR notch (Q={NOTCH_Q:g}), zero-phase" if self.notch else "none",
            "highpass_hz": self.highpass or None,
            "highpass": f"Butterworth order {HIGHPASS_ORDER}, zero-phase (sosfiltfilt)" if self.highpass else "none",
            "lowpass_hz": self.lowpass or None,
            "lowpass": f"Butterworth order {LOWPASS_ORDER}, zero-phase (sosfiltfilt)" if self.lowpass else "none",
        }

    def pad_s(self) -> float:
        """Seconds of context needed on each side of a window so its filtered interior is exact."""
        p = 2.0
        if self.highpass:
            p = max(p, 8.0 / self.highpass)       # 16 s at 0.5 Hz: the order-8 response has died out
        if self.notch:
            p = max(p, 3.0)
        return p


def plan_filters(sfreq: float, line_freq: float, highpass: float, lowpass: float) -> FilterPlan:
    nyq = sfreq / 2.0
    notch = []
    if line_freq > 0:
        k = 1
        while k * line_freq < nyq - 1.0:   # harmonics strictly below Nyquist (with 1 Hz room)
            notch.append(float(k * line_freq))
            k += 1
    hp = highpass if 0 < highpass < nyq else 0.0
    lp = lowpass if 0 < lowpass < nyq * 0.98 else 0.0
    return FilterPlan(sfreq, line_freq, notch, hp, lp)


class Filter:
    """Applies a FilterPlan along the last axis of a (channels, samples) window."""

    def __init__(self, plan: FilterPlan):
        self.plan = plan
        self.notches = [signal.iirnotch(f, NOTCH_Q, fs=plan.sfreq) for f in plan.notch]
        self.hp = (signal.butter(HIGHPASS_ORDER, plan.highpass, btype="highpass", fs=plan.sfreq, output="sos")
                   if plan.highpass else None)
        self.lp = (signal.butter(LOWPASS_ORDER, plan.lowpass, btype="lowpass", fs=plan.sfreq, output="sos")
                   if plan.lowpass else None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        n = y.shape[-1]
        if n < 32:
            return y.copy()
        if self.hp is not None:
            y = y - np.median(y, axis=-1, keepdims=True)   # the high-pass removes DC anyway; smaller edge transients
        for b, a in self.notches:
            y = signal.filtfilt(b, a, y, axis=-1, padlen=min(n - 1, 3 * max(len(a), len(b))))
        if self.hp is not None:
            pad = int(min(n - 1, 3 * self.plan.sfreq / self.plan.highpass))
            y = signal.sosfiltfilt(self.hp, y, axis=-1, padlen=pad)
        if self.lp is not None:
            y = signal.sosfiltfilt(self.lp, y, axis=-1)
        return y


def bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    hi = min(hi, 0.98 * nyq)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


@dataclass
class Resampler:
    up: int
    down: int

    @staticmethod
    def make(sfreq: float, target: float) -> "Resampler | None":
        if not target or abs(target - sfreq) < 1e-9:
            return None
        fr = Fraction(target / sfreq).limit_denominator(1000)
        if abs(float(fr) * sfreq - target) > 1e-6:
            raise ValueError(f"cannot resample {sfreq:g} Hz to {target:g} Hz by a rational factor")
        return Resampler(fr.numerator, fr.denominator)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return signal.resample_poly(x, self.up, self.down, axis=-1)


def detect_line_frequency(x: np.ndarray, sfreq: float) -> tuple[float, str]:
    """50 or 60 Hz from the spectrum of a (channels, samples) segment, or 0 when neither stands out."""
    if sfreq < 130:
        return 0.0, "sampling rate too low to see 50/60 Hz"
    nper = int(min(x.shape[-1], 4 * sfreq))
    f, p = signal.welch(x, fs=sfreq, nperseg=nper, axis=-1)
    p = np.median(p, axis=0)
    ratios = {}
    for lf in (50.0, 60.0):
        peak = p[np.abs(f - lf) <= 0.5].max()
        flank = np.median(p[(np.abs(f - lf) >= 2) & (np.abs(f - lf) <= 5)])
        ratios[lf] = peak / flank if flank > 0 else 0.0
    best = max(ratios, key=ratios.get)
    if ratios[best] >= 3.0 and ratios[best] >= 2.0 * ratios[110.0 - best]:
        return best, f"detected from the spectrum ({best:g} Hz peak {ratios[best]:.1f}x its flanks)"
    return 0.0, (f"no clear 50/60 Hz peak in the spectrum (50 Hz {ratios[50.0]:.1f}x, 60 Hz {ratios[60.0]:.1f}x "
                 "the flanks); no notch")


@dataclass
class Metrics:
    sd_raw: float               # µV, of the detrended unfiltered signal
    zero_diff_fraction: float   # fraction of consecutive samples that are identical
    line_length: float          # median |dx| per sample, µV, after a 1 Hz high-pass + notch
    amplitude: float            # robust standard deviation (1.4826 × MAD), µV, after the same filter
    line_noise_ratio: float     # power within ±1 Hz of the line frequency and harmonics / power 1 Hz – min(Nyquist, 100 Hz)


def metrics(x: np.ndarray, sfreq: float, line_freq: float, qcf: Filter) -> Metrics:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return Metrics(0.0, 1.0, 0.0, 0.0, 0.0)
    xd = signal.detrend(x, type="linear")
    sd_raw = float(np.std(xd))
    zdf = float(np.mean(np.diff(x) == 0))
    y = qcf(x)
    ll = float(np.median(np.abs(np.diff(y))))
    amp = float(1.4826 * np.median(np.abs(y - np.median(y))))
    lnr = 0.0
    if line_freq > 0 and sd_raw > 0:
        nper = int(min(n, max(256, 2 ** int(np.ceil(np.log2(sfreq * 2))))))
        f, p = signal.welch(xd, fs=sfreq, nperseg=nper)
        top = min(sfreq / 2.0 - 1.0, 100.0)
        band = (f >= 1.0) & (f <= top)
        total = float(np.sum(p[band]))
        line = np.zeros_like(f, dtype=bool)
        k = 1
        while k * line_freq <= top:
            line |= np.abs(f - k * line_freq) <= 1.0
            k += 1
        lnr = float(np.sum(p[band & line]) / total) if total > 0 else 0.0
    return Metrics(sd_raw, zdf, ll, amp, lnr)


MIN_LOG_SPREAD = 0.05   # smallest spread (decades) for the robust z of log10 line length / amplitude


def robust_z(values, min_spread: float = 0.0) -> np.ndarray:
    """(x - median) / max(1.4826 · MAD, min_spread); NaN in, NaN out."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    med = np.median(v[ok])
    mad = max(1.4826 * np.median(np.abs(v[ok] - med)), min_spread)
    if mad <= 0:
        return np.where(ok, 0.0, np.nan)
    return (v - med) / mad
