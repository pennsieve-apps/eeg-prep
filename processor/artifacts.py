"""
Artifact handling.

ICA + ICLabel: an ICA (MNE; extended Infomax by default, the decomposition ICLabel was trained
on, or Picard) is fitted on a sample of the recording (chunks spread over it, seizures and
very high-amplitude chunks left out), band-passed 1-100 Hz and in average reference as ICLabel
expects; mne-icalabel's port of ICLabel (Pion-Tonachini et al. 2019; Li et al. 2022) labels
every component, and components labelled eye blink, muscle, heart, line noise or channel noise
with probability >= ICLABEL_THRESHOLD are removed. ICA cleaning is linear, so it is turned into
one matrix (x_clean = A x + b) and applied to every window of the recording.

ASR: python-meegkit's Artifact Subspace Reconstruction (Mullen et al. 2015; Chang et al. 2020),
calibrated on the same sample, applied online in 0.5 s chunks across the recording.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("eeg-prep")

ICLABEL_CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]
ARTIFACT_CLASSES = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}


def make_info(names: list[str], sfreq: float, positions: dict[str, tuple[float, float, float]]):
    import mne
    info = mne.create_info(names, sfreq, "eeg")
    mont = mne.channels.make_dig_montage(ch_pos={n: np.array(positions[n]) for n in names}, coord_frame="head")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(mont)
    return info


@dataclass
class IcaResult:
    names: list[str]                     # channels the ICA works on (good, with positions)
    n_components: int
    method: str
    labels: list[str]                    # ICLabel class of each component (argmax)
    probabilities: np.ndarray            # (n_components, 7)
    removed: list[int]
    operator: np.ndarray                 # A: x_clean = A @ x + b, over `names`
    offset: np.ndarray                   # b
    topographies: np.ndarray             # (n_channels, n_components) mixing matrix columns
    info: object                         # mne.Info of `names` (for topomaps)
    fit_seconds: float
    explained_removed: float = 0.0       # fraction of the fit data's variance removed
    notes: list[str] = field(default_factory=list)


def fit_ica(data: np.ndarray, names: list[str], sfreq: float, positions: dict, method: str, seed: int,
            threshold: float, max_components: int, threads: int) -> IcaResult:
    """data: (channels, samples) µV, 1-100 Hz band-passed, in average reference."""
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel.iclabel import iclabel_label_components

    info = make_info(names, sfreq, positions)
    raw = mne.io.RawArray(data * 1e-6, info, verbose="error")
    with raw.info._unlock():
        raw.info["highpass"] = 1.0
        raw.info["lowpass"] = min(100.0, sfreq / 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_eeg_reference("average", projection=False, verbose="error")
    n_comp = len(names) - 1                              # average reference removes one dimension
    if max_components > 0:
        n_comp = min(n_comp, max_components)
    if method == "picard":
        ica = ICA(n_components=n_comp, method="picard", fit_params={"ortho": False, "extended": True},
                  random_state=seed, max_iter="auto", verbose="error")
    else:
        ica = ICA(n_components=n_comp, method="infomax", fit_params={"extended": True}, random_state=seed,
                  max_iter="auto", verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose="error")
        probs = iclabel_label_components(raw, ica, inplace=False, backend="onnx")
    labels = [ICLABEL_CLASSES[int(k)] for k in np.argmax(probs, axis=1)]
    removed = [k for k in range(probs.shape[0])
               if any(probs[k, ICLABEL_CLASSES.index(c)] >= threshold for c in ARTIFACT_CLASSES)]
    # the cleaning as one affine map, by applying it to the unit vectors
    n = len(names)
    probe = mne.io.RawArray(np.hstack([np.zeros((n, 1)), np.eye(n)]) * 1e-6, raw.info.copy(), verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.apply(probe, exclude=removed, verbose="error")
    out = probe.get_data() * 1e6
    b = out[:, 0].copy()
    A = out[:, 1:] - b[:, None]
    fit_data = raw.get_data() * 1e6
    cleaned = A @ fit_data + b[:, None]
    var = float(np.var(fit_data))
    frac = float(np.var(fit_data - cleaned) / var) if var > 0 else 0.0
    return IcaResult(names=names, n_components=n_comp, method="extended Infomax" if method != "picard" else
                     "Picard (extended, non-orthogonal)", labels=labels, probabilities=probs, removed=removed,
                     operator=A, offset=b, topographies=ica.get_components(), info=raw.info,
                     fit_seconds=data.shape[1] / sfreq, explained_removed=frac)


class AsrCleaner:
    """meegkit ASR, applied online in 0.5 s chunks; keeps its state between windows."""

    def __init__(self, sfreq: float, cutoff: float, calibration: np.ndarray):
        from meegkit.asr import ASR
        self.asr = ASR(sfreq=sfreq, cutoff=cutoff, method="euclid")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.asr.fit(calibration)
        self.chunk = max(int(round(0.5 * sfreq)), 1)
        self.changed = 0
        self.total = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for a in range(0, x.shape[1], self.chunk):
                b = min(a + self.chunk, x.shape[1])
                out[:, a:b] = self.asr.transform(x[:, a:b])
        diff = np.max(np.abs(out - x), axis=0)
        self.changed += int(np.count_nonzero(diff > 0.01))
        self.total += x.shape[1]
        return out
