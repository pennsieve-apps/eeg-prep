"""
A small synthetic input for eeg-prep that exercises the contract, written with numpy, scipy and
edfio only (so it also runs in the platform's review generator image, which has no MNE):

  <out>/eeg/bids/dataset_description.json, participants.tsv
  <out>/eeg/bids/sub-01/ses-01/eeg/sub-01_ses-01_task-routine_run-01_{eeg.edf,eeg.json,channels.tsv,events.tsv}
  <out>/eeg/chbx/chbx01_01.edf          a CHB-MIT style bipolar-only recording (loose EDF)
  <out>/eeg/chbx/chbx01-summary.txt     its seizure, CHB-MIT summary format
  <out>/eeg/labels.csv                  a recording-level label (lateralisation)

Referential recording (sub-01, 256 Hz, 180 s, 19 channels named as an exporter writes them,
'EEG Fp1-REF' with the old names T3/T4/T5/T6, plus an EKG and a Photic channel): spatially
correlated 1/f background with a common reference signal, a posterior 10 Hz alpha rhythm,
eye blinks every ~6 s (frontal), left temporal spikes at SPIKE_TIMES (peak on F7/T7), a
left temporal 5 Hz seizure-like discharge SEIZURE (annotated in events.tsv as HED-SCORE
`sz_foc_ia` on T3,T5), a sleep-like segment SLEEP with 1 Hz slow waves and 13 Hz spindles,
50 Hz line noise, a noisy channel (C4, NOISY), and a movement artifact ARTIFACT (2000 µV on
the frontal channels).

Bipolar recording (chbx01_01, 256 Hz, 120 s): the 23 CHB-MIT channels (FP1-F7, ..., T8-P8
twice, P7-T7 the inverse of T7-P7) derived from the same kind of signals, a '-' dummy and an
ECG channel; a seizure in BIPOLAR_SEIZURE listed in the summary file.

Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from edfio import Edf, EdfSignal, Patient, Recording
from scipy import signal

SFREQ = 256.0
DURATION = 180.0
SPIKE_TIMES = [21.3, 27.8, 33.1, 41.6, 48.2, 55.9, 70.4, 77.7, 84.1, 91.5]
SPIKE_CHANNELS = {"F7": 1.0, "T7": 0.8, "Fp1": 0.35, "P7": 0.35, "F3": 0.2}
SPIKE_UV = -180.0
SEIZURE = (100.0, 130.0)
SLEEP = (140.0, 175.0)
ARTIFACT = (62.0, 62.6)
NOISY = "C4"
BIPOLAR_DURATION = 120.0
BIPOLAR_SEIZURE = (60.0, 80.0)
SEED = 7

# Approximate 2-D scalp positions (x right, y front; unit radius at the 10-20 equator).
POS2D = {
    "Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95), "F7": (-0.81, 0.59), "F3": (-0.41, 0.52), "Fz": (0.0, 0.5),
    "F4": (0.41, 0.52), "F8": (0.81, 0.59), "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0), "T8": (1.0, 0.0), "P7": (-0.81, -0.59), "P3": (-0.41, -0.52), "Pz": (0.0, -0.5),
    "P4": (0.41, -0.52), "P8": (0.81, -0.59), "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
    "FT9": (-1.05, 0.25), "FT10": (1.05, 0.25),
}
ELECTRODES = [e for e in POS2D if e not in ("FT9", "FT10")]
OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
CHBMIT = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4",
          "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
          "T8-P8"]


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SFREQ)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std()


def spike_wave(fs: float) -> np.ndarray:
    """A sharp 70 ms spike followed by a 200 ms slow wave, peak 1 (negative-going is applied by the caller)."""
    t = np.arange(int(0.4 * fs)) / fs
    spike = np.exp(-0.5 * ((t - 0.05) / 0.012) ** 2)
    wave = -0.35 * np.exp(-0.5 * ((t - 0.17) / 0.05) ** 2)
    return spike + wave


def make_signals(duration: float, seed: int, seizure: tuple[float, float] | None, spikes: list[float],
                 sleep: tuple[float, float] | None, blinks: bool = True) -> dict[str, np.ndarray]:
    """Electrode potentials (µV, against an ideal reference) for every electrode in POS2D."""
    rng = np.random.default_rng(seed)
    n = int(duration * SFREQ)
    t = np.arange(n) / SFREQ
    names = list(POS2D)
    xy = np.array([POS2D[e] for e in names])
    # background: 8 latent 1/f sources with smooth random topographies, plus sensor noise
    out = {e: 4.0 * rng.standard_normal(n) for e in names}
    for _ in range(8):
        c = rng.uniform(-0.9, 0.9, size=2)
        w = np.exp(-np.sum((xy - c) ** 2, axis=1) / (2 * 0.6 ** 2))
        s = 12.0 * pink(n, rng)
        for k, e in enumerate(names):
            out[e] += w[k] * s
    # posterior alpha, waxing and waning, reduced during sleep
    env = 0.6 + 0.4 * np.sin(2 * np.pi * t / 9.0)
    if sleep:
        env[(t >= sleep[0]) & (t < sleep[1])] *= 0.15
    alpha = 30.0 * env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    for e, g in (("O1", 1.0), ("O2", 1.0), ("P3", 0.5), ("P4", 0.5), ("Pz", 0.6), ("P7", 0.4), ("P8", 0.4)):
        out[e] += g * alpha
    # eye blinks: frontal, falling off towards the back
    if blinks:
        bt = np.arange(3.0, duration - 1, 6.0) + rng.uniform(-1, 1, size=len(np.arange(3.0, duration - 1, 6.0)))
        blink = np.zeros(n)
        shape = np.hanning(int(0.35 * SFREQ))
        for b in bt:
            i = int(b * SFREQ)
            blink[i:i + len(shape)] += 160.0 * shape[: n - i]
        for k, e in enumerate(names):
            y = xy[k, 1]
            out[e] += blink * max(0.0, (y - 0.1) / 0.85) ** 2
    # spikes
    sw = spike_wave(SFREQ)
    for st in spikes:
        i = int(st * SFREQ) - int(0.05 * SFREQ)
        for e, g in SPIKE_CHANNELS.items():
            out[e][i:i + len(sw)] += SPIKE_UV * g * sw
    # seizure: 5 Hz rhythmic sharp waves on the left temporal chain, growing and spreading
    if seizure:
        a, b = int(seizure[0] * SFREQ), int(seizure[1] * SFREQ)
        ts = t[a:b] - seizure[0]
        ramp = np.minimum(1.0, ts / 5.0)
        freq = 6.0 - 2.0 * ts / (seizure[1] - seizure[0])
        phase = 2 * np.pi * np.cumsum(freq) / SFREQ
        wave = np.sign(np.sin(phase)) * np.abs(np.sin(phase)) ** 3
        for e, g in (("F7", 1.0), ("T7", 1.0), ("P7", 0.8), ("Fp1", 0.5), ("F3", 0.4), ("C3", 0.4), ("O1", 0.3)):
            out[e][a:b] += 140.0 * g * ramp * wave
    # sleep: 1 Hz slow waves (frontal-central) and 13 Hz spindles (central)
    if sleep:
        a, b = int(sleep[0] * SFREQ), int(sleep[1] * SFREQ)
        ts = t[a:b]
        sw_ = 90.0 * np.sin(2 * np.pi * 0.9 * ts)
        for e in ("Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz", "F7", "F8"):
            out[e][a:b] += sw_ * (1.0 if e in ("Fz", "F3", "F4", "Cz") else 0.6)
        for c0 in np.arange(sleep[0] + 2, sleep[1] - 2, 6.0):
            i = int(c0 * SFREQ)
            m = int(1.0 * SFREQ)
            sp = 25.0 * np.hanning(m) * np.sin(2 * np.pi * 13.0 * np.arange(m) / SFREQ)
            for e in ("C3", "C4", "Cz"):
                out[e][i:i + m] += sp
    return out


def _sig(x: np.ndarray, label: str, lim: float = 3000.0) -> EdfSignal:
    return EdfSignal(np.clip(x, -lim, lim), SFREQ, label=label, physical_dimension="uV",
                     physical_range=(-lim, lim), digital_range=(-32768, 32767))


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_referential(root: Path) -> None:
    rng = np.random.default_rng(SEED + 1)
    sig = make_signals(DURATION, SEED, SEIZURE, SPIKE_TIMES, SLEEP)
    n = int(DURATION * SFREQ)
    t = np.arange(n) / SFREQ
    common = 15.0 * pink(n, rng)            # the recording reference's own activity
    line = 20.0 * np.sin(2 * np.pi * 50.0 * t)
    labels, data = [], []
    for e in ELECTRODES:
        x = sig[e] - common + line
        if e == NOISY:
            x = x + 70.0 * rng.standard_normal(n)
        if e in ("Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"):
            a, b = int(ARTIFACT[0] * SFREQ), int(ARTIFACT[1] * SFREQ)
            x[a:b] += 2000.0 * np.hanning(b - a)
        labels.append(f"EEG {OLD.get(e, e)}-REF")
        data.append(x)
    ekg = np.zeros(n)
    for bt in np.arange(0.4, DURATION, 0.85):
        i = int(bt * SFREQ)
        ekg[i:i + 10] += 800.0 * np.hanning(10)[: n - i]
    photic = np.zeros(n)
    labels += ["EKG", "Photic"]
    data += [ekg, photic]
    d = root / "bids"
    eeg = d / "sub-01" / "ses-01" / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)
    stem = "sub-01_ses-01_task-routine_run-01"
    start = dt.datetime(2020, 3, 1, 10, 0, 0)
    Edf([_sig(x, lb) for x, lb in zip(data, labels)], patient=Patient(), recording=Recording(startdate=start.date()),
        starttime=start.time()).write(eeg / f"{stem}_eeg.edf")
    with open(d / "dataset_description.json", "w") as fh:
        json.dump({"Name": "eeg-prep synthetic fixture", "BIDSVersion": "1.8.0", "DatasetType": "raw",
                   "License": "CC0"}, fh, indent=2)
    write_tsv(d / "participants.tsv", ["participant_id", "age", "sex"], [["sub-01", "34", "f"]])
    with open(eeg / f"{stem}_eeg.json", "w") as fh:
        json.dump({"TaskName": "routine", "SamplingFrequency": SFREQ, "PowerLineFrequency": 50,
                   "EEGReference": "Ref", "SoftwareFilters": "n/a", "EEGChannelCount": 19, "ECGChannelCount": 1,
                   "TriggerChannelCount": 1, "RecordingDuration": DURATION, "RecordingType": "continuous",
                   "EEGPlacementScheme": "10-20"}, fh, indent=2)
    write_tsv(eeg / f"{stem}_channels.tsv", ["name", "type", "units", "status"],
              [[lb, "EEG", "uV", "good"] for lb in labels[:-2]] + [["EKG", "ECG", "uV", "good"],
                                                                    ["Photic", "TRIG", "n/a", "good"]])
    write_tsv(eeg / f"{stem}_events.tsv",
              ["onset", "duration", "eventType", "confidence", "channels", "dateTime", "recordingDuration"],
              [["0.00", f"{DURATION:.2f}", "bckg", "n/a", "n/a", "2020-03-01 10:00:00", f"{DURATION:.2f}"],
               [f"{SEIZURE[0]:.2f}", f"{SEIZURE[1] - SEIZURE[0]:.2f}", "sz_foc_ia", "n/a", "T3,T5",
                "2020-03-01 10:00:00", f"{DURATION:.2f}"]])


def write_bipolar(root: Path) -> None:
    rng = np.random.default_rng(SEED + 2)
    sig = make_signals(BIPOLAR_DURATION, SEED + 3, BIPOLAR_SEIZURE, [], None, blinks=True)
    n = int(BIPOLAR_DURATION * SFREQ)
    t = np.arange(n) / SFREQ
    line = 15.0 * np.sin(2 * np.pi * 60.0 * t)
    key = {e.upper(): e for e in POS2D}
    sigs = []
    for name in CHBMIT:
        a, b = name.split("-")
        sigs.append(_sig(sig[key[a]] - sig[key[b]] + line, name))
    sigs.append(_sig(np.zeros(n), "-"))
    sigs.append(_sig(300.0 * np.sin(2 * np.pi * 1.2 * t) ** 21 + 5 * rng.standard_normal(n), "ECG"))
    d = root / "chbx"
    d.mkdir(parents=True, exist_ok=True)
    start = dt.datetime(2076, 11, 6, 11, 42, 54)
    Edf(sigs, patient=Patient(), recording=Recording(startdate=start.date()), starttime=start.time()).write(
        d / "chbx01_01.edf")
    with open(d / "chbx01-summary.txt", "w") as fh:
        fh.write("Data Sampling Rate: 256 Hz\n*************************\n\nChannels in EDF Files:\n"
                 "**********************\n")
        for k, name in enumerate(CHBMIT, start=1):
            fh.write(f"Channel {k}: {name}\n")
        fh.write("\nFile Name: chbx01_01.edf\nFile Start Time: 11:42:54\nFile End Time: 11:44:54\n"
                 f"Number of Seizures in File: 1\nSeizure Start Time: {BIPOLAR_SEIZURE[0]:.0f} seconds\n"
                 f"Seizure End Time: {BIPOLAR_SEIZURE[1]:.0f} seconds\n")


def main(out: Path) -> None:
    root = out / "eeg"
    root.mkdir(parents=True, exist_ok=True)
    write_referential(root)
    write_bipolar(root)
    with open(root / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["subject", "recording_id", "onset_s", "duration_s", "trial_type", "lateralisation"])
        w.writerow(["01", "", "", "", "", "left"])
        w.writerow(["01", "sub-01_ses-01_task-routine_run-01", "150", "", "lights_off", ""])
    print(f"wrote {root}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(Path(sys.argv[1]))
