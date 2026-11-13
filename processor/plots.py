"""
QC figures: one per recording (spectra, channel status on the head, the removed ICA
components, a timeline of labels and bad segments, and 10 s of the output), and an overview
of all recordings.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GOOD = "#3a6ea5"
BAD = "#c0392b"
SEIZURE = "#e67e22"
BADSEG = "#7f8c8d"


def _project(xyz: tuple[float, float, float]) -> tuple[float, float]:
    """Azimuthal equidistant projection of a head position (as MNE's topomaps), radius 1 at the equator."""
    x, y, z = xyz
    r = np.sqrt(x * x + y * y + z * z) or 1.0
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    rho = theta / (np.pi / 2) * 0.5
    return rho * np.cos(phi), rho * np.sin(phi)


def _head(ax) -> None:
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color="k", lw=1)
    ax.plot([-0.05, 0, 0.05], [0.49, 0.56, 0.49], color="k", lw=1)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")


def _channel_map(ax, p: dict) -> None:
    from processor.channels import position
    _head(ax)
    if p["layout"] == "referential":
        for n, st in zip(p["names"], p["status"]):
            xyz = p["positions"].get(n)
            if not xyz:
                continue
            u, v = _project(xyz)
            ax.scatter([u], [v], s=110, color=GOOD if st == "good" else BAD, zorder=3)
            ax.text(u, v, n, fontsize=6, ha="center", va="center", color="white", zorder=4)
    else:
        for n, st in zip(p["names"], p["status"]):
            a, b = p["contacts"][n]
            pa, pb = position(a), position(b)
            if not (pa and pb):
                continue
            (u1, v1), (u2, v2) = _project(pa), _project(pb)
            ax.plot([u1, u2], [v1, v2], color=GOOD if st == "good" else BAD, lw=2.5 if st == "bad" else 1.5)
            for e, (u, v) in ((a, (u1, v1)), (b, (u2, v2))):
                ax.scatter([u], [v], s=60, color="white", edgecolor="k", zorder=3)
                ax.text(u, v, e, fontsize=5, ha="center", va="center", zorder=4)
    nbad = sum(1 for s in p["status"] if s == "bad")
    ax.set_title(f"Channels: {len(p['names']) - nbad} good, {nbad} bad (red)", fontsize=9)


def qc_figure(p: dict, path: Path) -> None:
    ica = p.get("ica")
    removed = list(ica.removed) if ica is not None else []
    n_topo = min(len(removed), 8)
    fig = plt.figure(figsize=(14, 11), dpi=150)
    gs = fig.add_gridspec(4, 8, height_ratios=[3.2, 1.6 if n_topo else 0.5, 0.9, 4.0], hspace=0.6, wspace=0.4)
    h = p["headline"]
    fig.suptitle(f"{p['recording_id']} - eeg-prep QC ({p['montage']} montage, {p['sfreq']:g} Hz, "
                 f"{p['duration_s'] / 60:.1f} min, {p['kind']})", fontsize=11, fontweight="bold")

    # spectra
    ax = fig.add_subplot(gs[0, :5])
    f, psd = p["psd_f"], p["psd"]
    top = min(f[-1], 70.0) if p["sfreq"] < 300 else 100.0
    sel = (f >= 0.5) & (f <= top)
    with np.errstate(divide="ignore"):
        lp = 10 * np.log10(np.maximum(psd[:, sel], 1e-12))
    for j, st in enumerate(p["status"]):
        if st == "good":
            ax.plot(f[sel], lp[j], color=GOOD, lw=0.6, alpha=0.6)
    for j, st in enumerate(p["status"]):
        if st == "bad":
            ax.plot(f[sel], lp[j], color=BAD, lw=1.0, label=p["names"][j])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re 1 µV²/Hz)")
    ax.set_title(f"Spectrum of every channel after preprocessing ({p['psd_epochs']} clean 4-s epochs)", fontsize=9)
    if any(st == "bad" for st in p["status"]):
        ax.legend(fontsize=6, ncol=2, loc="upper right", title="bad channels", title_fontsize=6)
    ax.grid(alpha=0.3)

    _channel_map(fig.add_subplot(gs[0, 5:]), p)

    # removed components
    if n_topo:
        import mne
        comps = ica.topographies
        for k in range(n_topo):
            ic = removed[k]
            axk = fig.add_subplot(gs[1, k])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mne.viz.plot_topomap(comps[:, ic], ica.info, axes=axk, show=False, contours=0, sensors=False)
            axk.set_title(f"IC{ic:02d}\n{ica.labels[ic]} {ica.probabilities[ic].max():.2f}", fontsize=7)
    else:
        axn = fig.add_subplot(gs[1, :])
        axn.axis("off")
        txt = ("ICA: no component removed" if ica is not None else
               "ICA/ICLabel not run (bipolar-only recording, too few channels with positions, or ARTIFACTS setting)")
        axn.text(0.0, 0.5, txt, fontsize=8, va="center")

    # timeline
    axt = fig.add_subplot(gs[2, :])
    dur = p["duration_s"]
    for s0, s1, _ in p["bad_segments"]:
        axt.axvspan(s0 / 60, s1 / 60, ymin=0.0, ymax=0.45, color=BADSEG)
    for on, du, tt in p["events"]:
        if tt == "seizure":
            axt.axvspan(on / 60, (on + (du or 10.0)) / 60, ymin=0.55, ymax=1.0, color=SEIZURE)
        else:
            axt.axvline(on / 60, ymin=0.55, ymax=1.0, color="#8e44ad", lw=0.8)
    axt.set_xlim(0, dur / 60)
    axt.set_yticks([0.25, 0.75])
    axt.set_yticklabels(["bad segments", "labels"], fontsize=7)
    axt.set_xlabel("Time (min)")
    axt.set_title(f"Timeline: seizures (orange), other labels (purple), bad segments (grey, "
                  f"{h['bad_segment_pct']:.2f}% of the recording)", fontsize=9)

    # snippet
    axs = fig.add_subplot(gs[3, :])
    snip = p["snippet"]
    n = snip.shape[0]
    t = p["snippet_start_s"] + np.arange(snip.shape[1]) / p["sfreq"]
    good_rows = [j for j, st in enumerate(p["status"]) if st == "good"]
    scale = np.median([np.percentile(np.abs(snip[j]), 99) for j in good_rows]) if good_rows else 50.0
    scale = max(scale, 1.0) * 1.5
    for j in range(n):
        axs.plot(t, snip[j] / scale - j, color=GOOD if p["status"][j] == "good" else BAD, lw=0.5)
    axs.set_yticks(-np.arange(n))
    axs.set_yticklabels(p["names"], fontsize=6)
    axs.set_ylim(-n, 1)
    axs.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1)
    axs.set_xlabel("Time (s)")
    axs.set_title(f"10 s of the output ({p['reference']} reference; one trace step = {scale:.0f} µV)", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def overview_figure(rows: list[dict], path: Path) -> None:
    """One row per recording: channels good/bad, ICA components removed, % bad segments."""
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, max(2.5, 0.28 * n + 1.5)), dpi=150, sharey=True)
    y = np.arange(n)
    ids = [r["recording_id"] for r in rows]
    good = [r["n_good"] for r in rows]
    bad = [r["n_bad_channels"] for r in rows]
    axes[0].barh(y, good, color=GOOD, label="good")
    axes[0].barh(y, bad, left=good, color=BAD, label="bad")
    axes[0].set_xlabel("Channels")
    axes[0].legend(fontsize=7)
    axes[1].barh(y, [r["n_ica_removed"] for r in rows], color="#16a085")
    axes[1].set_xlabel("ICA components removed")
    axes[2].barh(y, [r["bad_segment_pct"] for r in rows], color=BADSEG)
    axes[2].set_xlabel("Bad segments (% of recording)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ids, fontsize=6)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("eeg-prep: data quality per recording", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
