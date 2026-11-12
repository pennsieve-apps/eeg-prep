"""
One recording, start to finish: channel selection and canonical names, labels, bad channels,
ICA/ASR calibration, then one windowed pass (filter, resample, re-reference, ASR, ICA, bad
segments) into a disk-backed array, and the BIDS-EEG derivative files written from it.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from processor import artifacts as A
from processor import badchannels as BC
from processor import bids as B
from processor import channels as C
from processor import dsp
from processor import edfout as E
from processor import labels as L
from processor import segments as SG
from processor import sources as S

log = logging.getLogger("eeg-prep")

ICA_CHUNK_S = 10.0
SEIZURE_MARGIN_S = 30.0
SNIPPET_S = 10.0
VERIFY_S = 5.0
PSD_EPOCH_S = 4.0
SIMPLE_FLAT_UV = 0.5
LINE_NOISE_MAX = 0.5
PROGRESS_EVERY_S = 60.0


@dataclass
class Params:
    reference: str = "average"
    line_freq: str = "auto"
    highpass: float = 0.5
    lowpass: float = 0.0
    resample_hz: float = 0.0
    artifacts: str = "iclabel"
    iclabel_threshold: float = 0.8
    ica_method: str = "infomax"
    ica_max_s: float = 300.0
    ica_components: int = 0
    asr_cutoff: float = 20.0
    bad_channels: str = "pyprep"
    ransac: bool = True
    qc_segment_s: float = 300.0
    bad_z: float = 5.0
    amplitude_uv: float = 500.0
    flat_uv: float = 0.1
    flat_s: float = 5.0
    muscle_z: float = 5.0
    kind: str = "auto"
    window_s: float = 300.0
    seed: int = 42
    verify: bool = True


@dataclass
class Result:
    recording_row: dict[str, Any]
    qc_rows: list[dict[str, Any]]
    details: dict[str, Any]
    plot: dict[str, Any]
    headline: dict[str, Any]


class Progress:
    def __init__(self, rid: str, what: str):
        self.rid, self.what = rid, what
        self.t0 = self.last = time.monotonic()

    def __call__(self, done: float, total: float) -> None:
        now = time.monotonic()
        if now - self.last >= PROGRESS_EVERY_S:
            self.last = now
            log.info("%s: %s %.0f%% (%.0f s elapsed)", self.rid, self.what, 100.0 * done / max(total, 1e-9),
                     now - self.t0)


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def line_frequency(param: str, sidecar: dict, src: S.Source) -> tuple[float | None, str]:
    if param != "auto":
        return float(param), "LINE_FREQ parameter"
    f = B.num(sidecar.get("PowerLineFrequency"))
    if f and f > 0:
        return f, "PowerLineFrequency in the BIDS sidecar"
    if src.line_freq:
        return float(src.line_freq), "line frequency recorded in the file"
    return None, ""


def spread_starts(duration: float, length: float, n: int, avoid: list[tuple[float, float]],
                  edge: float = 1.0) -> list[float]:
    """n starts of `length`-second segments spread over the recording, moved off the `avoid` intervals when possible."""
    usable = duration - 2 * edge
    if usable <= length or n <= 0:
        return [max(0.0, (duration - length) / 2.0)] if duration >= length else [0.0]
    starts = []
    grid = np.linspace(edge, duration - edge - length, num=max(n * 8, 16))
    ok = [s for s in grid if not any(s < b + SEIZURE_MARGIN_S and s + length > a - SEIZURE_MARGIN_S
                                     for a, b in avoid)]
    pool = ok if len(ok) >= n else list(grid)
    for k in range(n):
        target = edge + (k + 0.5) * (usable - length) / n
        best = min(pool, key=lambda s: abs(s - target))
        starts.append(float(best))
        pool = [s for s in pool if abs(s - best) >= length] or pool
    return sorted(set(starts))


def _age(si: dict, meas_date) -> str:
    bd = si.get("birthday")
    if isinstance(bd, (dt.date, dt.datetime)) and isinstance(meas_date, dt.datetime):
        try:
            days = (meas_date.date() - (bd if isinstance(bd, dt.date) else bd.date())).days
            if 0 < days < 130 * 365:
                return fmt(days / 365.25, 1)
        except (TypeError, ValueError):
            return ""
    return ""


def process(c: S.Candidate, params: Params, labelset: L.LabelSet, summaries: dict, participants: dict,
            out_root: Path, input_dir: Path) -> Result:
    t0 = time.monotonic()
    src, notes = S.open_source(c)
    try:
        return _process(c, src, notes, params, labelset, summaries, participants, out_root, input_dir, t0)
    finally:
        src.close()


def _process(c: S.Candidate, src: S.Source, notes: list[str], params: Params, labelset: L.LabelSet,
             summaries: dict, participants: dict, out_root: Path, input_dir: Path, t0: float) -> Result:
    rid = c.recording_id
    sfreq = src.sfreq
    if sfreq < 64:
        raise ValueError(f"sampling rate {sfreq:g} Hz is too low for EEG")
    duration = src.n_times / sfreq
    if duration < 10:
        raise ValueError(f"recording is shorter than 10 s ({duration:.1f} s)")

    # ---- sidecars ---------------------------------------------------------------------------------
    ch_path = B.sidecar(c.path, "channels.tsv") if c.entities else None
    ch_tsv = B.read_channels(ch_path)
    ev_path = B.sidecar(c.path, "events.tsv") if c.entities else None
    events_in = B.read_events(ev_path)
    js_path = B.sidecar(c.path, "eeg.json") if c.entities else None
    sidecar = B.read_json(js_path)

    # ---- channels ---------------------------------------------------------------------------------
    names = src.ch_names
    types = {n: (ch_tsv[n].type if n in ch_tsv and ch_tsv[n].type else src.file_types.get(n, "")) for n in names}
    trusted = src.types_trusted or bool(ch_tsv)
    classes, layout, rec_ref = C.classify(names, types, trusted)
    kept = [cl for cl in classes if cl.keep]
    if len(kept) < 2:
        raise ValueError("fewer than 2 EEG channels (dropped: " +
                         ", ".join(sorted({cl.dropped_type for cl in classes if not cl.keep})) + ")")
    picks = [cl.index for cl in kept]
    onames = [cl.name for cl in kept]
    nch = len(kept)
    pos = {n: C.position(n) for n in onames} if layout == "referential" else {}
    positioned = [n for n in onames if pos.get(n)]

    # ---- labels -----------------------------------------------------------------------------------
    labs: list[L.Label] = []
    canon = lambda ch: C.canonical_electrode(C.parse(ch).a or ch) or ch  # noqa: E731
    for e in events_in:
        tt = "seizure" if e.hed.lower().startswith("sz") else L.map_trial_type(e.trial_type)
        if tt is None:
            continue
        labs.append(L.Label(e.onset, e.duration, tt, "annotation", ev_path.name if ev_path else "events.tsv",
                            [canon(ch) for ch in e.channels], e.hed))
    if not events_in:
        labs.extend(L.seizures_from_annotations(src.annotations))
    summ = summaries.get(c.path.parent, {}).get(c.path.name)
    if summ:
        for a, b in summ["seizures"]:
            labs.append(L.Label(a, b - a, "seizure", "annotation", f"{c.subject}-summary.txt (CHB-MIT)"))
    ids = {rid, L.norm_recording(c.path.name)}
    label_rows = labelset.for_recording(c.subject, ids)
    csv_events, extra_cols = L.resolve(label_rows, labelset.path.name if labelset.path else "labels.csv")
    for ev in csv_events:
        ev.channels = [canon(ch) for ch in ev.channels]
    labs.extend(csv_events)
    deduped: list[L.Label] = []
    for lb in sorted(labs, key=lambda x: (x.onset, x.source != "annotation")):
        if any(d.trial_type == lb.trial_type and abs(d.onset - lb.onset) < 1.0 for d in deduped):
            notes.append(f"{lb.trial_type} at {lb.onset:.1f} s from {lb.detail} duplicates an earlier label; kept once")
            continue
        if not (0 <= lb.onset < duration):
            notes.append(f"{lb.trial_type} at {lb.onset:.1f} s ({lb.detail}) is outside the recording; ignored")
            continue
        deduped.append(lb)
    labs = deduped
    seizures = [(lb.onset, lb.onset + (lb.duration or 60.0)) for lb in labs if lb.trial_type == "seizure"]

    # ---- line frequency and filters ----------------------------------------------------------------
    lf, lf_src = line_frequency(params.line_freq, sidecar, src)
    seg_len = min(params.qc_segment_s, duration)
    n_seg = 1 if duration < 3 * seg_len else 3
    seg_starts = spread_starts(duration, seg_len, n_seg, seizures)

    def read(a: int, b: int) -> np.ndarray:
        return src.get(picks, a, b)

    t_read = time.monotonic()
    raw_segs = []
    for s0 in seg_starts:
        a = int(s0 * sfreq)
        x = read(a, a + int(seg_len * sfreq))
        # leave annotated seizures (+-10 s) out of the bad-channel check: ictal channels are not bad channels
        keep = np.ones(x.shape[1], dtype=bool)
        for z0, z1 in seizures:
            i0, i1 = int((z0 - 10.0) * sfreq) - a, int((z1 + 10.0) * sfreq) - a
            keep[max(0, i0):max(0, min(x.shape[1], i1))] = False
        if keep.sum() >= 60 * sfreq and not keep.all():
            x = x[:, keep]
            notes.append(f"seizures left out of the bad-channel check (QC segment at {s0:.0f} s)")
        raw_segs.append(x)
    t_read = time.monotonic() - t_read
    if lf is None:
        lf, lf_src = dsp.detect_line_frequency(raw_segs[0], sfreq)
    plan = dsp.plan_filters(sfreq, lf, params.highpass, params.lowpass)
    filt = dsp.Filter(plan)
    notch_only = dsp.Filter(dsp.plan_filters(sfreq, lf, 0.0, 0.0))

    # ---- bad channels ------------------------------------------------------------------------------
    reasons: dict[str, list[str]] = {n: [] for n in onames}
    metrics: dict[str, dict[str, float]] = {n: {} for n in onames}
    suspect: dict[str, list[str]] = {}
    bad_method = "none"
    if params.bad_channels != "none":
        segs_notched = [notch_only(s) for s in raw_segs]
        use_pyprep = (params.bad_channels == "pyprep" and layout == "referential" and len(positioned) >= 4)
        if use_pyprep:
            jpos = [onames.index(n) for n in positioned]
            med_segs = [s[jpos] - np.median(s[jpos], axis=0) for s in segs_notched]
            info = A.make_info(positioned, sfreq, pos)
            try:
                res = BC.pyprep_bads(med_segs, positioned, sfreq, params.seed, params.ransac, info)
                notes.extend(res.notes)
                bad_method = res.method
                for n in positioned:
                    reasons[n].extend(res.reasons.get(n, []))
                    if n in res.suspect:
                        suspect[n] = res.suspect[n]
                    metrics[n].update(res.metrics.get(n, {}))
            except Exception as exc:  # noqa: BLE001 - pyprep refuses some inputs; fall back
                notes.append(f"pyprep failed ({type(exc).__name__}: {exc}); robust metrics used instead")
                use_pyprep = False
        rest = [n for n in onames if not (use_pyprep and n in positioned)]
        if rest:
            seg = np.concatenate(segs_notched, axis=1)
            res = BC.simple_bads(seg, onames, sfreq, lf, SIMPLE_FLAT_UV, params.bad_z, LINE_NOISE_MAX,
                                 against_median=(layout == "referential"))
            bad_method = (bad_method + "; " if use_pyprep else "") + res.method + \
                (f" (channels without template position: {', '.join(rest)})" if use_pyprep else "")
            for n in rest:
                reasons[n].extend(res.reasons.get(n, []))
            for n in onames:
                metrics[n].update({k: v for k, v in res.metrics.get(n, {}).items() if k not in metrics[n]})
        del segs_notched
    for cl in kept:
        row = ch_tsv.get(cl.original)
        if row and row.status == "bad":
            reasons[cl.name].append("bad in source" + (f": {row.status_description}" if row.status_description else ""))
        elif cl.original in src.file_bads:
            reasons[cl.name].append("bad in source")
    status = {n: ("bad" if reasons[n] else "good") for n in onames}
    good = [j for j, n in enumerate(onames) if status[n] == "good"]
    if len(good) < 2:
        raise ValueError(f"fewer than 2 good channels ({nch} EEG channels, {nch - len(good)} bad)")
    if len(good) < 0.5 * nch:
        notes.append(f"{nch - len(good)} of {nch} channels were found bad; check the QC figure")
    del raw_segs

    # ---- reference --------------------------------------------------------------------------------
    ear_idx: list[int] = []
    if layout == "bipolar":
        montage, reference = "bipolar-longitudinal", "bipolar"
        if params.reference != "average":
            notes.append(f"REFERENCE={params.reference} does not apply to a bipolar-only recording; pairs kept")
    else:
        montage = params.reference
        if montage == "linked-ears":
            for pair in (("A1", "A2"), ("M1", "M2")):
                if all(p in onames and status[p] == "good" for p in pair):
                    ear_idx = [onames.index(p) for p in pair]
                    reference = "|".join(pair)
                    break
            if not ear_idx:
                notes.append("REFERENCE=linked-ears but no good A1/A2 or M1/M2 pair; average reference used")
                montage = "average"
        if montage == "average":
            reference = "average"
        elif montage == "as-recorded":
            sc_ref = str(sidecar.get("EEGReference") or "")
            if rec_ref:
                reference = "average" if rec_ref.lower() in ("avg", "av", "ave", "average", "ar", "car") else (
                    sc_ref if (rec_ref.lower().startswith("ref") and sc_ref and sc_ref.lower() != "n/a") else rec_ref)
            else:
                reference = sc_ref if sc_ref and sc_ref.lower() != "n/a" else "unknown"
    good_names = [onames[j] for j in good]

    # ---- resampling plan --------------------------------------------------------------------------
    rs = dsp.Resampler.make(sfreq, params.resample_hz) if params.resample_hz else None
    up, down = (rs.up, rs.down) if rs else (1, 1)
    osf = sfreq * up / down
    n_out = (src.n_times * up) // down
    pad = int(math.ceil(plan.pad_s() * sfreq / down)) * down
    win = max(int(params.window_s * sfreq) // down * down, down)

    def filtered(a: int, b: int) -> tuple[np.ndarray, int, int]:
        """Filtered (and resampled) data of input samples [a, b): (array, output start, output length)."""
        pa, pb = max(0, a - pad), min(src.n_times, b + pad)
        x = filt(read(pa, pb))
        if rs:
            x = rs(x)
        o0 = (a * up) // down
        off = o0 - (pa * up) // down
        length = min(((b - a) * up) // down if b < src.n_times else n_out - o0, x.shape[1] - off)
        return x[:, off:off + length], o0, length

    # ---- artifact handling: calibration ----------------------------------------------------------
    ica: A.IcaResult | None = None
    asr: A.AsrCleaner | None = None
    art_notes: list[str] = []
    want_ica = params.artifacts in ("iclabel", "both")
    want_asr = params.artifacts in ("asr", "both")
    ica_names = [n for n in good_names if pos.get(n)]
    if want_ica and layout == "bipolar":
        art_notes.append("ICA/ICLabel skipped: ICLabel needs single-electrode channels with positions; "
                         "a bipolar-only recording has neither")
        want_ica = False
    if want_ica and len(ica_names) < 8:
        art_notes.append(f"ICA/ICLabel skipped: {len(ica_names)} good channels with positions (needs >= 8)")
        want_ica = False
    t_ica = time.monotonic()
    if want_ica or want_asr:
        n_chunks = int(max(1, math.ceil(min(params.ica_max_s, duration) / ICA_CHUNK_S)))
        chunk_len = min(ICA_CHUNK_S, duration)
        starts = spread_starts(duration, chunk_len, n_chunks, seizures)
        prog = Progress(rid, "reading the ICA/ASR calibration sample")
        calib, fit_chunks, peak = [], [], []
        for k, s in enumerate(starts):
            a = int(s * sfreq)
            b = min(a + int(chunk_len * sfreq), src.n_times)
            x, _, _ = filtered(a, b)
            if layout == "referential":
                x = x - x[good].mean(axis=0)
            calib.append(x[good])
            peak.append(float(np.max(np.abs(x[good]))) if x.size else 0.0)
            prog(k + 1, len(starts))
        limit = 2.0 * params.amplitude_uv if params.amplitude_uv > 0 else np.inf
        keep = [k for k in range(len(calib)) if peak[k] <= limit]
        if len(keep) < max(1, len(calib) // 2):
            keep = sorted(np.argsort(peak)[: max(1, len(calib) // 2)].tolist())
        dropped_chunks = len(calib) - len(keep)
        if dropped_chunks:
            art_notes.append(f"{dropped_chunks} of {len(calib)} calibration chunks left out "
                             f"(peak above {limit:g} µV)")
        calib = [calib[k] for k in keep]
        if want_ica:
            jsub = [good_names.index(n) for n in ica_names]
            q = int(osf // 250) if osf >= 500 else 1
            for x in calib:
                y = x[jsub]
                y = dsp.bandpass(y, osf, 1.0, 100.0)
                if q > 1:
                    y = signal.resample_poly(y, 1, q, axis=-1)
                fit_chunks.append(y)
            fit = np.concatenate(fit_chunks, axis=1)
            try:
                ica = A.fit_ica(fit, ica_names, osf / q, pos, params.ica_method, params.seed,
                                params.iclabel_threshold, params.ica_components, 1)
            except Exception as exc:  # noqa: BLE001 - an ICA that does not converge must not sink the recording
                art_notes.append(f"ICA failed ({type(exc).__name__}: {exc}); no ICA cleaning")
                ica = None
            del fit, fit_chunks
        if want_asr:
            cal = np.concatenate(calib, axis=1)
            if cal.shape[1] < 30 * osf:
                art_notes.append(f"ASR skipped: calibration sample {cal.shape[1] / osf:.0f} s < 30 s")
            else:
                try:
                    asr = A.AsrCleaner(osf, params.asr_cutoff, cal)
                except Exception as exc:  # noqa: BLE001
                    art_notes.append(f"ASR calibration failed ({type(exc).__name__}: {exc}); no ASR")
            del cal
        del calib
    t_ica = time.monotonic() - t_ica

    # muscle detection looks at 70+ Hz; meaningless when the source or LOWPASS already removed that band
    eff_lp = min(([plan.lowpass] if plan.lowpass else []) + [osf / 2.0])
    muscle_z = params.muscle_z
    band = SG.muscle_band(osf)
    if muscle_z > 0 and (band is None or eff_lp < band[0] + 10):
        art_notes.append(f"muscle bad segments not searched: the data are low-passed at {eff_lp:g} Hz "
                         "(the muscle detector looks at 70 Hz and above)")
        muscle_z = 0.0

    # ---- main pass --------------------------------------------------------------------------------
    tmpdir = tempfile.gettempdir()
    fd, mm_path = tempfile.mkstemp(prefix="eeg-prep-", suffix=".f32", dir=tmpdir)
    os.close(fd)
    try:
        mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(nch, max(n_out, 1)))
        bad_ivs: list[tuple[float, float, str]] = []
        nper = int(PSD_EPOCH_S * osf)
        psd_sum = np.zeros((nch, nper // 2 + 1))
        psd_n = 0
        hann = signal.get_window("hann", nper)
        sub_idx = [onames.index(n) for n in ica.names] if ica else []
        prog = Progress(rid, "filtering/cleaning")
        t_main = time.monotonic()
        for a in range(0, src.n_times, win):
            b = min(a + win, src.n_times)
            x, o0, length = filtered(a, b)
            if length <= 0:
                continue
            if layout == "referential":
                m = x[good].mean(axis=0)
                x = x - m
                if asr is not None:
                    x[good] = asr(x[good])
                if ica is not None:
                    y = x[sub_idx]
                    mu = y.mean(axis=0)
                    x[sub_idx] = ica.operator @ (y - mu) + ica.offset[:, None] + mu
                if montage == "average":
                    x = x - x[good].mean(axis=0)
                elif montage == "linked-ears":
                    x = x - x[ear_idx].mean(axis=0)
                elif montage == "as-recorded":
                    x = x + m
            elif asr is not None:
                x[good] = asr(x[good])
            mm[:, o0:o0 + length] = x
            t_win = o0 / osf
            ivs = SG.find(x[good], good_names, osf, t_win, params.amplitude_uv, params.flat_uv, params.flat_s,
                          muscle_z)
            bad_ivs.extend(ivs)
            # Welch PSD over 4 s epochs that do not touch a bad interval of this window
            for e0 in range(0, length - nper + 1, nper):
                ts, te = t_win + e0 / osf, t_win + (e0 + nper) / osf
                if any(s < te and e > ts for s, e, _ in ivs):
                    continue
                seg = x[:, e0:e0 + nper]
                seg = seg - seg.mean(axis=1, keepdims=True)
                psd_sum += np.abs(np.fft.rfft(seg * hann, axis=1)) ** 2
                psd_n += 1
            prog(b, src.n_times)
        mm.flush()
        t_main = time.monotonic() - t_main
        psd_f = np.fft.rfftfreq(nper, 1.0 / osf)
        psd = psd_sum / max(psd_n, 1) / (osf * np.sum(hann ** 2)) * 2.0
        bad_segs = SG.merge(bad_ivs, n_out / osf)

        # ---- ranges, EDF+ ---------------------------------------------------------------------------
        rec_s, spr = E.record_duration(osf)
        n_rec = n_out // spr
        if n_rec < 1:
            raise ValueError("recording shorter than one EDF data record")
        n_edf = n_rec * spr
        if n_edf < n_out:
            notes.append(f"last {(n_out - n_edf) / osf:.3f} s dropped to fill whole EDF data records")
        dur_out = n_edf / osf
        ranges = [E.channel_range(np.asarray(mm[j, :n_edf])) for j in range(nch)]
        prefilter = (f"HP:{plan.highpass:g}Hz" if plan.highpass else "HP:DC") + \
                    (f" LP:{plan.lowpass:g}Hz" if plan.lowpass else "") + \
                    (" N:" + "/".join(f"{f:g}" for f in plan.notch) + "Hz" if plan.notch else "")
        sigs = [E.Signal(n, r.limit, prefilter, "EEG electrode" if layout == "referential" else "EEG bipolar")
                for n, r in zip(onames, ranges)]

        ev_rows: list[dict[str, Any]] = []
        for lb in labs:
            ev_rows.append({"onset": lb.onset, "duration": lb.duration, "trial_type": lb.trial_type,
                            "source": lb.source, "channels": ",".join(lb.channels), "hed": lb.hed})
        for s0, s1, why in bad_segs:
            if s0 < dur_out:
                ev_rows.append({"onset": s0, "duration": min(s1, dur_out) - s0, "trial_type": "bad_segment",
                                "source": "eeg-prep", "channels": "", "hed": "", "why": why})
        ev_rows.sort(key=lambda r: (r["onset"], r["trial_type"]))
        edf_ann = [(r["onset"], r["duration"], r["trial_type"] + (f" ({r['hed']})" if r["hed"] else ""))
                   for r in ev_rows if r["onset"] < dur_out][:20000]

        rel_dir = Path(f"sub-{c.subject}") / (f"ses-{c.session}" if c.session else "") / "eeg"
        out_dir = out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        edf_name = f"{rid}_desc-preproc_eeg.edf"
        start = src.meas_date if isinstance(src.meas_date, dt.datetime) else None
        t_write = time.monotonic()
        E.write_edf(out_dir / edf_name, mm, osf, sigs, edf_ann, start, rec_s,
                    progress=Progress(rid, "writing EDF+"))
        t_write = time.monotonic() - t_write
        verification = None
        snip_start = 0
        if seizures:
            snip_start = int(max(0.0, seizures[0][0] - 2.0) * osf)
        else:
            snip_start = int(max(0.0, dur_out / 2 - SNIPPET_S / 2) * osf)
        snip_len = int(min(SNIPPET_S * osf, n_edf))
        snip_start = min(snip_start, n_edf - snip_len)
        snippet = np.clip(np.asarray(mm[:, snip_start:snip_start + snip_len]),
                          [[-r.limit] for r in ranges], [[r.limit] for r in ranges])
        if params.verify:
            vlen = int(min(VERIFY_S * osf, n_edf))
            checks = []
            for vs in sorted({0, max(0, n_edf // 2 - vlen // 2), n_edf - vlen}):
                block = np.asarray(mm[:, vs:vs + vlen], dtype=np.float64)
                for j in range(nch):
                    checks.append((j, vs, np.clip(block[j], -ranges[j].limit, ranges[j].limit)))
            verification = E.verify(out_dir / edf_name, checks, onames, [r.step for r in ranges],
                                    n_annotations=len(edf_ann))
            if verification["max_error_in_steps"] > 0.5 + 1e-3:
                raise ValueError(f"EDF read-back error {verification['max_abs_error_uv']:.4g} µV on "
                                 f"{verification['worst_channel']} exceeds half a quantisation step")
        del mm
    finally:
        try:
            os.remove(mm_path)
        except OSError:
            pass

    # ---- sidecars of the derivative ----------------------------------------------------------------
    ch_rows = []
    for cl, r in zip(kept, ranges):
        n = cl.name
        srow = ch_tsv.get(cl.original)
        # the derivative's own filtering: what eeg-prep applied, bounded by Nyquist. The source's declared
        # filters (sidecar SoftwareFilters, EDF prefiltering, channels.tsv cutoffs) are often display settings
        # never applied to the stored samples (Siena declares 1.6-30 Hz on full-band data); they are kept in
        # eeg.json (SourceSoftwareFilters, SourceDeclaredFilters), not copied here.
        lows = [plan.highpass or 0.0]
        highs = [plan.lowpass or osf / 2.0, osf / 2.0]
        if cl.kind == "pair":
            a, b = cl.contacts
            pa, pb = C.position(a), C.position(b)
            xyz = tuple((pa[k] + pb[k]) / 2 for k in range(3)) if pa and pb else None
            hemi, reg = C.pair_hemisphere(a, b), C.pair_region(a, b)
        else:
            xyz = pos.get(n)
            hemi, reg = C.hemisphere(n), C.region(n)
        ch_rows.append({
            "name": n, "type": "EEG", "units": "µV", "low_cutoff": fmt(max(lows)), "high_cutoff": fmt(min(highs)),
            "sampling_frequency": fmt(osf), "status": status[n], "status_description": "; ".join(reasons[n]),
            "contacts": "|".join(cl.contacts), "hemisphere": hemi, "region": reg,
            "x": fmt(xyz[0], 5) if xyz else "", "y": fmt(xyz[1], 5) if xyz else "",
            "z": fmt(xyz[2], 5) if xyz else "", "original_name": cl.original,
        })
    write_tsv(out_dir / f"{rid}_desc-preproc_channels.tsv", list(ch_rows[0].keys()), ch_rows)
    write_tsv(out_dir / f"{rid}_desc-preproc_events.tsv",
              ["onset", "duration", "trial_type", "source", "channels", "hed"],
              [{**r, "onset": fmt(r["onset"], 3), "duration": "n/a" if r["duration"] is None else fmt(r["duration"], 3),
                "channels": r["channels"] or "n/a", "hed": r["hed"] or "n/a"} for r in ev_rows])
    electrodes = []
    seen: set[str] = set()
    for cl in kept:
        for e in cl.contacts:
            if e in seen:
                continue
            seen.add(e)
            p = C.position(e)
            electrodes.append({"name": e, "x": fmt(p[0], 5) if p else "n/a", "y": fmt(p[1], 5) if p else "n/a",
                               "z": fmt(p[2], 5) if p else "n/a", "space": C.TEMPLATE_SPACE if p else "n/a"})
    write_tsv(out_dir / f"{rid}_desc-preproc_electrodes.tsv", ["name", "x", "y", "z", "space"], electrodes)
    with open(out_dir / f"{rid}_desc-preproc_coordsystem.json", "w", encoding="utf-8") as fh:
        json.dump({"EEGCoordinateSystem": "Other", "EEGCoordinateUnits": "m",
                   "EEGCoordinateSystemDescription": (
                       "Template positions, not measured: MNE-Python's 10-05 montage 'standard_1005' (named "
                       "'colin27_1005' since MNE 1.13) in MNE head coordinates (x towards the right preauricular "
                       "point, y towards the nasion, z up).")}, fh, indent=2)

    ref_text = {"average": "average of the good EEG channels (computed by eeg-prep)",
                "linked-ears": f"linked ears: mean of {reference.replace('|', ' and ')} (computed by eeg-prep)",
                "as-recorded": f"as recorded: {reference}",
                "bipolar-longitudinal": "bipolar: each channel is the first electrode minus the second "
                                        "(see contacts in channels.tsv)"}[montage]
    js: dict[str, Any] = {
        "TaskName": c.task or sidecar.get("TaskName") or "eeg",
        "SamplingFrequency": osf,
        "EEGReference": ref_text,
        "PowerLineFrequency": lf if lf else "n/a",
        "SoftwareFilters": {
            **({"Highpass": {"CutoffFrequency": plan.highpass, "FilterType": "Butterworth",
                             "Order": dsp.HIGHPASS_ORDER, "ZeroPhase": True}} if plan.highpass else {}),
            **({"Notch": {"Frequencies": plan.notch, "FilterType": "IIR notch", "Q": dsp.NOTCH_Q,
                          "ZeroPhase": True}} if plan.notch else {}),
            **({"Lowpass": {"CutoffFrequency": plan.lowpass, "FilterType": "Butterworth",
                            "Order": dsp.LOWPASS_ORDER, "ZeroPhase": True}} if plan.lowpass else {}),
            **({"Resample": {"From": sfreq, "To": osf, "Method": "polyphase (scipy resample_poly)"}} if rs else {}),
            **({"ICA": {"Method": ica.method, "Classifier": "ICLabel (mne-icalabel, ONNX)",
                        "Threshold": params.iclabel_threshold, "RemovedComponents": len(ica.removed),
                        "Components": ica.n_components}} if ica else {}),
            **({"ASR": {"Implementation": "python-meegkit", "Cutoff": params.asr_cutoff}} if asr else {}),
        } or "n/a",
        "RecordingDuration": round(dur_out, 6),
        "RecordingType": "continuous",
        "EEGChannelCount": nch, "ECGChannelCount": 0, "EMGChannelCount": 0, "EOGChannelCount": 0,
        "MiscChannelCount": 0, "TriggerChannelCount": 0,
        "EEGPlacementScheme": sidecar.get("EEGPlacementScheme") or "10-20",
        **{k: sidecar[k] for k in ("TaskDescription", "Manufacturer", "ManufacturersModelName", "InstitutionName",
                                   "InstitutionAddress", "EEGGround", "CapManufacturer", "HardwareFilters")
           if k in sidecar},
        "SourceSoftwareFilters": sidecar.get("SoftwareFilters", "n/a"),
        "SourceDeclaredFilters": {"highpass_hz": src.highpass or None, "lowpass_hz": src.lowpass or None,
                                  "note": "as declared by the source file header; not verified, not used for "
                                          "low_cutoff/high_cutoff"},
        "Sources": [c.origin],
        "PennsieveEegPrep": {
            "montage": montage, "reference": reference, "layout": layout, "lineFrequencySource": lf_src,
            "sourceFormat": source_format(c), "reader": src.reader, "badChannelMethod": bad_method,
            "edfPhysicalRange": "per channel, symmetric ±1.1 × min(max|x|, 3 × p99.99|x|), at least 1 µV; "
                                "digital ±32767",
            "positions": "template (MNE standard_1005); not measured",
        },
    }
    with open(out_dir / f"{rid}_desc-preproc_eeg.json", "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=2, ensure_ascii=False)

    # ---- qc rows ----------------------------------------------------------------------------------
    qc_rows: list[dict[str, Any]] = []
    for cl in classes:
        if not cl.keep:
            qc_rows.append({"recording_id": rid, "item": "dropped", "name": cl.original, "original_name": cl.original,
                            "type": cl.dropped_type, "status": "dropped", "reason": cl.reason})
    for cl in kept:
        mt = metrics[cl.name]
        qc_rows.append({"recording_id": rid, "item": "channel", "name": cl.name, "original_name": cl.original,
                        "type": "EEG", "status": status[cl.name], "reason": "; ".join(reasons[cl.name]) or (
                            "suspect only: " + "; ".join(suspect[cl.name]) if cl.name in suspect else ""),
                        **({"reason": ("; ".join(reasons[cl.name]) + "; " if reasons[cl.name] else "")
                            + "no physical dimension in the EDF header: values taken as µV"}
                           if cl.original in src.unitless else {}),
                        "robust_sd_uv": fmt(mt.get("robust_sd_uv"), 2),
                        "line_noise_ratio": fmt(mt.get("line_noise_ratio"), 4)})
    if ica:
        for k in range(ica.n_components):
            pr = ica.probabilities[k]
            lab = ica.labels[k]
            qc_rows.append({"recording_id": rid, "item": "ica_component", "name": f"IC{k:02d}", "original_name": "",
                            "type": "ICA", "status": "removed" if k in ica.removed else "kept",
                            "reason": ", ".join(f"{cls} {p:.2f}" for cls, p in zip(A.ICLABEL_CLASSES, pr) if p >= 0.05),
                            "iclabel_label": lab, "iclabel_probability": fmt(float(pr.max()), 3)})
    n_bad_s = sum(min(s1, dur_out) - s0 for s0, s1, _ in bad_segs if s0 < dur_out)
    qc_rows.append({"recording_id": rid, "item": "bad_segments", "name": "total", "original_name": "",
                    "type": "segments", "status": "", "reason": f"{len(bad_segs)} segment(s), {n_bad_s:.1f} s "
                    f"({100 * n_bad_s / max(dur_out, 1e-9):.2f}%)" + (
                        "; " + "; ".join(sorted({p for _, _, w in bad_segs for p in w.split("; ")})) if bad_segs else "")})
    if asr:
        qc_rows.append({"recording_id": rid, "item": "asr", "name": "ASR", "original_name": "", "type": "ASR",
                        "status": "applied", "reason": f"cutoff {params.asr_cutoff:g}; "
                        f"{100 * asr.changed / max(asr.total, 1):.1f}% of samples modified"})

    # ---- recording row ----------------------------------------------------------------------------
    part = participants.get(c.subject, {})
    age = part.get("age", "")
    sex = part.get("sex", "")
    if age.lower() in B.NA:
        age = ""
    if sex.lower() in B.NA:
        sex = ""
    if not age:
        age = _age(src.subject_info, src.meas_date)
    if not sex:
        sex = {1: "m", 2: "f"}.get(src.subject_info.get("sex"), "")
    if params.kind != "auto":
        kind = params.kind
    else:
        text = f"{c.task} {sidecar.get('TaskName', '')} {sidecar.get('TaskDescription', '')}".lower()
        if re.search(r"sleep|psg|polysomno|nap|night", text):
            kind = "sleep"
        else:
            kind = "routine" if dur_out <= 7200 else "long-term"
    start_iso = start.isoformat() if start is not None else ""
    if not start_iso:
        dts = [e.date_time for e in events_in if e.date_time]
        if dts:
            start_iso = dts[0].replace(" ", "T")
    n_good = len(good)
    rec_row = {
        "recording_id": rid, "subject": c.subject, "session": c.session, "task": c.task, "run": c.run,
        "path": str(rel_dir / edf_name), "sfreq": fmt(osf), "n_channels": nch, "n_good": n_good,
        "duration_s": fmt(dur_out, 3), "montage": montage, "reference": reference, "kind": kind,
        "start_time": start_iso, "n_seizures": sum(1 for lb in labs if lb.trial_type == "seizure"),
        "source_format": source_format(c), "age": age, "sex": sex,
    }
    rec_row.update({k: v for k, v in extra_cols.items() if k not in rec_row})

    dropped: dict[str, int] = {}
    for cl in classes:
        if not cl.keep:
            dropped[cl.dropped_type] = dropped.get(cl.dropped_type, 0) + 1
    seconds = time.monotonic() - t0
    bad_list = [n for n in onames if status[n] == "bad"]
    details = {
        "status": "ok", "input": c.origin, "reader": src.reader, "sfreq": sfreq, "output_sfreq": osf,
        "montage": montage, "reference": reference, "layout": layout,
        "source_channels": len(names), "eeg_channels": nch, "dropped": dropped, "bad_channels": bad_list,
        "bad_channel_method": bad_method, "line_freq": lf, "line_freq_source": lf_src, "filters": plan.describe(),
        "qc_segments_s": [round(s, 1) for s in seg_starts], "qc_segment_length_s": seg_len,
        "ica": None if not ica else {"method": ica.method, "components": ica.n_components,
                                     "removed": [f"IC{k:02d} {ica.labels[k]}" for k in ica.removed],
                                     "fit_seconds": round(ica.fit_seconds, 1),
                                     "variance_removed_fraction": round(ica.explained_removed, 4)},
        "asr": None if not asr else {"cutoff": params.asr_cutoff,
                                     "modified_fraction": round(asr.changed / max(asr.total, 1), 4)},
        "bad_segments": len(bad_segs), "bad_segment_seconds": round(n_bad_s, 1),
        "n_seizures": rec_row["n_seizures"], "labels_csv_rows": len(label_rows),
        "clipped_samples": int(sum(r.clipped for r in ranges)),
        "quantisation_step_uv": {"median": float(np.median([r.step for r in ranges])),
                                 "max": float(max(r.step for r in ranges))},
        "verification": verification, "notes": notes + art_notes,
        "unitless_channels_taken_as_uv": [cl.name for cl in kept if cl.original in src.unitless],
        "seconds": {"total": round(seconds, 2), "read_qc": round(t_read, 2), "calibration": round(t_ica, 2),
                    "main_pass": round(t_main, 2), "write": round(t_write, 2)},
    }
    headline = {
        "duration_s": round(dur_out, 1), "n_channels": nch, "n_good": n_good, "n_bad_channels": len(bad_list),
        "montage": montage, "n_ica_components": ica.n_components if ica else 0,
        "n_ica_removed": len(ica.removed) if ica else 0,
        "ica_variance_removed_pct": round(100 * ica.explained_removed, 1) if ica else 0.0,
        "bad_segment_pct": round(100 * n_bad_s / max(dur_out, 1e-9), 2), "n_seizures": rec_row["n_seizures"],
    }
    plot = {"recording_id": rid, "sfreq": osf, "names": onames, "status": [status[n] for n in onames],
            "reasons": [reasons[n] for n in onames], "psd_f": psd_f, "psd": psd, "psd_epochs": psd_n,
            "snippet": snippet, "snippet_start_s": snip_start / osf, "duration_s": dur_out,
            "bad_segments": bad_segs, "events": [(lb.onset, lb.duration, lb.trial_type) for lb in labs],
            "montage": montage, "reference": reference, "layout": layout,
            "positions": {n: (C.position(n) if layout == "referential" else None) for n in onames},
            "contacts": {cl.name: cl.contacts for cl in kept},
            "ica": ica, "line_freq": lf, "headline": headline, "kind": kind}
    return Result(rec_row, qc_rows, details, plot, headline)


def source_format(c: S.Candidate) -> str:
    base = S.SOURCE_FORMAT[c.fmt]
    return f"bids-{base}" if c.bids else base


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
