"""
eeg-prep: the first step of the scalp EEG workflows.

Finds every recording under INPUT_DIR (BIDS-EEG, EDF/EDF+/BDF, BrainVision, EEGLAB, FIF, Nihon
Kohden, Natus/Nicolet) and an optional labels.csv, and writes one BIDS-EEG derivative:

  OUTPUT_DIR/eeg/dataset_description.json, participants.tsv, recordings.csv, provenance.json
  OUTPUT_DIR/eeg/sub-<label>/[ses-<label>/]eeg/<stem>_desc-preproc_{eeg.edf,eeg.json,channels.tsv,
                                                 events.tsv,electrodes.tsv,coordsystem.json}
  OUTPUT_DIR/qc/eeg-prep/qc.csv, <recording_id>_qc.png, overview_qc.png, summary.json, provenance.json

One recording at a time, read in windows; a recording that fails is logged and skipped.
Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from processor import resources


def _setup_environment() -> tuple[dict, str]:
    """Writable HOME/TMPDIR (Lambda replaces both), library caches under TMPDIR, thread counts
    from the cgroup limits; all before numpy is imported."""
    for var, default in (("TMPDIR", tempfile.gettempdir()), ("HOME", None)):
        v = os.environ.get(var) or default
        if v:
            try:
                Path(v).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    tempfile.tempdir = tmp
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tmp) / "mpl"))
    cache = os.environ.get("EEG_PREP_MPL_CACHE")   # font cache built into the image (read-only)
    if cache and Path(cache).is_dir():
        import shutil
        try:
            shutil.copytree(cache, os.environ["MPLCONFIGDIR"], dirs_exist_ok=True)
        except OSError:
            pass
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_DATA", str(Path(tmp) / "mne_data"))
    os.environ.setdefault("MNE_USE_CUDA", "false")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    raw = (os.environ.get("THREADS") or "0").strip()
    try:
        requested = int(raw)
    except ValueError:
        requested = -1
    if requested < 0:
        return {"requested": raw, "used": 1, "why": "THREADS is not a whole number >= 0"}, "bad"
    used, why = resources.resolve_workers(requested, per_worker_mb=150.0, reserve_mb=900.0)
    used = max(1, min(used, 8))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(used)
    return {"requested": requested, "used": used, "why": why + " (BLAS threads; recordings run one at a time)"}, ""


THREAD_INFO, THREAD_ERR = _setup_environment()

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

from processor import bids as B  # noqa: E402
from processor import labels as L  # noqa: E402
from processor import plots  # noqa: E402
from processor import prep as P  # noqa: E402
from processor import sources as S  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("eeg-prep")
log.setLevel(logging.INFO)
for noisy in ("mne", "mne_bids", "matplotlib", "pyprep", "mne_icalabel"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

APP_NAME = "eeg-prep"
REPO_URL = "https://github.com/pennsieve-apps/eeg-prep"
REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING_COLUMNS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
                     "duration_s", "montage", "reference", "kind", "start_time", "n_seizures", "source_format",
                     "age", "sex"]
QC_COLUMNS = ["recording_id", "item", "name", "original_name", "type", "status", "reason", "iclabel_label",
              "iclabel_probability", "robust_sd_uv", "line_noise_ratio"]
CITATIONS_ALWAYS = ["10.3389/fnins.2013.00267", "10.21105/joss.01896", "10.3389/fninf.2015.00016",
                    "10.5281/zenodo.6363575"]
CITATIONS_ICA = ["10.1016/j.neuroimage.2019.05.026", "10.21105/joss.04484", "10.1162/089976699300016719"]
CITATIONS_PICARD = ["10.1109/TSP.2018.2844203"]
CITATIONS_ASR = ["10.1109/TBME.2015.2481482", "10.1109/TBME.2019.2930186", "10.5281/zenodo.5643659"]


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("mne", "mne-bids", "pyprep", "mne-icalabel", "onnxruntime", "meegkit", "python-picard", "numpy",
                "scipy", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def env_params() -> dict[str, str]:
    return {
        "RECORDINGS": env("RECORDINGS", "*"),
        "LABELS_FILE": env("LABELS_FILE", "labels.csv"),
        "REFERENCE": env("REFERENCE", "average").lower(),
        "LINE_FREQ": env("LINE_FREQ", "auto").lower(),
        "HIGHPASS": env("HIGHPASS", "0.5"),
        "LOWPASS": env("LOWPASS", "0"),
        "RESAMPLE_HZ": env("RESAMPLE_HZ", "0"),
        "ARTIFACTS": env("ARTIFACTS", "iclabel").lower(),
        "ICLABEL_THRESHOLD": env("ICLABEL_THRESHOLD", "0.8"),
        "ICA_METHOD": env("ICA_METHOD", "infomax").lower(),
        "ICA_MAX_S": env("ICA_MAX_S", "300"),
        "ICA_COMPONENTS": env("ICA_COMPONENTS", "0"),
        "ASR_CUTOFF": env("ASR_CUTOFF", "20"),
        "BAD_CHANNELS": env("BAD_CHANNELS", "pyprep").lower(),
        "RANSAC": env("RANSAC", "true").lower(),
        "QC_SEGMENT_S": env("QC_SEGMENT_S", "300"),
        "BAD_Z": env("BAD_Z", "5"),
        "AMPLITUDE_UV": env("AMPLITUDE_UV", "500"),
        "AMPLITUDE_CHANNELS": env("AMPLITUDE_CHANNELS", "0.5"),
        "EXTREME_UV": env("EXTREME_UV", "2000"),
        "FLAT_UV": env("FLAT_UV", "0.1"),
        "FLAT_S": env("FLAT_S", "5"),
        "MUSCLE_Z": env("MUSCLE_Z", "7"),
        "RECORDING_KIND": env("RECORDING_KIND", "auto").lower(),
        "PROTECT_ANNOTATED_SEIZURES": env("PROTECT_ANNOTATED_SEIZURES", "true").lower(),
        "WINDOW_S": env("WINDOW_S", "300"),
        "SEED": env("SEED", "42"),
        "THREADS": env("THREADS", "0"),
        "VERIFY_EDF": env("VERIFY_EDF", "true").lower(),
    }


def parse_params(raw: dict[str, str]) -> P.Params:
    def num(k: str, lo: float | None = None, hi: float | None = None) -> float:
        try:
            v = float(raw[k])
        except ValueError:
            raise ValueError(f"{k} must be a number (got {raw[k]!r})") from None
        if lo is not None and v < lo:
            raise ValueError(f"{k} must be >= {lo:g} (got {v:g})")
        if hi is not None and v > hi:
            raise ValueError(f"{k} must be <= {hi:g} (got {v:g})")
        return v

    def integer(k: str, lo: int) -> int:
        v = num(k, lo)
        if v != int(v):
            raise ValueError(f"{k} must be a whole number (got {raw[k]!r})")
        return int(v)

    def flag(k: str) -> bool:
        if raw[k] not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"{k} must be true or false (got {raw[k]!r})")
        return raw[k] in ("true", "1", "yes")

    def choice(k: str, options: tuple[str, ...]) -> str:
        if raw[k] not in options:
            raise ValueError(f"{k} must be one of {', '.join(options)} (got {raw[k]!r})")
        return raw[k]

    if raw["LINE_FREQ"] != "auto":
        if num("LINE_FREQ", 0) not in (0.0, 50.0, 60.0):
            log.warning("LINE_FREQ=%s is neither 50 nor 60 Hz", raw["LINE_FREQ"])
    hp, lp = num("HIGHPASS", 0), num("LOWPASS", 0)
    if hp and lp and lp <= hp:
        raise ValueError(f"LOWPASS ({lp:g}) must be above HIGHPASS ({hp:g})")
    integer("THREADS", 0)
    return P.Params(
        reference=choice("REFERENCE", ("average", "linked-ears", "as-recorded")),
        line_freq=raw["LINE_FREQ"], highpass=hp, lowpass=lp, resample_hz=num("RESAMPLE_HZ", 0),
        artifacts=choice("ARTIFACTS", ("iclabel", "asr", "both", "none")),
        iclabel_threshold=num("ICLABEL_THRESHOLD", 0.5, 1.0),
        ica_method=choice("ICA_METHOD", ("infomax", "picard")), ica_max_s=num("ICA_MAX_S", 30),
        ica_components=integer("ICA_COMPONENTS", 0), asr_cutoff=num("ASR_CUTOFF", 1),
        bad_channels=choice("BAD_CHANNELS", ("pyprep", "simple", "none")), ransac=flag("RANSAC"),
        qc_segment_s=num("QC_SEGMENT_S", 30), bad_z=num("BAD_Z", 1), amplitude_uv=num("AMPLITUDE_UV", 0),
        amplitude_channels=num("AMPLITUDE_CHANNELS", 0.01, 1.0), extreme_uv=num("EXTREME_UV", 0),
        flat_uv=num("FLAT_UV", 0), flat_s=num("FLAT_S", 0), muscle_z=num("MUSCLE_Z", 0),
        kind=choice("RECORDING_KIND", ("auto", "routine", "long-term", "sleep", "unknown")),
        protect_seizures=flag("PROTECT_ANNOTATED_SEIZURES"), window_s=num("WINDOW_S", 30), seed=integer("SEED", 0), verify=flag("VERIFY_EDF"),
    )


def write_csv(path: Path, header: list[str], rows: list[dict], delimiter: str = ",") -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", lineterminator="\n", delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)


def find_labels(files: list[Path], name: str, input_dir: Path) -> Path | None:
    if not name:
        return None
    hits = sorted((p for p in files if p.name == name), key=lambda p: (len(p.relative_to(input_dir).parts), str(p)))
    if len(hits) > 1:
        log.warning("%d files named %s; using %s", len(hits), name, hits[0].relative_to(input_dir))
    return hits[0] if hits else None


def source_datasets(files: list[Path]) -> list[dict]:
    out = []
    for p in files:
        if p.name != "dataset_description.json":
            continue
        d = B.read_json(p)
        if d.get("DatasetType") == "derivative":
            continue
        entry = {k: d[k] for k in ("Name",) if d.get(k)}
        if d.get("DatasetDOI"):
            doi = str(d["DatasetDOI"])
            entry["DOI"] = doi if doi.startswith("doi:") else f"doi:{doi.split('doi.org/')[-1]}"
        if entry and entry not in out:
            out.append(entry)
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    raw_params = env_params()
    if THREAD_ERR:
        log.error("bad parameter: THREADS must be a whole number >= 0 (got %r)", raw_params["THREADS"])
        return 2
    try:
        params = parse_params(raw_params)
    except ValueError as exc:
        log.error("bad parameter: %s", exc)
        return 2
    log.info("threads: %d (%s)", THREAD_INFO["used"], THREAD_INFO["why"])

    eeg_dir, qc_dir = output_dir / "eeg", output_dir / "qc" / APP_NAME
    eeg_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    app_version = version_of(REPO_ROOT)
    rel = lambda p: str(p.relative_to(input_dir))  # noqa: E731

    candidates, files, notes = S.find_recordings(input_dir, raw_params["RECORDINGS"])
    for n in notes:
        log.info(n)
    labels_path = find_labels(files, raw_params["LABELS_FILE"], input_dir)
    try:
        labelset = L.read_labels(labels_path)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        log.error("cannot use %s: %s", labels_path, exc)
        return 1
    summaries = L.find_chbmit_summaries(files)
    log.info("input: %d recording(s) (%s), labels: %s, CHB-MIT summaries: %d", len(candidates),
             ", ".join(f"{k} {v}" for k, v in Counter(c.fmt for c in candidates).items()) or "none",
             f"{rel(labels_path)} ({len(labelset.rows)} rows)" if labels_path else "none", len(summaries))
    if not candidates:
        listing = [rel(p) for p in files]
        log.error("no recording (EDF, BDF, BrainVision .vhdr, EEGLAB .set, FIF, Nihon Kohden .EEG, Natus .e) under "
                  "INPUT_DIR; it holds %d file(s): %s", len(listing),
                  ", ".join(listing[:40]) + (" …" if len(listing) > 40 else ""))
        return 1

    ids = Counter(c.recording_id for c in candidates)
    prov_recs: dict[str, dict] = {}
    for c in candidates:
        if ids[c.recording_id] > 1:
            log.error("%s: recording id %s is used by %d inputs, skipped", c.origin, c.recording_id,
                      ids[c.recording_id])
            prov_recs.setdefault(c.recording_id, {"status": "skipped", "reason": "duplicate recording id",
                                                  "inputs": []})["inputs"].append(c.origin)
    candidates = [c for c in candidates if ids[c.recording_id] == 1]

    participants_by_root: dict[Path | None, dict] = {}
    rec_rows: list[dict] = []
    qc_rows: list[dict] = []
    headlines: dict[str, dict] = {}
    figures: list[dict] = []
    warnings_out: list[str] = []
    part_rows: dict[str, dict] = {}
    used_ica = used_asr = used_picard = False
    clash = [k for k in labelset.extra_columns if k in RECORDING_COLUMNS]
    if clash:
        log.warning("labels.csv column(s) %s name recordings.csv columns eeg-prep computes; ignored", ", ".join(clash))
    extra_cols = [k for k in labelset.extra_columns if k not in RECORDING_COLUMNS]

    def write_tables() -> None:
        """recordings.csv, participants.tsv and qc.csv of everything done so far (rewritten after each recording,
        so a partial or still-running output is always indexed)."""
        write_csv(eeg_dir / "recordings.csv", RECORDING_COLUMNS + extra_cols,
                  sorted(rec_rows, key=lambda r: r["recording_id"]))
        pcols = ["participant_id"]
        for r in part_rows.values():
            pcols.extend(k for k in r if k not in pcols)
        write_csv(eeg_dir / "participants.tsv", pcols,
                  sorted(part_rows.values(), key=lambda r: r["participant_id"]), "\t")
        write_csv(qc_dir / "qc.csv", QC_COLUMNS, qc_rows)

    dd = {
        "Name": "eeg-prep derivative",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": APP_NAME, "Version": app_version, "CodeURL": REPO_URL,
                         "Description": "EEG channel selection and 10-20 names, bad channels (pyprep), notch/high-pass "
                                        "filters, re-reference, ICA + ICLabel and/or ASR artifact removal, bad "
                                        "segments, labels"}],
        "SourceDatasets": source_datasets(files) or [{"URL": "n/a"}],
    }
    with open(eeg_dir / "dataset_description.json", "w", encoding="utf-8") as fh:
        json.dump(dd, fh, indent=2, ensure_ascii=False)
    write_tables()
    for i, c in enumerate(candidates, start=1):
        if c.bids_root not in participants_by_root:
            participants_by_root[c.bids_root] = B.read_participants(c.bids_root)
        participants = participants_by_root[c.bids_root]
        log.info("[%d/%d] %s (%s)", i, len(candidates), c.recording_id, c.origin)
        try:
            res = P.process(c, params, labelset, summaries, participants, eeg_dir, input_dir)
        except Exception as exc:  # noqa: BLE001 - one recording's failure must not sink the run
            log.error("%s: FAILED: %s\n%s", c.origin, exc, traceback.format_exc())
            prov_recs[c.recording_id] = {"status": "failed", "input": c.origin, "error": f"{type(exc).__name__}: {exc}"}
            warnings_out.append(f"{c.recording_id}: failed ({type(exc).__name__}: {exc})")
            continue
        rec_rows.append(res.recording_row)
        qc_rows.extend(res.qc_rows)
        headlines[c.recording_id] = res.headline
        d = res.details
        used_ica |= d["ica"] is not None
        used_picard |= d["ica"] is not None and params.ica_method == "picard"
        used_asr |= d["asr"] is not None
        fig_name = f"{c.recording_id}_qc.png"
        try:
            plots.qc_figure(res.plot, qc_dir / fig_name)
            h = res.headline
            figures.append({"path": fig_name, "caption": (
                f"{c.recording_id}: data quality after preprocessing. Top left: the spectrum of every channel "
                f"(bad channels in red; a peak near 10 Hz in posterior channels is the alpha rhythm, a line at "
                f"{d['line_freq'] or 50:g} Hz would mean the notch did not work). Top right: which electrodes or "
                f"pairs were judged bad ({h['n_bad_channels']} of {h['n_channels']}). Middle: scalp maps of the "
                f"{h['n_ica_removed']} ICA component(s) removed as artifacts (eye, muscle, heart, line or channel "
                f"noise per ICLabel). Timeline: annotated seizures and the {h['bad_segment_pct']:.2f}% of the "
                f"recording marked bad (left out by later steps). Bottom: 10 s of the cleaned output.")})
        except Exception as exc:  # noqa: BLE001 - a plot must not fail the recording
            log.warning("%s: QC figure failed: %s", c.recording_id, exc)
        prov_recs[c.recording_id] = {"status": "ok", "seconds": d["seconds"]["total"], "montage": d["montage"],
                                     "input": c.origin, "output": res.recording_row["path"], **{
                                         k: d[k] for k in ("reference", "sfreq", "output_sfreq", "eeg_channels",
                                                           "bad_channels", "bad_channel_method", "dropped",
                                                           "line_freq", "line_freq_source", "ica", "asr",
                                                           "bad_segments", "bad_segment_seconds",
                                                           "bad_segment_seconds_by_criterion", "n_seizures", "unitless_channels_taken_as_uv",
                                                           "verification", "notes", "filters")},
                                     "timings": d["seconds"]}
        part = participants.get(c.subject, {})
        prow = part_rows.setdefault(c.subject, {"participant_id": f"sub-{c.subject}"})
        for k, v in part.items():
            if k != "participant_id":
                prow.setdefault(k, v)
        for k in ("age", "sex"):
            if res.recording_row.get(k) and not prow.get(k):
                prow[k] = res.recording_row[k]
        v = d.get("verification") or {}
        log.info("%s: %d source ch -> %d EEG (%s, %d bad: %s; dropped %s) -> %s, ref %s; line %s Hz (%s); "
                 "ICA %s; bad segments %d (%.1f s); seizures %d; EDF max error %.3g µV (%.2f steps); "
                 "%.1f s; peak RSS %.0f MB",
                 c.recording_id, d["source_channels"], d["eeg_channels"], d["layout"], len(d["bad_channels"]),
                 ", ".join(d["bad_channels"]) or "-", dict(d["dropped"]) or "none", d["montage"], d["reference"],
                 d["line_freq"], d["line_freq_source"],
                 f"{len(d['ica']['removed'])}/{d['ica']['components']} removed" if d["ica"] else "not run",
                 d["bad_segments"], d["bad_segment_seconds"], d["n_seizures"],
                 v.get("max_abs_error_uv", float("nan")), v.get("max_error_in_steps", float("nan")),
                 d["seconds"]["total"], peak_rss_mb())
        for n in d["notes"]:
            log.info("%s: %s", c.recording_id, n)
        write_tables()

    write_tables()
    if headlines:
        try:
            plots.overview_figure([{"recording_id": k, **v} for k, v in sorted(headlines.items())],
                                  qc_dir / "overview_qc.png")
            figures.insert(0, {"path": "overview_qc.png", "caption": (
                "Data quality of every recording: good and bad channels, ICA components removed as artifacts, "
                "and the share of the recording marked as bad segments (excluded by the analysis steps). A "
                "recording with many bad channels or a large bad share deserves a look at its own QC figure.")})
        except Exception as exc:  # noqa: BLE001
            log.warning("overview figure failed: %s", exc)
    finished = datetime.now(timezone.utc)
    citations = list(CITATIONS_ALWAYS)
    if used_ica:
        citations += CITATIONS_ICA
    if used_picard:
        citations += CITATIONS_PICARD
    if used_asr:
        citations += CITATIONS_ASR
    inputs = sorted({rec.get("input") for rec in prov_recs.values() if rec.get("input")} |
                    {i for rec in prov_recs.values() for i in rec.get("inputs", [])})
    if labels_path:
        inputs.append(rel(labels_path))
    provenance = {
        "app": APP_NAME, "version": app_version, "variant": None, "tools": tool_versions(),
        "parameters": raw_params, "inputs": inputs,
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": round(peak_rss_mb(), 1),
        "threads": THREAD_INFO, "recordings": prov_recs, "citations": citations, "licence_notice": None,
    }
    summary = {
        "app": APP_NAME, "version": app_version, "recordings": headlines, "figures": figures,
        "tables": [{"path": "qc.csv", "caption": (
            "One row per channel (kept: good/bad with the reason; dropped: the non-EEG channels such as ECG, "
            "EOG, triggers), per ICA component (ICLabel class and probability; removed or kept), and the bad "
            "segment total of each recording.")}],
        "warnings": warnings_out,
    }
    for d in (eeg_dir, qc_dir):
        with open(d / "provenance.json", "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2, ensure_ascii=False, default=str)
    with open(qc_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    n_fail = sum(1 for r in prov_recs.values() if r["status"] != "ok")
    log.info("%d of %d recording(s) prepared, %d failed/skipped, %.1f s, peak RSS %.0f MB; wrote %s and %s",
             len(rec_rows), len(prov_recs), n_fail, time.monotonic() - t_start, peak_rss_mb(), eeg_dir, qc_dir)
    if not rec_rows:
        log.error("every recording failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
