"""
eeg-prep on the synthetic fixture (tests/make_fixture.py), run in-process (no Docker).
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests import make_fixture as F

REF_ID = "sub-01_ses-01_task-routine_run-01"
BIP_ID = "sub-chbx01_task-eeg_run-01"
PROVENANCE_KEYS = {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                   "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def read_edf(path: Path):
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_edf(path, preload=True, verbose="error")


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixture")
    F.main(d)
    return d


def run(input_dir: Path, out: Path, tmp: Path, **params) -> int:
    keep = dict(os.environ)
    try:
        os.environ.update({"INPUT_DIR": str(input_dir), "OUTPUT_DIR": str(out), "TMPDIR": str(tmp)})
        for k, v in params.items():
            os.environ[k] = str(v)
        from processor import main as M
        importlib.reload(M)
        return M.main()
    finally:
        os.environ.clear()
        os.environ.update(keep)


@pytest.fixture(scope="session")
def default_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    tmp = tmp_path_factory.mktemp("tmp")
    assert run(fixture_dir, out, tmp) == 0
    assert not list(tmp.glob("eeg-prep-*.f32")), "scratch array left behind"
    return out


def test_layout(default_run):
    eeg = default_run / "eeg"
    for f in ("dataset_description.json", "participants.tsv", "recordings.csv", "provenance.json"):
        assert (eeg / f).is_file()
    d = eeg / "sub-01" / "ses-01" / "eeg"
    for suffix in ("eeg.edf", "eeg.json", "channels.tsv", "events.tsv", "electrodes.tsv", "coordsystem.json"):
        assert (d / f"{REF_ID}_desc-preproc_{suffix}").is_file(), suffix
    assert (eeg / "sub-chbx01" / "eeg" / f"{BIP_ID}_desc-preproc_eeg.edf").is_file()
    qc = default_run / "qc" / "eeg-prep"
    for f in ("qc.csv", "summary.json", "provenance.json", f"{REF_ID}_qc.png", f"{BIP_ID}_qc.png", "overview_qc.png"):
        assert (qc / f).is_file(), f
    dd = json.loads((eeg / "dataset_description.json").read_text())
    assert dd["DatasetType"] == "derivative" and dd["GeneratedBy"][0]["Name"] == "eeg-prep"


def test_recordings_csv(default_run):
    rows = {r["recording_id"]: r for r in read_csv(default_run / "eeg" / "recordings.csv")}
    assert set(rows) == {REF_ID, BIP_ID}
    r = rows[REF_ID]
    assert r["montage"] == "average" and r["reference"] == "average"
    assert r["n_channels"] == "19" and r["n_good"] == "18"
    assert r["kind"] == "routine" and r["n_seizures"] == "1"
    assert r["start_time"].startswith("2020-03-01T10:00:00")
    assert (r["age"], r["sex"]) == ("34", "f")
    assert r["lateralisation"] == "left"            # labels.csv recording-level column
    assert r["source_format"] == "bids-edf" and float(r["duration_s"]) == F.DURATION
    b = rows[BIP_ID]
    assert b["montage"] == "bipolar-longitudinal" and b["reference"] == "bipolar"
    assert b["n_seizures"] == "1" and b["source_format"] == "edf" and b["n_channels"] == "21"
    assert b["lateralisation"] == ""


def test_channels_referential(default_run):
    rows = read_csv(default_run / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_channels.tsv", "\t")
    names = [r["name"] for r in rows]
    assert names == F.ELECTRODES                     # canonical names, old T3/T5 renamed, EKG/Photic dropped
    by = {r["name"]: r for r in rows}
    assert by["T7"]["original_name"] == "EEG T3-REF"
    assert by[F.NOISY]["status"] == "bad" and by[F.NOISY]["status_description"]
    assert all(r["status"] == "good" for r in rows if r["name"] != F.NOISY)
    assert by["O1"]["hemisphere"] == "L" and by["O2"]["hemisphere"] == "R" and by["Cz"]["hemisphere"] == "Z"
    assert by["T7"]["region"] == "temporal" and by["O1"]["region"] == "occipital" and by["Cz"]["region"] == "midline"
    assert all(r["x"] and r["y"] and r["z"] for r in rows)
    assert float(by["T7"]["x"]) < 0 < float(by["T8"]["x"])
    assert by["Fp1"]["units"] == "µV" and float(by["Fp1"]["low_cutoff"]) == 0.5 and float(by["Fp1"]["high_cutoff"]) == 128
    el = read_csv(default_run / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_electrodes.tsv", "\t")
    assert [e["name"] for e in el] == F.ELECTRODES and el[0]["space"] == "standard_1005"


def test_channels_bipolar(default_run):
    rows = read_csv(default_run / "eeg" / "sub-chbx01" / "eeg" / f"{BIP_ID}_desc-preproc_channels.tsv", "\t")
    names = [r["name"] for r in rows]
    assert names[:4] == ["Fp1-F7", "F7-T7", "T7-P7", "P7-O1"]
    assert "P7-T7" not in names and names.count("T8-P8") == 1 and len(names) == 21
    by = {r["name"]: r for r in rows}
    assert by["Fp1-F7"]["contacts"] == "Fp1|F7" and by["Fp1-F7"]["hemisphere"] == "L"
    assert by["Fz-Cz"]["hemisphere"] == "Z" and by["F8-T8"]["region"] == "temporal"
    qc = [r for r in read_csv(default_run / "qc" / "eeg-prep" / "qc.csv") if r["recording_id"] == BIP_ID]
    dropped = {r["name"]: r["type"] for r in qc if r["item"] == "dropped"}
    assert dropped.get("P7-T7") == "duplicate" and dropped.get("ECG") == "ecg" and dropped.get("-") == "empty"
    assert not [r for r in qc if r["item"] == "ica_component"]


def test_events(default_run):
    ev = read_csv(default_run / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_events.tsv", "\t")
    sz = [e for e in ev if e["trial_type"] == "seizure"]
    assert len(sz) == 1 and float(sz[0]["onset"]) == F.SEIZURE[0]
    assert float(sz[0]["duration"]) == F.SEIZURE[1] - F.SEIZURE[0]
    assert sz[0]["hed"] == "sz_foc_ia" and sz[0]["channels"] == "T7,P7" and sz[0]["source"] == "annotation"
    assert any(e["trial_type"] == "lights_off" and e["source"] == "labels.csv" for e in ev)
    bad = [e for e in ev if e["trial_type"] == "bad_segment"]
    assert any(float(e["onset"]) <= F.ARTIFACT[0] + 0.1 and float(e["onset"]) + float(e["duration"]) >= F.ARTIFACT[1] - 0.1
               for e in bad), bad
    assert all(e["source"] == "eeg-prep" for e in bad)
    assert sum(float(e["duration"]) for e in bad) < 10           # the rest of the recording is not thrown away
    evb = read_csv(default_run / "eeg" / "sub-chbx01" / "eeg" / f"{BIP_ID}_desc-preproc_events.tsv", "\t")
    szb = [e for e in evb if e["trial_type"] == "seizure"]
    assert [(float(e["onset"]), float(e["duration"])) for e in szb] == [
        (F.BIPOLAR_SEIZURE[0], F.BIPOLAR_SEIZURE[1] - F.BIPOLAR_SEIZURE[0])]


def test_edf_signals(default_run):
    d = default_run / "eeg" / "sub-01" / "ses-01" / "eeg"
    raw = read_edf(d / f"{REF_ID}_desc-preproc_eeg.edf")
    assert raw.ch_names == F.ELECTRODES and raw.info["sfreq"] == F.SFREQ
    x = raw.get_data() * 1e6
    good = [k for k, n in enumerate(raw.ch_names) if n != F.NOISY]
    # average reference of the good channels
    assert np.max(np.abs(x[good].mean(axis=0))) < 0.5
    # line noise removed: 50 Hz power far below the planted 20 µV sine
    from scipy import signal
    f, p = signal.welch(x[raw.ch_names.index("O1")], fs=F.SFREQ, nperseg=1024)
    assert p[np.argmin(np.abs(f - 50))] < 0.01 * p[np.argmin(np.abs(f - 10))]
    # alpha kept: 10 Hz peak on O1
    band = (f > 5) & (f < 15)
    assert abs(f[band][np.argmax(p[band])] - 10.0) < 0.5
    # spikes kept on F7 (negative peaks of > 100 µV at the planted times)
    f7 = x[raw.ch_names.index("F7")]
    for st in F.SPIKE_TIMES:
        i = int(st * F.SFREQ)
        assert f7[i - 10:i + 10].min() < -100, st
    # blinks reduced on Fp1 by ICA (removed eye component): Fp1 below its planted 160 µV blinks
    fp1 = x[raw.ch_names.index("Fp1")]
    a = int(10 * F.SFREQ)
    assert np.percentile(np.abs(fp1[a:int(55 * F.SFREQ)]), 99.5) < 120
    # EDF+ annotations carry the seizure
    assert any("seizure" in d_ for d_ in raw.annotations.description)


def test_qc_and_ica(default_run):
    qc = [r for r in read_csv(default_run / "qc" / "eeg-prep" / "qc.csv") if r["recording_id"] == REF_ID]
    dropped = {r["name"]: r["type"] for r in qc if r["item"] == "dropped"}
    assert dropped == {"EKG": "ecg", "Photic": "trigger"}
    comps = [r for r in qc if r["item"] == "ica_component"]
    assert len(comps) == 17                                      # 18 good channels - 1 (average reference)
    removed = [r for r in comps if r["status"] == "removed"]
    assert removed and all(r["iclabel_label"] == "eye blink" for r in removed)
    assert any(r["item"] == "bad_segments" for r in qc)


def test_summary_and_provenance(default_run):
    qc = default_run / "qc" / "eeg-prep"
    s = json.loads((qc / "summary.json").read_text())
    assert s["app"] == "eeg-prep" and set(s["recordings"]) == {REF_ID, BIP_ID}
    assert s["figures"][0]["path"] == "overview_qc.png"
    assert all((qc / f["path"]).is_file() and f["caption"] for f in s["figures"])
    assert isinstance(s["recordings"][REF_ID]["n_ica_removed"], int)
    for d in (default_run / "eeg", qc):
        p = json.loads((d / "provenance.json").read_text())
        assert set(p) == PROVENANCE_KEYS
        assert p["recordings"][REF_ID]["status"] == "ok" and p["recordings"][REF_ID]["montage"] == "average"
        assert p["recordings"][BIP_ID]["montage"] == "bipolar-longitudinal"
        assert "10.3389/fnins.2013.00267" in p["citations"] and "10.1016/j.neuroimage.2019.05.026" in p["citations"]
        assert p["threads"]["used"] >= 1 and p["parameters"]["REFERENCE"] == "average"


def test_as_recorded_and_linked_ears(fixture_dir, tmp_path):
    assert run(fixture_dir, tmp_path / "o", tmp_path, REFERENCE="as-recorded", ARTIFACTS="none",
               RECORDINGS="sub-01*") == 0
    r = read_csv(tmp_path / "o" / "eeg" / "recordings.csv")[0]
    assert r["montage"] == "as-recorded" and r["reference"] == "Ref"   # EEGReference of the sidecar
    raw = read_edf(tmp_path / "o" / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_eeg.edf")
    x = raw.get_data() * 1e6
    assert np.std(x.mean(axis=0)) > 5          # the recording reference's activity is still in every channel
    assert run(fixture_dir, tmp_path / "o2", tmp_path, REFERENCE="linked-ears", RECORDINGS="sub-01*") == 0
    r = read_csv(tmp_path / "o2" / "eeg" / "recordings.csv")[0]
    assert r["montage"] == "average"           # no A1/A2 in the fixture: falls back, with a note
    p = json.loads((tmp_path / "o2" / "eeg" / "provenance.json").read_text())
    assert any("linked-ears" in n for n in p["recordings"][REF_ID]["notes"])


def test_asr_and_resample(fixture_dir, tmp_path):
    assert run(fixture_dir, tmp_path / "o", tmp_path, ARTIFACTS="both", RESAMPLE_HZ="128", LOWPASS="40") == 0
    rows = {r["recording_id"]: r for r in read_csv(tmp_path / "o" / "eeg" / "recordings.csv")}
    assert rows[REF_ID]["sfreq"] == "128" and float(rows[REF_ID]["duration_s"]) == F.DURATION
    qc = read_csv(tmp_path / "o" / "qc" / "eeg-prep" / "qc.csv")
    assert any(r["item"] == "asr" and r["recording_id"] == REF_ID for r in qc)
    raw = read_edf(tmp_path / "o" / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_eeg.edf")
    assert raw.info["sfreq"] == 128 and raw.n_times == int(F.DURATION * 128)
    ch = read_csv(tmp_path / "o" / "eeg" / "sub-01" / "ses-01" / "eeg" / f"{REF_ID}_desc-preproc_channels.tsv", "\t")
    assert float(ch[0]["high_cutoff"]) == 40


def test_exit_codes(fixture_dir, tmp_path):
    assert run(fixture_dir, tmp_path / "a", tmp_path, REFERENCE="bogus") == 2
    assert run(fixture_dir, tmp_path / "b", tmp_path, HIGHPASS="x") == 2
    assert run(fixture_dir, tmp_path / "c", tmp_path, THREADS="-1") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing here")
    assert run(empty, tmp_path / "d", tmp_path) == 1


def test_edf_writer_roundtrip(tmp_path):
    from processor import edfout as E
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, 2560)) * [[20], [300], [0.2]]).astype(np.float32)
    ranges = [E.channel_range(r) for r in x]
    sigs = [E.Signal(n, r.limit) for n, r in zip(["Fp1", "Cz-Pz", "O2"], ranges)]
    ann = [(1.5, 2.0, "seizure"), (3.25, None, "lights_off")] + [(5 + 0.01 * k, 0.1, "bad_segment") for k in range(200)]
    E.write_edf(tmp_path / "t.edf", x, 256.0, sigs, ann, None, 1.0)
    checks = [(j, 0, np.clip(x[j], -ranges[j].limit, ranges[j].limit)) for j in range(3)]
    v = E.verify(tmp_path / "t.edf", checks, ["Fp1", "Cz-Pz", "O2"], [r.step for r in ranges], len(ann))
    assert v["max_error_in_steps"] <= 0.5 + 1e-6
    assert v["annotations_read"] == len(ann)


def test_channel_names():
    from processor import channels as C
    assert C.parse("EEG Fp1-REF").a == "Fp1"
    assert C.parse("Fp1-Avg").a == "Fp1" and C.parse("Fp1-Avg").kind == "electrode"
    assert C.parse("T3").a == "T7" and C.parse("T5-LE").a == "P7"
    p = C.parse("FP1-F7")
    assert (p.kind, p.a, p.b) == ("pair", "Fp1", "F7")
    assert C.parse("T8-P8-1").dup == 1
    assert C.drop_type_by_name("EKG1-CHIN") and C.drop_type_by_name("LOC-ROC") == "eog"
    assert C.drop_type_by_name("C3") is None
    classes, layout, ref = C.classify(["C3-A1", "C4-A1", "O1-A1", "O2-A1"], {}, False)
    assert layout == "referential" and ref == "A1" and [c.name for c in classes] == ["C3", "C4", "O1", "O2"]


def test_labels():
    from processor import labels as L
    assert L.is_seizure("Seizure onset") and L.is_seizure("sz_foc_ia") and L.is_seizure("SZ")
    assert not L.is_seizure("interictal spike") and not L.is_seizure("seizure end") and not L.is_seizure("bckg")
    assert L.map_trial_type("Eyes closed") == "eyes_closed" and L.map_trial_type("HV") == "hyperventilation"
    ann = [(10.0, 0.0, "Seizure start"), (55.0, 0.0, "seizure end"), (70.0, 0.0, "pt talking")]
    out = L.seizures_from_annotations(ann)
    assert [(o.onset, o.duration, o.trial_type) for o in out] == [(10.0, 45.0, "seizure")]


def test_edf_without_physical_dimension(tmp_path):
    """ds003555-style EDF: blank physical dimension; MNE reads volts, eeg-prep takes the numbers as µV."""
    from edfio import Edf, EdfSignal

    from processor import sources as S
    x = 50.0 * np.sin(2 * np.pi * 10 * np.arange(2560) / 256.0)
    Edf([EdfSignal(x, 256.0, label=n, physical_dimension="", physical_range=(-500, 500)) for n in ("C3", "C4")]
        ).write(tmp_path / "nounit.edf")
    c = S.describe(tmp_path / "nounit.edf", "edf")
    src, notes = S.open_source(c)
    try:
        assert src.unitless == ["C3", "C4"] and notes
        assert abs(np.max(src.get([0], 0, 2560)) - 50.0) < 0.1
    finally:
        src.close()


def test_loose_file_ids(tmp_path):
    from processor import sources as S
    assert S.describe(tmp_path / "chb01_03.edf", "edf").recording_id == "sub-chb01_task-eeg_run-03"
    assert S.describe(tmp_path / "patient02.edf", "edf").recording_id == "sub-patient02_task-eeg"
