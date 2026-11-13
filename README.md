# eeg-prep

**eeg-prep is built on [MNE-Python](https://github.com/mne-tools/mne-python)** (developed by
Alexandre Gramfort and the MNE community; [Gramfort et al. 2013](https://doi.org/10.3389/fnins.2013.00267))
and **[MNE-BIDS](https://github.com/mne-tools/mne-bids)** (Stefan Appelhoff and colleagues;
[Appelhoff et al. 2019](https://doi.org/10.21105/joss.01896)) for reading EEG, on
**[pyprep](https://github.com/sappelhoff/pyprep)** (Stefan Appelhoff and the PyPREP developers),
the Python implementation of the **PREP pipeline** of Nima Bigdely-Shamlo, Tim Mullen, Christian
Kothe, Kyung-Min Su and Kay Robbins ([Bigdely-Shamlo et al. 2015](https://doi.org/10.3389/fninf.2015.00016)),
for bad channels, on **[ICLabel](https://github.com/sccn/ICLabel)**, the independent-component
classifier of Luca Pion-Tonachini, Ken Kreutz-Delgado and Scott Makeig at the Swartz Center for
Computational Neuroscience, UC San Diego ([Pion-Tonachini et al. 2019](https://doi.org/10.1016/j.neuroimage.2019.05.026)),
run through **[mne-icalabel](https://github.com/mne-tools/mne-icalabel)** (Adam Li, Jacob Feitelberg,
Anand Prakash Saini, Richard Höchenberger and Mathieu Scheltienne; [Li et al. 2022](https://doi.org/10.21105/joss.04484)),
and, optionally, on **Artifact Subspace Reconstruction** (Tim Mullen, Christian Kothe and colleagues;
[Mullen et al. 2015](https://doi.org/10.1109/TBME.2015.2481482)) as implemented in
**[python-meegkit](https://github.com/nbara/python-meegkit)** by Nicolas Barascud and contributors.
EDF files are written in the EDF+ format; test files are made with [edfio](https://github.com/the-siesta-group/edfio)
(The Siesta Group). **Pennsieve packages these methods into an app; it did not develop them.**
The demo data are the Siena Scalp EEG Database of Paolo Detti and colleagues at the University of
Siena ([Detti 2020](https://doi.org/10.13026/5d4a-j060)) in the BIDS conversion by Jonathan Dan
and Paolo Detti (EPFL; [doi:10.5281/zenodo.10640762](https://doi.org/10.5281/zenodo.10640762)),
the CHB-MIT Scalp EEG Database of Ali Shoeb and John Guttag (MIT, with Children's Hospital Boston;
[doi:10.13026/C2K01R](https://doi.org/10.13026/C2K01R)), and OpenNeuro ds003555 of Dorottya
Cserpan, Johannes Sarnthein and colleagues at the University Hospital Zurich
([doi:10.18112/openneuro.ds003555.v1.0.1](https://doi.org/10.18112/openneuro.ds003555.v1.0.1)).

## What it does

The first step of both scalp EEG workflows (*Scalp EEG Interictal Biomarkers* and *Scalp EEG
Seizure Detection & Monitoring*): it turns clinical scalp EEG as a hospital exports it, or any
BIDS-EEG dataset, into one clean, uniform **BIDS-EEG derivative** that every later step
(eeg-spectral, eeg-spikes, eeg-seizures, eeg-sleep, eeg-connectivity, eeg-report) reads.

**For an epileptologist.** Every recording comes out with the same channel names (10-20 / 10-10:
`T7` for `T3`, `P7` for `T5`), in average reference (or linked ears, or as recorded), with mains
interference and slow drift filtered out, with eye blinks and eye movements, scalp muscle and ECG
artifact removed by ICA where the classifier is confident, broken electrodes marked bad (and left
out of the average), stretches of the recording that are still unreadable (electrode pops,
movement, very high amplitude, flat signal) marked as bad segments, and the seizures annotated in
the source (BIDS events, EDF+ annotations, CHB-MIT summary files, or your own `labels.csv`)
carried along. A QC figure per recording shows what was done. Bipolar-only exports (such as
CHB-MIT) cannot be turned back into a referential montage: they stay bipolar, with each pair's two
electrodes listed. **Nothing in this step is a clinical finding**; the bad-channel, artifact and
bad-segment decisions are automatic and should be checked on the QC figures.

**For a researcher.** Readers: MNE-Python (EDF/EDF+, BDF, BrainVision, EEGLAB, FIF, Nihon Kohden,
Natus/Nicolet) and MNE-BIDS. Channels: non-EEG channels (ECG, EMG, EOG, triggers, annotation and
empty channels) are dropped and listed; names are canonicalised to MNE's 10-05 spelling and given
template positions (MNE `standard_1005`, called `colin27_1005` since MNE 1.13). Bad channels:
pyprep `NoisyChannels` criteria on up to three QC segments with annotated seizures left out
(details below). Filters: zero-phase IIR notch at the mains frequency and harmonics, Butterworth
high-pass 0.5 Hz, optional low-pass and resampling, applied to padded windows so the result
equals filtering the whole recording. Re-reference: average of the good channels. Artifacts: ICA
(extended Infomax, or Picard) fitted on a 300 s sample, components labelled by ICLabel as eye
blink, muscle, heart, line noise or channel noise with probability >= 0.8 removed; optional ASR.
Bad segments mark *technically* unreadable stretches only: high amplitude on most channels at
once (movement) or extreme amplitude on any channel (saturation, electrode pops), flat signal (MNE
`annotate_amplitude`), clear muscle bursts (MNE `annotate_muscle_zscore`, z >= 7); spikes and
seizure discharges are not bad segments, and annotated seizures are never covered by one. Output: EDF+ written incrementally (16-bit, per-channel range from the
data), BIDS sidecars, `recordings.csv`, QC table and figures. Recordings are processed one at a
time and read in windows, so a 24 h recording does not need 24 h in memory.

## Ports

| Port | Direction | What |
|---|---|---|
| `eeg` | input (subfolder) | BIDS-EEG dataset or loose EEG files (see above), optional `labels.csv`, CHB-MIT `*-summary.txt`. Found recursively, following symlinks. |
| `eeg` | output | the BIDS-EEG derivative (below) |
| `qc` | output | `eeg-prep/qc.csv`, `eeg-prep/<recording_id>_qc.png`, `eeg-prep/overview_qc.png`, `eeg-prep/summary.json`, `eeg-prep/provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `RECORDINGS` | `*` | Comma-separated globs on recording ids, file names or paths |
| `LABELS_FILE` | `labels.csv` | Name of the optional labels file anywhere in the input |
| `REFERENCE` | `average` | `average` (good EEG channels), `linked-ears` (A1/A2 or M1/M2; falls back to average with a note), `as-recorded`. Bipolar-only recordings stay bipolar |
| `LINE_FREQ` | `auto` | Sidecar `PowerLineFrequency`, else the file, else detected (50/60 Hz) from the spectrum; or `50`, `60`, `0` (no notch) |
| `HIGHPASS` | `0.5` | Hz, Butterworth order 4, zero phase; 0 = none |
| `LOWPASS` | `0` | Hz; 0 = none (full band for spikes and spectra) |
| `RESAMPLE_HZ` | `0` | e.g. 256 for seizure models; 0 = keep |
| `ARTIFACTS` | `iclabel` | `iclabel`, `asr`, `both` (ASR then ICA), `none` |
| `ICLABEL_THRESHOLD` | `0.8` | ICLabel probability for removing a component |
| `ICA_METHOD` | `infomax` | `infomax` (extended; what ICLabel was trained on) or `picard` |
| `ICA_MAX_S` | `300` | Seconds of data (10 s chunks spread over the recording) the ICA and ASR are fitted on |
| `ICA_COMPONENTS` | `0` | 0 = good channels - 1 |
| `ASR_CUTOFF` | `20` | ASR threshold in SD (Chang et al. 2020) |
| `BAD_CHANNELS` | `pyprep` | `pyprep`, `simple` (robust z of line length / amplitude, flat, line noise), `none` |
| `RANSAC` | `true` | Use pyprep's RANSAC criterion |
| `QC_SEGMENT_S` | `300` | Length of each bad-channel QC segment (up to 3) |
| `BAD_Z` | `5` | Robust z threshold of the simple criteria |
| `AMPLITUDE_UV` | `500` | Bad segment (+-0.5 s) where at least `AMPLITUDE_CHANNELS` of the good channels exceed this at once; 0 = off |
| `AMPLITUDE_CHANNELS` | `0.5` | Fraction of good channels for the amplitude criterion |
| `EXTREME_UV` | `2000` | Any good channel above this marks a bad segment (saturation, electrode pop); 0 = off |
| `FLAT_UV`, `FLAT_S` | `0.1`, `5` | A good channel changing less than 0.1 µV per sample for 5 s marks a bad segment |
| `MUSCLE_Z` | `7` | MNE muscle z-score threshold (70 Hz and up, z within each window); 0 = off |
| `RECORDING_KIND` | `auto` | `recordings.csv` `kind`: auto (sleep from the task/sidecar, else routine <= 2 h / long-term), or forced |
| `PROTECT_ANNOTATED_SEIZURES` | `true` | `true`: annotated seizures are never bad segments and are left out of the bad-channel check and the ICA/ASR calibration sample. `false`: annotations are still read and written (events.tsv, recordings.csv) but processing ignores them, as on an unannotated recording; use it to evaluate a seizure detector that excludes bad segments without the bias of protected annotations |
| `WINDOW_S` | `300` | Processing window |
| `SEED` | `42` | ICA and RANSAC seed |
| `THREADS` | `0` | BLAS threads; 0 = from the cgroup CPU quota |
| `VERIFY_EDF` | `true` | Read every EDF back and check it (half a quantisation step) |

## Outputs

`eeg/` follows `workflow-scalp-eeg/docs/conventions.md`:

- `recordings.csv`: `recording_id, subject, session, task, run, path, sfreq, n_channels, n_good,
  duration_s, montage` (`average` | `linked-ears` | `as-recorded` | `bipolar-longitudinal`),
  `reference` (`average`, `A1|A2`, the recorded reference, or `bipolar`), `kind`, `start_time`,
  `n_seizures`, `source_format`, `age`, `sex`, then recording-level columns of `labels.csv`.
  Rewritten after every recording, so it always lists exactly the recordings written so far.
- `participants.tsv`, `dataset_description.json` (derivative, `GeneratedBy` eeg-prep), `provenance.json`.
- per recording, `sub-<label>/[ses-<label>/]eeg/<stem>_desc-preproc_*`:
  - `eeg.edf`: EDF+ in µV, 16-bit with a symmetric per-channel range of 1.1 x the robust maximum,
    EDF+ annotations for the events below;
  - `channels.tsv`: `name, type, units, low_cutoff, high_cutoff, sampling_frequency, status,
    status_description, contacts, hemisphere, region, x, y, z, original_name`. Cutoffs describe the
    derivative's own filtering (what eeg-prep applied, bounded by Nyquist); filters a source only
    declares are kept in `eeg.json` (`SourceSoftwareFilters`, `SourceDeclaredFilters`). `hemisphere`
    is empty for a bipolar pair across the midline (FT9-FT10);
  - `events.tsv`: `onset, duration, trial_type, source, channels, hed`: `seizure` (HED-SCORE type in
    `hed`), `artifact`, `eyes_open`, `eyes_closed`, `photic`, `hyperventilation`, `sleep`, `wake`,
    `lights_off`, `lights_on`, and `bad_segment` (source `eeg-prep`);
  - `electrodes.tsv` + `coordsystem.json`: template positions (not measured);
  - `eeg.json`: sidecar with the filters, ICA/ASR settings and the reference.

`qc/eeg-prep/qc.csv`: `recording_id, item, name, original_name, type, status, reason,
iclabel_label, iclabel_probability, robust_sd_uv, line_noise_ratio`, one row per kept channel
(good/bad and why; a channel flagged only by the weak criteria is `good` with "suspect only: ..."),
per dropped channel, per ICA component (removed/kept with its ICLabel probabilities), plus the
bad-segment total and the ASR share.

### Bad channels in detail

On referential recordings with template positions, pyprep's criteria run on up to three QC
segments (300 s, spread over the recording, annotated seizures +-10 s cut out), after a notch
filter and a robust (median) reference, as PREP does before its robust average reference:
NaN/flat, deviation (robust amplitude z > 5), high-frequency noise (z > 5), low SNR, dropouts,
low correlation with every other channel, and RANSAC. Two changes for clinical scalp EEG:

1. pyprep's correlation criterion flags a channel below 0.4 correlation in **1 %** of 1-s windows;
   eeg-prep uses **20 %**, since transient single-channel artifacts are common and are handled as
   bad segments.
2. Low correlation and RANSAC also respond to *focal activity* (frequent spikes, focal slowing,
   temporal muscle), which on epilepsy EEG is signal, not a broken electrode. So a channel is bad
   when a technical criterion (flat, NaN, dropout, deviation, HF noise, SNR) fails in most QC
   segments, or when correlation/RANSAC alone fail in **every** segment (at least two). Otherwise
   it stays good and qc.csv notes "suspect only".

PREP's new PSD criterion (pyprep's `find_bad_by_PSD`) is not used: it flags frontal channels with
eye movements and temporal channels with muscle. Bad channels are excluded from the average
reference and from ICA, are still written (status `bad`), and are not interpolated.

## Runtime

Chosen: **2 vCPU (`cpu: 2048`), 4 GiB (`memory: 4096`), timeout 12 h**. Measured in Docker with
`--cpus 2 --memory 3g`, **linux/amd64 emulated on an M-series Mac** (native amd64 is faster), defaults
(`ARTIFACTS=iclabel`, pyprep with RANSAC), peak memory from the container's `memory.peak`:

| Input | Recording | Wall time | Peak memory |
|---|---|---|---|
| Siena sub-00 run-00 + sub-13 run-00 | 44 min + 2.6 h, 19 ch, 256 Hz | 48 s (whole run) | 777 MiB |
| Siena sub-01 run-00 (longest Siena recording) | 13.5 h, 19 ch, 256 Hz | 72 s | 1.77 GiB |
| ds003555 sub-01 | 3 h, 23 ch, 1024 Hz | 201 s | 2.13 GiB |

Time grows with the number of samples (about 5 s per hour of 19-channel 256 Hz EEG, ICA fit
included); memory with the samples of one channel (the per-channel EDF range is computed on a whole
channel) plus the disk-backed array of the cleaned recording in `TMPDIR`, whose pages the kernel
counts while they are being written. A 24 h, 25-channel, 512 Hz recording was not measured; 4 GiB
is the margin for it. Scratch disk: 4 bytes per sample of the output (24 h, 25 ch, 512 Hz = 4.4 GB),
plus the EDF output (2 bytes per sample). The whole Siena demo (41 recordings, 141 h) took 463 s
natively (arm64, 2 BLAS threads, no Docker), CHB-MIT chb01 (42 x 1 h) 118 s, ds003555 (3 x 3 h at
1024 Hz) 181 s. The image is 796 MB (200 MB compressed). Lambda: possible for routine EEG within
15 minutes and 10 GB of `/tmp` (the image is Lambda-safe: tested with a read-only root, uid 65534,
HOME and TMPDIR under `/tmp`).

## Limitations

- Automatic decisions to review, not clinical findings: a channel can be marked bad because it is
  noisy throughout. Annotated seizures (up to 300 s after an onset without a known end) are cut out
  of bad segments, but an *unannotated* seizure with myogenic fast activity can still be marked as
  muscle; eeg-seizures keeps bad segments by default.
- ICLabel was trained on lab EEG (mostly 32+ channels, extended Infomax); on 19-channel clinical
  EEG its components are coarser. Components it labels "muscle" may carry temporal ictal or
  interictal activity mixed with muscle; set `ARTIFACTS=none` or raise `ICLABEL_THRESHOLD` when
  in doubt and compare the QC figures. Bipolar-only recordings get no ICA.
- Bad channels are found on up to three 5-minute segments: an electrode that fails later in a
  24 h recording is caught as bad segments, not as a bad channel.
- Electrode positions are a template (MNE `standard_1005`), not measured; T1/T2, Sp1/Sp2 and
  other non-10-05 positions have no position.
- Source filters declared in headers or sidecars are not trusted (they are often display
  settings); the derivative's cutoffs describe eeg-prep's own filters. EDF signals with a blank
  physical dimension are taken as µV (noted in qc.csv and provenance.json).
- The muscle detector looks at 70 Hz and above, so it is off for data sampled below ~200 Hz or
  low-passed below 80 Hz.

## Licences

eeg-prep: Apache-2.0. Bundled components (THIRD_PARTY.md): MNE-Python, MNE-BIDS, mne-icalabel,
python-meegkit, python-picard (BSD-3-Clause), pyprep (MIT), ICLabel weights (BSD-2-Clause
upstream, shipped in mne-icalabel under BSD-3-Clause), ONNX Runtime (MIT), edfio (Apache-2.0).
The input data keep their own licences.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

- Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T,
  Parkkonen L, Hämäläinen M. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*
  2013;7:267. doi:10.3389/fnins.2013.00267, PMID: 24431986
- Appelhoff S, Sanderson M, Brooks T, van Vliet M, Quentin R, Holdgraf C, Chaumon M, Mikulan E,
  Tavabi K, Höchenberger R, Welke D, Brunner C, Rockhill A, Larson E, Gramfort A, Jas M. MNE-BIDS:
  Organizing electrophysiological data into the BIDS format and facilitating their analysis.
  *Journal of Open Source Software* 2019;4(44):1896. doi:10.21105/joss.01896, PMID: 35990374
- Pernet CR, Appelhoff S, Gorgolewski KJ, Flandin G, Phillips C, Delorme A, Oostenveld R. EEG-BIDS,
  an extension to the brain imaging data structure for electroencephalography. *Scientific Data*
  2019;6:103. doi:10.1038/s41597-019-0104-8, PMID: 31239435
- Bigdely-Shamlo N, Mullen T, Kothe C, Su KM, Robbins KA. The PREP pipeline: standardized
  preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics* 2015;9:16.
  doi:10.3389/fninf.2015.00016, PMID: 26150785
- Appelhoff S, Hurst AJ, Lawrence A, Li A, Mantilla Ramos YJ, O'Reilly C, et al. PyPREP: A Python
  implementation of the preprocessing pipeline (PREP) for EEG data. Zenodo. doi:10.5281/zenodo.6363575
- Pion-Tonachini L, Kreutz-Delgado K, Makeig S. ICLabel: An automated electroencephalographic
  independent component classifier, dataset, and website. *NeuroImage* 2019;198:181-197.
  doi:10.1016/j.neuroimage.2019.05.026, PMID: 31103785
- Li A, Feitelberg J, Saini AP, Höchenberger R, Scheltienne M. MNE-ICALabel: Automatically
  annotating ICA components with ICLabel in Python. *Journal of Open Source Software* 2022;7(76):4484.
  doi:10.21105/joss.04484
- Lee TW, Girolami M, Sejnowski TJ. Independent Component Analysis Using an Extended Infomax
  Algorithm for Mixed Subgaussian and Supergaussian Sources. *Neural Computation* 1999;11(2):417-441.
  doi:10.1162/089976699300016719, PMID: 9950738
- Ablin P, Cardoso JF, Gramfort A. Faster Independent Component Analysis by Preconditioning With
  Hessian Approximations. *IEEE Transactions on Signal Processing* 2018;66(15):4040-4049.
  doi:10.1109/TSP.2018.2844203 (when `ICA_METHOD=picard`)
- Mullen TR, Kothe CAE, Chi YM, Ojeda A, Kerth T, Makeig S, Jung TP, Cauwenberghs G. Real-time
  neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Transactions on Biomedical
  Engineering* 2015;62(11):2553-2567. doi:10.1109/TBME.2015.2481482, PMID: 26415149 (ASR)
- Chang CY, Hsu SH, Pion-Tonachini L, Jung TP. Evaluation of Artifact Subspace Reconstruction for
  Automatic Artifact Components Removal in Multi-Channel EEG Recordings. *IEEE Transactions on
  Biomedical Engineering* 2020;67(4):1114-1121. doi:10.1109/TBME.2019.2930186, PMID: 31329105 (ASR)
- Barascud N, et al. nbara/python-meegkit: v0.1.1. Zenodo 2021. doi:10.5281/zenodo.5643659 (ASR)
- Dan J, Pale U, Amirshahi A, et al. SzCORE: Seizure Community Open-Source Research Evaluation
  framework for the validation of electroencephalography-based automated seizure detection
  algorithms. *Epilepsia* 2024;66(S3):14-24. doi:10.1111/epi.18113, PMID: 39292446 (the HED-SCORE event format read here)

Demo data:

- Detti P. Siena Scalp EEG Database. PhysioNet 2020. doi:10.13026/5d4a-j060; Detti P, Vatti G,
  Zabalo Manrique de Lara G. EEG Synchronization Analysis for Seizure Prediction: A Study on Data of
  Noninvasive Recordings. *Processes* 2020;8(7):846. doi:10.3390/pr8070846; Dan J, Detti P. BIDS
  Siena Scalp EEG Database. Zenodo 2024. doi:10.5281/zenodo.10640762
- Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet 2010. doi:10.13026/C2K01R; Shoeb AH. Application of
  machine learning to epileptic seizure onset detection and treatment. PhD thesis, Massachusetts
  Institute of Technology, 2009. https://dspace.mit.edu/handle/1721.1/54669
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PC, Mark RG, et al. PhysioBank,
  PhysioToolkit, and PhysioNet. *Circulation* 2000;101(23):e215-e220. doi:10.1161/01.CIR.101.23.e215,
  PMID: 10851218
- Cserpan D, Boran E, Rosch R, Lo Biundo SP, Ramantani G, Sarnthein J. Dataset of EEG recordings of
  pediatric patients with epilepsy based on the 10-20 system. OpenNeuro 2021.
  doi:10.18112/openneuro.ds003555.v1.0.1; Cserpan D, Boran E, Lo Biundo SP, Rosch R, Sarnthein J,
  Ramantani G. Scalp high-frequency oscillation rates are higher in younger children. *Brain
  Communications* 2021;3(2):fcab052. doi:10.1093/braincomms/fcab052, PMID: 33870193

## Acknowledgements

MNE-Python, MNE-BIDS, pyprep and mne-icalabel are community projects of the MNE-tools
organisation. ICLabel was developed at the Swartz Center for Computational Neuroscience, UC San
Diego. PhysioNet asks users of its databases to cite the database and Goldberger et al. 2000; the
Siena database asks for Detti et al. 2020. ds003555 was funded by the Swiss National Science
Foundation (CRSK-3_190895, per its dataset_description.json).

## Development

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python     # synthetic fixture, no Docker
make run PYTHON=.venv/bin/python      # Docker over the fixture into data/output/
```
