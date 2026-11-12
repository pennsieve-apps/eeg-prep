# Getting real input for eeg-prep

eeg-prep reads any folder of scalp EEG: a BIDS-EEG dataset, or loose EDF/EDF+/BDF, BrainVision
(.vhdr + .eeg + .vmrk), EEGLAB (.set/.fdt), FIF, Nihon Kohden (.EEG with .PNT/.21E/.LOG) or
Natus/Nicolet (.e) files, with an optional `labels.csv`. Link the folder to the `eeg` input port.

The demo data of the scalp EEG workflows (`workflow-scalp-eeg/scripts/fetch-sample-data.sh`):

- **Siena Scalp EEG Database, BIDS version** (14 adults with focal epilepsy, 41 recordings,
  19 channels in average reference, 256 Hz, 47 seizures as HED-SCORE events):
  Zenodo record 10640762, `BIDS_Siena.zip` (4.5 GB), doi:10.5281/zenodo.10640762. The original is on
  PhysioNet, https://physionet.org/content/siena-scalp-eeg/1.0.0/ (doi:10.13026/5d4a-j060).
  Use the unzipped `BIDS_Siena/` folder as input as is.
- **CHB-MIT Scalp EEG Database** (paediatric, bipolar-only, 256 Hz): one subject folder from
  https://physionet.org/content/chbmit/1.0.0/, e.g. `chb01/` (`chb01_01.edf` ... and
  `chb01-summary.txt`, which eeg-prep reads for the seizure times). Recording ids are made from the
  file names: `chb01_03.edf` -> `sub-chb01_task-eeg_run-03`.
- **OpenNeuro ds003555** (children with epilepsy, 3 h of sleep EEG, 10-20, 1024 Hz):
  https://openneuro.org/datasets/ds003555 (doi:10.18112/openneuro.ds003555.v1.0.1), e.g. `sub-01/`
  to `sub-03/` with the top-level `participants.tsv`. Set `RECORDING_KIND=sleep`. The `derivatives/`
  folder holds the authors' HFO clips and is skipped.

Example `labels.csv` (recording-level columns go into recordings.csv; rows with `onset_s` become events):

```
subject,recording_id,onset_s,duration_s,trial_type,lateralisation
PN06,,,,,left
PN06,sub-PN06_ses-01_task-szMonitoring_run-01,1520,75,seizure,
```
