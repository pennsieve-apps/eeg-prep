# Third-party components

eeg-prep's own code is Apache-2.0 (LICENSE). The Docker image bundles the following
components, installed from PyPI as binary wheels pinned in `requirements.txt`, on the
`python:3.12.16-slim-bookworm` base image (Debian, Python Software Foundation License).

## Methods and model weights

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| MNE-Python | 1.13.2 | MNE-Python developers (Alexandre Gramfort and the MNE community) | BSD-3-Clause | https://github.com/mne-tools/mne-python |
| MNE-BIDS | 0.20.0 | MNE-BIDS developers | BSD-3-Clause | https://github.com/mne-tools/mne-bids |
| pyprep (PyPREP) | 0.10.0 | PyPREP developers (Stefan Appelhoff and contributors) | MIT | https://github.com/sappelhoff/pyprep |
| mne-icalabel | 0.10.0 | MNE developers (Adam Li, Mathieu Scheltienne and contributors) | BSD-3-Clause | https://github.com/mne-tools/mne-icalabel |
| ICLabel network weights (`mne_icalabel/iclabel/network/assets/ICLabelNet.onnx`, SHA-256 `b7dc72bc3fa5960b2843fc6d5ba60add842b0a240c6371598be09b595182c04b`, checked at build time) | shipped in mne-icalabel 0.10.0 | Swartz Center for Computational Neuroscience, UC San Diego (Luca Pion-Tonachini, Ken Kreutz-Delgado, Scott Makeig); ported by mne-icalabel | upstream ICLabel: BSD-2-Clause; redistributed in the mne-icalabel wheel under its BSD-3-Clause | https://github.com/sccn/ICLabel |
| ONNX Runtime | 1.31.1 | Microsoft Corporation | MIT | https://github.com/microsoft/onnxruntime |
| python-meegkit (`meegkit`) | 0.2.0 | Nicolas Barascud and contributors | BSD-3-Clause | https://github.com/nbara/python-meegkit |
| python-picard | 0.8.2 | Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort | BSD-3-Clause | https://github.com/pierreablin/picard |
| edfio | 0.4.18 | The Siesta Group | Apache-2.0 | https://github.com/the-siesta-group/edfio |

The ICLabel `.pt` (PyTorch) copy of the same weights is also in the wheel; it is not used (no PyTorch in the image).
The wheel also contains `mne_icalabel/megnet/assets/megnet.onnx`, the MEGnet weights for MEG
(Treacher et al. 2021); their upstream licence (https://github.com/DeepLearningForPrecisionHealthLab/MegNET_2020,
University of Texas Southwestern Medical Center) permits academic and research use only, so the
Dockerfile deletes the file: it is not in the image and eeg-prep never uses it.

## Libraries

| Component | Version | Licence |
|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (and bundled 0BSD, MIT, Zlib, CC0-1.0 parts) |
| scipy | 1.18.1 | BSD-3-Clause |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based, BSD-compatible) |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| pandas | 3.1.1 | BSD-3-Clause |
| statsmodels | 0.15.0 | BSD-3-Clause |
| pyriemann | 0.12 | BSD-3-Clause |
| psutil | 7.2.2 | BSD-3-Clause |
| pooch | 1.9.0 | BSD-3-Clause |
| joblib 1.6.0, threadpoolctl 3.8.0 | | BSD-3-Clause |
| protobuf 7.36.2 | | BSD-3-Clause |
| flatbuffers 25.12.19 | | Apache-2.0 |
| formulaic 1.2.2, interface_meta 2.0.1, narwhals 2.28.0, patsy 1.0.3, wrapt 2.5.2 | | MIT / BSD-2-Clause (patsy, wrapt) |
| contourpy, cycler, kiwisolver, fonttools, pillow, pyparsing, python-dateutil, lazy-loader, decorator, packaging, platformdirs, tqdm (MPL-2.0 AND MIT), Jinja2, MarkupSafe, requests, urllib3, idna, charset-normalizer, certifi, tzdata, typing_extensions, cloudpickle, array-api-compat, array-api-extra | see requirements.txt | BSD / MIT / Apache-2.0 / MPL-2.0 (certifi) / HPND (pillow) / PSF (typing_extensions) |

## Demo data (not bundled; used for testing and the README's numbers)

| Dataset | Creators | Licence |
|---|---|---|
| Siena Scalp EEG Database (PhysioNet, doi:10.13026/5d4a-j060) and its BIDS conversion (Zenodo, doi:10.5281/zenodo.10640762) | Paolo Detti (University of Siena); BIDS conversion Jonathan Dan, Paolo Detti | PhysioNet: CC BY 4.0; BIDS conversion: ODC-By 1.0 in its dataset_description.json, CC BY 4.0 on the Zenodo record (both allow redistribution with attribution) |
| CHB-MIT Scalp EEG Database (PhysioNet, doi:10.13026/C2K01R) | Ali Shoeb, John Guttag (MIT), Children's Hospital Boston | ODC-By 1.0 |
| OpenNeuro ds003555 v1.0.1 (doi:10.18112/openneuro.ds003555.v1.0.1) | Dorottya Cserpan, Ece Boran, Richard Rosch, San Pietro Lo Biundo, Georgia Ramantani, Johannes Sarnthein | CC0 |
