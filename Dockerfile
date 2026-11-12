FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package). No apt packages, no compiler.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name tests -o -name testing \) -path "*/mne/*" -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/pandas/tests /usr/local/lib/python3.12/site-packages/scipy/*/tests \
              /usr/local/lib/python3.12/site-packages/numpy/*/tests /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/statsmodels/*/tests /usr/local/lib/python3.12/site-packages/sklearn/*/tests \
              /usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data \
    # ICLabel weights ship inside the mne-icalabel wheel (BSD-3; upstream ICLabel BSD-2): check them
    && echo "b7dc72bc3fa5960b2843fc6d5ba60add842b0a240c6371598be09b595182c04b  /usr/local/lib/python3.12/site-packages/mne_icalabel/iclabel/network/assets/ICLabelNet.onnx" | sha256sum -c - \
    # Matplotlib's font cache, built once; copied into $TMPDIR/mpl at start (HOME and /tmp are replaced on Lambda)
    && mkdir -p /opt/eeg-prep && MPLCONFIGDIR=/opt/eeg-prep/mpl python3 -c "import matplotlib.font_manager" \
    && chmod -R a+rX /opt/eeg-prep

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EEG_PREP_MPL_CACHE=/opt/eeg-prep/mpl \
    MNE_DONTWRITE_HOME=true \
    MNE_USE_CUDA=false \
    MNE_LOGGING_LEVEL=ERROR

# Runs as uid 1000 on ECS (and as an arbitrary user on Lambda); nothing here needs root.
RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
