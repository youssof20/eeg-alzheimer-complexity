"""
EEG preprocessing: load .set/.edf, band-pass filter, ICA for EOG, epoch, channel subset.
"""
import numpy as np

try:
    import mne
    from mne.preprocessing import ICA
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Channels to keep (10-20)
TARGET_CHANNELS = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
EPOCH_LEN_SEC = 4.0
OVERLAP_RATIO = 0.5  # 50% overlap -> step = 2 s


def load_raw_eeg(path: str, preload: bool = True):
    """Load raw EEG from .set or .edf. Returns MNE Raw or None if MNE not available."""
    if not MNE_AVAILABLE:
        return None
    path = str(path).lower()
    if path.endswith(".set"):
        raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)
    elif path.endswith(".edf"):
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    else:
        raise ValueError("Unsupported format; use .set or .edf")
    return raw


def bandpass_filter(raw, l_freq=0.5, h_freq=45.0):
    """Apply band-pass filter (default 0.5–45 Hz)."""
    if not MNE_AVAILABLE:
        return raw
    return raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)


def run_ica_for_eog(raw, n_components=15, eog_ch_names=None):
    """Fit ICA and remove EOG-like components. Modifies raw in place if possible."""
    if not MNE_AVAILABLE:
        return raw
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    ica = ICA(n_components=min(n_components, len(picks)), random_state=97, max_iter="auto")
    ica.fit(raw, picks=picks, verbose=False)
    eog_inds = []
    if eog_ch_names:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch_names, verbose=False)
    if not eog_inds:
        eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
    if eog_inds:
        ica.exclude = eog_inds
        ica.apply(raw)
    return raw


def pick_channels(raw, ch_names=None):
    """Restrict to target channels. ch_names defaults to TARGET_CHANNELS."""
    if not MNE_AVAILABLE:
        return raw
    ch_names = ch_names or TARGET_CHANNELS
    available = [c for c in ch_names if c in raw.ch_names]
    if not available:
        return raw
    return raw.pick(available)


def make_epochs(raw, tmin=0.0, tmax=None, overlap_ratio=OVERLAP_RATIO):
    """Create fixed-length epochs with overlap. tmax = epoch length in seconds."""
    if not MNE_AVAILABLE:
        return None
    tmax = tmax or EPOCH_LEN_SEC
    duration = tmax - tmin
    step = duration * (1 - overlap_ratio)
    events = mne.make_fixed_length_events(raw, id=1, duration=step, start=tmin, stop=raw.times[-1] - duration)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


def preprocess_pipeline(
    path: str,
    apply_ica: bool = True,
    n_ica_components: int = 15,
):
    """
    Full pipeline: load -> filter -> (optional ICA) -> pick channels -> epoch.
    Returns (epochs_array, ch_names, sfreq) or (None, [], 0) on failure.
    """
    if not MNE_AVAILABLE:
        return None, [], 0
    raw = load_raw_eeg(path)
    if raw is None:
        return None, [], 0
    raw = bandpass_filter(raw)
    if apply_ica:
        raw = run_ica_for_eog(raw, n_components=n_ica_components)
    raw = pick_channels(raw)
    epochs = make_epochs(raw)
    if epochs is None or len(epochs) == 0:
        return None, [], raw.info["sfreq"]
    data = epochs.get_data()  # (n_epochs, n_chans, n_times)
    ch_names = list(epochs.ch_names)
    sfreq = epochs.info["sfreq"]
    return data, ch_names, sfreq


def epochs_to_npz_dict(data: np.ndarray, ch_names: list, sfreq: float, subject_id: str, group: str):
    """Pack preprocessed epochs and metadata for saving as .npz."""
    return {
        "epochs": data,
        "ch_names": np.array(ch_names, dtype=object),
        "sfreq": np.array(sfreq),
        "subject_id": np.array(subject_id, dtype=object),
        "group": np.array(group, dtype=object),
    }


def load_preprocessed_npz(path: str):
    """Load a preprocessed .npz and return (data, ch_names, sfreq, subject_id, group)."""
    o = np.load(path, allow_pickle=True)
    data = o["epochs"]
    ch_names = o["ch_names"].tolist() if hasattr(o["ch_names"], "tolist") else list(o["ch_names"])
    sfreq = float(o["sfreq"])
    subject_id = str(o["subject_id"]) if np.isscalar(o["subject_id"]) else o["subject_id"].item()
    group = str(o["group"]) if np.isscalar(o["group"]) else o["group"].item()
    return data, ch_names, sfreq, subject_id, group
