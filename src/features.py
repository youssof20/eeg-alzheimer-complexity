"""
Complexity features: Sample Entropy (antropy), Higuchi FD and DFA from scratch in NumPy.
"""
import numpy as np

try:
    import antropy as ant
    ANTORPY_AVAILABLE = True
except ImportError:
    ANTORPY_AVAILABLE = False


def sample_entropy(x: np.ndarray, order: int = 2, tolerance: float = None) -> float:
    """Sample Entropy via antropy. x: 1D array."""
    if not ANTORPY_AVAILABLE:
        return np.nan
    try:
        out = ant.sample_entropy(x, order=order, tolerance=tolerance)
        return float(out) if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """
    Higuchi Fractal Dimension from scratch (NumPy).
    x: 1D time series. Returns HFD (typically 1–2).
    """
    n = len(x)
    if n < 2 * kmax:
        kmax = max(1, n // 2)
    lk = np.zeros(kmax)
    for k in range(1, kmax + 1):
        lm = np.zeros(k)
        for m in range(k):
            ll = 0.0
            n_max = (n - m - 1) // k
            if n_max < 2:
                continue
            for i in range(1, n_max):
                ll += abs(x[m + i * k] - x[m + (i - 1) * k])
            ll *= (n - 1) / (n_max * k * k)
            lm[m] = ll
        lk[k - 1] = np.mean(lm)
    # log(L(k)) vs log(1/k); slope = HFD
    x_log = np.log(1.0 / np.arange(1, kmax + 1))
    y_log = np.log(lk + 1e-12)
    slope = np.polyfit(x_log, y_log, 1)[0]
    return float(slope)


def dfa_alpha(x: np.ndarray, min_scale: int = 4, max_scale: int = None) -> float:
    """
    Detrended Fluctuation Analysis from scratch (NumPy).
    Returns DFA exponent alpha (Hurst-like).
    """
    n = len(x)
    if max_scale is None:
        max_scale = max(min_scale + 1, n // 4)
    max_scale = min(max_scale, n // 4)
    if max_scale <= min_scale:
        return np.nan
    # Integrate: cumulative sum of mean-centered series
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    fluct = np.zeros(len(scales))
    for si, s in enumerate(scales):
        n_win = len(y) // s
        if n_win < 2:
            continue
        rms_list = []
        for i in range(n_win):
            seg = y[i * s : (i + 1) * s]
            t = np.arange(len(seg))
            coef = np.polyfit(t, seg, 1)
            fit = np.polyval(coef, t)
            rms_list.append(np.sqrt(np.mean((seg - fit) ** 2)))
        fluct[si] = np.mean(rms_list)
    # Remove any zero/NaN
    valid = np.isfinite(fluct) & (fluct > 1e-12)
    if np.sum(valid) < 2:
        return np.nan
    scales_v = scales[valid]
    fluct_v = fluct[valid]
    log_s = np.log(scales_v)
    log_f = np.log(fluct_v)
    alpha = np.polyfit(log_s, log_f, 1)[0]
    return float(alpha)


def compute_epoch_features(epoch_1d: np.ndarray) -> dict:
    """Compute SampEn, HFD, DFA for one 1D epoch. Returns dict."""
    return {
        "SampEn": sample_entropy(epoch_1d),
        "HFD": higuchi_fd(epoch_1d),
        "DFA": dfa_alpha(epoch_1d),
    }


def extract_subject_features(epochs_3d: np.ndarray, ch_names: list = None) -> dict:
    """
    epochs_3d: (n_epochs, n_chans, n_times).
    Compute per (epoch, channel) then average across channels per epoch, then median across epochs.
    Returns dict with SampEn, HFD, DFA (subject-level).
    """
    n_epochs, n_chans, n_times = epochs_3d.shape
    epoch_sampen = []
    epoch_hfd = []
    epoch_dfa = []
    for ep in range(n_epochs):
        ch_sampen, ch_hfd, ch_dfa = [], [], []
        for ch in range(n_chans):
            x = epochs_3d[ep, ch, :].ravel()
            f = compute_epoch_features(x)
            if not np.isnan(f["SampEn"]):
                ch_sampen.append(f["SampEn"])
            if not np.isnan(f["HFD"]):
                ch_hfd.append(f["HFD"])
            if not np.isnan(f["DFA"]):
                ch_dfa.append(f["DFA"])
        epoch_sampen.append(np.nanmean(ch_sampen) if ch_sampen else np.nan)
        epoch_hfd.append(np.nanmean(ch_hfd) if ch_hfd else np.nan)
        epoch_dfa.append(np.nanmean(ch_dfa) if ch_dfa else np.nan)
    def _safe_median(arr):
        m = np.nanmedian(arr)
        return float(m) if np.isfinite(m) else 0.0

    return {
        "SampEn": _safe_median(epoch_sampen),
        "HFD": _safe_median(epoch_hfd),
        "DFA": _safe_median(epoch_dfa),
    }


def build_features_table(processed_list: list) -> "pd.DataFrame":
    """
    processed_list: list of dicts from data_loader.load_processed_folder (each has 'data', 'subject_id', 'group').
    Returns DataFrame with columns subject_id, group, SampEn, HFD, DFA.
    """
    import pandas as pd
    rows = []
    for rec in processed_list:
        feats = extract_subject_features(rec["data"], rec.get("ch_names"))
        rows.append({
            "subject_id": rec["subject_id"],
            "group": rec["group"],
            "SampEn": feats["SampEn"],
            "HFD": feats["HFD"],
            "DFA": feats["DFA"],
        })
    return pd.DataFrame(rows)
