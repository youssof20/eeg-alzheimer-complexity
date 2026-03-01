"""
Unified loading: from preprocessed .npz or from raw .set/.edf.
"""
import os
from pathlib import Path

from .preprocessing import load_preprocessed_npz, preprocess_pipeline

TARGET_CHANNELS = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
GROUP_LABELS = {"C": "Healthy", "A": "Alzheimer's", "F": "FTD"}
GROUP_ORDER = ["Healthy", "FTD", "Alzheimer's"]


def load_processed_folder(processed_dir: str):
    """
    Load all .npz files from data/processed/ and return list of dicts with
    keys: data, ch_names, sfreq, subject_id, group (display name).
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.is_dir():
        return []
    results = []
    for p in sorted(processed_dir.glob("*.npz")):
        data, ch_names, sfreq, subject_id, group = load_preprocessed_npz(str(p))
        group_display = GROUP_LABELS.get(group, group)
        results.append({
            "data": data,
            "ch_names": ch_names,
            "sfreq": sfreq,
            "subject_id": subject_id,
            "group": group_display,
            "group_code": group,
        })
    return results


def load_participants_tsv(tsv_path: str):
    """Load participants.tsv and return dict participant_id -> group (A/C/F)."""
    path = Path(tsv_path)
    if not path.is_file():
        return {}
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns=lambda c: c.strip())
    if "participant_id" not in df.columns or "Group" not in df.columns:
        return {}
    return dict(zip(df["participant_id"].str.strip(), df["Group"].str.strip()))


def load_raw_and_preprocess(file_path: str, subject_id: str = None, group: str = None):
    """
    Load a single .set or .edf, run preprocessing, return same structure as one
    entry from load_processed_folder: dict with data, ch_names, sfreq, subject_id, group.
    """
    data, ch_names, sfreq = preprocess_pipeline(file_path, apply_ica=True)
    if data is None or len(ch_names) == 0:
        return None
    subject_id = subject_id or os.path.splitext(os.path.basename(file_path))[0]
    group = group or "Unknown"
    group_display = GROUP_LABELS.get(group, group)
    return {
        "data": data,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "subject_id": subject_id,
        "group": group_display,
        "group_code": group,
    }
