"""
Generate 5–6 preprocessed sample .npz files so the app runs without downloading ds004504.
Uses synthetic EEG-like epochs with group-dependent complexity so the gradient is visible.
"""
import os
import sys
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import TARGET_CHANNELS, epochs_to_npz_dict

SFREQ = 500.0
EPOCH_LEN_SAMPLES = int(4.0 * SFREQ)  # 2000
N_EPOCHS = 40
N_CHANS = len(TARGET_CHANNELS)


def _make_epoch(complexity: float, rng: np.random.Generator):
    """
    One epoch: more complexity -> more irregular/higher entropy.
    complexity in (0, 1); higher = more random/less periodic.
    """
    # Base: mix of oscillations and noise
    t = np.arange(EPOCH_LEN_SAMPLES) / SFREQ
    # Low-freq + alpha-ish
    periodic = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    # Noise scaled by complexity
    noise = rng.standard_normal(EPOCH_LEN_SAMPLES) * (0.3 + 0.7 * complexity)
    return (periodic + noise).astype(np.float64)


def generate_subject(subject_id: str, group: str, rng: np.random.Generator):
    """
    Generate epochs for one subject. Group affects mean complexity so that
    Healthy > FTD > Alzheimer's (complexity decreases).
    """
    if group == "Healthy" or group == "C":
        mean_complexity = 0.75
    elif group == "FTD" or group == "F":
        mean_complexity = 0.5
    else:
        mean_complexity = 0.25  # Alzheimer's
    # Slight per-epoch variation
    complexities = mean_complexity + rng.uniform(-0.15, 0.15, N_EPOCHS)
    complexities = np.clip(complexities, 0.05, 0.95)
    data = np.stack([_make_epoch(c, rng) for c in complexities], axis=0)
    # Shape (n_epochs, n_times) -> need (n_epochs, n_chans, n_times); replicate across chans with small variation
    data = data[:, np.newaxis, :] + rng.uniform(-0.05, 0.05, (N_EPOCHS, N_CHANS, EPOCH_LEN_SAMPLES))
    group_code = "C" if group in ("Healthy", "C") else "F" if group in ("FTD", "F") else "A"
    return data, group_code


def main():
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # 2 Healthy, 2 FTD, 2 Alzheimer's
    subjects = [
        ("sub-037", "C"),
        ("sub-038", "C"),
        ("sub-066", "F"),
        ("sub-067", "F"),
        ("sub-001", "A"),
        ("sub-002", "A"),
    ]
    for subject_id, group_code in subjects:
        group_name = "Healthy" if group_code == "C" else "FTD" if group_code == "F" else "Alzheimer's"
        data, _ = generate_subject(subject_id, group_name, rng)
        d = epochs_to_npz_dict(data, TARGET_CHANNELS, SFREQ, subject_id, group_code)
        np.savez_compressed(out_dir / f"{subject_id}.npz", **d)
        print(f"Wrote {out_dir / (subject_id + '.npz')}")
    # participants.tsv for processed subset
    tsv_path = ROOT / "data" / "participants.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w") as f:
        f.write("participant_id\tGroup\n")
        for sid, g in subjects:
            f.write(f"{sid}\t{g}\n")
    print(f"Wrote {tsv_path}")


if __name__ == "__main__":
    main()
