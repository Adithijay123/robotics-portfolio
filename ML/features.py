"""
features.py
Loads raw UCI HAR sensor data and extracts time-domain + freq-domain features.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Activity labels from UCI HAR
LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_raw_signals(split: str, data_root: str = "./data/UCI HAR Dataset") -> np.ndarray:
    """Load raw inertial signals for 'train' or 'test' split.
    Returns array of shape (n_samples, window_length=128, n_signals=9)
    """
    root = Path(data_root) / split / "Inertial Signals"
    signals = []
    for name in SIGNAL_NAMES:
        path = root / f"{name}_{split}.txt"
        arr = pd.read_csv(path, sep=r"\s+", header=None).values
        signals.append(arr)
    # stack → (n_samples, 9, 128) then transpose → (n_samples, 128, 9)
    return np.stack(signals, axis=1).transpose(0, 2, 1)


def load_labels(split: str, data_root: str = "./data/UCI HAR Dataset") -> np.ndarray:
    """Load activity labels (1-indexed integers)."""
    path = Path(data_root) / split / f"y_{split}.txt"
    return pd.read_csv(path, header=None).values.ravel()


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    From raw windows (n_samples, 128, 9) extract a flat feature vector per sample.
    Features: mean, std, min, max, RMS, peak FFT frequency — per signal channel.
    Returns (n_samples, n_features)
    """
    n_samples, win_len, n_ch = windows.shape
    features = []

    for i in range(n_samples):
        w = windows[i]  # (128, 9)
        row = []
        for ch in range(n_ch):
            sig = w[:, ch]
            row.append(np.mean(sig))
            row.append(np.std(sig))
            row.append(np.min(sig))
            row.append(np.max(sig))
            row.append(np.sqrt(np.mean(sig ** 2)))           # RMS
            fft_mag = np.abs(np.fft.rfft(sig))[1:]           # drop DC
            row.append(np.argmax(fft_mag) / len(fft_mag))    # dominant freq (normalised)
            row.append(np.max(fft_mag))                       # FFT peak magnitude
        features.append(row)

    return np.array(features, dtype=np.float32)


def load_precomputed_features(split: str, data_root: str = "./data/UCI HAR Dataset") -> np.ndarray:
    """
    UCI HAR also ships pre-computed 561-feature vectors.
    We load those as an alternative (already engineered by the dataset authors).
    """
    path = Path(data_root) / split / f"X_{split}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None).values.astype(np.float32)
