from typing import Optional

import librosa
import numpy as np


def extract_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    include_deltas: bool = True,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    feats = [mfcc]
    if include_deltas:
        feats.append(librosa.feature.delta(mfcc))
        feats.append(librosa.feature.delta(mfcc, order=2))

    stacked = np.vstack(feats)
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1)

    return np.concatenate([mean, std], axis=0)
