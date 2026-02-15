from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from weld_audio_classifier.features import extract_mfcc_features
from weld_audio_classifier.segmenter import load_audio_segment


@dataclass
class FeatureConfig:
    n_mfcc: int = 40
    include_deltas: bool = True
    n_fft: int = 1024
    hop_length: int = 512


def _build_cache(cache_dir: Path) -> joblib.Memory:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return joblib.Memory(location=str(cache_dir), verbose=0)


def load_features(
    csv_path: Path,
    audio_root: Path,
    segment_duration: float,
    overlap_ratio: float,
    sample_rate: int,
    feature_cfg: FeatureConfig,
    cache_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    cache = _build_cache(cache_dir)

    feature_multiplier = 3 if feature_cfg.include_deltas else 1
    feature_size = feature_cfg.n_mfcc * feature_multiplier * 2

    @cache.cache
    def compute_feature(
        audio_path: str,
        segment_index: int,
        segment_duration_value: float,
        overlap_ratio_value: float,
        sample_rate_value: int,
        n_mfcc_value: int,
        include_deltas_value: bool,
        n_fft_value: int,
        hop_length_value: int,
    ) -> np.ndarray:
        full_path = audio_root / audio_path
        segment = load_audio_segment(
            full_path,
            segment_duration=segment_duration_value,
            segment_index=segment_index,
            sr=sample_rate_value,
            overlap_ratio=overlap_ratio_value,
        )
        if segment is None:
            return np.zeros(feature_size)

        return extract_mfcc_features(
            segment,
            sr=sample_rate_value,
            n_mfcc=n_mfcc_value,
            include_deltas=include_deltas_value,
            n_fft=n_fft_value,
            hop_length=hop_length_value,
        )

    features = []
    for _, row in df.iterrows():
        features.append(
            compute_feature(
                row["Audio Path"],
                int(row["Segment Index"]),
                segment_duration,
                overlap_ratio,
                sample_rate,
                feature_cfg.n_mfcc,
                feature_cfg.include_deltas,
                feature_cfg.n_fft,
                feature_cfg.hop_length,
            )
        )

    X = np.vstack(features)

    return X, df


def fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in ["Plate Thickness", "Electrode", "Type of Current"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str).values)
        label_encoders[col] = le
    return label_encoders


def encode_labels(
    df: pd.DataFrame, label_encoders: Dict[str, LabelEncoder]
) -> Dict[str, np.ndarray]:
    y: Dict[str, np.ndarray] = {}
    for col, le in label_encoders.items():
        y[col] = le.transform(df[col].astype(str).values)
    return y
