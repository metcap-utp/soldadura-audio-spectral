"""Extracción de características MFCC con caching."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import librosa
import numpy as np


def extract_mfcc_features(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    include_deltas: bool = True,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Extrae características MFCC de un audio.
    
    Returns:
        Vector de características: [mean, std] de MFCC + deltas
    """
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


def compute_features_hash(
    audio_paths: List[str],
    segment_indices: List[int],
    segment_duration: float,
    overlap_ratio: float,
    n_mfcc: int = 40,
) -> str:
    """Calcula hash para invalidación de caché."""
    data_str = f"dur={segment_duration}|overlap={overlap_ratio}|n_mfcc={n_mfcc}|" + \
               "".join([f"{p}:{s}" for p, s in zip(audio_paths, segment_indices)])
    return hashlib.md5(data_str.encode()).hexdigest()


def get_cache_path(
    duration_dir: Path,
    segment_duration: float,
    overlap_ratio: float,
) -> Path:
    """Obtiene ruta de caché de features."""
    cache_dir = duration_dir / "mfcc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"mfcc_features_{segment_duration}s_overlap_{overlap_ratio}.pkl"


def save_features_cache(
    features: List[np.ndarray],
    audio_paths: List[str],
    segment_indices: List[int],
    labels: Dict[str, List],
    duration_dir: Path,
    segment_duration: float,
    overlap_ratio: float,
    n_mfcc: int = 40,
):
    """Guarda features en caché."""
    cache_path = get_cache_path(duration_dir, segment_duration, overlap_ratio)
    
    cache_data = {
        "hash": compute_features_hash(audio_paths, segment_indices, segment_duration, overlap_ratio, n_mfcc),
        "features": features,
        "audio_paths": audio_paths,
        "segment_indices": segment_indices,
        "labels": labels,
        "segment_duration": segment_duration,
        "overlap_ratio": overlap_ratio,
        "n_mfcc": n_mfcc,
        "num_samples": len(features),
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    
    print(f"  [CACHE] Guardados {len(features)} features en {cache_path}")


def load_features_cache(
    audio_paths: List[str],
    segment_indices: List[int],
    duration_dir: Path,
    segment_duration: float,
    overlap_ratio: float,
    n_mfcc: int = 40,
) -> Tuple[Optional[Dict], bool]:
    """Carga features del caché si es válido."""
    cache_path = get_cache_path(duration_dir, segment_duration, overlap_ratio)
    
    if not cache_path.exists():
        return None, False
    
    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        
        # Verificar hash
        current_hash = compute_features_hash(audio_paths, segment_indices, segment_duration, overlap_ratio, n_mfcc)
        if cache_data.get("hash") != current_hash:
            print("  [CACHE] Hash no coincide, regenerando features...")
            return None, False
        
        print(f"  [CACHE] Cargados {len(cache_data['features'])} features desde {cache_path}")
        return cache_data, True
        
    except Exception as e:
        print(f"  [CACHE] Error leyendo caché: {e}")
        return None, False
