from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np


def load_audio_segment(
    audio_path: Path,
    segment_duration: float,
    segment_index: int,
    sr: int,
    overlap_ratio: float,
) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        segment_samples = int(segment_duration * sr)
        hop_samples = max(1, int(segment_samples * (1.0 - overlap_ratio)))

        start = segment_index * hop_samples
        end = start + segment_samples

        if start >= len(y):
            return np.zeros(segment_samples, dtype=np.float32)

        if end > len(y):
            segment = np.zeros(segment_samples, dtype=np.float32)
            segment[: len(y) - start] = y[start:]
        else:
            segment = y[start:end]

        return segment.astype(np.float32)
    except Exception as exc:
        print(f"Error loading {audio_path}: {exc}")
        return None


def count_segments_in_file(
    audio_path: Path,
    segment_duration: float,
    sr: int,
    overlap_ratio: float,
) -> int:
    try:
        duration = librosa.get_duration(path=audio_path)
        hop_seconds = max(1e-6, segment_duration * (1.0 - overlap_ratio))

        if duration < segment_duration:
            return 1

        num_segments = int((duration - segment_duration) / hop_seconds) + 1
        return max(1, num_segments)
    except Exception:
        return 0


def get_all_segments_from_session(
    audio_root: Path,
    session_path: str,
    segment_duration: float,
    sr: int,
    overlap_ratio: float,
) -> List[Tuple[Path, int]]:
    session_dir = audio_root / session_path
    audio_files = sorted(session_dir.glob("*.wav"))
    segments: List[Tuple[Path, int]] = []

    for audio_file in audio_files:
        num_segments = count_segments_in_file(
            audio_file, segment_duration, sr=sr, overlap_ratio=overlap_ratio
        )
        for seg_idx in range(num_segments):
            segments.append((audio_file, seg_idx))

    return segments
