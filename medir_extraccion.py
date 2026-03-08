#!/usr/bin/env python3
"""
Mide el tiempo de extracción de features espectrales (MFCC) para cada duración.
Genera embeddings_extraction_times.json con los tiempos medidos.

Uso:
    python medir_extraccion.py
    python medir_extraccion.py --duraciones 10 20
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from weld_audio_classifier.features import extract_mfcc_features
from utils.audio_utils import AUDIO_BASE_DIR

warnings.filterwarnings("ignore")

N_MFCC = 40
OVERLAP_RATIO = 0.5
DURACIONES = [1, 2, 5, 10, 20, 30, 50]
PROJECT_DIR = Path(__file__).parent
OUTPUT_FILE = PROJECT_DIR / "embeddings_extraction_times.json"


def extract_segment_features(audio_path, segment_idx, duration, overlap, sr=16000):
    """Carga un segmento de audio y extrae MFCC features."""
    full_path = AUDIO_BASE_DIR / audio_path
    y, _ = librosa.load(str(full_path), sr=sr)

    hop = int(duration * (1 - overlap) * sr)
    samples = int(duration * sr)
    start = segment_idx * hop

    if start + samples > len(y):
        segment = np.zeros(samples)
        available = len(y) - start
        if available > 0:
            segment[:available] = y[start : start + available]
    else:
        segment = y[start : start + samples]

    return extract_mfcc_features(segment, sr=sr, n_mfcc=N_MFCC)


def main():
    parser = argparse.ArgumentParser(description="Medir tiempo de extracción Spectral/MFCC")
    parser.add_argument(
        "--duraciones",
        type=int,
        nargs="+",
        default=DURACIONES,
        help="Duraciones a medir (default: todas)",
    )
    args = parser.parse_args()

    # Cargar resultados existentes si hay
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
    else:
        results = []

    measured = {r["segment_duration"] for r in results}

    for duration in args.duraciones:
        dur_key = f"{duration:02d}seg" if duration < 10 else f"{duration}seg"
        csv_path = PROJECT_DIR / dur_key / "completo.csv"

        if not csv_path.exists():
            print(f"[SKIP] {dur_key}: No existe {csv_path}")
            continue

        if duration in measured:
            print(f"[SKIP] {dur_key}: Ya medido previamente")
            continue

        df = pd.read_csv(csv_path)
        n_segments = len(df)
        overlap_seconds = duration * OVERLAP_RATIO

        print(f"{'='*60}")
        print(f"Duración: {duration}s ({n_segments} segmentos)")
        print(f"{'='*60}")

        start_time = time.perf_counter()

        for i, (_, row) in enumerate(df.iterrows()):
            if i % 200 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Procesando {i}/{n_segments}... ({elapsed:.1f}s)")
            extract_segment_features(
                row["audio_path"],
                int(row["segment_index"]),
                duration,
                OVERLAP_RATIO,
            )

        extraction_time = round(time.perf_counter() - start_time, 2)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "segment_duration": duration,
            "overlap_ratio": OVERLAP_RATIO,
            "overlap_seconds": overlap_seconds,
            "num_segments": n_segments,
            "num_embeddings": n_segments,
            "extraction_time_seconds": extraction_time,
            "extraction_time_minutes": round(extraction_time / 60, 2),
        }
        results.append(entry)

        # Guardar después de cada duración
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  -> {extraction_time}s ({extraction_time/60:.2f}min)")
        print(f"  Guardado en {OUTPUT_FILE}\n")

    print("\nResultados finales:")
    for r in sorted(results, key=lambda x: x["segment_duration"]):
        print(
            f"  {r['segment_duration']:2d}s: {r['extraction_time_seconds']:.2f}s "
            f"({r['extraction_time_minutes']:.2f}min) - {r['num_segments']} segmentos"
        )


if __name__ == "__main__":
    main()
