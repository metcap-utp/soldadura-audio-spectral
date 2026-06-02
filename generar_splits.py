"""
Generación de splits estratificados para entrenamiento, validación y test.

Adaptado de vggish-backbone/generar_splits.py para usar MFCC.

Crea conjuntos:
- train.csv: 72% de los datos
- validation.csv: 18% de los datos  
- test.csv: 10% de los datos (evaluación final)

Los splits son estratificados por sesión para evitar data leakage.

Uso:
    python generar_splits.py --duration 5 --overlap 0.5
    python generar_splits.py --duration 10 --overlap 0.0 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Directorio raíz
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from utils.audio_utils import AUDIO_BASE_DIR, get_audio_files
from utils.timing import Timer, timer


RANDOM_SEED = 42
TEST_FRACTION = 0.10  # 10% para evaluación final
VAL_FRACTION = 0.18   # 18% para validación
TRAIN_FRACTION = 0.72  # 72% para entrenamiento


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generación de splits estratificados para SMAW"
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        choices=[1, 2, 5, 10, 20, 30, 50],
        help="Duración de segmento en segundos",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap entre segmentos como ratio (0.0 a 0.75, default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Semilla para reproducibilidad (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def create_session_based_split(audio_files: list, seed: int = RANDOM_SEED):
    """
    Crea splits basados en sesiones (carpetas YYMMDD-HHMMSS_Audio).
    Esto garantiza que todos los segmentos de una misma grabación
    vayan al mismo conjunto, evitando data leakage.
    """
    # Agrupar por sesión
    sessions = {}
    for audio_info in audio_files:
        session = audio_info['sesion']
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(audio_info)
    
    session_names = list(sessions.keys())
    
    # Split sesiones: 10% test, 18% validation, 72% train
    # Primero separar test (10%)
    trainval_sessions, test_sessions = train_test_split(
        session_names,
        test_size=TEST_FRACTION,
        random_state=seed,
    )
    
    # Luego separar validación de train (18% del total = 20% de 90% restante)
    val_fraction_of_remaining = VAL_FRACTION / (1 - TEST_FRACTION)
    train_sessions, val_sessions = train_test_split(
        trainval_sessions,
        test_size=val_fraction_of_remaining,
        random_state=seed,
    )
    
    # Crear listas de archivos
    train_files = []
    val_files = []
    test_files = []
    
    for session in train_sessions:
        train_files.extend(sessions[session])
    
    for session in val_sessions:
        val_files.extend(sessions[session])
    
    for session in test_sessions:
        test_files.extend(sessions[session])
    
    return train_files, val_files, test_files


def generate_segment_csv(audio_files: list, segment_duration: float, overlap_ratio: float):
    """
    Genera un DataFrame con todos los segmentos de los archivos de audio.
    """
    import librosa
    
    rows = []
    
    for audio_info in audio_files:
        audio_path = AUDIO_BASE_DIR / audio_info['path']
        
        try:
            # Cargar audio para saber su duración
            audio, sr = librosa.load(str(audio_path), sr=16000)
            audio_duration = len(audio) / sr
        except Exception as e:
            print(f"Error cargando {audio_path}: {e}")
            continue
        
        # Calcular número de segmentos
        hop_duration = segment_duration * (1 - overlap_ratio)
        num_segments = max(1, int((audio_duration - segment_duration) / hop_duration) + 1)
        
        for seg_idx in range(num_segments):
            rows.append({
                'audio_path': str(audio_info['path']),
                'placa': audio_info['placa'],
                'electrodo': audio_info['electrodo'],
                'corriente': audio_info['corriente'],
                'sesion': audio_info['sesion'],
                'segment_index': seg_idx,
                'segment_duration': segment_duration,
                'overlap_ratio': overlap_ratio,
            })
    
    return pd.DataFrame(rows)


def compute_data_stats(df: pd.DataFrame):
    """Calcula estadísticas del dataset."""
    stats = {
        'total_samples': len(df),
        'unique_sessions': df['sesion'].nunique(),
        'unique_audio_files': df['audio_path'].nunique(),
        'plate_distribution': df['placa'].value_counts().to_dict(),
        'electrode_distribution': df['electrodo'].value_counts().to_dict(),
        'current_distribution': df['corriente'].value_counts().to_dict(),
    }
    return stats


def main():
    args = parse_args()
    total_start = time.time()

    duration = args.duration
    overlap = args.overlap
    seed = args.seed

    print("=" * 60)
    print("GENERACIÓN DE SPLITS ESTRATIFICADOS")
    print("=" * 60)
    print(f"Duración: {duration}s")
    print(f"Overlap: {overlap}")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Set seed
    np.random.seed(seed)

    # Directorio de duración
    duration_dir = ROOT_DIR / f"{duration:02d}seg"
    duration_dir.mkdir(exist_ok=True)

    # Obtener archivos de audio
    print("\nDescubriendo archivos de audio...")
    with Timer("Descubrimiento archivos"):
        audio_files = get_audio_files()
    print(f"Total de archivos: {len(audio_files)}")

    # Crear splits por sesión
    print("\nCreando splits estratificados por sesión...")
    with Timer("Creación splits"):
        train_files, val_files, test_files = create_session_based_split(audio_files, seed)

    print(f"\nSesiones:")
    print(f"  Train: {len(set(f['sesion'] for f in train_files))} sesiones, {len(train_files)} archivos")
    print(f"  Validation: {len(set(f['sesion'] for f in val_files))} sesiones, {len(val_files)} archivos")
    print(f"  Test: {len(set(f['sesion'] for f in test_files))} sesiones, {len(test_files)} archivos")

    # Generar CSVs con segmentos
    print(f"\nGenerando CSVs con segmentos (duración={duration}s, overlap={overlap})...")

    with Timer("Generación CSV Train"):
        train_df = generate_segment_csv(train_files, float(duration), overlap)
    with Timer("Generación CSV Validation"):
        val_df = generate_segment_csv(val_files, float(duration), overlap)
    with Timer("Generación CSV Test"):
        test_df = generate_segment_csv(test_files, float(duration), overlap)
    
    # Guardar CSVs
    train_path = duration_dir / "train.csv"
    val_path = duration_dir / "validation.csv"
    test_path = duration_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nCSVs guardados:")
    print(f"  Train: {train_path} ({len(train_df)} segmentos)")
    print(f"  Validation: {val_path} ({len(val_df)} segmentos)")
    print(f"  Test: {test_path} ({len(test_df)} segmentos)")
    
    # Calcular y guardar estadísticas
    print("\nCalculando estadísticas...")
    
    train_stats = compute_data_stats(train_df)
    val_stats = compute_data_stats(val_df)
    test_stats = compute_data_stats(test_df)
    
    total_time = time.time() - total_start

    stats = {
        'segment_duration': duration,
        'overlap_ratio': overlap,
        'random_seed': seed,
        'timing': {
            'total_seconds': total_time,
            'total_minutes': total_time / 60,
        },
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }

    stats_path = duration_dir / "data_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEstadísticas guardadas: {stats_path}")

    # Crear CSV completo
    complete_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    complete_path = duration_dir / "completo.csv"
    complete_df.to_csv(complete_path, index=False)
    print(f"CSV completo: {complete_path} ({len(complete_df)} segmentos)")

    print("\n" + "=" * 60)
    print("RESUMEN DE TIEMPOS")
    print("=" * 60)
    print(f"  Tiempo total: {stats['timing']['total_seconds']:.2f}s ({stats['timing']['total_minutes']:.2f} min)")

    print("\n" + "=" * 60)
    print("¡Splits generados exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
