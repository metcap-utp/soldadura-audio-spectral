"""
Generación de splits estratificados para entrenamiento, test y blind.

Adaptado de vggish-backbone/generar_splits.py para usar MFCC.

Crea conjuntos:
- train.csv: 72% de los datos
- test.csv: 18% de los datos  
- blind.csv: 10% de los datos (evaluación ciega)

Los splits son estratificados por sesión para evitar data leakage.

Uso:
    python generar_splits.py --duration 5 --overlap 0.5
    python generar_splits.py --duration 10 --overlap 0.0 --seed 42
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Directorio raíz
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from utils.audio_utils import AUDIO_BASE_DIR, get_audio_files


# =============================================================================
# Configuración
# =============================================================================

RANDOM_SEED = 42
BLIND_FRACTION = 0.10  # 10% para evaluación ciega
TEST_FRACTION = 0.18   # 18% para test
TRAIN_FRACTION = 0.72  # 72% para entrenamiento


# =============================================================================
# Parseo de argumentos
# =============================================================================

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


# =============================================================================
# Funciones
# =============================================================================

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
    
    # Split sesiones: 10% blind, 18% test, 72% train
    # Primero separar blind (10%)
    trainval_sessions, blind_sessions = train_test_split(
        session_names,
        test_size=BLIND_FRACTION,
        random_state=seed,
    )
    
    # Luego separar test de train (18% del total = 20% de 90% restante)
    test_fraction_of_remaining = TEST_FRACTION / (1 - BLIND_FRACTION)
    train_sessions, test_sessions = train_test_split(
        trainval_sessions,
        test_size=test_fraction_of_remaining,
        random_state=seed,
    )
    
    # Crear listas de archivos
    train_files = []
    test_files = []
    blind_files = []
    
    for session in train_sessions:
        train_files.extend(sessions[session])
    
    for session in test_sessions:
        test_files.extend(sessions[session])
    
    for session in blind_sessions:
        blind_files.extend(sessions[session])
    
    return train_files, test_files, blind_files


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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
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
    audio_files = get_audio_files()
    print(f"Total de archivos: {len(audio_files)}")
    
    # Crear splits por sesión
    print("\nCreando splits estratificados por sesión...")
    train_files, test_files, blind_files = create_session_based_split(audio_files, seed)
    
    print(f"\nSesiones:")
    print(f"  Train: {len(set(f['sesion'] for f in train_files))} sesiones, {len(train_files)} archivos")
    print(f"  Test: {len(set(f['sesion'] for f in test_files))} sesiones, {len(test_files)} archivos")
    print(f"  Blind: {len(set(f['sesion'] for f in blind_files))} sesiones, {len(blind_files)} archivos")
    
    # Generar CSVs con segmentos
    print(f"\nGenerando CSVs con segmentos (duración={duration}s, overlap={overlap})...")
    
    train_df = generate_segment_csv(train_files, float(duration), overlap)
    test_df = generate_segment_csv(test_files, float(duration), overlap)
    blind_df = generate_segment_csv(blind_files, float(duration), overlap)
    
    # Guardar CSVs
    train_path = duration_dir / "train.csv"
    test_path = duration_dir / "test.csv"
    blind_path = duration_dir / "blind.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    blind_df.to_csv(blind_path, index=False)
    
    print(f"\nCSVs guardados:")
    print(f"  Train: {train_path} ({len(train_df)} segmentos)")
    print(f"  Test: {test_path} ({len(test_df)} segmentos)")
    print(f"  Blind: {blind_path} ({len(blind_df)} segmentos)")
    
    # Calcular y guardar estadísticas
    print("\nCalculando estadísticas...")
    
    train_stats = compute_data_stats(train_df)
    test_stats = compute_data_stats(test_df)
    blind_stats = compute_data_stats(blind_df)
    
    stats = {
        'segment_duration': duration,
        'overlap_ratio': overlap,
        'random_seed': seed,
        'train': train_stats,
        'test': test_stats,
        'blind': blind_stats,
    }
    
    stats_path = duration_dir / "data_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEstadísticas guardadas: {stats_path}")
    
    # Crear CSV completo
    complete_df = pd.concat([train_df, test_df, blind_df], ignore_index=True)
    complete_path = duration_dir / "completo.csv"
    complete_df.to_csv(complete_path, index=False)
    print(f"CSV completo: {complete_path} ({len(complete_df)} segmentos)")
    
    print("\n" + "=" * 60)
    print("¡Splits generados exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
