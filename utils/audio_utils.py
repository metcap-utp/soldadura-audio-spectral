"""Utilidades para carga y procesamiento de audio."""

import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_BASE_DIR = Path("/home/luis/projects/vggish-backbone/audio")


def extract_session_from_path(audio_path: str) -> str:
    """Extrae el identificador de sesión del path del audio.
    
    El path tiene estructura: audio/Placa_Xmm/EXXXX/AC|DC/YYMMDD-HHMMSS_Audio/file.wav
    La sesión es la carpeta con formato YYMMDD-HHMMSS_Audio
    """
    parts = Path(audio_path).parts
    for part in parts:
        if part.endswith("_Audio"):
            return part
    return Path(audio_path).parent.name


def load_audio_segment(
    audio_path: Path,
    segment_duration: float,
    segment_index: int = 0,
    sr: int = 16000,
    overlap_seconds: float = 0.0,
) -> np.ndarray:
    """Carga un segmento específico de un archivo de audio.
    
    Args:
        audio_path: Ruta al archivo de audio
        segment_duration: Duración del segmento en segundos
        segment_index: Índice del segmento a cargar
        sr: Sample rate objetivo
        overlap_seconds: Segundos de solapamiento entre segmentos
    
    Returns:
        Array numpy con el segmento de audio
    """
    try:
        audio, file_sr = sf.read(str(audio_path))
        
        # Convertir a mono si es estéreo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resamplear si es necesario
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        # Calcular índices de inicio y fin
        hop_size = int((segment_duration - overlap_seconds) * sr)
        start_sample = segment_index * hop_size
        end_sample = start_sample + int(segment_duration * sr)
        
        # Padding si es necesario
        if end_sample > len(audio):
            audio = np.pad(audio, (0, end_sample - len(audio)), mode='constant')
        
        return audio[start_sample:end_sample]
        
    except Exception as e:
        print(f"Error cargando {audio_path}: {e}")
        return None


def get_audio_files(base_dir: Path = None) -> list:
    """Descubre todos los archivos de audio en el directorio base.
    
    Returns:
        Lista de diccionarios con información de cada archivo
    """
    if base_dir is None:
        base_dir = AUDIO_BASE_DIR
    
    audio_files = []
    
    for placa_dir in sorted(base_dir.glob("Placa_*")):
        placa = placa_dir.name
        for electrodo_dir in sorted(placa_dir.glob("E*")):
            electrodo = electrodo_dir.name
            for corriente_dir in sorted(electrodo_dir.glob("*C")):
                corriente = corriente_dir.name
                for sesion_dir in sorted(corriente_dir.glob("*_Audio")):
                    sesion = sesion_dir.name
                    for wav_file in sorted(sesion_dir.glob("*.wav")):
                        rel_path = wav_file.relative_to(base_dir)
                        audio_files.append({
                            'path': str(rel_path),
                            'placa': placa,
                            'electrodo': electrodo,
                            'corriente': corriente,
                            'sesion': sesion,
                            'full_path': wav_file
                        })
    
    return audio_files
