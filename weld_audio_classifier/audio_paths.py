from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def extract_labels_from_session_path(
    audio_root: Path, session_path: Path
) -> Optional[Dict]:
    try:
        relative_path = session_path.relative_to(audio_root)
    except ValueError:
        return None

    parts = relative_path.parts
    # Esperado: Placa_Xmm/EXXXX/AC|DC/TIMESTAMP_Audio
    if len(parts) != 4:
        return None

    plate_thickness = parts[0]
    electrode = parts[1]
    current_type = parts[2]
    session = parts[3]

    return {
        "Session Path": str(relative_path),
        "Plate Thickness": plate_thickness,
        "Electrode": electrode,
        "Type of Current": current_type,
        "Session": session,
    }


def discover_sessions(audio_root: Path) -> pd.DataFrame:
    sessions_data = []

    for session_dir in audio_root.rglob("*_Audio"):
        if not session_dir.is_dir():
            continue
        labels = extract_labels_from_session_path(audio_root, session_dir)
        if labels:
            wav_files = list(session_dir.glob("*.wav"))
            labels["Num Files"] = len(wav_files)
            if wav_files:
                sessions_data.append(labels)

    return pd.DataFrame(sessions_data)


def get_session_audio_files(audio_root: Path, session_path: str) -> List[Path]:
    full_path = audio_root / session_path
    if not full_path.exists():
        return []
    return sorted(full_path.glob("*.wav"))
