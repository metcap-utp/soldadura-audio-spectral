import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    audio_root: Path
    cache_dir: Path
    sample_rate: int


def load_config(config_path: Optional[Path] = None) -> Config:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    data = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    audio_root_env = os.getenv("AUDIO_ROOT")
    audio_root = (
        Path(audio_root_env) if audio_root_env else Path(data.get("audio_root", ""))
    )

    if not audio_root:
        raise ValueError(
            "audio_root no esta configurado. Ajusta config.yaml o AUDIO_ROOT."
        )

    cache_dir = Path(data.get("cache_dir", ".cache_features"))
    sample_rate = int(data.get("sample_rate", 16000))

    return Config(audio_root=audio_root, cache_dir=cache_dir, sample_rate=sample_rate)
