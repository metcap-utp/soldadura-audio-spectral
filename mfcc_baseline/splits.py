from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mfcc_baseline.audio_paths import discover_sessions
from mfcc_baseline.segmenter import get_all_segments_from_session


@dataclass
class SplitConfig:
    duration: float
    overlap_ratio: float
    blind_fraction: float = 0.10
    test_fraction: float = 0.18
    val_fraction: float = 0.0
    seed: int = 42


def create_strat_label(df: pd.DataFrame) -> pd.Series:
    return df["Plate Thickness"] + "_" + df["Electrode"] + "_" + df["Type of Current"]


def load_all_sessions(audio_root: Path, cfg: SplitConfig, sr: int) -> pd.DataFrame:
    sessions_df = discover_sessions(audio_root)
    if sessions_df.empty:
        return sessions_df

    segment_counts = []
    for _, row in sessions_df.iterrows():
        segments = get_all_segments_from_session(
            audio_root,
            row["Session Path"],
            cfg.duration,
            sr=sr,
            overlap_ratio=cfg.overlap_ratio,
        )
        segment_counts.append(len(segments))

    sessions_df["Num Segments"] = segment_counts
    return sessions_df


def expand_sessions_to_segments(
    sessions_df: pd.DataFrame, audio_root: Path, cfg: SplitConfig, sr: int
) -> pd.DataFrame:
    segments_data = []
    for _, row in sessions_df.iterrows():
        segments = get_all_segments_from_session(
            audio_root,
            row["Session Path"],
            cfg.duration,
            sr=sr,
            overlap_ratio=cfg.overlap_ratio,
        )
        for audio_path, seg_idx in segments:
            rel_path = audio_path.relative_to(audio_root)
            segments_data.append(
                {
                    "Audio Path": str(rel_path),
                    "Segment Index": seg_idx,
                    "Plate Thickness": row["Plate Thickness"],
                    "Electrode": row["Electrode"],
                    "Type of Current": row["Type of Current"],
                    "Session": row["Session"],
                }
            )

    return pd.DataFrame(segments_data)


def split_by_session(df: pd.DataFrame, cfg: SplitConfig) -> Dict[str, str]:
    sessions_df = df.groupby("Session").first().reset_index()
    sessions_df["Strat_Label"] = create_strat_label(sessions_df)

    strat_counts = sessions_df["Strat_Label"].value_counts()
    min_samples = 3 if cfg.blind_fraction > 0 else 2
    rare_classes = strat_counts[strat_counts < min_samples].index.tolist()

    sessions_df["is_rare"] = sessions_df["Strat_Label"].isin(rare_classes)
    rare_sessions = sessions_df[sessions_df["is_rare"]]["Session"].tolist()
    normal_sessions_df = sessions_df[~sessions_df["is_rare"]]

    session_splits: Dict[str, str] = {}
    for session in rare_sessions:
        session_splits[session] = "train"

    if len(normal_sessions_df) == 0:
        return session_splits

    sessions = normal_sessions_df["Session"].values
    strat_labels = normal_sessions_df["Strat_Label"].values

    total_sessions = len(sessions_df)
    normal_sessions = len(normal_sessions_df)

    remaining_sessions = sessions
    remaining_labels = strat_labels

    if cfg.blind_fraction > 0:
        adjusted_blind_frac = min(
            cfg.blind_fraction * total_sessions / normal_sessions, 0.4
        )
        try:
            remaining_sessions, blind_sessions, remaining_labels, _ = train_test_split(
                remaining_sessions,
                remaining_labels,
                test_size=adjusted_blind_frac,
                random_state=cfg.seed,
                stratify=remaining_labels,
            )
        except ValueError:
            remaining_sessions, blind_sessions = train_test_split(
                remaining_sessions,
                test_size=adjusted_blind_frac,
                random_state=cfg.seed,
            )

        for session in blind_sessions:
            session_splits[session] = "blind"

    remaining_total = len(remaining_sessions)
    adjusted_test_frac = (
        min(cfg.test_fraction * total_sessions / remaining_total, 0.5)
        if remaining_total > 0
        else 0
    )

    adjusted_val_frac = (
        min(cfg.val_fraction * total_sessions / remaining_total, 0.5)
        if cfg.val_fraction > 0 and remaining_total > 0
        else 0
    )

    if adjusted_test_frac > 0:
        try:
            train_sessions, test_sessions, train_labels, _ = train_test_split(
                remaining_sessions,
                remaining_labels,
                test_size=adjusted_test_frac,
                random_state=cfg.seed,
                stratify=remaining_labels,
            )
        except ValueError:
            train_sessions, test_sessions = train_test_split(
                remaining_sessions,
                test_size=adjusted_test_frac,
                random_state=cfg.seed,
            )
            train_labels = sessions_df[sessions_df["Session"].isin(train_sessions)][
                "Strat_Label"
            ].values

        for session in test_sessions:
            session_splits[session] = "test"

        remaining_sessions = train_sessions
        remaining_labels = train_labels

    if adjusted_val_frac > 0 and len(remaining_sessions) > 0:
        try:
            train_sessions, val_sessions = train_test_split(
                remaining_sessions,
                test_size=adjusted_val_frac,
                random_state=cfg.seed,
                stratify=remaining_labels,
            )
        except ValueError:
            train_sessions, val_sessions = train_test_split(
                remaining_sessions,
                test_size=adjusted_val_frac,
                random_state=cfg.seed,
            )

        for session in val_sessions:
            session_splits[session] = "val"

        remaining_sessions = train_sessions

    for session in remaining_sessions:
        session_splits[session] = "train"

    return session_splits


def generate_splits(
    audio_root: Path,
    cfg: SplitConfig,
    output_dir: Path,
    sr: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions_df = load_all_sessions(audio_root, cfg, sr=sr)
    if sessions_df.empty:
        raise RuntimeError("No se encontraron sesiones de audio.")

    segments_df = expand_sessions_to_segments(sessions_df, audio_root, cfg, sr=sr)
    session_splits = split_by_session(segments_df, cfg)

    segments_df["Split"] = segments_df["Session"].map(session_splits)
    train_df = segments_df[segments_df["Split"] == "train"].drop(columns=["Split"])
    test_df = segments_df[segments_df["Split"] == "test"].drop(columns=["Split"])
    blind_df = segments_df[segments_df["Split"] == "blind"].drop(columns=["Split"])

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    blind_df.to_csv(output_dir / "blind.csv", index=False)
    segments_df.to_csv(output_dir / "completo.csv", index=False)

    return train_df, test_df, blind_df, segments_df
