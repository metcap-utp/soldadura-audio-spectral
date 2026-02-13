import json
from pathlib import Path
from typing import Dict

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mfcc_baseline.dataset import (
    FeatureConfig,
    encode_labels,
    fit_label_encoders,
    load_features,
)
from mfcc_baseline.metrics import (
    compute_confusion,
    compute_metrics,
    save_confusion_matrix,
)


def _build_model(model_type: str) -> Pipeline:
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        return Pipeline([("clf", clf)])

    clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_eval(
    train_csv: Path,
    test_csv: Path,
    audio_root: Path,
    segment_duration: float,
    overlap_ratio: float,
    sample_rate: int,
    feature_cfg: FeatureConfig,
    cache_dir: Path,
    output_dir: Path,
    model_type: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, train_df = load_features(
        train_csv,
        audio_root=audio_root,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cache_dir,
    )

    X_test, test_df = load_features(
        test_csv,
        audio_root=audio_root,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cache_dir,
    )

    label_encoders = fit_label_encoders(train_df)
    y_train = encode_labels(train_df, label_encoders)
    y_test = encode_labels(test_df, label_encoders)

    results: Dict = {
        "duration": segment_duration,
        "overlap": overlap_ratio,
        "model": model_type,
        "tasks": {},
    }

    for task, labels in y_train.items():
        model = _build_model(model_type)
        model.fit(X_train, labels)
        y_pred = model.predict(X_test)

        label_names = list(label_encoders[task].classes_)
        metrics = compute_metrics(y_test[task], y_pred, label_names)
        cm = compute_confusion(y_test[task], y_pred)

        results["tasks"][task] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
        }

        model_path = output_dir / f"model_{task.replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)

        plot_path = output_dir / f"confusion_{task.replace(' ', '_')}.png"
        save_confusion_matrix(cm, label_names, plot_path, title=task)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
