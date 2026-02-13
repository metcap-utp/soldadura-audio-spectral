from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        ),
    }


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Pred")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
