import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# ============================================================
# RUTA DE AUDIOS - Cambiar aquí la ruta de los archivos de audio
# ============================================================
DEFAULT_AUDIO_ROOT = Path("/home/luis/projects/tesis/audio/soldadura/audio")

from mfcc_baseline.config import load_config
from mfcc_baseline.dataset import (
    FeatureConfig,
    encode_labels,
    fit_label_encoders,
    load_features,
)
from mfcc_baseline.models import XVectorModel
from mfcc_baseline.pytorch_dataset import AudioDataset, MultiTaskDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch models (X-Vector, ECAPA-TDNN)")
    parser.add_argument("--splits-dir", type=str, required=True, help="Path to splits directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for models")
    parser.add_argument("--audio-root", type=str, default=None, 
                        help="Path to audio files directory (default: uses DEFAULT_AUDIO_ROOT constant)")
    parser.add_argument("--duration", type=int, default=10, choices=[1, 2, 5, 10, 15, 20, 30, 50])
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio (e.g., 0.0, 0.5)")
    parser.add_argument("--k-fold", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--model-type", type=str, default="xvector", choices=["xvector", "ecapa"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--task", type=str, default="Plate Thickness",
                        choices=["Plate Thickness", "Electrode", "Type of Current"])
    parser.add_argument("--multi-task", action="store_true", help="Enable multi-task training (all 3 tasks)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_folder_name(k_fold: int, overlap: float) -> str:
    return f"k{k_fold:02d}_overlap_{overlap}"


def build_model(model_type: str, input_size: int, num_classes: int, embedding_dim: int, device: str):
    if model_type == "xvector":
        model = XVectorModel(
            input_size=input_size,
            embedding_size=embedding_dim,
            num_classes=num_classes,
        )
    elif model_type == "ecapa":
        from mfcc_baseline.models.ecapa_tdnn import ECAPA_TDNNClassifier
        model = ECAPA_TDNNClassifier(
            input_size=input_size,
            num_classes=num_classes,
            lin_neurons=embedding_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


class MultiTaskModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes_dict: dict, embedding_dim: int):
        super().__init__()
        self.base_model = base_model
        self.num_classes_dict = num_classes_dict
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(embedding_dim, num_classes)
            for task, num_classes in num_classes_dict.items()
        })

    def forward(self, x, task: str = None):
        embedding = self.base_model(x, return_embedding=True)

        if task is not None:
            return self.classifiers[task](embedding)

        outputs = {task: clf(embedding) for task, clf in self.classifiers.items()}
        return outputs


def train_epoch_single_task(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        X, y = batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_epoch_multi_task(model, dataloader, optimizer, device, tasks: list):
    model.train()
    total_loss = 0

    for batch in dataloader:
        X, labels = batch
        X = X.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        outputs = model(X)

        loss = 0
        for task in tasks:
            criterion = nn.CrossEntropyLoss()
            loss += criterion(outputs[task], labels[task])

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def eval_model_single_task(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def eval_model_multi_task(model, dataloader, device, tasks: list):
    model.eval()
    results = {task: {"preds": [], "labels": [], "acc": 0.0} for task in tasks}

    with torch.no_grad():
        for batch in dataloader:
            X, labels = batch
            X = X.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            outputs = model(X)

            for task in tasks:
                preds = outputs[task].argmax(dim=1)
                results[task]["preds"].extend(preds.cpu().numpy())
                results[task]["labels"].extend(labels[task].cpu().numpy())

    for task in tasks:
        results[task]["acc"] = accuracy_score(
            results[task]["labels"],
            results[task]["preds"]
        )

    return results


def train_single_fold(
    X_train: np.ndarray,
    y_train: dict,
    X_test: np.ndarray,
    y_test: dict,
    label_encoders: dict,
    input_size: int,
    num_classes: int,
    model_type: str,
    task: str,
    embedding_dim: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    fold_idx: int,
    output_dir: Path,
):
    train_dataset = AudioDataset(X_train, y_train, task=task)
    test_dataset = AudioDataset(X_test, y_test, task=task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, input_size, num_classes, embedding_dim, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_single_task(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = eval_model_single_task(model, test_loader, criterion, device)

        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Fold {fold_idx + 1} - Epoch {epoch+1}/{epochs} - "
                  f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / f"model_fold_{fold_idx}.pt")

    _, _, preds, labels = eval_model_single_task(model, test_loader, criterion, device)
    label_names = label_encoders[task].classes_.tolist()
    report = classification_report(labels, preds, target_names=label_names, output_dict=True)

    return {
        "fold": fold_idx,
        "best_accuracy": best_acc,
        "classification_report": report,
    }


def train_with_kfold(
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
    task: str,
    k_fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    embedding_dim: int,
    device: str,
):
    print(f"Loading data from {train_csv}...")
    X_train_full, train_df = load_features(
        train_csv,
        audio_root=audio_root,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cache_dir,
    )

    print(f"Loading test data from {test_csv}...")
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
    y_train_full = encode_labels(train_df, label_encoders)
    y_test = encode_labels(test_df, label_encoders)

    num_classes = len(label_encoders[task].classes_)
    input_size = X_train_full.shape[1]

    print(f"Input size: {input_size}, Num classes: {num_classes}")
    print(f"Total train samples: {len(X_train_full)}, Test samples: {len(X_test)}")
    print(f"Running {k_fold}-fold cross-validation...")

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    fold_results = []
    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n--- Fold {fold_idx + 1}/{k_fold} ---")
        fold_start = time.time()

        X_train_fold = X_train_full[train_idx]
        X_val_fold = X_train_full[val_idx]

        y_train_fold = {task: y_train_full[task][train_idx] for task in y_train_full}
        y_val_fold = {task: y_train_full[task][val_idx] for task in y_train_full}

        result = train_single_fold(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_test=X_val_fold,
            y_test=y_val_fold,
            label_encoders=label_encoders,
            input_size=input_size,
            num_classes=num_classes,
            model_type=model_type,
            task=task,
            embedding_dim=embedding_dim,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            fold_idx=fold_idx,
            output_dir=output_dir,
        )

        fold_time = time.time() - fold_start
        result["time_seconds"] = fold_time
        fold_results.append(result)
        print(f"  Fold {fold_idx + 1} - Best Val Acc: {result['best_accuracy']:.4f} - Time: {fold_time:.1f}s")

    total_time = time.time() - start_time

    print(f"\n=== K-Fold Results ({k_fold} folds) ===")
    accuracies = [r["best_accuracy"] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Total training time: {total_time:.1f}s")

    final_results = {
        "model": model_type,
        "task": task,
        "k_fold": k_fold,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "fold_results": fold_results,
        "total_time_seconds": total_time,
        "config": {
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "overlap": overlap_ratio,
        }
    }

    with open(output_dir / "results_kfold.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to {output_dir}")

    return final_results


def train_model_single_task(
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
    task: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    embedding_dim: int,
    device: str,
    k_fold: int,
):
    if k_fold > 1:
        return train_with_kfold(
            train_csv=train_csv,
            test_csv=test_csv,
            audio_root=audio_root,
            segment_duration=segment_duration,
            overlap_ratio=overlap_ratio,
            sample_rate=sample_rate,
            feature_cfg=feature_cfg,
            cache_dir=cache_dir,
            output_dir=output_dir,
            model_type=model_type,
            task=task,
            k_fold=k_fold,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            embedding_dim=embedding_dim,
            device=device,
        )

    print(f"Loading training data from {train_csv}...")
    X_train, train_df = load_features(
        train_csv,
        audio_root=audio_root,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cache_dir,
    )

    print(f"Loading test data from {test_csv}...")
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

    num_classes = len(label_encoders[task].classes_)
    input_size = X_train.shape[1]

    print(f"Input size: {input_size}, Num classes: {num_classes}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    train_dataset = AudioDataset(X_train, y_train, task=task)
    test_dataset = AudioDataset(X_test, y_test, task=task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, input_size, num_classes, embedding_dim, device)
    print(f"Model: {model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_single_task(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = eval_model_single_task(model, test_loader, criterion, device)

        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    train_time = time.time() - start_time

    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    _, _, preds, labels = eval_model_single_task(model, test_loader, criterion, device)

    label_names = label_encoders[task].classes_.tolist()
    report = classification_report(labels, preds, target_names=label_names, output_dict=True)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=label_names))

    final_results = {
        "model": model_type,
        "task": task,
        "best_accuracy": best_acc,
        "classification_report": report,
        "training_time_seconds": train_time,
        "config": {
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "overlap": overlap_ratio,
        }
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Training time: {train_time:.1f}s")
    print(f"Results saved to {output_dir}")

    return final_results


def main():
    args = parse_args()

    cfg = load_config()
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    # Usar --audio-root si se proporciona, sino usar DEFAULT_AUDIO_ROOT
    if args.audio_root:
        audio_root = Path(args.audio_root)
    else:
        audio_root = DEFAULT_AUDIO_ROOT

    model_folder = get_model_folder_name(args.k_fold, args.overlap)
    models_dir = output_dir / f"{args.duration:02d}seg" / model_folder
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(n_mfcc=args.n_mfcc, include_deltas=True)

    print(f"=== Training Configuration ===")
    print(f"Audio Root: {audio_root}")
    print(f"Duration: {args.duration}s")
    print(f"Overlap: {args.overlap}")
    print(f"K-Fold: {args.k_fold}")
    print(f"Model: {args.model_type}")
    print(f"Task: {args.task}")
    print(f"Output: {models_dir}")
    print("=" * 35)

    if args.multi_task:
        print("Multi-task training not implemented with k-fold yet")
        return

    results = train_model_single_task(
        train_csv=splits_dir / f"train_{args.duration}s.csv",
        test_csv=splits_dir / f"test_{args.duration}s.csv",
        audio_root=audio_root,
        segment_duration=args.duration,
        overlap_ratio=args.overlap,
        sample_rate=cfg.sample_rate,
        feature_cfg=feature_cfg,
        cache_dir=cfg.cache_dir,
        output_dir=models_dir,
        model_type=args.model_type,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        device=args.device,
        k_fold=args.k_fold,
    )

    print(f"\nDone! Results in {models_dir}")


if __name__ == "__main__":
    main()
