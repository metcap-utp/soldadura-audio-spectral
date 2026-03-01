#!/usr/bin/env python3
"""
Entrenamiento ECAPA-TDNN para clasificación SMAW (multi-task).

Lee splits desde CSVs, generándolos automáticamente si no existen.
Evalúa en conjunto blind al finalizar.

Uso:
    python entrenar_ecapa.py --duration 10 --overlap 0.5 --k-folds 10
"""

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from utils.timing import timer

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from models.modelo_ecapa import ECAPAMultiTask
from utils.audio_utils import AUDIO_BASE_DIR
from utils.logging_utils import setup_log_file

warnings.filterwarnings("ignore")

# Hiperparámetros
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SWA_START = 5
N_MFCC = 40


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento ECAPA-TDNN SMAW")
    parser.add_argument("--duration", type=int, required=True, choices=[1, 2, 5, 10, 20, 30, 50])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def ensure_csvs_exist(duration, overlap, seed):
    """Genera CSVs si no existen usando generar_splits.py."""
    duration_dir = Path(f"{duration:02d}seg")
    
    # Intentar con nombres específicos de overlap
    train_csv = duration_dir / f"train_overlap_{overlap}.csv"
    test_csv = duration_dir / f"test_overlap_{overlap}.csv"
    blind_csv = duration_dir / f"blind_overlap_{overlap}.csv"
    
    # Si no existen, usar nombres genéricos
    if not train_csv.exists():
        train_csv = duration_dir / "train.csv"
    if not test_csv.exists():
        test_csv = duration_dir / "test.csv"
    if not blind_csv.exists():
        blind_csv = duration_dir / "blind.csv"
    
    if not train_csv.exists() or not test_csv.exists() or not blind_csv.exists():
        print(f"CSVs no encontrados. Generando con generar_splits.py...")
        print(f"  Duración: {duration}s, Overlap: {overlap}, Seed: {seed}")
        
        result = subprocess.run(
            [
                sys.executable, 
                "generar_splits.py",
                "--duration", str(duration),
                "--overlap", str(overlap),
                "--seed", str(seed)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error generando CSVs: {result.stderr}")
            sys.exit(1)
        
        print(f"  CSVs generados exitosamente")
    else:
        print(f"  [CACHE] Usando CSVs existentes")
    
    return train_csv, test_csv, blind_csv


def extract_mfcc_raw(y, sr=16000, n_mfcc=40):
    """Extrae MFCC sin agregar estadísticas (para ECAPA)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc  # (n_mfcc, time)


def load_segments_from_csv(df, duration, overlap):
    """Carga segmentos de audio definidos en el CSV."""
    features = []
    failed = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"    Procesados {idx}/{len(df)} segmentos...")
        
        audio_path = AUDIO_BASE_DIR / row['audio_path']
        segment_idx = int(row['segment_index'])
        
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Calcular posición del segmento
            hop = int(duration * (1 - overlap) * sr)
            samples = int(duration * sr)
            start = segment_idx * hop
            
            # Verificar límites
            if start + samples > len(y):
                segment = np.zeros(samples)
                available = len(y) - start
                if available > 0:
                    segment[:available] = y[start:start + available]
            else:
                segment = y[start:start + samples]
            
            # Extraer MFCC raw (sin agregar)
            mfcc = extract_mfcc_raw(segment, sr=16000, n_mfcc=N_MFCC)
            features.append(mfcc)
            
        except Exception as e:
            print(f"    Error en {audio_path}, segmento {segment_idx}: {e}")
            failed += 1
            # Usar MFCC ceros como fallback con dimensiones correctas
            # MFCC hop_length=512 con 16kHz, entonces time_steps ≈ (samples / 512) + 1
            samples = int(duration * 16000)
            n_time_steps = (samples // 512) + 1
            features.append(np.zeros((N_MFCC, n_time_steps)))
    
    if failed > 0:
        print(f"    ⚠️  {failed} segmentos fallaron, usando ceros")
    
    return features


def extract_features_from_csv(train_csv, test_csv, blind_csv, duration, overlap, cache_path):
    """Extrae features de los CSVs."""
    if cache_path.exists():
        print(f"  [CACHE] Cargando features desde {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return (data['X_train'], data['y_train'], data['sessions_train'],
                data['X_test'], data['y_test'], data['sessions_test'],
                data['X_blind'], data['y_blind'], data['sessions_blind'])
    
    # Cargar CSVs
    print(f"Cargando CSVs...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    blind_df = pd.read_csv(blind_csv)
    
    print(f"  Train: {len(train_df)} segmentos")
    print(f"  Test: {len(test_df)} segmentos")
    print(f"  Blind: {len(blind_df)} segmentos")
    
    # Extraer features
    print(f"\nExtrayendo MFCC raw de train...")
    X_train = load_segments_from_csv(train_df, duration, overlap)
    
    print(f"Extrayendo MFCC raw de test...")
    X_test = load_segments_from_csv(test_df, duration, overlap)
    
    print(f"Extrayendo MFCC raw de blind...")
    X_blind = load_segments_from_csv(blind_df, duration, overlap)
    
    # Preparar labels y sessions
    sessions_train = train_df['sesion'].values
    sessions_test = test_df['sesion'].values
    sessions_blind = blind_df['sesion'].values
    
    y_train = {
        'plate': train_df['placa'].values,
        'electrode': train_df['electrodo'].values,
        'current': train_df['corriente'].values
    }
    
    y_test = {
        'plate': test_df['placa'].values,
        'electrode': test_df['electrodo'].values,
        'current': test_df['corriente'].values
    }
    
    y_blind = {
        'plate': blind_df['placa'].values,
        'electrode': blind_df['electrodo'].values,
        'current': blind_df['corriente'].values
    }
    
    # Guardar cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'X_train': X_train, 'y_train': y_train, 'sessions_train': sessions_train,
        'X_test': X_test, 'y_test': y_test, 'sessions_test': sessions_test,
        'X_blind': X_blind, 'y_blind': y_blind, 'sessions_blind': sessions_blind,
    }, cache_path)
    print(f"\n  [CACHE] Features guardadas en {cache_path}")
    
    return (X_train, y_train, sessions_train,
            X_test, y_test, sessions_test,
            X_blind, y_blind, sessions_blind)


class MFCCRawDataset(Dataset):
    """Dataset para MFCC raw (no agregados) con padding."""
    
    def __init__(self, features, labels_plate, labels_electrode, labels_current):
        self.features = features  # Lista de arrays (n_mfcc, time)
        self.labels_plate = torch.LongTensor(labels_plate)
        self.labels_electrode = torch.LongTensor(labels_electrode)
        self.labels_current = torch.LongTensor(labels_current)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.features[idx])
        return feat, self.labels_plate[idx], self.labels_electrode[idx], self.labels_current[idx]


def collate_fn(batch):
    """Collation function para manejar tamaños variables."""
    feats = [item[0] for item in batch]
    labels_plate = torch.stack([item[1] for item in batch])
    labels_electrode = torch.stack([item[2] for item in batch])
    labels_current = torch.stack([item[3] for item in batch])
    
    # Encontrar tamaño máximo
    max_time = max(feat.shape[1] for feat in feats)
    
    # Padding
    padded = []
    for feat in feats:
        if feat.shape[1] < max_time:
            pad = torch.zeros(feat.shape[0], max_time - feat.shape[1])
            feat = torch.cat([feat, pad], dim=1)
        padded.append(feat)
    
    return torch.stack(padded), labels_plate, labels_electrode, labels_current


def get_system_info(device):
    """Recolecta información del sistema y dependencias."""
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return info


def count_model_parameters(model):
    """Cuenta parámetros totales y entrenables del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def train_model(model, train_loader, val_loader, device):
    """Entrena un modelo con tracking completo de métricas."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    
    best_acc = 0
    best_epoch = 0
    patience = 0
    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    acc_plate = acc_electrode = acc_current = 0.0
    training_history = []
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x, y_plate, y_electrode, y_current = batch
            x, y_plate = x.to(device), y_plate.to(device)
            y_electrode, y_current = y_electrode.to(device), y_current.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            loss = (criterion(out['plate'], y_plate) + 
                   criterion(out['electrode'], y_electrode) + 
                   criterion(out['current'], y_current)) / 3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds_plate, all_labels_plate = [], []
        all_preds_electrode, all_labels_electrode = [], []
        all_preds_current, all_labels_current = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_plate, y_electrode, y_current = batch
                x = x.to(device)
                out = model(x)
                
                # Calculate loss
                loss_p = criterion(out['plate'], y_plate.to(device))
                loss_e = criterion(out['electrode'], y_electrode.to(device))
                loss_c = criterion(out['current'], y_current.to(device))
                val_loss += (loss_p + loss_e + loss_c).item()
                
                all_preds_plate.extend(out['plate'].argmax(dim=1).cpu().numpy())
                all_labels_plate.extend(y_plate.numpy())
                all_preds_electrode.extend(out['electrode'].argmax(dim=1).cpu().numpy())
                all_labels_electrode.extend(y_electrode.numpy())
                all_preds_current.extend(out['current'].argmax(dim=1).cpu().numpy())
                all_labels_current.extend(y_current.numpy())
        
        acc_plate = accuracy_score(all_labels_plate, all_preds_plate)
        acc_electrode = accuracy_score(all_labels_electrode, all_preds_electrode)
        acc_current = accuracy_score(all_labels_current, all_preds_current)
        
        f1_plate = f1_score(all_labels_plate, all_preds_plate, average='macro')
        f1_electrode = f1_score(all_labels_electrode, all_preds_electrode, average='macro')
        f1_current = f1_score(all_labels_current, all_preds_current, average='macro')
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Track epoch history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss / len(train_loader), 4),
            "val_loss": round(avg_val_loss, 4),
            "learning_rate": round(current_lr, 2e-6),
            "val_acc_plate": round(acc_plate, 4),
            "val_acc_electrode": round(acc_electrode, 4),
            "val_acc_current": round(acc_current, 4),
            "val_f1_plate": round(f1_plate, 4),
            "val_f1_electrode": round(f1_electrode, 4),
            "val_f1_current": round(f1_current, 4),
        })
        
        # Update scheduler and SWA
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Early stopping with best epoch tracking
        if acc_plate > best_acc:
            best_acc = acc_plate
            best_epoch = epoch + 1
            patience = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        
        if patience >= EARLY_STOP_PATIENCE:
            break
    
    # Load best model state
    model.load_state_dict(best_state_dict)
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    
    return model, swa_model, {
        'accuracy_plate': float(acc_plate),
        'accuracy_electrode': float(acc_electrode),
        'accuracy_current': float(acc_current),
        'f1_plate': float(f1_plate),
        'f1_electrode': float(f1_electrode),
        'f1_current': float(f1_current),
    }, best_epoch, training_history



def evaluate_blind(models, X_blind, y_blind, le_plate, le_electrode, le_current, device):
    """Evalúa el ensemble en el conjunto blind."""
    print("\n" + "="*60)
    print("EVALUACIÓN EN CONJUNTO BLIND")
    print("="*60)
    
    # Crear dataset y loader para blind
    blind_dataset = MFCCRawDataset(
        X_blind,
        le_plate.transform(y_blind['plate']),
        le_electrode.transform(y_blind['electrode']),
        le_current.transform(y_blind['current'])
    )
    blind_loader = DataLoader(blind_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Soft voting
    all_logits_plate = []
    all_logits_electrode = []
    all_logits_current = []
    
    with torch.no_grad():
        for batch in blind_loader:
            x, _, _, _ = batch
            x = x.to(device)
            
            batch_logits_plate = []
            batch_logits_electrode = []
            batch_logits_current = []
            
            for model in models:
                out = model(x)
                batch_logits_plate.append(out['plate'].cpu().numpy())
                batch_logits_electrode.append(out['electrode'].cpu().numpy())
                batch_logits_current.append(out['current'].cpu().numpy())
            
            all_logits_plate.extend(np.mean(batch_logits_plate, axis=0))
            all_logits_electrode.extend(np.mean(batch_logits_electrode, axis=0))
            all_logits_current.extend(np.mean(batch_logits_current, axis=0))
    
    # Calcular métricas
    y_true_plate = le_plate.transform(y_blind['plate'])
    y_true_electrode = le_electrode.transform(y_blind['electrode'])
    y_true_current = le_current.transform(y_blind['current'])
    
    avg_plate = np.array(all_logits_plate).argmax(axis=1)
    avg_electrode = np.array(all_logits_electrode).argmax(axis=1)
    avg_current = np.array(all_logits_current).argmax(axis=1)
    
    results = {
        'plate': {
            'accuracy': float(accuracy_score(y_true_plate, avg_plate)),
            'f1': float(f1_score(y_true_plate, avg_plate, average='macro')),
            'precision': float(precision_score(y_true_plate, avg_plate, average='macro')),
            'recall': float(recall_score(y_true_plate, avg_plate, average='macro')),
            'confusion_matrix': confusion_matrix(y_true_plate, avg_plate).tolist(),
        },
        'electrode': {
            'accuracy': float(accuracy_score(y_true_electrode, avg_electrode)),
            'f1': float(f1_score(y_true_electrode, avg_electrode, average='macro')),
            'precision': float(precision_score(y_true_electrode, avg_electrode, average='macro')),
            'recall': float(recall_score(y_true_electrode, avg_electrode, average='macro')),
            'confusion_matrix': confusion_matrix(y_true_electrode, avg_electrode).tolist(),
        },
        'current': {
            'accuracy': float(accuracy_score(y_true_current, avg_current)),
            'f1': float(f1_score(y_true_current, avg_current, average='macro')),
            'precision': float(precision_score(y_true_current, avg_current, average='macro')),
            'recall': float(recall_score(y_true_current, avg_current, average='macro')),
            'confusion_matrix': confusion_matrix(y_true_current, avg_current).tolist(),
        },
    }
    
    exact_match = np.mean(
        (avg_plate == y_true_plate) & 
        (avg_electrode == y_true_electrode) & 
        (avg_current == y_true_current)
    )
    
    hamming_accuracy = np.mean(
        (avg_plate == y_true_plate).astype(int) + 
        (avg_electrode == y_true_electrode).astype(int) + 
        (avg_current == y_true_current).astype(int)
    ) / 3
    
    results['global'] = {
        'exact_match': float(exact_match),
        'hamming_accuracy': float(hamming_accuracy),
    }
    
    print(f"\nResultados Blind (Ensemble de {len(models)} modelos):")
    for task, metrics in results.items():
        if task == 'global':
            print(f"  {'Global':12s} - Exact Match: {metrics['exact_match']:.4f}, Hamming: {metrics['hamming_accuracy']:.4f}")
        else:
            print(f"  {task:12s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Set up logging
    log_file, log_path = setup_log_file(
        Path(".") / "logs", "entrenar_ecapa", suffix=f"_{int(args.duration):02d}seg"
    )
    sys.stdout = log_file
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("ENTRENAMIENTO ECAPA-TDNN SMAW (Multi-Task)")
    print("="*60)
    print(f"Duración: {args.duration}s")
    print(f"Overlap: {args.overlap}")
    print(f"K-folds: {args.k_folds}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    start_time = time.time()
    
    # Directorios
    duration_dir = Path(f"{args.duration:02d}seg")
    duration_dir.mkdir(exist_ok=True)
    cache_path = duration_dir / "mfcc_cache" / f"ecapa_features_overlap_{args.overlap}.pt"
    models_dir = duration_dir / "modelos" / "ecapa" / f"k{args.k_folds:02d}_overlap_{args.overlap}"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Asegurar que existan CSVs
    print("\nVerificando CSVs...")
    train_csv, test_csv, blind_csv = ensure_csvs_exist(args.duration, args.overlap, args.seed)
    
    # Cargar features
    print("\nCargando features desde CSVs...")
    with timer("Extracción de características MFCC") as get_extraction_time:
        (X_train, y_train, sessions_train,
         X_test, y_test, sessions_test,
         X_blind, y_blind, sessions_blind) = extract_features_from_csv(
            train_csv, test_csv, blind_csv, args.duration, args.overlap, cache_path
        )
    feature_extraction_time = get_extraction_time().seconds
    
    # Combinar train+test para K-Fold CV
    X_all = X_train + X_test  # Lista concatenada
    y_all = {
        'plate': np.concatenate([np.array(y_train['plate']), np.array(y_test['plate'])]),
        'electrode': np.concatenate([np.array(y_train['electrode']), np.array(y_test['electrode'])]),
        'current': np.concatenate([np.array(y_train['current']), np.array(y_test['current'])]),
    }
    sessions_all = np.concatenate([np.array(sessions_train), np.array(sessions_test)])
    
    # Codificar labels
    le_plate = LabelEncoder()
    le_electrode = LabelEncoder()
    le_current = LabelEncoder()
    
    y_plate = le_plate.fit_transform(y_all['plate'])
    y_electrode = le_electrode.fit_transform(y_all['electrode'])
    y_current = le_current.fit_transform(y_all['current'])
    
    print(f"\nTotal samples: {len(X_all)}")
    print(f"Clases: Plate={len(le_plate.classes_)}, Electrode={len(le_electrode.classes_)}, Current={len(le_current.classes_)}")
    
    # K-Fold CV
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION")
    print("="*60)
    
    training_start_time = time.time()  # Measure training time from here
    
    skf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    trained_models = []
    fold_training_times = []
    fold_best_epochs = []
    all_fold_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(X_all)), y_plate, sessions_all)):
        print(f"\nFold {fold_idx + 1}/{args.k_folds}")
        
        # Obtener features por índice
        X_tr = [X_all[i] for i in train_idx]
        X_val = [X_all[i] for i in val_idx]
        yp_tr, yp_val = y_plate[train_idx], y_plate[val_idx]
        ye_tr, ye_val = y_electrode[train_idx], y_electrode[val_idx]
        yc_tr, yc_val = y_current[train_idx], y_current[val_idx]
        
        train_dataset = MFCCRawDataset(X_tr, yp_tr, ye_tr, yc_tr)
        val_dataset = MFCCRawDataset(X_val, yp_val, ye_val, yc_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        
        model = ECAPAMultiTask(
            input_size=40,
            lin_neurons=192,
            num_classes_plate=len(le_plate.classes_),
            num_classes_electrode=len(le_electrode.classes_),
            num_classes_current=len(le_current.classes_)
        ).to(device)
        
        start = time.time()
        model, swa_model, metrics, best_epoch, fold_history = train_model(model, train_loader, val_loader, device)
        fold_time = time.time() - start
        
        # Guardar modelo
        torch.save(model.state_dict(), models_dir / f"model_fold_{fold_idx}.pt")
        torch.save(swa_model.state_dict(), models_dir / f"model_fold_{fold_idx}_swa.pt")
        
        # Compute confusion matrices on validation set
        model.eval()
        val_preds = {"plate": [], "electrode": [], "current": []}
        val_labels_all = {"plate": [], "electrode": [], "current": []}
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_plate, y_electrode, y_current = batch
                x = x.to(device)
                out = model(x)
                
                val_preds["plate"].extend(out["plate"].argmax(dim=1).cpu().numpy())
                val_preds["electrode"].extend(out["electrode"].argmax(dim=1).cpu().numpy())
                val_preds["current"].extend(out["current"].argmax(dim=1).cpu().numpy())
                val_labels_all["plate"].extend(y_plate.numpy())
                val_labels_all["electrode"].extend(y_electrode.numpy())
                val_labels_all["current"].extend(y_current.numpy())
        
        cm_plate = confusion_matrix(val_labels_all["plate"], val_preds["plate"])
        cm_electrode = confusion_matrix(val_labels_all["electrode"], val_preds["electrode"])
        cm_current = confusion_matrix(val_labels_all["current"], val_preds["current"])
        
        # Guardar para ensemble
        trained_models.append(model)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'time_seconds': round(fold_time, 2),
            'accuracy_plate': round(metrics['accuracy_plate'], 4),
            'accuracy_electrode': round(metrics['accuracy_electrode'], 4),
            'accuracy_current': round(metrics['accuracy_current'], 4),
            'f1_plate': round(metrics['f1_plate'], 4),
            'f1_electrode': round(metrics['f1_electrode'], 4),
            'f1_current': round(metrics['f1_current'], 4),
            'precision_plate': round(precision_score(val_labels_all["plate"], val_preds["plate"], average='macro'), 4),
            'precision_electrode': round(precision_score(val_labels_all["electrode"], val_preds["electrode"], average='macro'), 4),
            'precision_current': round(precision_score(val_labels_all["current"], val_preds["current"], average='macro'), 4),
            'recall_plate': round(recall_score(val_labels_all["plate"], val_preds["plate"], average='macro'), 4),
            'recall_electrode': round(recall_score(val_labels_all["electrode"], val_preds["electrode"], average='macro'), 4),
            'recall_current': round(recall_score(val_labels_all["current"], val_preds["current"], average='macro'), 4),
            'confusion_matrix_plate': cm_plate.tolist(),
            'confusion_matrix_electrode': cm_electrode.tolist(),
            'confusion_matrix_current': cm_current.tolist(),
            'best_epoch': best_epoch,
        })
        
        fold_training_times.append(round(fold_time, 2))
        fold_best_epochs.append(best_epoch)
        all_fold_histories.append(fold_history)
        
        print(f"  Fold {fold_idx + 1}: Plate={metrics['accuracy_plate']:.4f} | "
              f"Electrode={metrics['accuracy_electrode']:.4f} | "
              f"Current={metrics['accuracy_current']:.4f} | "
              f"Best epoch={best_epoch}")
        print(f"  Time: {fold_time:.1f}s")
    
    # Evaluar en blind
    blind_results = evaluate_blind(
        trained_models, X_blind, y_blind, 
        le_plate, le_electrode, le_current, device
    )
    
    # Calculate ensemble metrics on blind set
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600
    training_time = end_time - training_start_time  # Wall-clock training time, not sum of folds
    training_time_minutes = training_time / 60
    
    # Calculate average metrics from blind evaluation
    avg_acc_p = np.mean([blind_results['plate']['accuracy']])
    avg_acc_e = np.mean([blind_results['electrode']['accuracy']])
    avg_acc_c = np.mean([blind_results['current']['accuracy']])
    
    acc_p = blind_results['plate']['accuracy']
    acc_e = blind_results['electrode']['accuracy']
    acc_c = blind_results['current']['accuracy']
    f1_p = blind_results['plate']['f1']
    f1_e = blind_results['electrode']['f1']
    f1_c = blind_results['current']['f1']
    
    segments_per_class = {
        "plate": {str(k): int(v) for k, v in pd.Series(y_all['plate']).value_counts().to_dict().items()},
        "electrode": {str(k): int(v) for k, v in pd.Series(y_all['electrode']).value_counts().to_dict().items()},
        "current": {str(k): int(v) for k, v in pd.Series(y_all['current']).value_counts().to_dict().items()},
    }
    
    # Get first model to extract parameters
    first_model = trained_models[0]
    model_params = count_model_parameters(first_model)
    
    # Guardar resultados con esquema completo
    new_entry = {
        "id": f"{int(args.duration)}seg_{args.k_folds}fold_overlap_{args.overlap}_ecapa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "model_type": "ecapa",
        "backbone": "spectral-mfcc",
        "execution_time": {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_minutes, 2),
            "hours": round(elapsed_hours, 4),
        },
        "training_time": {
            "seconds": round(training_time, 2),
            "minutes": round(training_time_minutes, 2),
        },
        "feature_extraction": {
            "from_cache": cache_path.exists() if cache_path else False,
            "extraction_time_seconds": round(feature_extraction_time, 2),
            "extraction_time_minutes": round(feature_extraction_time / 60, 2),
        },
        "config": {
            "segment_duration": args.duration,
            "overlap_ratio": args.overlap,
            "overlap_seconds": args.duration * args.overlap,
            "n_folds": args.k_folds,
            "models_dir": str(models_dir.name),
            "random_seed": args.seed,
            "voting_method": "soft",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "swa_start": SWA_START,
            "swa_lr": 1e-4,
            "n_mfcc": N_MFCC,
        },
        "system_info": get_system_info(torch.device(device)),
        "model_parameters": model_params,
        "data": {
            "total_segments": len(X_all),
            "unique_sessions": int(len(set(sessions_all))),
            "segments_per_class": segments_per_class,
            "classes": {
                "plate": list(le_plate.classes_),
                "electrode": list(le_electrode.classes_),
                "current": list(le_current.classes_),
            },
        },
        "fold_results": fold_results,
        "fold_best_epochs": fold_best_epochs,
        "fold_training_times_seconds": fold_training_times,
        "ensemble_results": {
            "plate": {
                "accuracy": round(acc_p, 4),
                "f1": round(f1_p, 4),
                "precision": round(blind_results['plate'].get('precision', 0.0), 4),
                "recall": round(blind_results['plate'].get('recall', 0.0), 4),
            },
            "electrode": {
                "accuracy": round(acc_e, 4),
                "f1": round(f1_e, 4),
                "precision": round(blind_results['electrode'].get('precision', 0.0), 4),
                "recall": round(blind_results['electrode'].get('recall', 0.0), 4),
            },
            "current": {
                "accuracy": round(acc_c, 4),
                "f1": round(f1_c, 4),
                "precision": round(blind_results['current'].get('precision', 0.0), 4),
                "recall": round(blind_results['current'].get('recall', 0.0), 4),
            },
            "global_metrics": {
                "exact_match": blind_results['global'].get('exact_match', 0.0),
                "hamming_accuracy": blind_results['global'].get('hamming_accuracy', 0.0),
            }
        },
        "individual_avg": {
            "plate": round(avg_acc_p, 4),
            "electrode": round(avg_acc_e, 4),
            "current": round(avg_acc_c, 4),
        },
        "improvement_vs_individual": {
            "plate": round(acc_p - avg_acc_p, 4),
            "electrode": round(acc_e - avg_acc_e, 4),
            "current": round(acc_c - avg_acc_c, 4),
        },
        "training_history": all_fold_histories,
    }
    
    results_path = duration_dir / "resultados.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        if not isinstance(all_results, list):
            all_results = [all_results]
    else:
        all_results = []
    
    all_results.append(new_entry)
    results_path.write_text(json.dumps(all_results, indent=2))
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Resultados guardados: {results_path}")
    print(f"Modelos guardados: {models_dir}")
    print(f"Logs guardados en: {log_path}")
    
    # Close log file
    log_file.close()


if __name__ == "__main__":
    main()
