#!/usr/bin/env python3
"""
Entrenamiento de modelos SMAW usando MFCC - Adaptado para spectral-analysis.

Emula la estructura de vggish-backbone pero usa los modelos existentes.

Uso:
    python entrenar.py --duration 10 --overlap 0.5 --k-folds 5
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, TensorDataset

# Imports del proyecto actual
sys.path.insert(0, str(Path(__file__).parent))
from weld_audio_classifier.models.xvector import XVectorModel
from weld_audio_classifier.features import extract_mfcc_features
from utils.audio_utils import AUDIO_BASE_DIR, get_audio_files
from utils.timing import timer

warnings.filterwarnings("ignore")

# Hiperparámetros (de vggish-backbone)
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SWA_START = 5
N_MFCC = 40


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento SMAW con MFCC")
    parser.add_argument("--duration", type=int, required=True, choices=[1, 2, 5, 10, 20, 30, 50])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def segment_audio(audio_path, duration, overlap, sr=16000):
    """Segmenta audio en clips."""
    try:
        y, file_sr = librosa.load(str(audio_path), sr=sr)
    except:
        return []
    
    hop = int(duration * (1 - overlap) * sr)
    samples = int(duration * sr)
    segments = []
    
    for start in range(0, len(y) - samples + 1, hop):
        segments.append(y[start:start + samples])
    
    return segments


def extract_features(audio_files, duration, overlap, cache_path):
    """Extrae MFCC de todos los audios."""
    if not args.no_cache and cache_path.exists():
        print(f"  [CACHE] Cargando features desde {cache_path}")
        data = torch.load(cache_path)
        return data['features'], data['labels_plate'], data['labels_electrode'], data['labels_current'], data['sessions']
    
    features, labels_plate, labels_electrode, labels_current, sessions = [], [], [], [], []
    
    print(f"Extrayendo MFCC de {len(audio_files)} archivos...")
    for i, info in enumerate(audio_files):
        if i % 50 == 0:
            print(f"  {i}/{len(audio_files)}")
        
        audio_path = AUDIO_BASE_DIR / info['path']
        segments = segment_audio(audio_path, duration, overlap)
        
        for segment in segments:
            feat = extract_mfcc_features(segment, sr=16000, n_mfcc=N_MFCC)
            features.append(feat)
            labels_plate.append(info['placa'])
            labels_electrode.append(info['electrodo'])
            labels_current.append(info['corriente'])
            sessions.append(info['sesion'])
    
    features = np.array(features)
    print(f"  Total segmentos: {len(features)}")
    
    # Guardar cache
    if not args.no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'features': features,
            'labels_plate': labels_plate,
            'labels_electrode': labels_electrode,
            'labels_current': labels_current,
            'sessions': sessions,
        }, cache_path)
        print(f"  [CACHE] Guardado en {cache_path}")
    
    return features, labels_plate, labels_electrode, labels_current, sessions


def train_model(model, train_loader, val_loader, device):
    """Entrena un modelo."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    
    best_acc = 0
    patience = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
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
        
        # Evaluar
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y_plate, _, _ = batch
                x = x.to(device)
                out = model(x)
                preds = out['plate'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_plate.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        
        if patience >= EARLY_STOP_PATIENCE:
            break
    
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    return model, swa_model, {'accuracy': best_acc}


def main():
    global args
    args = parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenamiento: {args.duration}s, k={args.k_folds}, overlap={args.overlap}")
    print(f"Device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Directorios
    duration_dir = Path(f"{args.duration:02d}seg")
    duration_dir.mkdir(exist_ok=True)
    cache_path = duration_dir / "mfcc_cache" / f"features_overlap_{args.overlap}.pt"
    models_dir = duration_dir / "modelos" / f"k{args.k_folds:02d}_overlap_{args.overlap}"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    audio_files = get_audio_files()
    features, labels_plate, labels_electrode, labels_current, sessions = \
        extract_features(audio_files, args.duration, args.overlap, cache_path)
    
    # Codificar labels
    le_plate = LabelEncoder()
    le_electrode = LabelEncoder()
    le_current = LabelEncoder()
    
    y_plate = le_plate.fit_transform(labels_plate)
    y_electrode = le_electrode.fit_transform(labels_electrode)
    y_current = le_current.fit_transform(labels_current)
    
    print(f"\nClases: Plate={len(le_plate.classes_)}, Electrode={len(le_electrode.classes_)}, Current={len(le_current.classes_)}")
    
    # K-Fold CV
    skf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, y_plate, sessions)):
        print(f"\nFold {fold_idx + 1}/{args.k_folds}")
        
        X_train, X_val = features[train_idx], features[val_idx]
        yp_train, yp_val = y_plate[train_idx], y_plate[val_idx]
        ye_train, ye_val = y_electrode[train_idx], y_electrode[val_idx]
        yc_train, yc_val = y_current[train_idx], y_current[val_idx]
        
        # Datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(yp_train),
            torch.LongTensor(ye_train),
            torch.LongTensor(yc_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(yp_val),
            torch.LongTensor(ye_val),
            torch.LongTensor(yc_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Modelo
        model = XVectorModel(
            input_size=240,
            num_classes_plate=len(le_plate.classes_),
            num_classes_electrode=len(le_electrode.classes_),
            num_classes_current=len(le_current.classes_)
        ).to(device)
        
        # Entrenar
        start = time.time()
        model, swa_model, metrics = train_model(model, train_loader, val_loader, device)
        fold_time = time.time() - start
        
        # Guardar
        torch.save(model.state_dict(), models_dir / f"model_fold_{fold_idx}.pt")
        torch.save(swa_model.state_dict(), models_dir / f"model_fold_{fold_idx}_swa.pt")
        
        fold_results.append({
            'fold': fold_idx,
            'time_seconds': fold_time,
            'accuracy': metrics['accuracy']
        })
        
        print(f"  Acc: {metrics['accuracy']:.4f}, Time: {fold_time:.1f}s")
    
    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_folds': args.k_folds,
            'duration': args.duration,
            'overlap': args.overlap,
            'seed': args.seed,
        },
        'fold_results': fold_results,
    }
    
    results_path = duration_dir / "resultados.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        if not isinstance(all_results, list):
            all_results = [all_results]
    else:
        all_results = []
    
    all_results.append(results)
    results_path.write_text(json.dumps(all_results, indent=2))
    
    print(f"\nResultados guardados: {results_path}")
    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()
