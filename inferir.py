#!/usr/bin/env python3
"""
Inferencia con ensemble voting - Evaluación en conjunto blind.

Genera inferencia.json con métricas completas y matrices de confusión.

Uso:
    python inferir.py --duration 10 --model xvector --k-folds 10 --overlap 0.5
    python inferir.py --duration 10 --model all --k-folds 10
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.audio_utils import AUDIO_BASE_DIR
from utils.timing import Timer
from utils.logging_utils import setup_log_file

N_MFCC = 40


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia SMAW - Evaluación blind")
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--k-folds", type=int, default=10)
    parser.add_argument("--model", type=str, default="all", 
                        choices=["xvector", "ecapa", "feedforward", "all"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_model_class(model_type):
    """Retorna la clase del modelo según el tipo."""
    if model_type == "xvector":
        from models.modelo_xvector import XVectorModel
        return XVectorModel
    elif model_type == "ecapa":
        from models.modelo_ecapa import ECAPAMultiTask
        return ECAPAMultiTask
    elif model_type == "feedforward":
        from models.modelo_feedforward import FeedForwardMultiTask
        return FeedForwardMultiTask
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")


def load_models(duration, overlap, k_folds, model_type, device):
    """Carga los modelos del ensemble."""
    models_dir = Path(f"{duration:02d}seg/modelos/{model_type}/k{k_folds:02d}_overlap_{overlap}")
    
    ModelClass = get_model_class(model_type)
    models = []
    
    for fold_idx in range(k_folds):
        model_path = models_dir / f"model_fold_{fold_idx}_swa.pt"
        is_swa = True
        if not model_path.exists():
            model_path = models_dir / f"model_fold_{fold_idx}.pt"
            is_swa = False
        
        if model_path.exists():
            input_size = {
                "xvector": 40,
                "ecapa": N_MFCC,
                "feedforward": N_MFCC * 6,
            }[model_type]
            
            model = ModelClass(input_size=input_size)
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            
            if is_swa and 'n_averaged' in state_dict:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() 
                              if k != 'n_averaged'}
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
    
    return models


def load_cached_data(duration_dir, model_type, overlap):
    """Carga datos desde caché si existe."""
    cache_names = {
        "xvector": f"xvector_features_overlap_{overlap}.pt",
        "ecapa": f"ecapa_features_overlap_{overlap}.pt",
        "feedforward": f"feedforward_features_overlap_{overlap}.pt"
    }
    cache_path = duration_dir / "mfcc_cache" / cache_names.get(model_type, f"features_overlap_{overlap}.pt")
    
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        return data
    return None


def extract_features_for_model(df, model_type, duration, overlap):
    """Extrae features para un modelo específico."""
    import librosa
    
    features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"    Procesados {idx}/{len(df)} segmentos...")
        
        audio_path = AUDIO_BASE_DIR / row['audio_path']
        segment_idx = int(row['segment_index'])
        
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            hop = int(duration * (1 - overlap) * sr)
            samples = int(duration * sr)
            start = segment_idx * hop
            
            if start + samples > len(y):
                segment = np.zeros(samples)
                available = len(y) - start
                if available > 0:
                    segment[:available] = y[start:start + available]
            else:
                segment = y[start:start + samples]
            
            if model_type == "xvector":
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                features.append(mfcc)
            elif model_type == "ecapa":
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                features.append(mfcc)
            elif model_type == "feedforward":
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                stacked = np.vstack([mfcc, delta1, delta2])
                feat = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
                features.append(feat)
                
        except Exception as e:
            print(f"    Error en {audio_path}: {e}")
            if model_type == "feedforward":
                features.append(np.zeros(N_MFCC * 6))
            else:
                features.append(np.zeros((N_MFCC, int(duration * 16000 / 512))))
    
    return features


def predict_ensemble(models, features, model_type, device):
    """Predice usando ensemble voting (soft voting)."""
    all_preds_plate = []
    all_preds_electrode = []
    all_preds_current = []
    
    with torch.no_grad():
        for feat in features:
            if model_type == "xvector":
                x = torch.FloatTensor(feat).unsqueeze(0).to(device)
            elif model_type == "ecapa":
                x = torch.FloatTensor(feat).unsqueeze(0).to(device)  # (1, n_mfcc, time)
            else:  # feedforward
                x = torch.FloatTensor(feat).unsqueeze(0).to(device)
            
            fold_logits_plate = []
            fold_logits_electrode = []
            fold_logits_current = []
            
            for model in models:
                out = model(x)
                fold_logits_plate.append(out['plate'].cpu().numpy())
                fold_logits_electrode.append(out['electrode'].cpu().numpy())
                fold_logits_current.append(out['current'].cpu().numpy())
            
            avg_plate = np.mean(fold_logits_plate, axis=0).argmax()
            avg_electrode = np.mean(fold_logits_electrode, axis=0).argmax()
            avg_current = np.mean(fold_logits_current, axis=0).argmax()
            
            all_preds_plate.append(avg_plate)
            all_preds_electrode.append(avg_electrode)
            all_preds_current.append(avg_current)
    
    return np.array(all_preds_plate), np.array(all_preds_electrode), np.array(all_preds_current)


def evaluate_model(y_true_plate, y_true_electrode, y_true_current,
                   y_pred_plate, y_pred_electrode, y_pred_current):
    """Calcula métricas completas incluyendo exact match y hamming accuracy."""
    n_samples = len(y_true_plate)
    
    metrics = {
        "plate": {
            "accuracy": accuracy_score(y_true_plate, y_pred_plate),
            "f1_macro": f1_score(y_true_plate, y_pred_plate, average='macro'),
        },
        "electrode": {
            "accuracy": accuracy_score(y_true_electrode, y_pred_electrode),
            "f1_macro": f1_score(y_true_electrode, y_pred_electrode, average='macro'),
        },
        "current": {
            "accuracy": accuracy_score(y_true_current, y_pred_current),
            "f1_macro": f1_score(y_true_current, y_pred_current, average='macro'),
        },
        "global": {
            "exact_match": 0.0,
            "hamming_accuracy": 0.0,
        }
    }
    
    exact_match_count = 0
    hamming_correct = 0
    
    for i in range(n_samples):
        plate_correct = (y_true_plate[i] == y_pred_plate[i])
        electrode_correct = (y_true_electrode[i] == y_pred_electrode[i])
        current_correct = (y_true_current[i] == y_pred_current[i])
        
        if plate_correct and electrode_correct and current_correct:
            exact_match_count += 1
        
        hamming_correct += int(plate_correct) + int(electrode_correct) + int(current_correct)
    
    metrics["global"]["exact_match"] = exact_match_count / n_samples
    metrics["global"]["hamming_accuracy"] = hamming_correct / (n_samples * 3)
    
    confusion_matrices = {
        "plate": confusion_matrix(y_true_plate, y_pred_plate).tolist(),
        "electrode": confusion_matrix(y_true_electrode, y_pred_electrode).tolist(),
        "current": confusion_matrix(y_true_current, y_pred_current).tolist(),
    }
    
    return metrics, confusion_matrices


def save_confusion_matrix_plot(confusion_matrices, output_dir, model_type, label_encoders):
    """Guarda matrices de confusión como imágenes."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_labels = {
        "plate": list(label_encoders['plate'].classes_),
        "electrode": list(label_encoders['electrode'].classes_),
        "current": list(label_encoders['current'].classes_),
    }
    
    for task, matrix in confusion_matrices.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = task_labels[task]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusión - {task.capitalize()} ({model_type})')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = output_dir / f"{model_type}_{task}_confusion.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_inference(duration, overlap, k_folds, model_type, device):
    """Ejecuta inferencia completa para un modelo."""
    import pandas as pd
    duration_dir = Path(f"{duration:02d}seg")
    
    print(f"\n{'='*60}")
    print(f"INFERENCIA: {model_type.upper()} - {duration}s")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    cached_data = load_cached_data(duration_dir, model_type, overlap)
    
    if cached_data:
        print(f"Usando caché existente...")
        y_blind = cached_data['y_blind']
        y_train = cached_data['y_train']
        y_test = cached_data['y_test']
        features = cached_data['X_blind']
        
        le_plate = LabelEncoder()
        le_electrode = LabelEncoder()
        le_current = LabelEncoder()
        
        all_plate = np.concatenate([y_train['plate'], y_test['plate'], y_blind['plate']])
        all_electrode = np.concatenate([y_train['electrode'], y_test['electrode'], y_blind['electrode']])
        all_current = np.concatenate([y_train['current'], y_test['current'], y_blind['current']])
        
        le_plate.fit(all_plate)
        le_electrode.fit(all_electrode)
        le_current.fit(all_current)
        
        y_true_plate = le_plate.transform(y_blind['plate'])
        y_true_electrode = le_electrode.transform(y_blind['electrode'])
        y_true_current = le_current.transform(y_blind['current'])
        
        n_samples = len(y_true_plate)
        print(f"Muestras blind: {n_samples}")
    else:
        print("Extrayendo features desde CSVs...")
        blind_csv = duration_dir / "blind.csv"
        train_csv = duration_dir / "train.csv"
        test_csv = duration_dir / "test.csv"
        
        blind_df = pd.read_csv(blind_csv)
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        le_plate = LabelEncoder()
        le_electrode = LabelEncoder()
        le_current = LabelEncoder()
        
        all_plate = np.concatenate([train_df['placa'], test_df['placa'], blind_df['placa']])
        all_electrode = np.concatenate([train_df['electrodo'], test_df['electrodo'], blind_df['electrodo']])
        all_current = np.concatenate([train_df['corriente'], test_df['corriente'], blind_df['corriente']])
        
        le_plate.fit(all_plate)
        le_electrode.fit(all_electrode)
        le_current.fit(all_current)
        
        y_true_plate = le_plate.transform(blind_df['placa'])
        y_true_electrode = le_electrode.transform(blind_df['electrodo'])
        y_true_current = le_current.transform(blind_df['corriente'])
        
        n_samples = len(y_true_plate)
        print(f"Muestras blind: {n_samples}")
        
        with Timer("Extracción de features"):
            features = extract_features_for_model(blind_df, model_type, duration, overlap)
    
    with Timer("Carga de modelos"):
        models = load_models(duration, overlap, k_folds, model_type, device)
        print(f"Modelos cargados: {len(models)}")
    
    if len(models) == 0:
        print(f"ERROR: No se encontraron modelos para {model_type}")
        return None
    
    if not cached_data:
        with Timer("Extracción de features"):
            blind_df = pd.read_csv(duration_dir / "blind.csv")
            features = extract_features_for_model(blind_df, model_type, duration, overlap)
    
    with Timer("Predicción ensemble"):
        y_pred_plate, y_pred_electrode, y_pred_current = predict_ensemble(
            models, features, model_type, device
        )
    
    with Timer("Evaluación"):
        metrics, confusion_matrices = evaluate_model(
            y_true_plate, y_true_electrode, y_true_current,
            y_pred_plate, y_pred_electrode, y_pred_current
        )
    
    total_time = time.time() - start_time
    
    label_encoders = {
        'plate': le_plate,
        'electrode': le_electrode,
        'current': le_current,
    }
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "config": {
            "duration": duration,
            "overlap": overlap,
            "k_folds": k_folds,
            "n_models": len(models),
            "n_samples": n_samples,
        },
        # Schema canónico (compatible con vggish/yamnet)
        "execution_time": {
            "seconds": round(total_time, 2),
            "minutes": round(total_time / 60, 2),
            "hours": round(total_time / 3600, 4),
        },
        # Mantener timing para compatibilidad hacia atrás
        "timing": {
            "total_seconds": total_time,
        },
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "label_classes": {
            "plate": list(le_plate.classes_),
            "electrode": list(le_electrode.classes_),
            "current": list(le_current.classes_),
        }
    }
    
    print(f"\nResultados {model_type}:")
    print(f"  Plate: Acc={metrics['plate']['accuracy']:.4f}, F1={metrics['plate']['f1_macro']:.4f}")
    print(f"  Electrode: Acc={metrics['electrode']['accuracy']:.4f}, F1={metrics['electrode']['f1_macro']:.4f}")
    print(f"  Current: Acc={metrics['current']['accuracy']:.4f}, F1={metrics['current']['f1_macro']:.4f}")
    print(f"  Exact Match: {metrics['global']['exact_match']:.4f}")
    print(f"  Hamming Accuracy: {metrics['global']['hamming_accuracy']:.4f}")
    print(f"  Tiempo total: {total_time:.1f}s")
    
    matrices_dir = duration_dir / "matrices_confusion"
    save_confusion_matrix_plot(confusion_matrices, matrices_dir, model_type, label_encoders)
    print(f"  Matrices guardadas en: {matrices_dir}")
    
    return results


def main():
    args = parse_args()
    
    # Set up logging
    log_file, log_path = setup_log_file(
        Path(".") / "logs", "inferir", suffix=f"_{int(args.duration):02d}seg_{args.model}"
    )
    sys.stdout = log_file
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    duration_dir = Path(f"{args.duration:02d}seg")
    
    model_types = ["xvector", "ecapa", "feedforward"] if args.model == "all" else [args.model]
    
    all_results = []
    
    for model_type in model_types:
        result = run_inference(args.duration, args.overlap, args.k_folds, model_type, device)
        if result:
            all_results.append(result)
    
    if all_results:
        inferencia_path = duration_dir / "inferencia.json"
        
        existing = []
        if inferencia_path.exists():
            with open(inferencia_path) as f:
                existing = json.load(f)
        
        existing.extend(all_results)
        
        with open(inferencia_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Resultados guardados en: {inferencia_path}")
        print(f"Logs guardados en: {log_path}")
        print(f"{'='*60}")
    
    # Close log file
    log_file.close()


if __name__ == "__main__":
    main()
