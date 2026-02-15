#!/usr/bin/env python3
"""
Inferencia con ensemble voting - Adaptado para spectral-analysis.

Uso:
    python inferir.py --duration 10 --k-folds 10 --evaluar
    python inferir.py --duration 10 --audio ruta/al/audio.wav
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from weld_audio_classifier.models.xvector import XVectorModel
from weld_audio_classifier.features import extract_mfcc_features
from utils.audio_utils import AUDIO_BASE_DIR

N_MFCC = 40


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia SMAW")
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--evaluar", action="store_true")
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_models(duration, overlap, k_folds, device):
    """Carga los modelos del ensemble."""
    models_dir = Path(f"{duration:02d}seg/modelos/k{k_folds:02d}_overlap_{overlap}")
    
    models = []
    for fold_idx in range(k_folds):
        model_path = models_dir / f"model_fold_{fold_idx}.pt"
        if model_path.exists():
            model = XVectorModel(input_size=240).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
    
    return models


def predict_audio(audio_path, models, duration, overlap, device):
    """Predice un archivo de audio."""
    y, sr = librosa.load(str(audio_path), sr=16000)
    
    # Segmentar
    hop = int(duration * (1 - overlap) * sr)
    samples = int(duration * sr)
    
    all_logits_plate = []
    all_logits_electrode = []
    all_logits_current = []
    
    for start in range(0, len(y) - samples + 1, hop):
        segment = y[start:start + samples]
        feat = extract_mfcc_features(segment, sr=16000, n_mfcc=N_MFCC)
        feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)
        
        # Soft voting: promediar logits
        fold_logits_plate = []
        fold_logits_electrode = []
        fold_logits_current = []
        
        with torch.no_grad():
            for model in models:
                out = model(feat_tensor)
                fold_logits_plate.append(out['plate'].cpu().numpy())
                fold_logits_electrode.append(out['electrode'].cpu().numpy())
                fold_logits_current.append(out['current'].cpu().numpy())
        
        all_logits_plate.append(np.mean(fold_logits_plate, axis=0))
        all_logits_electrode.append(np.mean(fold_logits_electrode, axis=0))
        all_logits_current.append(np.mean(fold_logits_current, axis=0))
    
    # Promediar sobre todos los segmentos
    final_plate = np.mean(all_logits_plate, axis=0).argmax()
    final_electrode = np.mean(all_logits_electrode, axis=0).argmax()
    final_current = np.mean(all_logits_current, axis=0).argmax()
    
    return final_plate, final_electrode, final_current


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Cargando ensemble: {args.duration}s, k={args.k_folds}")
    models = load_models(args.duration, args.overlap, args.k_folds, device)
    print(f"Modelos cargados: {len(models)}")
    
    if args.audio:
        plate, electrode, current = predict_audio(args.audio, models, args.duration, args.overlap, device)
        print(f"\nPredicción:")
        print(f"  Plate: {plate}")
        print(f"  Electrode: {electrode}")
        print(f"  Current: {current}")
    elif args.evaluar:
        print("Modo evaluación - implementar según necesidad")
    else:
        print("Use --audio o --evaluar")


if __name__ == "__main__":
    main()
