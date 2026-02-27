#!/usr/bin/env python3
"""
Script para entrenar ECAPA y FeedForward con configuraciones de K y Overlap
para permitir comparación con X-Vector.

Configuraciones disponibles en X-Vector (10seg):
- k=2, 3, 10 con overlap=0.5 (comparar K)
- overlap=0.0, 0.25, 0.5, 0.75 con k=10 (comparar Overlap)

Uso:
    python scripts/entrenar_comparacion.py --duration 10 --k-folds 2,3,10 --overlap 0.5
    python scripts/entrenar_comparacion.py --duration 10 --k-folds 10 --overlap 0.0,0.25,0.5,0.75
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODELS = ["ecapa", "feedforward"]

K_VALUES_DEFAULT = [2, 3, 10]
OVERLAP_VALUES_DEFAULT = [0.5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena modelos para comparar K y Overlap"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duración en segundos (default: 10)",
    )
    parser.add_argument(
        "--k-folds",
        type=str,
        default="2,3,10",
        help="Valores de K separados por coma (default: 2,3,10)",
    )
    parser.add_argument(
        "--overlap",
        type=str,
        default="0.5",
        help="Valores de overlap separados por coma (default: 0.5)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ecapa,feedforward",
        help="Modelos a entrenar separados por coma (default: ecapa,feedforward)",
    )
    return parser.parse_args()


def parse_list(value: str) -> list:
    """Convierte string '1,2,3' a lista [1, 2, 3]."""
    return [item.strip() for item in value.split(",")]


def train_model(model: str, duration: int, k_folds: int, overlap: float):
    """Entrena un modelo específico."""
    if model == "ecapa":
        script = ROOT_DIR / "entrenar_ecapa.py"
    elif model == "feedforward":
        script = ROOT_DIR / "entrenar_feedforward.py"
    else:
        print(f"Modelo desconocido: {model}")
        return False
    
    cmd = [
        sys.executable,
        str(script),
        "--duration", str(duration),
        "--overlap", str(overlap),
        "--k-folds", str(k_folds),
    ]
    
    print(f"\n{'='*60}")
    print(f"Entrenando: {model.upper()}")
    print(f"Duración: {duration}s, K={k_folds}, Overlap={overlap}")
    print(f"Comando: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    return result.returncode == 0


def main():
    args = parse_args()
    
    k_values = parse_list(args.k_folds)
    overlap_values = parse_list(args.overlap)
    models = parse_list(args.models)
    
    k_values = [int(k) for k in k_values]
    overlap_values = [float(o) for o in overlap_values]
    
    print(f"\nConfiguración:")
    print(f"  Duración: {args.duration} segundos")
    print(f"  K-folds: {k_values}")
    print(f"  Overlaps: {overlap_values}")
    print(f"  Modelos: {models}")
    
    total = len(models) * len(k_values) * len(overlap_values)
    current = 0
    
    for model in models:
        for k in k_values:
            for overlap in overlap_values:
                current += 1
                print(f"\n[{current}/{total}] {model.upper()} - K={k}, overlap={overlap}")
                
                success = train_model(model, args.duration, k, overlap)
                
                if not success:
                    print(f"ERROR: Entrenamiento falló para {model} K={k} overlap={overlap}")
                    respuesta = input("¿Continuar con el siguiente? (s/n): ")
                    if respuesta.lower() != 's':
                        print("Entrenamiento cancelado.")
                        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTOS COMPLETADOS")
    print('='*60)


if __name__ == "__main__":
    main()
