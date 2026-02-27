#!/usr/bin/env python3
"""
Script para ejecutar entrenamientos en lote.

Uso:
    python batch_train.py
"""

import itertools
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "entrenar_xvector.py",
]

DURATION = 5
OVERLAP_BASE = 0.0

KFOLDS_VARIOS = [3, 5, 7, 10, 15, 20]
KFOLDS_BASE = 10
OVERLAPS = [0.0, 0.25, 0.5, 0.75]


def run_training(script: str, duration: int, overlap: float, k_folds: int) -> None:
    cmd = [
        sys.executable,
        script,
        "--duration", str(duration),
        "--overlap", str(overlap),
        "--k-folds", str(k_folds),
    ]
    print(f"\n{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: El entrenamiento falló con código {result.returncode}")
        sys.exit(1)


def main() -> None:
    print("Iniciando entrenamientos en lote...")
    print(f"Duración: {DURATION}s")
    print(f"K-folds (variando): {KFOLDS_VARIOS}")
    print(f"Overlaps (variando): {OVERLAPS}")
    print(f"Modelos: {SCRIPTS}")

    for script in SCRIPTS:
        print(f"\n{'#'*60}")
        print(f"# Modelo: {script}")
        print(f"{'#'*60}")

        print(f"\n--- K-folds Variable (overlap={OVERLAP_BASE}) ---")
        for kfolds in KFOLDS_VARIOS:
            run_training(script, DURATION, OVERLAP_BASE, kfolds)

        print(f"\n--- Overlaps Variable (k-folds={KFOLDS_BASE}) ---")
        for overlap in OVERLAPS:
            run_training(script, DURATION, overlap, KFOLDS_BASE)

    print("\n¡Todos los entrenamientos completados!")


if __name__ == "__main__":
    main()
