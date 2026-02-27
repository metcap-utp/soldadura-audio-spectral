#!/usr/bin/env python3
"""
Script para ejecutar inferencias en lote (evaluación blind).

Uso:
    python batch_inferir.py
"""

import subprocess
import sys
from pathlib import Path

MODEL = "xvector"
DURATION = 5
OVERLAP_BASE = 0.0

KFOLDS_VARIOS = [3, 5, 7, 10, 15, 20]
KFOLDS_BASE = 10
OVERLAPS = [0.0, 0.25, 0.5, 0.75]


def run_inference(duration: int, overlap: float, k_folds: int, model: str) -> None:
    cmd = [
        sys.executable,
        "inferir.py",
        "--duration", str(duration),
        "--overlap", str(overlap),
        "--k-folds", str(k_folds),
        "--model", model,
    ]
    print(f"\n{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: La inferencia falló con código {result.returncode}")
        sys.exit(1)


def main() -> None:
    print("Iniciando inferencias en lote...")
    print(f"Modelo: {MODEL}")
    print(f"Duración: {DURATION}s")
    print(f"K-folds (variando): {KFOLDS_VARIOS}")
    print(f"Overlaps (variando): {OVERLAPS}")

    print(f"\n--- K-folds Variable (overlap={OVERLAP_BASE}) ---")
    for kfolds in KFOLDS_VARIOS:
        run_inference(DURATION, OVERLAP_BASE, kfolds, MODEL)

    print(f"\n--- Overlaps Variable (k-folds={KFOLDS_BASE}) ---")
    for overlap in OVERLAPS:
        run_inference(DURATION, overlap, KFOLDS_BASE, MODEL)

    print("\n¡Todas las inferencias completadas!")


if __name__ == "__main__":
    main()
