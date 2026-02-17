#!/usr/bin/env python3
"""
Script para graficar tiempos de entrenamiento desde los JSON de resultados.

Uso:
    python scripts/graficar_tiempos.py
    python scripts/graficar_tiempos.py --output tiempos.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Graficar tiempos de entrenamiento")
    parser.add_argument("--output", type=str, default="graficas/tiempos_entrenamiento.png")
    return parser.parse_args()


def load_timing_data(durations=[1, 2, 5, 10, 20, 30, 50]):
    """Carga datos de timing de todos los resultados.json."""
    data = []

    for dur in durations:
        result_path = Path(f"{dur:02d}seg/resultados.json")
        if result_path.exists():
            with open(result_path) as f:
                results = json.load(f)
                if isinstance(results, list) and len(results) > 0:
                    # Tomar el resultado más reciente
                    result = results[-1]
                    if 'timing' in result:
                        data.append({
                            'duration': dur,
                            'total_seconds': result['timing']['total_seconds'],
                            'total_minutes': result['timing']['total_minutes'],
                            'cv_seconds': result['timing']['cv_seconds'],
                            'cv_minutes': result['timing']['cv_minutes'],
                            'avg_per_fold_seconds': result['timing']['avg_per_fold_seconds'],
                            'avg_per_epoch_seconds': result['timing']['avg_per_epoch_seconds'],
                        })

    return data


def plot_times(data, output_path):
    """Genera gráficas de tiempos."""
    if not data:
        print("No se encontraron datos de timing")
        return

    durations = [d['duration'] for d in data]
    total_mins = [d['total_minutes'] for d in data]
    cv_mins = [d['cv_minutes'] for d in data]
    avg_fold_secs = [d['avg_per_fold_seconds'] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Tiempo total por duración
    ax = axes[0, 0]
    ax.plot(durations, total_mins, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Duración del segmento (s)', fontsize=11)
    ax.set_ylabel('Tiempo total (min)', fontsize=11)
    ax.set_title('Tiempo Total de Entrenamiento', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 2. Tiempo de CV por duración
    ax = axes[0, 1]
    ax.plot(durations, cv_mins, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('Duración del segmento (s)', fontsize=11)
    ax.set_ylabel('Tiempo CV (min)', fontsize=11)
    ax.set_title('Tiempo de Cross-Validation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 3. Tiempo promedio por fold
    ax = axes[1, 0]
    ax.plot(durations, avg_fold_secs, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Duración del segmento (s)', fontsize=11)
    ax.set_ylabel('Tiempo por fold (s)', fontsize=11)
    ax.set_title('Tiempo Promedio por Fold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 4. Tabla de tiempos
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for d in data:
        table_data.append([
            f"{d['duration']}s",
            f"{d['total_minutes']:.1f} min",
            f"{d['cv_minutes']:.1f} min",
            f"{d['avg_per_fold_seconds']:.0f} s",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Duración', 'Tiempo Total', 'Tiempo CV', 'Tiempo/Fold'],
        loc='center',
        cellLoc='center',
        colColours=['#2c3e50'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color de header
    for i in range(4):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Resumen de Tiempos', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {output_path}")

    # Imprimir resumen en consola
    print("\n" + "="*60)
    print("RESUMEN DE TIEMPOS DE ENTRENAMIENTO")
    print("="*60)
    print(f"{'Duración':<12} {'Total (min)':<15} {'CV (min)':<15} {'Tiempo/Fold (s)':<15}")
    print("-"*60)
    for d in data:
        print(f"{d['duration']}s{'':<10} {d['total_minutes']:<15.1f} {d['cv_minutes']:<15.1f} {d['avg_per_fold_seconds']:<15.0f}")
    print("="*60)


if __name__ == "__main__":
    args = parse_args()
    data = load_timing_data()
    plot_times(data, args.output)
