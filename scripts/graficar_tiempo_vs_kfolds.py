"""
Grafica tiempo de entrenamiento vs cantidad de folds.

Analiza cómo escala el tiempo de entrenamiento con el número de folds.

Uso:
    python graficar_tiempo_vs_kfolds.py 05seg
    python graficar_tiempo_vs_kfolds.py 10seg --save
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent

I18N = {
    "es": {
        "title": "Tiempo de Entrenamiento vs Cantidad de Folds",
        "xlabel": "Cantidad de Folds (K)",
        "ylabel": "Tiempo de Entrenamiento (segundos)",
        "total_time": "Tiempo Total",
        "avg_time": "Tiempo Promedio por Fold",
    },
    "en": {
        "title": "Training Time vs Number of Folds",
        "xlabel": "Number of Folds (K)",
        "ylabel": "Training Time (seconds)",
        "total_time": "Total Time",
        "avg_time": "Average Time per Fold",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grafica tiempo vs folds")
    parser.add_argument("duration_dir", type=str, help="Directorio de duración (ej: 05seg)")
    parser.add_argument("--save", action="store_true", help="Guardar figura")
    parser.add_argument("--lang", type=str, default="es", choices=["es", "en"])
    return parser.parse_args()


def load_results(duration_dir: Path):
    """Carga resultados."""
    results_path = duration_dir / "resultados.json"
    
    if not results_path.exists():
        print(f"Error: No se encontró {results_path}")
        return []
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    if not isinstance(results, list):
        results = [results]
    
    return results


def extract_time_by_kfolds(results: list):
    """Extrae tiempos organizados por K."""
    data = {}
    
    for result in results:
        k = result.get("config", {}).get("n_folds", 5)
        
        if k not in data:
            data[k] = {
                "total_times": [],
                "fold_times": [],
            }
        
        # Tiempo total del experimento
        if "average_metrics" in result:
            total_time = result["average_metrics"].get("mean_time_seconds", 0)
            if total_time > 0:
                data[k]["total_times"].append(total_time)
        
        # Tiempos individuales de folds
        if "fold_results" in result:
            for fold_result in result["fold_results"]:
                if "time_seconds" in fold_result:
                    data[k]["fold_times"].append(fold_result["time_seconds"])
    
    return data


def plot_time_analysis(data: dict, duration: str, lang: str, save: bool):
    """Genera gráfica de tiempo vs folds."""
    i18n = I18N[lang]
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = sorted(data.keys())
    
    # Gráfica 1: Tiempo total
    ax = axes[0]
    total_means = []
    total_stds = []
    
    for k in k_values:
        times = data[k]["total_times"]
        if times:
            total_means.append(np.mean(times))
            total_stds.append(np.std(times) if len(times) > 1 else 0)
        else:
            total_means.append(0)
            total_stds.append(0)
    
    ax.bar(k_values, total_means, yerr=total_stds, capsize=5, alpha=0.7, color='steelblue')
    ax.plot(k_values, total_means, 'o-', color='darkblue', linewidth=2, markersize=8)
    ax.set_xlabel(i18n["xlabel"], fontweight='bold')
    ax.set_ylabel(i18n["ylabel"], fontweight='bold')
    ax.set_title(i18n["total_time"], fontweight='bold')
    ax.set_xticks(k_values)
    
    # Gráfica 2: Tiempo promedio por fold
    ax = axes[1]
    avg_means = []
    avg_stds = []
    
    for k in k_values:
        times = data[k]["fold_times"]
        if times:
            avg_means.append(np.mean(times))
            avg_stds.append(np.std(times) if len(times) > 1 else 0)
        else:
            avg_means.append(0)
            avg_stds.append(0)
    
    ax.bar(k_values, avg_means, yerr=avg_stds, capsize=5, alpha=0.7, color='coral')
    ax.plot(k_values, avg_means, 'o-', color='darkred', linewidth=2, markersize=8)
    ax.set_xlabel(i18n["xlabel"], fontweight='bold')
    ax.set_ylabel(i18n["ylabel"], fontweight='bold')
    ax.set_title(i18n["avg_time"], fontweight='bold')
    ax.set_xticks(k_values)
    
    fig.suptitle(f"{i18n['title']}\nDuración: {duration}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / "tiempo_vs_kfolds.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    duration_dir = ROOT_DIR / args.duration_dir
    
    if not duration_dir.exists():
        print(f"Error: No existe el directorio {duration_dir}")
        sys.exit(1)
    
    print(f"Cargando resultados de {duration_dir}...")
    results = load_results(duration_dir)
    
    if not results:
        print("No se encontraron resultados.")
        sys.exit(1)
    
    data = extract_time_by_kfolds(results)
    
    if not data:
        print("No se pudieron extraer tiempos.")
        sys.exit(1)
    
    print(f"K-folds encontrados: {sorted(data.keys())}")
    
    plot_time_analysis(data, args.duration_dir, args.lang, args.save)


if __name__ == "__main__":
    main()
