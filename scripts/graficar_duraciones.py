"""
Grafica métricas vs duración de clips de audio.

Compara el rendimiento del modelo entre diferentes duraciones de segmento
(01seg, 02seg, 05seg, 10seg, 20seg, 30seg, 50seg).

Uso:
    python graficar_duraciones.py                    # Todas las duraciones
    python graficar_duraciones.py --k-folds 5        # Solo 5-fold
    python graficar_duraciones.py --save             # Guarda imagen
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
DURATION_DIRS = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]

TASKS = ["plate", "electrode", "current"]
COLORS = {
    "plate": "#2ecc71",
    "electrode": "#3498db",
    "current": "#e74c3c",
}

I18N = {
    "es": {
        "task_names": {
            "plate": "Espesor de Placa",
            "electrode": "Tipo de Electrodo",
            "current": "Tipo de Corriente",
        },
        "xlabel_dur": "Duración del Clip (segundos)",
        "title": "Métricas vs Duración del Clip",
        "metric_names": {
            "accuracy": "Accuracy",
            "f1": "F1-Score",
        },
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel_dur": "Clip Duration (seconds)",
        "title": "Metrics vs Clip Duration",
        "metric_names": {
            "accuracy": "Accuracy",
            "f1": "F1-Score",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grafica métricas vs duración")
    parser.add_argument("--k-folds", type=int, default=None, help="Filtrar por K-folds")
    parser.add_argument("--save", action="store_true", help="Guardar figura")
    parser.add_argument("--lang", type=str, default="es", choices=["es", "en"])
    return parser.parse_args()


def get_duration_value(dir_name: str) -> float:
    """Extrae valor numérico del nombre del directorio."""
    match = re.match(r"(\d+)seg", dir_name)
    if match:
        return float(match.group(1))
    return 0


def load_all_results(k_folds: int = None):
    """Carga resultados de todas las duraciones."""
    results_by_duration = {}
    
    for duration_dir in DURATION_DIRS:
        results_path = ROOT_DIR / duration_dir / "resultados.json"
        
        if not results_path.exists():
            continue
        
        with open(results_path, "r") as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            results = [results]
        
        # Filtrar por k_folds
        if k_folds is not None:
            results = [r for r in results if r.get("config", {}).get("n_folds", 5) == k_folds]
        
        if results:
            duration_value = get_duration_value(duration_dir)
            results_by_duration[duration_value] = {
                "dir": duration_dir,
                "results": results,
            }
    
    return results_by_duration


def extract_best_metrics(results_by_duration: dict, k_folds: int = None):
    """Extrae mejores métricas para cada duración."""
    metrics = {
        "durations": [],
        "duration_dirs": [],
        "plate": {"accuracy": [], "f1": []},
        "electrode": {"accuracy": [], "f1": []},
        "current": {"accuracy": [], "f1": []},
    }
    
    for duration in sorted(results_by_duration.keys()):
        data = results_by_duration[duration]
        results = data["results"]
        
        # Buscar mejor resultado
        best_result = None
        best_acc = 0
        
        for result in results:
            if "ensemble_results" in result:
                avg_acc = np.mean([
                    result["ensemble_results"][task]["accuracy"]
                    for task in TASKS
                ])
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_result = result
        
        if best_result:
            metrics["durations"].append(duration)
            metrics["duration_dirs"].append(data["dir"])
            
            for task in TASKS:
                if "ensemble_results" in best_result:
                    for metric in ["accuracy", "f1"]:
                        value = best_result["ensemble_results"][task].get(metric, 0)
                        metrics[task][metric].append(value)
    
    return metrics


def plot_metrics(metrics: dict, k_folds: int = None, lang: str = "es", save: bool = False):
    """Genera gráfica."""
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
        'lines.linewidth': 2,
        'lines.markersize': 8,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    durations = metrics["durations"]
    
    for idx, metric_type in enumerate(["accuracy", "f1"]):
        ax = axes[idx]
        
        for task in TASKS:
            values = metrics[task][metric_type]
            ax.plot(
                durations,
                values,
                label=i18n["task_names"][task],
                color=COLORS[task],
                marker="o" if metric_type == "accuracy" else "s",
                linestyle="-" if metric_type == "accuracy" else "--",
                linewidth=2,
                markersize=8,
            )
        
        ax.set_xlabel(i18n["xlabel_dur"], fontweight='bold')
        ax.set_ylabel(i18n["metric_names"][metric_type], fontweight='bold')
        ax.set_title(f"{i18n['metric_names'][metric_type]} vs Duración", fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xticks(durations)
    
    k_text = f"K={k_folds}" if k_folds else "Todos los K"
    fig.suptitle(f"{i18n['title']}\n{k_text}", fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / "graficas" / f"metricas_vs_duracion_{k_folds or 'all'}folds.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    print(f"Cargando resultados...")
    results_by_duration = load_all_results(args.k_folds)
    
    if not results_by_duration:
        print("No se encontraron resultados.")
        sys.exit(1)
    
    print(f"Duraciones encontradas: {sorted(results_by_duration.keys())}")
    
    metrics = extract_best_metrics(results_by_duration, args.k_folds)
    
    if not metrics["durations"]:
        print("No se pudieron extraer métricas.")
        sys.exit(1)
    
    plot_metrics(metrics, args.k_folds, args.lang, args.save)


if __name__ == "__main__":
    main()
