"""
Grafica métricas vs overlap ratio.

Compara el rendimiento entre diferentes ratios de solapamiento
(0.0, 0.25, 0.5, 0.75).

Uso:
    python graficar_overlap.py                 # Todos los overlaps
    python graficar_overlap.py --k-folds 5     # Solo 5-fold
    python graficar_overlap.py --save          # Guardar figura
    python graficar_overlap.py --heatmap       # Generar heatmap
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT_DIR = Path(__file__).parent.parent
OVERLAP_RATIOS = [0.0, 0.25, 0.5, 0.75]
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
        "xlabel": "Overlap Ratio",
        "title": "Métricas vs Overlap Ratio",
        "heatmap_title": "Heatmap: Duración vs Overlap",
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel": "Overlap Ratio",
        "title": "Metrics vs Overlap Ratio",
        "heatmap_title": "Heatmap: Duration vs Overlap",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grafica métricas vs overlap")
    parser.add_argument("--k-folds", type=int, default=5, help="Número de folds")
    parser.add_argument("--duration", type=int, default=None, help="Filtrar por duración")
    parser.add_argument("--save", action="store_true", help="Guardar figura")
    parser.add_argument("--heatmap", action="store_true", help="Generar heatmap")
    parser.add_argument("--lang", type=str, default="es", choices=["es", "en"])
    return parser.parse_args()


def load_results_for_overlap(duration: int, overlap: float, k_folds: int):
    """Carga resultados para una configuración específica."""
    duration_dir = ROOT_DIR / f"{duration:02d}seg"
    results_path = duration_dir / "resultados.json"
    
    if not results_path.exists():
        return None
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    if not isinstance(results, list):
        results = [results]
    
    # Buscar resultado con overlap y k_folds específicos
    for result in results:
        config = result.get("config", {})
        if (config.get("overlap_ratio") == overlap and 
            config.get("n_folds") == k_folds):
            return result
    
    return None


def extract_metrics_by_overlap(k_folds: int, duration_filter: int = None):
    """Extrae métricas organizadas por overlap."""
    data = {}
    
    durations = [duration_filter] if duration_filter else [1, 2, 5, 10, 20, 30, 50]
    
    for duration in durations:
        data[duration] = {}
        
        for overlap in OVERLAP_RATIOS:
            result = load_results_for_overlap(duration, overlap, k_folds)
            
            if result and "ensemble_results" in result:
                data[duration][overlap] = {
                    task: {
                        "accuracy": result["ensemble_results"][task]["accuracy"],
                        "f1": result["ensemble_results"][task]["f1"],
                    }
                    for task in TASKS
                }
    
    return data


def plot_overlap_comparison(data: dict, k_folds: int, duration_filter: int, lang: str, save: bool):
    """Genera gráfica de métricas vs overlap."""
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
    
    for idx, metric_type in enumerate(["accuracy", "f1"]):
        ax = axes[idx]
        
        for task in TASKS:
            overlaps = []
            values = []
            
            # Usar primera duración disponible o la filtrada
            duration = duration_filter if duration_filter else list(data.keys())[0]
            
            for overlap in OVERLAP_RATIOS:
                if overlap in data.get(duration, {}):
                    overlaps.append(overlap)
                    values.append(data[duration][overlap][task][metric_type])
            
            if overlaps:
                ax.plot(
                    overlaps,
                    values,
                    label=i18n["task_names"][task],
                    color=COLORS[task],
                    marker="o" if metric_type == "accuracy" else "s",
                    linestyle="-" if metric_type == "accuracy" else "--",
                    linewidth=2,
                    markersize=8,
                )
        
        ax.set_xlabel(i18n["xlabel"], fontweight='bold')
        ax.set_ylabel(metric_type.capitalize(), fontweight='bold')
        ax.set_title(f"{metric_type.capitalize()} vs Overlap", fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xticks(OVERLAP_RATIOS)
    
    dur_text = f"{duration_filter}s" if duration_filter else "Todas las duraciones"
    fig.suptitle(f"{i18n['title']}\nK={k_folds}, {dur_text}", fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        suffix = f"_{duration_filter}s" if duration_filter else ""
        output_path = ROOT_DIR / "graficas" / f"metricas_vs_overlap_k{k_folds}{suffix}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_heatmap(data: dict, metric: str, k_folds: int, lang: str, save: bool):
    """Genera heatmap de duración vs overlap."""
    i18n = I18N[lang]
    
    # Preparar matriz de datos
    durations = sorted(data.keys())
    overlaps = OVERLAP_RATIOS
    
    # Para cada tarea
    for task in TASKS:
        matrix = np.zeros((len(durations), len(overlaps)))
        
        for i, duration in enumerate(durations):
            for j, overlap in enumerate(overlaps):
                if overlap in data.get(duration, {}):
                    matrix[i, j] = data[duration][overlap][task][metric]
                else:
                    matrix[i, j] = np.nan
        
        # Crear heatmap
        plt.figure(figsize=(10, 6))
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[f"{o:.2f}" for o in overlaps],
            yticklabels=[f"{d}s" for d in durations],
            cbar_kws={'label': f'{metric.capitalize()}'},
            vmin=0,
            vmax=1,
        )
        
        plt.xlabel("Overlap Ratio", fontweight='bold')
        plt.ylabel("Duración (segundos)", fontweight='bold')
        plt.title(f"{i18n['heatmap_title']}\n{i18n['task_names'][task]} - K={k_folds}", fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = ROOT_DIR / "graficas" / f"heatmap_{task}_{metric}_k{k_folds}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap guardado: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    args = parse_args()
    
    print(f"Cargando resultados para K={args.k_folds}...")
    data = extract_metrics_by_overlap(args.k_folds, args.duration)
    
    if not data:
        print("No se encontraron resultados.")
        sys.exit(1)
    
    print(f"Duraciones encontradas: {sorted(data.keys())}")
    
    if args.heatmap:
        print("Generando heatmaps...")
        for metric in ["accuracy", "f1"]:
            plot_heatmap(data, metric, args.k_folds, args.lang, args.save)
    else:
        plot_overlap_comparison(data, args.k_folds, args.duration, args.lang, args.save)


if __name__ == "__main__":
    main()
