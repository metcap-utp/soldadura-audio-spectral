"""
Grafica métricas vs cantidad de folds para una duración específica.

Genera figuras con Accuracy y F1 para las 3 etiquetas
(Plate Thickness, Electrode Type, Current Type) con estilo científico.

Uso:
    python scripts/graficar_folds.py 05seg              # Grafica para 5 segundos
    python scripts/graficar_folds.py 10seg --save       # Guarda la imagen
    python scripts/graficar_folds.py 05seg --metric f1  # Solo métrica F1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_styles import (
    COLORS,
    MARKERS,
    LINESTYLES,
    TAREAS_LABELS,
    METRIC_LABELS,
    setup_axis,
    save_figure,
)

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

TASKS = ["plate", "electrode", "current"]

# Configuración de idioma
I18N = {
    "es": {
        "task_names": {
            "plate": "Espesor de Placa",
            "electrode": "Tipo de Electrodo",
            "current": "Tipo de Corriente",
        },
        "xlabel_k": "Cantidad de Folds (K)",
        "title_metric": "{metric} vs Cantidad de Folds",
        "suptitle": "Métricas vs Cantidad de Folds (K-Fold CV)",
        "metric_names": {
            "accuracy": "Accuracy",
            "f1": "F1-Score",
            "precision": "Precisión",
            "recall": "Recall",
        },
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel_k": "Number of Folds (K)",
        "title_metric": "{metric} vs Number of Folds",
        "suptitle": "Metrics vs Number of Folds (K-Fold CV)",
        "metric_names": {
            "accuracy": "Accuracy",
            "f1": "F1-Score",
            "precision": "Precision",
            "recall": "Recall",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grafica métricas vs cantidad de folds"
    )
    parser.add_argument(
        "duration_dir",
        type=str,
        help="Directorio de duración (ej: 05seg, 10seg)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar la figura en lugar de mostrarla",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "accuracy", "f1", "precision", "recall"],
        help="Métrica a graficar (default: all)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Idioma (default: es)",
    )
    return parser.parse_args()


def load_results(duration_dir: Path):
    """Carga resultados del archivo JSON."""
    results_path = duration_dir / "resultados.json"
    
    if not results_path.exists():
        print(f"Error: No se encontró {results_path}")
        return []
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    if not isinstance(results, list):
        results = [results]
    
    return results


def extract_metrics_by_kfolds(results: list):
    """Extrae métricas organizadas por cantidad de folds."""
    data = {}
    
    for result in results:
        k = result.get("config", {}).get("n_folds", 5)
        
        if k not in data:
            data[k] = {
                "plate": {"accuracy": [], "f1": [], "precision": [], "recall": []},
                "electrode": {"accuracy": [], "f1": [], "precision": [], "recall": []},
                "current": {"accuracy": [], "f1": [], "precision": [], "recall": []},
            }
        
        # Extraer métricas de ensemble_results o fold_results
        if "ensemble_results" in result:
            for task in TASKS:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    if metric in result["ensemble_results"][task]:
                        data[k][task][metric].append(result["ensemble_results"][task][metric])
        elif "average_metrics" in result:
            # Usar promedios
            for task in TASKS:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    key = f"mean_{metric}_{task}"
                    if key in result["average_metrics"]:
                        data[k][task][metric].append(result["average_metrics"][key])
    
    return data


def plot_metrics(data: dict, duration: str, metric_type: str = "all", lang: str = "es", save: bool = False):
    """Genera la gráfica de métricas."""
    i18n_task = TAREAS_LABELS[lang]
    i18n_metric = METRIC_LABELS[lang]
    
    # Etiquetas
    xlabel_k = "Cantidad de Folds (K)" if lang == "es" else "Number of Folds (K)"
    title_metric = "{metric} vs Cantidad de Folds" if lang == "es" else "{metric} vs Number of Folds"
    suptitle = "Métricas vs Cantidad de Folds (K-Fold CV)" if lang == "es" else "Metrics vs Number of Folds (K-Fold CV)"
    
    # Determinar métricas a graficar
    if metric_type == "all":
        metrics_to_plot = ["accuracy", "f1"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        metrics_to_plot = [metric_type]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    
    # K values ordenados
    k_values = sorted(data.keys())
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx] if len(axes) > 1 else axes[0]
        
        for task in TASKS:
            values = []
            errors = []
            
            for k in k_values:
                metric_key = f"{metric}_{task}"
                if metric_key in data[k][task] and len(data[k][task][metric_key]) > 0:
                    vals = data[k][task][metric_key]
                    values.append(np.mean(vals))
                    errors.append(np.std(vals) if len(vals) > 1 else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Graficar línea con errorbars
            ax.errorbar(
                k_values,
                values,
                yerr=errors,
                label=i18n_task[task],
                color=COLORS[task],
                marker=MARKERS[metric],
                linestyle=LINESTYLES[metric],
                capsize=5,
                capthick=2,
                elinewidth=1.5,
            )
        
        ax.set_xlabel(xlabel_k, fontweight='bold')
        ax.set_ylabel(i18n_metric[metric], fontweight='bold')
        ax.set_title(title_metric.format(metric=i18n_metric[metric]), fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        setup_axis(ax, k_values, values)
    
    # Título general
    if len(axes) > 1:
        fig.suptitle(
            f"{suptitle}\nDuración: {duration}",
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
    
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / f"{duration}" / "graficas" / f"metricas_vs_folds_{metric_type}.png"
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
    
    print(f"Cargados {len(results)} experimentos")
    
    data = extract_metrics_by_kfolds(results)
    
    if not data:
        print("No se pudieron extraer métricas.")
        sys.exit(1)
    
    print(f"K-folds encontrados: {sorted(data.keys())}")
    
    plot_metrics(
        data,
        duration=args.duration_dir,
        metric_type=args.metric,
        lang=args.lang,
        save=args.save,
    )


if __name__ == "__main__":
    main()
