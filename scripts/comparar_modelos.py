#!/usr/bin/env python3
"""
Gráfica comparación de modelos (XVector, ECAPA-TDNN, FeedForward)
para una duración y overlap específicos.

Uso:
    python scripts/comparar_modelos.py 10seg --save
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent

TASKS = ["plate", "electrode", "current"]
MODEL_COLORS = {
    "xvector": "#e74c3c",
    "ecapa": "#3498db",
    "feedforward": "#2ecc71",
}
MODEL_NAMES = {
    "xvector": "X-Vector",
    "ecapa": "ECAPA-TDNN",
    "feedforward": "FeedForward",
}
TASK_NAMES = {
    "es": {
        "plate": "Espesor de Placa",
        "electrode": "Tipo de Electrodo",
        "current": "Tipo de Corriente",
    },
    "en": {
        "plate": "Plate Thickness",
        "electrode": "Electrode Type",
        "current": "Current Type",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Comparar modelos entrenados")
    parser.add_argument("duration_dir", type=str, help="Directorio (ej: 10seg)")
    parser.add_argument("--k-folds", type=int, default=None, help="Filtrar por K-folds")
    parser.add_argument("--overlap", type=float, default=None, help="Filtrar por overlap")
    parser.add_argument("--save", action="store_true", help="Guardar figura")
    parser.add_argument("--lang", type=str, default="es", choices=["es", "en"])
    return parser.parse_args()


def load_results(duration_dir: Path):
    results_path = duration_dir / "resultados.json"
    if not results_path.exists():
        print(f"Error: No se encontró {results_path}")
        return []
    with open(results_path, "r") as f:
        results = json.load(f)
    if not isinstance(results, list):
        results = [results]
    return results


def filter_results(results: list, k_folds: int = None, overlap: float = None):
    filtered = []
    for r in results:
        config = r.get("config", {})
        model_type = r.get("model_type", "")
        if not model_type:
            continue
        if k_folds is not None and config.get("n_folds") != k_folds:
            continue
        if overlap is not None and config.get("overlap", config.get("overlap_ratio")) != overlap:
            continue
        filtered.append(r)
    return filtered


def extract_metrics(results: list):
    """Extrae métricas de evaluación ciega (blind)."""
    metrics = {}
    for r in results:
        model_type = r.get("model_type", "")
        if not model_type or model_type not in MODEL_NAMES:
            continue
        
        blind = r.get("blind_evaluation", {})
        if not blind:
            continue
        
        m = {
            "plate": {"accuracy": blind.get("plate", {}).get("accuracy", 0)},
            "electrode": {"accuracy": blind.get("electrode", {}).get("accuracy", 0)},
            "current": {"accuracy": blind.get("current", {}).get("accuracy", 0)},
        }
        
        global_metrics = blind.get("global", {})
        if global_metrics:
            m["global"] = {
                "exact_match": global_metrics.get("exact_match", 0),
                "hamming": global_metrics.get("hamming_accuracy", 0),
            }
        
        metrics[model_type] = m
    return metrics


def extract_cv_metrics(results: list):
    """Extrae métricas de Cross-Validation (entrenamiento)."""
    metrics = {}
    for r in results:
        model_type = r.get("model_type", "")
        if not model_type or model_type not in MODEL_NAMES:
            continue
        
        fold_results = r.get("fold_results", [])
        if not fold_results:
            continue
        
        plate_acc = np.mean([f.get("accuracy_plate", 0) for f in fold_results])
        electrode_acc = np.mean([f.get("accuracy_electrode", 0) for f in fold_results])
        current_acc = np.mean([f.get("accuracy_current", 0) for f in fold_results])
        
        metrics[model_type] = {
            "plate": {"accuracy": plate_acc},
            "electrode": {"accuracy": electrode_acc},
            "current": {"accuracy": current_acc},
        }
    return metrics


def plot_comparison(metrics: dict, duration: str, lang: str = "es", save: bool = False):
    """Gráfica comparativa por modelo con métricas de evaluación ciega."""
    task_names = TASK_NAMES[lang]
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(MODEL_NAMES.keys())
    
    for i, model in enumerate(models):
        if model not in metrics:
            continue
        values = [metrics[model][task]["accuracy"] for task in TASKS]
        bars = axes[i].bar(TASKS, values, color=MODEL_COLORS[model], alpha=0.8, width=0.6)
        
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[i].set_title(MODEL_NAMES[model], fontweight='bold', fontsize=14)
        axes[i].set_ylim([0, 1.15])
        axes[i].set_ylabel('Accuracy', fontweight='bold')
        axes[i].set_xticks(range(len(TASKS)))
        axes[i].set_xticklabels([task_names[t][:12] for t in TASKS], rotation=15, ha='right')
    
    fig.suptitle(f'BLIND SET - {duration}', fontsize=16, fontweight='bold', y=1.02, color='black')
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / f"blind_modelos_{duration}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cv_comparison(metrics: dict, duration: str, lang: str = "es", save: bool = False):
    """Gráfica comparativa por modelo con métricas de Cross-Validation."""
    task_names = TASK_NAMES[lang]
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(MODEL_NAMES.keys())
    
    for i, model in enumerate(models):
        if model not in metrics:
            continue
        values = [metrics[model][task]["accuracy"] for task in TASKS]
        bars = axes[i].bar(TASKS, values, color=MODEL_COLORS[model], alpha=0.8, width=0.6)
        
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[i].set_title(MODEL_NAMES[model], fontweight='bold', fontsize=14)
        axes[i].set_ylim([0, 1.15])
        axes[i].set_ylabel('Accuracy', fontweight='bold')
        axes[i].set_xticks(range(len(TASKS)))
        axes[i].set_xticklabels([task_names[t][:12] for t in TASKS], rotation=15, ha='right')
    
    fig.suptitle(f'TEST SET - {duration}', fontsize=16, fontweight='bold', y=1.02, color='black')
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / f"cv_modelos_{duration}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_summary(metrics: dict, duration: str, lang: str = "es", save: bool = False):
    """Gráfico comparativo agrupado con métricas de evaluación ciega."""
    task_names = TASK_NAMES[lang]
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m for m in MODEL_NAMES.keys() if m in metrics]
    x = np.arange(len(TASKS))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [metrics[model][task]["accuracy"] for task in TASKS]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=MODEL_NAMES[model], 
                     color=MODEL_COLORS[model], alpha=0.85)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Tarea', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'BLIND SET - {duration}', fontweight='bold', fontsize=14, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([task_names[t] for t in TASKS])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / f"blind_resumen_{duration}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cv_summary(metrics: dict, duration: str, lang: str = "es", save: bool = False):
    """Gráfico comparativo agrupado con métricas de Cross-Validation."""
    task_names = TASK_NAMES[lang]
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m for m in MODEL_NAMES.keys() if m in metrics]
    x = np.arange(len(TASKS))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [metrics[model][task]["accuracy"] for task in TASKS]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=MODEL_NAMES[model], 
                     color=MODEL_COLORS[model], alpha=0.85)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Tarea', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'TEST SET - {duration}', fontweight='bold', fontsize=14, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([task_names[t] for t in TASKS])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / f"cv_resumen_{duration}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_global_metrics(metrics: dict, duration: str, lang: str = "es", save: bool = False):
    """Gráfico de métricas globales: Exact Match y Hamming Accuracy."""
    has_global = any('global' in m for m in metrics.values())
    if not has_global:
        return
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m for m in MODEL_NAMES.keys() if m in metrics and 'global' in metrics[m]]
    x = np.arange(len(models))
    width = 0.35
    
    exact_match = [metrics[m]['global']['exact_match'] for m in models]
    hamming = [metrics[m]['global']['hamming'] for m in models]
    
    bars1 = ax.bar(x - width/2, exact_match, width, label='Exact Match', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x + width/2, hamming, width, label='Hamming Accuracy', color='#3498db', alpha=0.85)
    
    for bar, val in zip(bars1, exact_match):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, hamming):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Modelo', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'BLIND SET - Métricas Globales - {duration}', fontweight='bold', fontsize=14, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    
    if save:
        output_path = ROOT_DIR / duration / "graficas" / f"global_metrics_{duration}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(metrics: dict, metric_type: str = "Test Set"):
    print("\n" + "="*60)
    print(f"RESUMEN: {metric_type}")
    print("="*60)
    print(f"{'Modelo':<15} {'Plate':<12} {'Electrode':<12} {'Current':<12} {'Promedio':<10}")
    if any('global' in m for m in metrics.values()):
        print(f"{'':15} {'':12} {'':12} {'':12} {'ExactMatch':<10} {'Hamming':<10}")
    print("-"*60)
    
    for model in MODEL_NAMES.keys():
        if model not in metrics:
            continue
        m = metrics[model]
        plate = m["plate"]["accuracy"]
        electrode = m["electrode"]["accuracy"]
        current = m["current"]["accuracy"]
        avg = (plate + electrode + current) / 3
        print(f"{MODEL_NAMES[model]:<15} {plate:>10.2%} {electrode:>10.2%} {current:>10.2%} {avg:>8.2%}", end="")
        
        if "global" in m:
            print(f"  {m['global']['exact_match']:>8.2%} {m['global']['hamming']:>8.2%}")
        else:
            print()
    
    print("="*60)
    
    avg_best = 0
    best_model = None
    for model in MODEL_NAMES.keys():
        if model not in metrics:
            continue
        m = metrics[model]
        avg = (m["plate"]["accuracy"] + m["electrode"]["accuracy"] + m["current"]["accuracy"]) / 3
        if avg > avg_best:
            avg_best = avg
            best_model = MODEL_NAMES[model]
    
    print(f"Mejor modelo: {best_model} ({avg_best:.2%})")
    print("="*60)


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
    
    results = filter_results(results, args.k_folds, args.overlap)
    print(f"Resultados filtrados: {len(results)}")
    
    blind_metrics = extract_metrics(results)
    cv_metrics = extract_cv_metrics(results)
    
    print(f"Modelos con métricas blind: {list(blind_metrics.keys())}")
    print(f"Modelos con métricas CV: {list(cv_metrics.keys())}")
    
    if not blind_metrics and not cv_metrics:
        print("No se pudieron extraer métricas.")
        sys.exit(1)
    
    if cv_metrics:
        print_summary(cv_metrics, "Test Set")
        plot_cv_comparison(cv_metrics, args.duration_dir, args.lang, args.save)
        plot_cv_summary(cv_metrics, args.duration_dir, args.lang, args.save)
    
    if blind_metrics:
        print_summary(blind_metrics, "Blind Set")
        plot_comparison(blind_metrics, args.duration_dir, args.lang, args.save)
        plot_summary(blind_metrics, args.duration_dir, args.lang, args.save)
        plot_global_metrics(blind_metrics, args.duration_dir, args.lang, args.save)


if __name__ == "__main__":
    main()
