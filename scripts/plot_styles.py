"""Configuración de estilos para gráficas del proyecto.

Estilos adaptados del proyecto graficas_tesis para mantener consistencia.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.markersize': 6,
    'lines.linewidth': 1.5,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    "plate": "#60a5fa",
    "electrode": "#3b82f6",
    "current": "#1d4ed8",
}

MARKERS = {
    "plate": "o",
    "electrode": "s",
    "current": "^",
    "accuracy": "o",
    "f1": "s",
}

LINESTYLES = {
    "accuracy": "-",
    "f1": "--",
}

TAREAS_LABELS = {
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

METRIC_LABELS = {
    "es": {
        "accuracy": "Accuracy",
        "f1": "F1-Score",
        "precision": "Precisión",
        "recall": "Recall",
    },
    "en": {
        "accuracy": "Accuracy",
        "f1": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
    },
}


def setup_axis(ax, x_values, y_values=None):
    """Configura los ejes con valores inteligentes."""
    if y_values is not None:
        y_values = list(y_values)
        y_lower, y_upper = _get_smart_ylim(y_values)
    else:
        y_lower, y_upper = 0, 1.0
    
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(y_lower, y_upper)


def _get_smart_ylim(values):
    """Calcula límites Y inteligentes basados en los datos."""
    if not values:
        return 0, 1.0
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val > 0.95:
        upper = 1.0
    else:
        upper = min(1.0, max_val + 0.05)
    
    if min_val > 0.9:
        lower = max(0, min_val - 0.1)
    elif min_val > 0.7:
        lower = max(0, min_val - 0.15)
    elif min_val > 0.5:
        lower = max(0, min_val - 0.2)
    else:
        lower = max(0, min_val - 0.1)
    
    return lower, upper


def save_figure(fig, output_path):
    """Guarda figura con configuración consistente."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format='png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.1
    )
    plt.close(fig)
