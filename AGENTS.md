# AGENTS.md - Agent Coding Guidelines

This file provides guidelines for agents working on this codebase.

## Project Overview

Audio spectral analysis project for weld sound classification using MFCC features and deep learning models (X-Vector, ECAPA-TDNN, FeedForward).

**Este proyecto está adaptado para comparación con vggish-backbone** - usa los mismos hiperparámetros, estructura de archivos y scripts de visualización.

## Project Structure

```
spectral-analysis/
├── entrenar.py              # Script principal de entrenamiento (adaptado de vggish)
├── inferir.py               # Inferencia y evaluación con ensemble voting
├── generar_splits.py        # Generación de splits estratificados
├── modelo.py                # Arquitectura X-Vector multi-task
├── entrenar_todos.sh        # Script batch para entrenar todas las combinaciones
├── evaluar_todos.sh         # Script batch para evaluar todos los modelos
│
├── {N}seg/                  # Directorios por duración: 01seg, 02seg, 05seg, etc.
│   ├── train.csv / test.csv / blind.csv
│   ├── train_overlap_{X}.csv         # Variantes con diferentes overlaps
│   ├── resultados.json               # Métricas de entrenamiento (acumulativo)
│   ├── inferencia.json               # Métricas de evaluación ciega
│   ├── data_stats.json               # Estadísticas de datos
│   ├── mfcc_cache/                   # Cache de features MFCC
│   ├── modelos/                      # Modelos entrenados
│   │   └── k{K:02d}_overlap_{ratio}/
│   │       ├── model_fold_{n}.pt
│   │       ├── model_fold_{n}_swa.pt
│   │       └── config.json
│   ├── matrices_confusion/           # Matrices de confusión generadas
│   └── graficas/                     # Figuras generadas
│
├── scripts/                 # Scripts de análisis y visualización
│   ├── graficar_folds.py    # Métricas vs K-folds
│   ├── graficar_duraciones.py  # Métricas vs duración
│   ├── graficar_overlap.py     # Comparación de overlaps
│   └── generar_confusion_matrices.py
│
├── utils/                   # Utilidades
│   ├── audio_utils.py       # Carga de audio y segmentación
│   └── timing.py            # Utilidades de timing
│
├── weld_audio_classifier/   # Paquete principal
│   ├── models/
│   │   ├── xvector.py       # X-Vector multi-task
│   │   ├── ecapa_tdnn.py    # ECAPA-TDNN
│   │   └── feedforward.py   # FeedForward
│   ├── features.py          # MFCC extraction con caching
│   └── ...
│
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md
```

- No usar este tipo de anotaciones: 
<!-- # =============================================================================
# Main
# =============================================================================
--> para separar secciones.
---

## Hiperparámetros (de vggish-backbone)

Los siguientes hiperparámetros son idénticos a los usados en vggish-backbone para permitir comparación justa:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `BATCH_SIZE` | 32 | Tamaño del batch |
| `NUM_EPOCHS` | 100 | Número máximo de épocas |
| `EARLY_STOP_PATIENCE` | 15 | Paciencia para early stopping |
| `LEARNING_RATE` | 1e-3 | Learning rate inicial |
| `WEIGHT_DECAY` | 1e-4 | Decay L2 |
| `LABEL_SMOOTHING` | 0.1 | Smoothing para CrossEntropy |
| `SWA_START` | 5 | Época de inicio de SWA |
| `N_MFCC` | 40 | Número de coeficientes MFCC |

**Optimizador:** AdamW  
**Schedulers:**
- `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)`
- `SWALR(swa_lr=1e-4)` desde epoch 5

**Pérdida Multi-Task:** Con incertidumbre aprendida (`log_vars`)

---

## Build/Lint/Test Commands

### Ejecución del Proyecto

```bash
# Generar splits para una duración específica
python generar_splits.py --duration 10 --overlap 0.5

# Entrenar modelo
python entrenar.py --duration 10 --overlap 0.5 --k-folds 5

# Evaluar en conjunto blind
python inferir.py --duration 10 --overlap 0.5 --k-folds 5 --evaluar

# Predicción de archivo individual
python inferir.py --duration 10 --overlap 0.5 --k-folds 5 --audio ruta/al/audio.wav
```

### Scripts de Visualización

```bash
# Métricas vs K-folds
python scripts/graficar_folds.py 10seg --save

# Métricas vs duración (comparar diferentes duraciones)
python scripts/graficar_duraciones.py --save

# Comparación de overlaps
python scripts/graficar_overlap.py --save --heatmap
```

### Scripts de Automatización

```bash
# Entrenar todas las combinaciones
./entrenar_todos.sh

# Evaluar todos los modelos entrenados
./evaluar_todos.sh
```

---

## Estructura de Directorios

### Features Cache
```
{N}seg/
└── mfcc_cache/
    └── mfcc_features_{duration}s_overlap_{ratio}.pkl
```

### Modelos Entrenados
```
{N}seg/
└── modelos/
    └── k{K:02d}_overlap_{ratio}/
        ├── model_fold_0.pt
        ├── model_fold_0_swa.pt
        ├── model_fold_1.pt
        ├── model_fold_1_swa.pt
        └── ...
```

### Resultados JSON

**`resultados.json`** (acumulativo):
```json
[
  {
    "timestamp": "2025-12-25T03:40:39.787190",
    "config": {
      "n_folds": 5,
      "random_seed": 42,
      "batch_size": 32,
      "epochs": 100,
      "learning_rate": 0.001,
      "weight_decay": 0.0001,
      "label_smoothing": 0.1,
      "overlap_ratio": 0.5
    },
    "fold_results": [
      {
        "fold": 0,
        "acc_plate": 0.7593,
        "acc_electrode": 0.8293,
        "acc_current": 0.9519,
        "f1_plate": 0.7595,
        "f1_electrode": 0.8295,
        "f1_current": 0.9519
      }
    ],
    "ensemble_results": {
      "plate": {"accuracy": 0.9899, "f1": 0.9899, "precision": 0.9899, "recall": 0.9899},
      "electrode": {...},
      "current": {...}
    },
    "improvement_vs_individual": {
      "plate": 0.1949,
      "electrode": 0.1358,
      "current": 0.0355
    },
    "global_metrics": {
      "exact_match_accuracy": 0.7204,
      "hamming_accuracy": 0.8620,
      "exact_match_improvement_vs_avg": 0.0862
    }
  }
]
```

**`inferencia.json`** (evaluación blind):
```json
[
  {
    "mode": "blind_evaluation",
    "segment_duration": 10.0,
    "n_samples": 447,
    "n_models": 5,
    "voting_method": "soft",
    "accuracy": {
      "plate_thickness": 0.7539,
      "electrode": 0.8613,
      "current_type": 0.9709
    },
    "macro_f1": {
      "plate_thickness": 0.7601,
      "electrode": 0.8525,
      "current_type": 0.9687
    },
    "confusion_matrices": {...},
    "global_metrics": {...}
  }
]
```

---

## Code Style Guidelines

### Imports

**Standard library first, then third-party, then local:**

```python
# 1. Standard library
import json
import time
from pathlib import Path
from typing import Optional

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# 3. Local
from weld_audio_classifier.config import load_config
from weld_audio_classifier.dataset import FeatureConfig
from weld_audio_classifier.models import XVectorModel
```

### Type Hints

Always use type hints for function parameters and return values:

```python
# Good
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    ...

# Avoid
def train_epoch(model, dataloader, criterion, optimizer, device):
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `XVectorModel`, `AudioDataset` |
| Functions | snake_case | `train_epoch`, `build_model` |
| Variables | snake_case | `learning_rate`, `num_classes` |
| Constants | UPPER_SNAKE | `DEFAULT_AUDIO_ROOT`, `MAX_EPOCHS` |
| Files | snake_case | `train_pytorch.py`, `pytorch_dataset.py` |

### Class Structure

```python
class AudioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: dict = None, task: str = None):
        self.X = torch.FloatTensor(X)
        self.y = y
        self.task = task
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.X[idx], self.labels[idx]
        return self.X[idx]
```

### Error Handling

Use specific exceptions and informative error messages:

```python
# Good
if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Avoid
if not audio_path.exists():
    raise Exception("Error")
```

### PyTorch Conventions

1. **Model forward pass**: Accept raw tensor, return logits o embeddings
2. **Device handling**: Always move data to device with `.to(device)`
3. **Loss computation**: Use `loss.detach().cpu().item()` for logging
4. **Evaluation**: Use `torch.no_grad()` context

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch in dataloader:
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
```

### Configuración

Ruta de audio configurable en `utils/audio_utils.py`:

```python
AUDIO_BASE_DIR = Path("/home/luis/projects/tesis/audio/vggish-backbone/audio")
```

### Estilo de Gráficas (Científico)

```python
import matplotlib.pyplot as plt

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

# Colores consistentes
COLORS = {
    "plate": "#2ecc71",      # Verde
    "electrode": "#3498db",  # Azul
    "current": "#e74c3c",    # Rojo
}
```

---

## Conventional Commits (Español)

Los mensajes de commit deben ser en español y seguir este formato:

```
<tipo>: <descripción>
```

**Tipos de commit:**
- `Agrega`: Nuevas funcionalidades o archivos
- `Arregla`: Corrección de bugs
- `Organiza`: Refactorización o mejora de código
- `Actualiza`: Actualización de dependencias o documentación
- `Elimina`: Eliminación de código o archivos

**Ejemplos:**
```
Agrega modelo X-Vector para clasificación de audio
Arregla error en dimensionamiento de tensores
Organiza estructura de archivos del proyecto
Actualiza AGENTS.md con reglas de commits
Elimina archivos temporales de caché
```

---

### Git Ignore

Archivos que deben ignorarse:
- `*/mfcc_cache/` - Cache de features MFCC
- `*/modelos/` - Modelos entrenados
- `*/matrices_confusion/` - Matrices de confusión
- `*/graficas/` - Figuras generadas
- `resultados.json` - Resultados acumulativos
- `inferencia.json` - Resultados de inferencia
- `.pyc__` - Python bytecode
- `.env` - Environment variables

---

## Running Tests

```bash
# Test imports de modelos
python -c "
from modelo import SMAWXVectorModel
import torch
model = SMAWXVectorModel()
x = torch.randn(2, 240)
out = model(x)
print(f'Output keys: {out.keys()}')
"

# Test extracción de features
python -c "
from utils.audio_utils import get_audio_files
files = get_audio_files()
print(f'Archivos encontrados: {len(files)}')
"

# Test generación de splits
python generar_splits.py --duration 1 --overlap 0.0

# Test entrenamiento (un solo fold para prueba)
python entrenar.py --duration 1 --overlap 0.0 --k-folds 3
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `entrenar.py` | Script principal de entrenamiento |
| `inferir.py` | Inferencia y evaluación con ensemble |
| `generar_splits.py` | Generación de splits estratificados |
| `modelo.py` | Arquitectura X-Vector multi-task |
| `weld_audio_classifier/models/xvector.py` | X-Vector implementation |
| `weld_audio_classifier/features.py` | MFCC extraction con caching |
| `scripts/graficar_folds.py` | Visualización de métricas vs folds |
| `utils/audio_utils.py` | Utilidades de audio |

---

## Common Issues

1. **Module not found**: Asegurarse de estar en el directorio raíz del proyecto
2. **Audio files not found**: Verificar ruta en `utils/audio_utils.py`
3. **CUDA out of memory**: Reducir `--batch-size` o usar `--device cpu`
4. **Missing splits**: Ejecutar `python generar_splits.py --duration X --overlap Y` primero
5. **Cache inválido**: Usar `--no-cache` para forzar recálculo de features
