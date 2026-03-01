# AGENTS.md

Guidelines for agents working on this audio spectral analysis codebase.

## Project Overview

Weld sound classification using MFCC features and deep learning (X-Vector, ECAPA-TDNN, FeedForward). Multi-task learning for plate thickness, electrode type, and current type.

## Build/Lint/Test Commands

```bash
# Generate splits
python generar_splits.py --duration 10 --overlap 0.5

# Train a model
python entrenar_xvector.py --duration 10 --overlap 0.5 --k-folds 10
python entrenar_ecapa.py --duration 10 --overlap 0.5 --k-folds 10
python entrenar_feedforward.py --duration 10 --overlap 0.5 --k-folds 10

# Run inference
python inferir.py --duration 10 --model xvector --k-folds 10 --overlap 0.5

# Quick test (minimal config)
python generar_splits.py --duration 1 --overlap 0.0
python entrenar_xvector.py --duration 1 --overlap 0.0 --k-folds 2

# Verify imports work
python -c "from weld_audio_classifier.models import XVectorModel, ECAPAMultiTask, FeedForwardMultiTask; print('OK')"

# Visualization scripts
python scripts/graficar_folds.py 10seg --save
python scripts/graficar_duraciones.py --save
```

## Execution Constraints

**IMPORTANTE: Las tareas de entrenamiento e inferencia deben ejecutarse de forma SECUENCIAL, nunca simultánea.**

- No ejecutar `entrenar_*.py` en paralelo con otros entrenamientos
- No ejecutar `inferir.py` en paralelo con entrenamientos
- Esperar a que termine un proceso antes de iniciar otro
- Esto se debe al uso intensivo de GPU y caché de archivos

## Log Files

All training and inference scripts automatically generate log files in the `logs/` directory.

### Log Locations

- **Training**: `logs/entrenar_[architecture]_[duration]seg_[timestamp].log`
  - Example: `logs/entrenar_ecapa_05seg_20250228_143000.log`
- **Inference**: `logs/inferir_[duration]seg_[model]_[timestamp].log`
  - Example: `logs/inferir_05seg_xvector_20250228_150000.log`

### Timestamp Format

Log files use `YYYYMMDD_HHMMSS` format:

- `YYYY`: Year (2025)
- `MM`: Month (01-12)
- `DD`: Day (01-31)
- `HH`: Hour (00-23)
- `MM`: Minute (00-59)
- `SS`: Second (00-59)

### Log Contents

Log files contain:

- All script output (print statements)
- Training metrics (loss, accuracy per fold)
- Execution times (total, per-fold, feature extraction)
- Backbone information (vggish, yamnet, spectral-mfcc)
- Any errors or warnings during execution

### Log Management

Log files are excluded from version control (`.gitignore`). To clean up old logs:

```bash
# Delete all logs
rm logs/*.log

# Delete logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete
```

### Reading Logs

```bash
# View last training log (last 50 lines)
tail -n 50 logs/entrenar_ecapa_05seg_*.log

# Search for errors
grep -i "error\|exception" logs/*.log

# View specific execution log
cat logs/entrenar_xvector_10seg_20250228_143000.log
```

## Project Structure

- `models/` - Model definitions (modelo_xvector.py, modelo_ecapa.py, modelo_feedforward.py)
  - Re-exports from `weld_audio_classifier.models` for consistency with vggish and yamnet backbones
- `logs/` - Training and inference log files
- `utils/` - Utilities (audio_utils.py, timing.py, logging_utils.py)
- `weld_audio_classifier/` - Main package with models, features, and utilities
- `{N}seg/` - Results per segment duration
  - `modelos/{architecture}/k{K}_overlap_{ratio}/` - Trained model checkpoints
  - `resultados.json` - Training metrics (cumulative)
  - `inferencia.json` - Evaluation metrics (cumulative)

### Model Imports

```python
from models.modelo_xvector import XVectorModel
from models.modelo_ecapa import ECAPAMultiTask
from models.modelo_feedforward import FeedForwardMultiTask
```

## File Naming Conventions

**Always check existing patterns before creating new files.**

| Type        | Pattern                                            | Example                                  |
| ----------- | -------------------------------------------------- | ---------------------------------------- |
| Cache files | `{model}_features_overlap_{overlap}.pt`            | `xvector_features_overlap_0.5.pt`        |
| Model dirs  | `{N}seg/modelos/{model}/k{K:02d}_overlap_{ratio}/` | `10seg/modelos/xvector/k10_overlap_0.5/` |
| Model files | `model_fold_{n}.pt`, `model_fold_{n}_swa.pt`       | `model_fold_0_swa.pt`                    |
| Splits      | `{N}seg/train.csv`, `test.csv`, `blind.csv`        | `10seg/train.csv`                        |

## Code Style

### Imports Order

```python
# 1. Standard library
import json
from pathlib import Path
from typing import Dict, Optional

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# 3. Local
from weld_audio_classifier.models import XVectorModel
from utils.audio_utils import AUDIO_BASE_DIR
```

### Type Hints

Always use type hints:

```python
def train_epoch(model: nn.Module, dataloader: DataLoader, device: str) -> tuple[float, float]:
    ...
```

### Naming Conventions

| Element   | Convention  | Example                           |
| --------- | ----------- | --------------------------------- |
| Classes   | PascalCase  | `XVectorModel`, `AudioDataset`    |
| Functions | snake_case  | `train_epoch`, `extract_features` |
| Variables | snake_case  | `learning_rate`, `num_classes`    |
| Constants | UPPER_SNAKE | `BATCH_SIZE`, `AUDIO_BASE_DIR`    |
| Files     | snake_case  | `entrenar_xvector.py`             |

### Error Handling

```python
# Good
if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Avoid
raise Exception("Error")
```

### PyTorch Patterns

```python
# Training
model.train()
for batch in dataloader:
    X, y = batch
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    for batch in dataloader:
        ...
loss_value = loss.detach().cpu().item()
```

## Hyperparameters

| Parameter           | Value |
| ------------------- | ----- |
| BATCH_SIZE          | 32    |
| NUM_EPOCHS          | 100   |
| EARLY_STOP_PATIENCE | 15    |
| LEARNING_RATE       | 1e-3  |
| WEIGHT_DECAY        | 1e-4  |
| LABEL_SMOOTHING     | 0.1   |
| SWA_START           | 5     |
| N_MFCC              | 40    |

Optimizer: AdamW. Schedulers: `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` + `SWALR(swa_lr=1e-4)` from epoch 5.

## Plot Style

Los estilos de gráfica se configuran centralmente en `scripts/plot_styles.py`. Este módulo define:

```python
# Estilos matplotlib (se aplican automáticamente al importar)
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
})

# Colores por tarea
COLORS = {"plate": "#2ecc71", "electrode": "#3498db", "current": "#e74c3c"}

# Para usar en nuevos scripts:
from plot_styles import COLORS, MARKERS, LINESTYLES, TAREAS_LABELS, METRIC_LABELS, setup_axis, save_figure
```

## Commit Messages (Spanish)

Format: `<tipo>: <descripción>`. Types: `Agrega`, `Arregla`, `Organiza`, `Actualiza`, `Elimina`.

## Key Files

| File                            | Purpose                        |
| ------------------------------- | ------------------------------ |
| `entrenar_xvector.py`           | X-Vector training              |
| `entrenar_ecapa.py`             | ECAPA-TDNN training            |
| `entrenar_feedforward.py`       | FeedForward training           |
| `inferir.py`                    | Inference with ensemble voting |
| `generar_splits.py`             | Stratified split generation    |
| `weld_audio_classifier/models/` | Model architectures            |
| `utils/audio_utils.py`          | Audio loading utilities        |

## Git Ignore

- `*/mfcc_cache/`, `*/modelos/`, `*/graficas/`
- `resultados.json`, `inferencia.json`

## Rules

1. No generar logs temporales, o eliminarlos después de uso.
2. No usar comentarios de separación tipo `# === Main ===`.
3. Verificar consistencia de nombres antes de crear archivos.
