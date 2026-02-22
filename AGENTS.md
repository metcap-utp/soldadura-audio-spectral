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

## File Naming Conventions

**Always check existing patterns before creating new files.**

| Type | Pattern | Example |
|------|---------|---------|
| Cache files | `{model}_features_overlap_{overlap}.pt` | `xvector_features_overlap_0.5.pt` |
| Model dirs | `{N}seg/modelos/{model}/k{K:02d}_overlap_{ratio}/` | `10seg/modelos/xvector/k10_overlap_0.5/` |
| Model files | `model_fold_{n}.pt`, `model_fold_{n}_swa.pt` | `model_fold_0_swa.pt` |
| Splits | `{N}seg/train.csv`, `test.csv`, `blind.csv` | `10seg/train.csv` |

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
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[float, float]:
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `XVectorModel`, `AudioDataset` |
| Functions | snake_case | `train_epoch`, `extract_features` |
| Variables | snake_case | `learning_rate`, `num_classes` |
| Constants | UPPER_SNAKE | `BATCH_SIZE`, `AUDIO_BASE_DIR` |
| Files | snake_case | `entrenar_xvector.py` |

### Error Handling

```python
# Good
if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Avoid
if not audio_path.exists():
    raise Exception("Error")
```

### PyTorch Patterns

```python
# Training loop
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

# Log loss
loss_value = loss.detach().cpu().item()
```

## Hyperparameters (from vggish-backbone)

| Parameter | Value |
|-----------|-------|
| BATCH_SIZE | 32 |
| NUM_EPOCHS | 100 |
| EARLY_STOP_PATIENCE | 15 |
| LEARNING_RATE | 1e-3 |
| WEIGHT_DECAY | 1e-4 |
| LABEL_SMOOTHING | 0.1 |
| SWA_START | 5 |
| N_MFCC | 40 |

Optimizer: AdamW. Schedulers: `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` + `SWALR(swa_lr=1e-4)` from epoch 5.

## Plot Style

```python
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
})

COLORS = {"plate": "#2ecc71", "electrode": "#3498db", "current": "#e74c3c"}
```

## Commit Messages (Spanish)

Format: `<tipo>: <descripción>`

Types: `Agrega`, `Arregla`, `Organiza`, `Actualiza`, `Elimina`

Examples:
```
Agrega modelo X-Vector para clasificación de audio
Arregla error en dimensionamiento de tensores
Organiza estructura de archivos del proyecto
```

## Key Files

| File | Purpose |
|------|---------|
| `entrenar_xvector.py` | X-Vector training |
| `entrenar_ecapa.py` | ECAPA-TDNN training |
| `entrenar_feedforward.py` | FeedForward training |
| `inferir.py` | Inference with ensemble voting |
| `generar_splits.py` | Stratified split generation |
| `weld_audio_classifier/models/xvector.py` | X-Vector architecture |
| `weld_audio_classifier/models/multitask.py` | ECAPA-TDNN, FeedForward |
| `utils/audio_utils.py` | Audio loading utilities |

## Git Ignore

- `*/mfcc_cache/` - MFCC feature cache
- `*/modelos/` - Trained models
- `*/graficas/` - Generated figures
- `resultados.json`, `inferencia.json`

## Rules

1. No generar logs temporales, o eliminarlos después de uso.
2. No usar comentarios de separación tipo `# === Main ===`.
3. Verificar consistencia de nombres antes de crear archivos.
