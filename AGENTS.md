# AGENTS.md - Agent Coding Guidelines

This file provides guidelines for agents working on this codebase.

## Project Overview

Audio spectral analysis project for weld sound classification using MFCC features and deep learning models (X-Vector, ECAPA-TDNN).

## Project Structure

```
spectral-analysis/
├── mfcc_baseline/
│   ├── models/           # X-Vector and ECAPA-TDNN PyTorch models
│   ├── features.py       # MFCC feature extraction
│   ├── segmenter.py      # Audio segmentation
│   ├── dataset.py        # Data loading utilities
│   ├── train.py          # Classic ML training (SVM, RF)
│   └── pytorch_dataset.py
├── scripts/
│   ├── train_pytorch.py  # PyTorch training script with k-fold CV
│   └── generate_splits.py
├── config.yaml           # Configuration file
├── requisitos.txt        # Python dependencies
└── ejemplos.md          # Usage examples
```

---

## Build/Lint/Test Commands

### Running the Project

```bash
# Set Python path
export PYTHONPATH=.

# Or run with inline path
PYTHONPATH=. python scripts/train_pytorch.py --help
```

### Training Examples

```bash
# X-Vector with 5-fold cross-validation
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector \
  --task "Plate Thickness"

# ECAPA-TDNN
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --model-type ecapa \
  --embedding-dim 192
```

### Testing Models

```bash
# Test model imports and shapes
PYTHONPATH=. python -c "
from mfcc_baseline.models import XVectorModel, ECAPA_TDNN
import torch

xvec = XVectorModel(input_size=40, embedding_size=512)
x = torch.randn(2, 40, 100)
out = xvec(x)
print(f'X-Vector: {out.shape}')

ecapa = ECAPA_TDNN(input_size=40, lin_neurons=192)
out = ecapa(x)
print(f'ECAPA: {out.shape}')
"
```

### Linting

No formal linter is configured. Manual code review is recommended. Key files to check:
- Type hints consistency
- Import organization
- Code formatting

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
from mfcc_baseline.config import load_config
from mfcc_baseline.dataset import FeatureConfig
from mfcc_baseline.models import XVectorModel
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

1. **Model forward pass**: Accept raw tensor, return logits or embeddings
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

### Configuration

Store configurable paths as constants at the top of files:

```python
# ============================================================
# RUTA DE AUDIOS - Cambiar aquí la ruta de los archivos de audio
# ============================================================
DEFAULT_AUDIO_ROOT = Path("/home/luis/projects/tesis/audio/soldadura/audio")
```

### Documentation

- Use docstrings for public functions and classes
- Keep docstrings concise but informative
- Include parameter types and return values

```python
def load_features(
    csv_path: Path,
    audio_root: Path,
    segment_duration: float,
    overlap_ratio: float,
    sample_rate: int,
    feature_cfg: FeatureConfig,
    cache_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load and extract MFCC features from audio segments.
    
    Args:
        csv_path: Path to CSV with audio file paths and labels
        audio_root: Root directory containing audio files
        segment_duration: Duration of each audio segment in seconds
        overlap_ratio: Overlap ratio between segments (0.0-1.0)
        sample_rate: Audio sample rate
        feature_cfg: Feature extraction configuration
        cache_dir: Directory for caching extracted features
    
    Returns:
        Tuple of (features array, dataframe with labels)
    """
```

### Git Ignore

Ensure sensitive files are ignored:
- `.cache_features/` - Feature cache
- `outputs/` - Model outputs
- `.pyc__` - Python bytecode
- `.env` - Environment variables

---

## Running Tests

Currently there are no formal tests. To manually test:

```bash
# Test model imports
PYTHONPATH=. python -c "from mfcc_baseline.models import XVectorModel, ECAPA_TDNN"

# Test training script help
PYTHONPATH=. python scripts/train_pytorch.py --help

# Test data loading
PYTHONPATH=. python -c "
from mfcc_baseline.config import load_config
cfg = load_config()
print(f'Audio root: {cfg.audio_root}')
"
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `mfcc_baseline/models/xvector.py` | X-Vector TDNN implementation |
| `mfcc_baseline/models/ecapa_tdnn.py` | ECAPA-TDNN implementation |
| `mfcc_baseline/features.py` | MFCC extraction |
| `scripts/train_pytorch.py` | Main training script |
| `config.yaml` | Configuration |

---

## Common Issues

1. **Module not found**: Use `PYTHONPATH=.` prefix
2. **Audio files not found**: Check `DEFAULT_AUDIO_ROOT` in training script
3. **CUDA out of memory**: Reduce `--batch-size`
4. **Missing splits**: Run `scripts/generate_splits.py` first
