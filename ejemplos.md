# Ejemplos de Entrenamiento

Este documento contiene ejemplos de comandos para entrenar los modelos X-Vector y ECAPA-TDNN.

## Estructura de directorios

```
spectral-analysis/
├── scripts/
│   └── train_pytorch.py
├── mfcc_baseline/
│   └── models/
├── outputs/                      # Carpeta de salida (se crea automáticamente)
│   └── 10seg/
│       └── k05_overlap_0.5/
│           ├── model_fold_0.pt
│           ├── model_fold_1.pt
│           └── results_kfold.json
```

---

## Parámetros Principales

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `--splits-dir` | Ruta a la carpeta con CSVs de splits | `splits` |
| `--output-dir` | Ruta de salida para modelos | `outputs` |
| `--audio-root` | Ruta a archivos de audio (opcional) | Por defecto usa `DEFAULT_AUDIO_ROOT` |
| `--duration` | Duración del segmento en segundos | `1, 2, 5, 10, 15, 20, 30, 50` |
| `--overlap` | Ratio de overlap entre segmentos | `0.0, 0.25, 0.5, 0.75` |
| `--k-fold` | Número de folds para cross-validation | `1, 5, 10` |
| `--model-type` | Tipo de modelo | `xvector`, `ecapa` |
| `--task` | Tarea a entrenar | `Plate Thickness`, `Electrode`, `Type of Current` |
| `--epochs` | Número de épocas | `50` |
| `--batch-size` | Tamaño del batch | `64` |
| `--lr` | Learning rate | `0.001` |
| `--embedding-dim` | Dimensión del embedding | `512` (xvector), `192` (ecapa) |

---

## Ejemplos de Comandos

### 1. X-Vector - K-Fold Cross Validation

Entrenar X-Vector con 5-fold cross-validation:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector \
  --task "Plate Thickness"
```

**Salida:**
```
outputs/10seg/k05_overlap_0.0/
├── model_fold_0.pt
├── model_fold_1.pt
├── model_fold_2.pt
├── model_fold_3.pt
├── model_fold_4.pt
└── results_kfold.json
```

---

### 2. X-Vector - Con Overlap

Entrenar con overlap de 50%:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.5 \
  --model-type xvector \
  --task "Electrode"
```

**Salida:**
```
outputs/10seg/k05_overlap_0.5/
├── model_fold_0.pt
├── ...
└── results_kfold.json
```

---

### 3. ECAPA-TDNN - K-Fold

Usar ECAPA-TDNN en lugar de X-Vector:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type ecapa \
  --embedding-dim 192 \
  --task "Type of Current"
```

---

### 4. Distintas Duraciones

Entrenar con diferentes duraciones de audio:

```bash
# 1 segundo
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 1 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector

# 5 segundos
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 5 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector

# 30 segundos
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 30 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector
```

---

### 5. Entrenamiento Simple (Sin K-Fold)

Usar `--k-fold 1` para entrenamiento sin cross-validation:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 1 \
  --overlap 0.0 \
  --model-type xvector \
  --task "Plate Thickness"
```

**Salida:**
```
outputs/10seg/k01_overlap_0.0/
├── best_model.pt
└── results.json
```

---

### 6. Especificar Ruta de Audio

Cambiar la ruta de los archivos de audio:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --audio-root /home/luis/otro/proyecto/audio \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector
```

---

### 7. Comparar Modelos

Comparar X-Vector vs ECAPA-TDNN:

```bash
# X-Vector
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
  --overlap 0.0 \
  --model-type ecapa \
  --embedding-dim 192 \
  --task "Plate Thickness"
```

---

### 8. Cambiar Hiperparámetros

Personalizar hyperparameters:

```bash
PYTHONPATH=. python scripts/train_pytorch.py \
  --splits-dir splits \
  --output-dir outputs \
  --duration 10 \
  --k-fold 5 \
  --overlap 0.0 \
  --model-type xvector \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0005 \
  --embedding-dim 256 \
  --task "Plate Thickness"
```

---

## Formato de Carpetas de Salida

El formato de las carpetas de salida es:

```
{output-dir}/{duration:02d}seg/k{k_fold:02d}_overlap_{overlap}/
```

Ejemplos:
- `outputs/01seg/k05_overlap_0.0/` - 1 segundo, 5-fold, sin overlap
- `outputs/10seg/k05_overlap_0.5/` - 10 segundos, 5-fold, 50% overlap
- `outputs/30seg/k10_overlap_0.75/` - 30 segundos, 10-fold, 75% overlap

---

## Resultados

Los resultados se guardan en `results_kfold.json` con el siguiente formato:

```json
{
  "model": "xvector",
  "task": "Plate Thickness",
  "k_fold": 5,
  "mean_accuracy": 0.85,
  "std_accuracy": 0.02,
  "fold_results": [
    {"fold": 0, "best_accuracy": 0.87},
    {"fold": 1, "best_accuracy": 0.84},
    ...
  ],
  "total_time_seconds": 120.5,
  "config": {
    "embedding_dim": 512,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "overlap": 0.0
  }
}
```

---

## Notas

- Usar `PYTHONPATH=.` si los módulos no se encuentran
- Por defecto usa `DEFAULT_AUDIO_ROOT` definido en el script
- Los modelos se guardan en formato PyTorch (`.pt`)
- El mejor modelo se selecciona por accuracy de validación
