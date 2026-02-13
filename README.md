# Soldadura Audio MFCC Baseline

Baseline espectral para comparar contra VGGish, usando MFCC + clasificadores clasicos.

## Objetivo

- Comparar desempeno por duracion y overlap con el pipeline VGGish.
- Usar los mismos splits y el mismo esquema de segmentacion on-the-fly.

## Estructura

```
.
├── mfcc_baseline/
│   ├── audio_paths.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── metrics.py
│   ├── segmenter.py
│   ├── splits.py
│   └── train.py
├── scripts/
│   ├── generate_splits.py
│   └── train_eval.py
├── config.yaml
└── requirements.txt
```

## Configuracion

1. Ajusta `config.yaml` con la ruta al directorio `audio/` del repo VGGish.
2. (Opcional) define `AUDIO_ROOT` como variable de entorno para sobreescribir `config.yaml`.

## Generar splits (10seg, overlap 0.5)

```
python scripts/generate_splits.py --duration 10 --overlap 0.5 --output-dir splits/10seg
```

## Entrenar y evaluar baseline

```
python scripts/train_eval.py \
  --duration 10 \
  --overlap 0.5 \
  --splits-dir splits/10seg \
  --output-dir outputs/10seg/overlap_0.5 \
  --model svm
```

## Referencias

- TODO: agregar citas a articulos sobre MFCC y clasificacion de audio para justificar el baseline.
